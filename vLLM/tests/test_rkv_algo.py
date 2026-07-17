"""CPU parity tests for the pure R-KV algorithm port (vLLM).

GPU-free and self-contained: an inline *reference* copy of the original R-KV
algorithm is compared against ``vLLM/rkv/algo.py``. The algorithm module is
loaded directly by file path so the test does not require ``vllm`` (or its
serving stack) to be installed.

Run directly::

    python tests/test_rkv_algo.py

or under a test runner (unittest / pytest).
"""

import importlib.util
import math
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the pure algorithm module directly by file path (no ``vllm`` import).
_ALGO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rkv", "algo.py")
)
_spec = importlib.util.spec_from_file_location("rkv_algo", _ALGO_PATH)
_algo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_algo)
R1KV = _algo.R1KV
cal_similarity = _algo.cal_similarity
compute_attention_scores = _algo.compute_attention_scores


# --------------------------------------------------------------------------- #
# Reference implementation (verbatim from the R-KV repo).                      #
# --------------------------------------------------------------------------- #
def ref_compute_attention_scores(query_states, key_states, pooling="max"):
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    if query_group_size == 1:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
    else:
        query_states = query_states.view(
            batch_size, kv_heads, query_group_size, q_len, head_dim
        )
        key_states = key_states.unsqueeze(2)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 4)
        ) / math.sqrt(head_dim)
        if pooling == "mean":
            attn_weights = attn_weights.mean(dim=2)
        elif pooling == "max":
            attn_weights = attn_weights.max(dim=2).values
        else:
            raise ValueError("Pooling method not supported")
    return attn_weights


def ref_cal_similarity(key_states, threshold=0.5, retain_ratio=0.2,
                       retain_direction="last"):
    _, _, seq_len, _ = key_states.shape
    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_states.device)
    similarity_cos.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)
    similarity_mask = similarity_cos > threshold
    k = max(1, int(seq_len * retain_ratio))
    indices = torch.where(
        similarity_mask,
        torch.arange(seq_len, device=similarity_mask.device).view(1, 1, 1, seq_len),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )
    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]
    else:
        raise ValueError("retain_direction not supported")
    similarity_cos.scatter_(-1, similarity_retain.unsqueeze(-1), 0)
    return similarity_cos.mean(dim=-2).softmax(dim=-1)


class RefR1KV:
    def __init__(self, budget=128, window_size=8, kernel_size=7, mix_lambda=0.07,
                 retain_ratio=0.1, retain_direction="last", **kwargs):
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction

    def update_kv(self, key_states, query_states, value_states):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]
        if kv_cache_len < self.budget:
            return key_states, value_states
        attn_weights = ref_compute_attention_scores(query_states, key_states)
        attn_weights_sum = (
            nn.functional.softmax(
                attn_weights[:, :, -self.window_size:, : -self.window_size],
                dim=-1, dtype=torch.float32,
            ).mean(dim=-2).to(query_states.dtype)
        )
        attn_cache = F.max_pool1d(
            attn_weights_sum, kernel_size=self.kernel_size,
            padding=self.kernel_size // 2, stride=1,
        )
        similarity_cos = ref_cal_similarity(
            key_states, retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )[:, :, : -self.window_size]
        final_score = attn_cache * self.mix_lambda - similarity_cos * (
            1 - self.mix_lambda
        )
        indices = final_score.topk(self.budget - self.window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past = key_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
        v_past = value_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        return (torch.cat([k_past, k_cur], dim=2),
                torch.cat([v_past, v_cur], dim=2))


class TestR1KVParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _rand(self, bsz, kv_heads, q_heads, kv_len, head_dim):
        q = torch.randn(bsz, q_heads, kv_len, head_dim)
        k = torch.randn(bsz, kv_heads, kv_len, head_dim)
        v = torch.randn(bsz, kv_heads, kv_len, head_dim)
        return q, k, v

    def test_parity_mha(self):
        q, k, v = self._rand(1, 4, 4, 200, 16)
        cfg = dict(budget=64, window_size=8)
        rk = R1KV(**cfg).update_kv(k, q, v)
        rf = RefR1KV(**cfg).update_kv(k.clone(), q.clone(), v.clone())
        self.assertTrue(torch.equal(rk[0], rf[0]))
        self.assertTrue(torch.equal(rk[1], rf[1]))

    def test_parity_gqa(self):
        q, k, v = self._rand(1, 2, 8, 160, 16)
        cfg = dict(budget=48, window_size=8)
        rk = R1KV(**cfg).update_kv(k, q, v)
        rf = RefR1KV(**cfg).update_kv(k.clone(), q.clone(), v.clone())
        self.assertTrue(torch.equal(rk[0], rf[0]))
        self.assertTrue(torch.equal(rk[1], rf[1]))

    def test_compresses_to_budget(self):
        q, k, v = self._rand(1, 4, 4, 300, 16)
        out_k, out_v = R1KV(budget=128, window_size=8).update_kv(k, q, v)
        self.assertEqual(out_k.shape[2], 128)
        self.assertEqual(out_v.shape[2], 128)

    def test_noop_below_budget(self):
        q, k, v = self._rand(1, 4, 4, 40, 16)
        out_k, out_v = R1KV(budget=128, window_size=8).update_kv(k, q, v)
        self.assertTrue(torch.equal(out_k, k))
        self.assertTrue(torch.equal(out_v, v))

    def test_keeps_trailing_window(self):
        q, k, v = self._rand(1, 1, 1, 100, 16)
        budget, window = 64, 8
        out_k, _ = R1KV(budget=budget, window_size=window).update_kv(k, q, v)
        # The trailing observation window is always retained verbatim.
        self.assertTrue(torch.equal(out_k[:, :, -window:, :], k[:, :, -window:, :]))

    def test_gqa_requires_divisible_heads(self):
        # q_heads not a multiple of kv_heads must raise, not silently mis-group.
        q = torch.randn(1, 6, 4, 16)   # 6 q heads
        k = torch.randn(1, 4, 8, 16)   # 4 kv heads, 6 % 4 != 0
        with self.assertRaises(ValueError):
            compute_attention_scores(q, k)

    def test_retain_first_finite_and_directional(self):
        # The fixed "first" retain uses a seq_len sentinel so it picks the
        # smallest MATCHED index (the old bug collapsed to position 0), so it
        # must differ from "last" on a mutually-similar key set. (The "*_percent"
        # modes are disabled at the config layer; only "last"/"first" are valid.)
        torch.manual_seed(0)
        seq = 6
        base = torch.randn(1, 1, 1, 16)
        key = base.expand(1, 1, seq, 16) + 0.02 * torch.randn(1, 1, seq, 16)
        last = cal_similarity(key.clone(), threshold=0.0, retain_direction="last")
        out = cal_similarity(key.clone(), threshold=0.0, retain_direction="first")
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.allclose(out, last))

    def test_retain_first_no_match_rows_do_not_crash(self):
        # Orthogonal keys + high threshold -> no matches -> every row holds the
        # out-of-range sentinel; the clamp must keep the scatter in bounds.
        key = torch.randn(1, 1, 8, 16)
        out = cal_similarity(key.clone(), threshold=0.999, retain_direction="first")
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(tuple(out.shape), (1, 1, 8))


if __name__ == "__main__":
    unittest.main()
