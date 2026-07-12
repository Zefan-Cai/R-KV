"""CPU parity tests for the pure R-KV algorithm port.

These tests are GPU-free and self-contained: an inline *reference* copy of the
original R-KV algorithm (from ``rkv/utils.py`` + ``rkv/compression/r1_kv.py``,
with the analysis-only ``record_kept_token_indices`` branch omitted since it is
disabled by default) is compared against
``sglang.srt.mem_cache.rkv.algo``.

Run directly::

    python test/srt/mem_cache/test_rkv_algo.py

or under a test runner (unittest / pytest).
"""

import importlib.util
import math
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the pure algorithm module directly by file path. It is intentionally free
# of SGLang dependencies, so we bypass ``sglang/__init__.py`` (which pulls in the
# full serving stack) to keep this test GPU-free and dependency-light.
_ALGO_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "rkv",
    "algo.py",
)
_spec = importlib.util.spec_from_file_location("rkv_algo", os.path.abspath(_ALGO_PATH))
_algo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_algo)
R1KV = _algo.R1KV


# --------------------------------------------------------------------------- #
# Reference implementation (verbatim from the R-KV repo, analysis branch off)  #
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


def ref_cal_similarity(
    key_states, threshold=0.5, retain_ratio=0.2, retain_direction="last"
):
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
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        **kwargs,
    ):
        assert budget - window_size > 0
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
                attn_weights[:, :, -self.window_size :, : -self.window_size],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)
            .to(query_states.dtype)
        )
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        similarity_cos = ref_cal_similarity(
            key_states,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )[:, :, : -self.window_size]
        final_score = attn_cache * self.mix_lambda - similarity_cos * (
            1 - self.mix_lambda
        )
        indices = final_score.topk(self.budget - self.window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
CONFIGS = [
    # (bsz, q_heads, kv_heads, seq_len, head_dim, budget, window_size)
    dict(
        bsz=1,
        q_heads=8,
        kv_heads=8,
        seq_len=256,
        head_dim=64,
        budget=128,
        window_size=8,
    ),
    dict(
        bsz=1,
        q_heads=32,
        kv_heads=8,
        seq_len=300,
        head_dim=128,
        budget=192,
        window_size=16,
    ),  # GQA
    dict(
        bsz=2,
        q_heads=8,
        kv_heads=8,
        seq_len=200,
        head_dim=64,
        budget=100,
        window_size=8,
    ),
    dict(
        bsz=1,
        q_heads=16,
        kv_heads=16,
        seq_len=100,
        head_dim=64,
        budget=128,
        window_size=8,
    ),  # below budget
]

KWARGS = dict(kernel_size=7, mix_lambda=0.1, retain_ratio=0.1, retain_direction="last")


def _make_inputs(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(
        cfg["bsz"], cfg["q_heads"], cfg["seq_len"], cfg["head_dim"], generator=g
    )
    k = torch.randn(
        cfg["bsz"], cfg["kv_heads"], cfg["seq_len"], cfg["head_dim"], generator=g
    )
    v = torch.randn(
        cfg["bsz"], cfg["kv_heads"], cfg["seq_len"], cfg["head_dim"], generator=g
    )
    return q, k, v


class TestRKVAlgoParity(unittest.TestCase):
    def test_update_kv_matches_reference(self):
        for i, cfg in enumerate(CONFIGS):
            with self.subTest(config=i):
                q, k, v = _make_inputs(cfg, seed=1000 + i)
                mine = R1KV(
                    budget=cfg["budget"], window_size=cfg["window_size"], **KWARGS
                )
                ref = RefR1KV(
                    budget=cfg["budget"], window_size=cfg["window_size"], **KWARGS
                )

                mk, mv = mine.update_kv(k.clone(), q.clone(), v.clone())
                rk, rv = ref.update_kv(k.clone(), q.clone(), v.clone())

                self.assertEqual(mk.shape, rk.shape)
                self.assertEqual(mv.shape, rv.shape)
                self.assertTrue(
                    torch.allclose(mk, rk, atol=1e-6, rtol=0),
                    msg=f"key mismatch cfg {i}",
                )
                self.assertTrue(
                    torch.allclose(mv, rv, atol=1e-6, rtol=0),
                    msg=f"value mismatch cfg {i}",
                )

    def test_below_budget_is_noop(self):
        cfg = CONFIGS[3]  # seq_len < budget
        q, k, v = _make_inputs(cfg, seed=7)
        mine = R1KV(budget=cfg["budget"], window_size=cfg["window_size"], **KWARGS)
        mk, mv = mine.update_kv(k.clone(), q.clone(), v.clone())
        self.assertTrue(torch.equal(mk, k))
        self.assertTrue(torch.equal(mv, v))
        self.assertIsNone(mine.select_indices(k, q))

    def test_select_indices_shape_and_membership(self):
        for i, cfg in enumerate(CONFIGS):
            if cfg["seq_len"] < cfg["budget"]:
                continue
            with self.subTest(config=i):
                q, k, v = _make_inputs(cfg, seed=2000 + i)
                mine = R1KV(
                    budget=cfg["budget"], window_size=cfg["window_size"], **KWARGS
                )

                idx = mine.select_indices(k, q, sort=True)
                self.assertEqual(
                    idx.shape, (cfg["bsz"], cfg["kv_heads"], cfg["budget"])
                )
                # sorted ascending, unique, within range
                self.assertTrue(torch.all(idx[..., 1:] > idx[..., :-1]))
                self.assertTrue(int(idx.min()) >= 0)
                self.assertTrue(int(idx.max()) < cfg["seq_len"])

                # kept set must equal the tokens produced by update_kv:
                # gather K at the selected indices and compare (as a set) to the
                # compressed keys from update_kv.
                mk, _ = mine.update_kv(k.clone(), q.clone(), v.clone())
                gathered = torch.gather(
                    k,
                    dim=2,
                    index=idx.unsqueeze(-1).expand(-1, -1, -1, cfg["head_dim"]),
                )
                # sort both along seq dim by a stable key to compare sets
                self.assertEqual(gathered.shape, mk.shape)
                gs, _ = torch.sort(
                    gathered.reshape(*gathered.shape[:2], -1).sum(-1), dim=-1
                )
                ms, _ = torch.sort(mk.reshape(*mk.shape[:2], -1).sum(-1), dim=-1)
                self.assertTrue(torch.allclose(gs, ms, atol=1e-4))

    def test_window_always_kept(self):
        cfg = CONFIGS[0]
        q, k, v = _make_inputs(cfg, seed=99)
        mine = R1KV(budget=cfg["budget"], window_size=cfg["window_size"], **KWARGS)
        idx = mine.select_indices(k, q, sort=True)
        window = set(range(cfg["seq_len"] - cfg["window_size"], cfg["seq_len"]))
        for b in range(cfg["bsz"]):
            for h in range(cfg["kv_heads"]):
                kept = set(idx[b, h].tolist())
                self.assertTrue(window.issubset(kept))


class TestFusedValidationMode(unittest.TestCase):
    """R4: the --rkv-fused-validation policy latched on R1KV construction."""

    def test_off_forces_reference(self):
        algo = R1KV(budget=64, window_size=8, fused_validation="off")
        # "off" latches the reference fallback immediately (never uses fused).
        self.assertIs(algo._fused_redundancy, False)

    def test_default_is_lazy(self):
        algo = R1KV(budget=64, window_size=8)  # default "first-request"
        self.assertIsNone(algo._fused_redundancy)

    def test_warmup_is_noop_on_cpu(self):
        # startup warmup must not crash and must no-op off CUDA (the tests are
        # CPU-only); the decision stays lazy for the CPU reference path.
        algo = R1KV(budget=64, window_size=8, fused_validation="startup")
        algo.warmup_fused_kernel(
            kv_heads=2, head_dim=16, device="cpu", dtype=torch.float32
        )
        self.assertIsNone(algo._fused_redundancy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
