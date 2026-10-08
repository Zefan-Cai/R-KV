import torch
import torch.nn.functional as F

from rkv import R1KV
from rkv.utils import cal_similarity, compute_attention_scores


def _reference_scores(policy, keys, queries):
    attn = compute_attention_scores(queries, keys)
    importance = (
        torch.softmax(
            attn[:, :, -policy.window_size :, : -policy.window_size],
            dim=-1,
            dtype=torch.float32,
        )
        .mean(dim=-2)
        .to(queries.dtype)
    )
    importance = F.max_pool1d(
        importance,
        kernel_size=policy.kernel_size,
        padding=policy.kernel_size // 2,
        stride=1,
    )
    redundancy = cal_similarity(
        keys,
        retain_ratio=policy.retain_ratio,
        retain_direction=policy.retain_direction,
    )[:, :, : -policy.window_size]
    return importance * policy.mix_lambda - redundancy * (1 - policy.mix_lambda)


def _policy(*, budget=12, window=4, buffer=8):
    return R1KV(
        budget=budget,
        window_size=window,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
        buffer=buffer,
    )


def test_update_kv_selection_matches_reference_formula():
    torch.manual_seed(1)
    policy = _policy()
    keys = torch.randn(1, 2, 24, 8)
    queries = torch.randn(1, 4, policy.window_size, 8)
    values = torch.randn_like(keys)

    scores = _reference_scores(policy, keys, queries)
    kept = scores.topk(policy.budget - policy.window_size, dim=-1).indices
    gather_idx = kept.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    expected_keys = torch.cat(
        [
            keys[:, :, : -policy.window_size, :].gather(2, gather_idx),
            keys[:, :, -policy.window_size :, :],
        ],
        dim=2,
    )
    expected_values = torch.cat(
        [
            values[:, :, : -policy.window_size, :].gather(2, gather_idx),
            values[:, :, -policy.window_size :, :],
        ],
        dim=2,
    )

    actual_keys, actual_values = policy.update_kv(keys, queries, values)
    assert torch.equal(actual_keys, expected_keys)
    assert torch.equal(actual_values, expected_values)
