import pytest
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


def test_select_kept_positions_owns_global_serving_selection():
    torch.manual_seed(2)
    policy = _policy()
    layer_keys = {
        f"layer-{i}": torch.randn(1, 2, 24, 8)
        for i in range(3)
    }
    observed = {
        name: torch.randn(1, 4, policy.window_size, 8)
        for name in layer_keys
    }
    for step in range(policy.window_size):
        policy.observe_query(
            {
                name: observed[name][:, :, step, :]
                for name in layer_keys
            }
        )

    shared_scores = None
    for name, keys in layer_keys.items():
        scores = _reference_scores(policy, keys, observed[name]).mean(dim=1)[0]
        shared_scores = scores if shared_scores is None else shared_scores + scores

    assert shared_scores is not None
    past_idx = shared_scores.topk(
        policy.budget - policy.window_size,
        dim=-1,
    ).indices
    window_idx = torch.arange(24 - policy.window_size, 24)
    expected = torch.sort(torch.cat([past_idx, window_idx], dim=-1)).values

    actual = policy.select_kept_positions(layer_keys)
    assert torch.equal(actual, expected)
    assert actual.shape == (policy.budget,)

def test_select_kept_positions_matches_legacy_retained_set_single_head():
    torch.manual_seed(3)
    policy = _policy()
    keys = torch.randn(1, 1, 24, 8)
    queries = torch.randn(1, 1, policy.window_size, 8)
    values = torch.randn_like(keys)

    for step in range(policy.window_size):
        policy.observe_query({"layer": queries[:, :, step, :]})

    legacy_keys, _ = policy.update_kv(keys, queries, values)
    legacy_positions = []
    for token in legacy_keys[0, 0]:
        matches = torch.all(keys[0, 0] == token, dim=-1).nonzero().flatten()
        assert matches.numel() == 1
        legacy_positions.append(int(matches.item()))

    expected = torch.tensor(sorted(legacy_positions))
    actual = policy.select_kept_positions({"layer": keys})
    assert torch.equal(actual, expected)

def test_select_kept_positions_requires_algorithm_owned_query_history():
    policy = _policy()
    keys = torch.randn(1, 2, 24, 8)

    with pytest.raises(RuntimeError, match="full query window"):
        policy.select_kept_positions({"layer": keys})
