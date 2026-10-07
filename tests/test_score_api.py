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


def test_query_history_is_owned_and_ordered_inside_rkv():
    policy = _policy(window=4)
    for step in range(6):
        query = torch.full((1, 2, 8), float(step))
        policy.observe_query("layer", query)

    actual = policy._serving_query_window("layer")
    assert actual.shape == (1, 2, 4, 8)
    assert torch.equal(
        actual[0, 0, :, 0],
        torch.tensor([2.0, 3.0, 4.0, 5.0]),
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
        for name in layer_keys:
            policy.observe_query(
                name,
                observed[name][:, :, step, :].permute(0, 1, 2),
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
        policy.observe_query("layer", queries[:, :, step, :])

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


def test_should_observe_query_tracks_only_rkv_scoring_window():
    policy = R1KV(
        budget=256,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
        buffer=128,
    )

    for decoded in range(1, 121):
        assert not policy.should_observe_query(
            num_decoded_tokens=decoded,
            num_new_tokens=1,
            is_genuine_decode=True,
        )
    for decoded in range(121, 129):
        assert policy.should_observe_query(
            num_decoded_tokens=decoded,
            num_new_tokens=1,
            is_genuine_decode=True,
        )
    assert not policy.should_observe_query(
        num_decoded_tokens=129,
        num_new_tokens=1,
        is_genuine_decode=True,
    )
    assert not policy.should_observe_query(
        num_decoded_tokens=128,
        num_new_tokens=1,
        is_genuine_decode=False,
    )


def test_should_compact_owns_rkv_trigger_and_readiness():
    policy = R1KV(
        budget=256,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
        buffer=128,
    )
    common = {
        "resident_len": 384,
        "num_new_tokens": 1,
        "is_genuine_decode": True,
    }

    assert not policy.should_compact(num_decoded_tokens=128, **common)
    for _ in range(policy.window_size):
        policy.observe_query("layer", torch.randn(1, 4, 8))

    assert not policy.should_compact(num_decoded_tokens=127, **common)
    assert policy.should_compact(num_decoded_tokens=128, **common)
    assert not policy.should_compact(
        resident_len=383,
        num_decoded_tokens=128,
        num_new_tokens=1,
        is_genuine_decode=True,
    )
    assert not policy.should_compact(
        resident_len=384,
        num_decoded_tokens=128,
        num_new_tokens=1,
        is_genuine_decode=False,
    )


def test_existing_positional_constructor_binding_is_unchanged():
    policy = R1KV(128, 8, 7, 0.07, 0.1, "last", True)
    assert policy.record_kept_token_indices is True
    assert policy.buffer == 128


def test_serving_config_defaults_match_vllm_port_without_changing_legacy_defaults():
    legacy = R1KV()
    serving = R1KV.from_serving_config(
        {
            "budget": 128,
            "buffer": 128,
        }
    )

    assert legacy.mix_lambda == 0.07
    assert serving.budget == 128
    assert serving.buffer == 128
    assert serving.window_size == 8
    assert serving.kernel_size == 7
    assert serving.mix_lambda == 0.1
    assert serving.retain_ratio == 0.1
    assert serving.retain_direction == "last"


def test_serving_config_overrides_match_vllm_port_algorithm_knobs():
    policy = R1KV.from_serving_config(
        {
            "budget": 64,
            "buffer": 40,
            "window_size": 4,
            "kernel_size": 5,
            "mix_lambda": 0.25,
            "retain_ratio": 0.2,
            "retain_direction": "first",
        }
    )

    assert policy.budget == 64
    assert policy.buffer == 40
    assert policy.window_size == 4
    assert policy.kernel_size == 5
    assert policy.mix_lambda == 0.25
    assert policy.retain_ratio == 0.2
    assert policy.retain_direction == "first"


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ([], "mapping"),
        ({"budget": 32}, "Missing required"),
        ({"buffer": 16}, "Missing required"),
        ({"budget": 0, "buffer": 16}, "positive integer"),
        ({"budget": True, "buffer": 16}, "positive integer"),
        ({"budget": 8, "buffer": 16}, "greater than window_size"),
        ({"budget": 32, "buffer": 4}, ">= window_size"),
        ({"budget": 32, "buffer": 16, "window_size": 0}, "window_size"),
        ({"budget": 32, "buffer": 16, "kernel_size": 4}, "odd integer"),
        ({"budget": 32, "buffer": 16, "mix_lambda": 2.0}, "mix_lambda"),
        ({"budget": 32, "buffer": 16, "retain_ratio": 0}, "retain_ratio"),
        (
            {"budget": 32, "buffer": 16, "retain_direction": "middle"},
            "retain_direction",
        ),
        ({"budget": 32, "buffer": 16, "unknown": 1}, "Unsupported"),
    ],
)
def test_serving_config_validation(config, match):
    with pytest.raises(ValueError, match=match):
        R1KV.from_serving_config(config)
