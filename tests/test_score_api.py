import pytest
import torch
import torch.nn.functional as F

from rkv import R1KV
from rkv.utils import cal_similarity, compute_attention_scores


def test_score_kv_matches_reference_formula():
    torch.manual_seed(0)
    window = 4
    policy = R1KV(
        budget=12,
        window_size=window,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
    )
    keys = torch.randn(2, 2, 24, 8)
    queries = torch.randn(2, 4, window, 8)

    attn = compute_attention_scores(queries, keys)
    importance = (
        torch.softmax(
            attn[:, :, -window:, :-window],
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
    )[:, :, :-window]
    expected = importance * policy.mix_lambda - redundancy * (
        1 - policy.mix_lambda
    )

    actual = policy.score_kv(keys, queries)
    assert torch.equal(actual, expected)
    assert actual.shape == (2, 2, 20)


def test_update_kv_selection_matches_score_kv():
    torch.manual_seed(1)
    window = 4
    budget = 12
    policy = R1KV(
        budget=budget,
        window_size=window,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
    )
    keys = torch.randn(1, 2, 24, 8)
    queries = torch.randn(1, 4, window, 8)
    values = torch.randn_like(keys)

    scores = policy.score_kv(keys, queries)
    kept = scores.topk(budget - window, dim=-1).indices
    gather_idx = kept.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    expected_keys = torch.cat(
        [
            keys[:, :, :-window, :].gather(2, gather_idx),
            keys[:, :, -window:, :],
        ],
        dim=2,
    )
    expected_values = torch.cat(
        [
            values[:, :, :-window, :].gather(2, gather_idx),
            values[:, :, -window:, :],
        ],
        dim=2,
    )

    actual_keys, actual_values = policy.update_kv(keys, queries, values)
    assert torch.equal(actual_keys, expected_keys)
    assert torch.equal(actual_values, expected_values)


def test_select_kept_positions_owns_global_serving_selection():
    torch.manual_seed(2)
    window = 4
    budget = 12
    policy = R1KV(
        budget=budget,
        window_size=window,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
    )
    layer_keys = [torch.randn(1, 2, 24, 8) for _ in range(3)]
    layer_queries = [torch.randn(1, 4, window, 8) for _ in range(3)]

    shared_scores = None
    for keys, queries in zip(layer_keys, layer_queries, strict=True):
        layer_score = policy.score_kv(keys, queries).mean(dim=1)[0]
        shared_scores = (
            layer_score if shared_scores is None else shared_scores + layer_score
        )

    assert shared_scores is not None
    past_idx = shared_scores.topk(budget - window, dim=-1).indices
    window_idx = torch.arange(24 - window, 24)
    expected = torch.sort(torch.cat([past_idx, window_idx], dim=-1)).values

    actual = policy.select_kept_positions(layer_keys, layer_queries)
    assert torch.equal(actual, expected)
    assert actual.shape == (budget,)
    assert policy.observation_window_tokens == window


def test_select_kept_positions_matches_legacy_retained_set_single_head():
    torch.manual_seed(3)
    policy = R1KV(
        budget=12,
        window_size=4,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
    )
    keys = torch.randn(1, 1, 24, 8)
    queries = torch.randn(1, 1, 4, 8)
    values = torch.randn_like(keys)

    legacy_keys, _ = policy.update_kv(keys, queries, values)
    legacy_positions = []
    for token in legacy_keys[0, 0]:
        matches = torch.all(keys[0, 0] == token, dim=-1).nonzero().flatten()
        assert matches.numel() == 1
        legacy_positions.append(int(matches.item()))

    expected = torch.tensor(sorted(legacy_positions))
    actual = policy.select_kept_positions([keys], [queries])
    assert torch.equal(actual, expected)


def test_select_kept_positions_rejects_non_finite_scores():
    policy = R1KV(
        budget=12,
        window_size=4,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
    )
    keys = torch.randn(1, 2, 24, 8)
    queries = torch.randn(1, 4, 4, 8)

    policy.score_kv = lambda *_: torch.full((1, 2, 20), float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        policy.select_kept_positions([keys], [queries])


def test_should_observe_query_tracks_only_the_scoring_window():
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


def test_should_compact_owns_serving_trigger_policy():
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
        "query_window_tokens": 8,
    }
    assert not policy.should_compact(num_decoded_tokens=127, **common)
    assert policy.should_compact(num_decoded_tokens=128, **common)
    assert not policy.should_compact(
        resident_len=383,
        num_decoded_tokens=128,
        num_new_tokens=1,
        is_genuine_decode=True,
        query_window_tokens=8,
    )
    assert not policy.should_compact(
        resident_len=384,
        num_decoded_tokens=128,
        num_new_tokens=1,
        is_genuine_decode=False,
        query_window_tokens=8,
    )
    assert not policy.should_compact(
        resident_len=384,
        num_decoded_tokens=128,
        num_new_tokens=1,
        is_genuine_decode=True,
        query_window_tokens=7,
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
    rkv = R1KV.from_serving_config(
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

    assert rkv.budget == 64
    assert rkv.buffer == 40
    assert rkv.window_size == 4
    assert rkv.kernel_size == 5
    assert rkv.mix_lambda == 0.25
    assert rkv.retain_ratio == 0.2
    assert rkv.retain_direction == "first"


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
