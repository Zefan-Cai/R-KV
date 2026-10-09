import pytest
import torch

from rkv import R1KV
from rkv.serving import RKVServing


def test_serving_policy_inherits_legacy_rkv():
    assert issubclass(RKVServing, R1KV)
    assert RKVServing().mix_lambda == R1KV().mix_lambda


def test_serving_config_defaults_do_not_change_legacy_algorithm():
    policy = RKVServing.from_serving_config({})

    assert type(policy) is RKVServing
    assert (policy.budget, policy.buffer, policy.window_size, policy.kernel_size) == (
        128, 128, 8, 7
    )
    assert policy.mix_lambda == 0.1
    assert policy.retain_ratio == 0.1
    assert policy.retain_direction == "last"
    assert R1KV().mix_lambda == 0.07
    assert policy._serving_query_history == {}
    assert policy._serving_layer_order is None


def test_serving_config_overrides():
    policy = RKVServing.from_serving_config({
        "budget": 64,
        "buffer": 40,
        "window_size": 4,
        "kernel_size": 5,
        "mix_lambda": 0.25,
        "retain_ratio": 0.2,
        "retain_direction": "first",
    })

    assert policy.budget == 64
    assert policy.buffer == 40
    assert policy.window_size == 4
    assert policy.kernel_size == 5
    assert policy.mix_lambda == 0.25
    assert policy.retain_ratio == 0.2
    assert policy.retain_direction == "first"


def test_serving_config_instances_have_independent_request_state():
    first = RKVServing.from_serving_config({})
    second = RKVServing.from_serving_config({})

    assert first is not second
    assert first._serving_query_history is not second._serving_query_history
    first._serving_query_history["layer0"] = object()
    assert second._serving_query_history == {}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ([], "must be a mapping"),
        ({"unknown": 1}, "Unsupported R-KV serving config keys"),
        ({"budget": 0}, "budget must be a positive integer"),
        ({"budget": True}, "budget must be a positive integer"),
        ({"buffer": 0}, "buffer must be a positive integer"),
        ({"buffer": False}, "buffer must be a positive integer"),
        ({"window_size": 0}, "window_size must be a positive integer"),
        ({"window_size": True}, "window_size must be a positive integer"),
        ({"budget": 8}, "budget must be greater than window_size"),
        ({"buffer": 4}, "buffer must be >= window_size"),
        ({"kernel_size": 4}, "kernel_size must be a positive odd integer"),
        ({"kernel_size": True}, "kernel_size must be a positive odd integer"),
        ({"mix_lambda": True}, "mix_lambda must be numeric"),
        ({"mix_lambda": 2.0}, "mix_lambda must be in"),
        ({"retain_ratio": False}, "retain_ratio must be numeric"),
        ({"retain_ratio": 0}, "retain_ratio must be in"),
        ({"retain_direction": "middle"}, "Unsupported retain_direction"),
        ({"retain_direction": "last_percent"}, "Unsupported retain_direction"),
        ({"retain_direction": "first_percent"}, "Unsupported retain_direction"),
    ],
)
def test_serving_config_rejects_invalid_settings(config, message):
    with pytest.raises(ValueError, match=message):
        RKVServing.from_serving_config(config)


def test_should_observe_token_queries_prefill():
    policy = RKVServing.from_serving_config({"window_size": 4, "buffer": 8})
    assert policy.should_observe_token_queries("prefill", 0) == 4


@pytest.mark.parametrize(
    ("buffer", "window_size", "expected"),
    [
        (8, 4, [0, 0, 0, 0, 1, 1, 1, 1]),
        (8, 1, [0, 0, 0, 0, 0, 0, 0, 1]),
        (4, 4, [1, 1, 1, 1]),
    ],
)
def test_should_observe_token_queries_decode_cadence(buffer, window_size, expected):
    policy = RKVServing.from_serving_config(
        {"buffer": buffer, "window_size": window_size}
    )
    assert [
        policy.should_observe_token_queries("decode", step)
        for step in range(buffer * 2)
    ] == expected * 2


def test_observe_token_queries_keeps_last_prefill_rows_per_layer():
    policy = RKVServing.from_serving_config({"window_size": 3})
    queries = {
        "layer0": torch.arange(5.0).reshape(5, 1, 1),
        "layer1": torch.arange(10.0, 15.0).reshape(5, 1, 1),
    }
    policy.observe_token_queries(queries)

    assert policy._serving_layer_order == ("layer0", "layer1")
    assert policy._serving_query_history["layer0"][:, 0, 0].tolist() == [2, 3, 4]
    assert policy._serving_query_history["layer1"][:, 0, 0].tolist() == [12, 13, 14]


def test_observe_token_queries_rolling_window_and_source_reuse():
    policy = RKVServing.from_serving_config({"window_size": 3})
    source = torch.tensor([[[1.0]]], requires_grad=True)
    policy.observe_token_queries({"layer0": source})
    with torch.no_grad():
        source.fill_(99.0)
    assert policy._serving_query_history["layer0"][0, 0, 0].item() == 1.0
    policy.observe_token_queries({"layer0": torch.tensor([[[2.0]]])})
    policy.observe_token_queries({"layer0": torch.tensor([[[3.0]], [[4.0]]])})

    history = policy._serving_query_history["layer0"]
    assert history[:, 0, 0].tolist() == [2, 3, 4]
    assert not history.requires_grad


def test_observe_token_queries_layer_mapping_order_does_not_matter():
    policy = RKVServing.from_serving_config({"window_size": 2})
    policy.observe_token_queries({
        "layer0": torch.tensor([[[1.0]]]),
        "layer1": torch.tensor([[[2.0]]]),
    })
    policy.observe_token_queries({
        "layer1": torch.tensor([[[4.0]]]),
        "layer0": torch.tensor([[[3.0]]]),
    })
    assert policy._serving_query_history["layer0"][:, 0, 0].tolist() == [1, 3]
    assert policy._serving_query_history["layer1"][:, 0, 0].tolist() == [2, 4]


def test_should_compact_kv_prefill_threshold():
    policy = RKVServing.from_serving_config({"budget": 12, "buffer": 8})
    assert not policy.should_compact_kv("prefill", 12, 0)
    assert policy.should_compact_kv("prefill", 13, 0)
    assert policy.should_compact_kv("prefill", 24, 0)


@pytest.mark.parametrize(
    ("decoded_before", "resident", "should_compact"),
    [
        (0, 20, False),
        (6, 20, False),
        (7, 19, False),
        (7, 20, True),
        (7, 21, True),
        (8, 20, False),
        (14, 20, False),
        (15, 19, False),
        (15, 20, True),
    ],
)
def test_should_compact_kv_decode_boundary_and_residency(
    decoded_before, resident, should_compact
):
    policy = RKVServing.from_serving_config({"budget": 12, "buffer": 8})
    assert (
        policy.should_compact_kv("decode", resident, decoded_before) is should_compact
    )


def test_compaction_follows_last_observation_step():
    policy = RKVServing.from_serving_config(
        {"budget": 12, "buffer": 8, "window_size": 4}
    )
    assert [
        (
            policy.should_observe_token_queries("decode", step),
            policy.should_compact_kv("decode", 20, step),
        )
        for step in range(8)
    ] == [(0, False)] * 4 + [(1, False)] * 3 + [(1, True)]


class _ReadOnlyKVView:
    def __init__(self, keys):
        self.keys = keys

    def get_keys(self):
        return self.keys

    def get_values(self):
        raise AssertionError("R-KV selection must not read V")


@pytest.mark.parametrize("resident_tokens", [20, 24])
def test_select_kept_positions_matches_legacy_kv_order(resident_tokens):
    torch.manual_seed(31 + resident_tokens)
    policy = RKVServing.from_serving_config(
        {"budget": 12, "buffer": 8, "window_size": 4}
    )
    keys_by_layer = {
        layer: torch.randn(1, 2, resident_tokens, 8)
        for layer in ("layer0", "layer1")
    }
    values_by_layer = {
        layer: torch.randn_like(keys) for layer, keys in keys_by_layer.items()
    }
    queries_by_layer = {
        layer: torch.randn(4, 4, 8) for layer in keys_by_layer
    }
    policy.observe_token_queries(queries_by_layer)
    kept = policy.select_kept_token_positions(
        {layer: _ReadOnlyKVView(keys) for layer, keys in keys_by_layer.items()}
    )

    assert set(kept) == set(keys_by_layer)
    for layer, keys in keys_by_layer.items():
        positions = kept[layer]
        assert positions.shape == (2, 12)
        assert positions.dtype == torch.long
        assert torch.equal(
            positions[:, -4:],
            torch.arange(resident_tokens - 4, resident_tokens).expand(2, -1),
        )

        gather = positions[None, :, :, None].expand(1, 2, 12, 8)
        selected_k = keys.gather(2, gather)
        selected_v = values_by_layer[layer].gather(2, gather)
        reference_k, reference_v = policy.update_kv(
            keys, queries_by_layer[layer].permute(1, 0, 2).unsqueeze(0),
            values_by_layer[layer],
        )
        assert torch.equal(selected_k, reference_k)
        assert torch.equal(selected_v, reference_v)

    assert not torch.equal(kept["layer0"], kept["layer1"])
    assert policy._serving_query_history == {}


def test_select_kept_positions_reuses_history_after_compaction():
    policy = RKVServing.from_serving_config(
        {"budget": 12, "buffer": 8, "window_size": 4}
    )
    view = {"layer0": _ReadOnlyKVView(torch.randn(1, 2, 20, 8))}
    for _ in range(2):
        policy.observe_token_queries({"layer0": torch.randn(4, 4, 8)})
        positions = policy.select_kept_token_positions(view)["layer0"]
        assert positions.shape == (2, 12)
        assert policy._serving_query_history == {}
