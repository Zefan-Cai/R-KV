"""Request-local serving API and configuration contract tests."""

import pytest
import torch

from rkv import R1KV
from rkv.serving import RKVServing


def test_serving_api_is_separate_from_legacy_rkv():
    assert issubclass(RKVServing, R1KV)
    assert not hasattr(R1KV, "from_serving_config")
    assert not hasattr(R1KV, "should_observe_token_queries")
    policy = RKVServing()
    assert policy._serving_query_history == {}
    assert policy._serving_layer_order is None


class KVView:
    def __init__(self, keys, values):
        self._keys = keys
        self._values = values
        self.values_read = 0

    def get_keys(self):
        return self._keys

    def get_values(self):
        self.values_read += 1
        return self._values


def make_policy():
    return RKVServing.from_serving_config(
        {
            "budget": 12,
            "buffer": 8,
            "window_size": 4,
            "kernel_size": 7,
        }
    )


def test_frozen_five_method_contract_and_phase_semantics():
    policy = make_policy()
    assert policy.should_observe_token_queries("prefill", 0) == 4
    assert [policy.should_observe_token_queries("decode", i) for i in range(8)] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert policy.should_compact_kv("prefill", 24, 0)
    assert not policy.should_compact_kv("prefill", 12, 0)
    assert policy.should_compact_kv("decode", 20, 7)
    assert not policy.should_compact_kv("decode", 19, 7)
    assert not policy.should_compact_kv("decode", 20, 6)


def test_full_rkv_per_layer_per_head_and_algorithm_order():
    torch.manual_seed(31)
    policy = make_policy()
    views = {
        "layer1": KVView(torch.randn(1, 2, 24, 8), torch.randn(1, 2, 24, 8)),
        "layer2": KVView(torch.randn(1, 2, 24, 8), torch.randn(1, 2, 24, 8)),
    }
    query_by_layer = {name: torch.randn(4, 4, 8) for name in views}
    policy.observe_token_queries(query_by_layer)
    result = policy.select_kept_token_positions(views)
    assert set(result) == set(views)
    for layer, view in views.items():
        queries = query_by_layer[layer].permute(1, 0, 2).unsqueeze(0)
        scores = policy.score_kv(queries, view.get_keys())
        expected_past = scores.topk(policy.budget - policy.window_size, dim=-1).indices
        recent = torch.arange(20, 24).view(1, 1, 4).expand(1, 2, 4)
        expected = torch.cat([expected_past, recent], dim=-1)[0]
        assert result[layer].shape == (2, 12)
        assert torch.equal(result[layer], expected)
        assert view.values_read == 0

    assert not torch.equal(result["layer1"], result["layer2"])
    with pytest.raises(RuntimeError, match="full query window"):
        policy.select_kept_token_positions(views)


def test_q_history_keeps_last_window_across_multiple_forwards():
    policy = make_policy()
    policy.observe_token_queries({"layer": torch.arange(3.0).view(3, 1, 1)})
    policy.observe_token_queries({"layer": torch.arange(3.0, 7.0).view(4, 1, 1)})
    assert policy._serving_query_history["layer"][:, 0, 0].tolist() == [
        3.0,
        4.0,
        5.0,
        6.0,
    ]


def test_rkv_legacy_update_kv_stays_available():
    torch.manual_seed(17)
    policy = make_policy()
    keys = torch.randn(1, 2, 24, 8)
    values = torch.randn_like(keys)
    queries = torch.randn(1, 4, 4, 8)
    new_keys, new_values = policy.update_kv(keys, queries, values)
    assert new_keys.shape == (1, 2, 12, 8)
    assert new_values.shape == (1, 2, 12, 8)


def test_serving_defaults_preserve_legacy_algorithm_defaults():
    legacy = R1KV()
    serving = RKVServing.from_serving_config(
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
    policy = RKVServing.from_serving_config(
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
        ({"budget": 32, "buffer": 16, "unused": 0}, "Unsupported"),
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
        RKVServing.from_serving_config(config)


def test_empty_serving_config_uses_rkv_owned_defaults():
    policy = RKVServing.from_serving_config({})
    assert policy.budget == 128
    assert policy.buffer == 128
    assert policy.window_size == 8


def test_serving_plugin_entry_point_resolves_factory():
    """The factory path exported in setup.py must remain importable."""
    from importlib.metadata import EntryPoint

    factory = EntryPoint(
        name="rkv",
        value="rkv.serving:RKVServing.from_serving_config",
        group="lmcache.token_drop_algorithms",
    ).load()
    serving_policy = factory({})
    assert isinstance(serving_policy, RKVServing)
    assert serving_policy.budget == 128
