import torch
import pytest

from rkv import R1KV


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
    return R1KV.from_serving_config({
        "budget": 12, "buffer": 8,
        "window_size": 4, "kernel_size": 7,
    })


def test_frozen_five_method_contract_and_phase_semantics():
    policy = make_policy()
    assert policy.should_observe_token_queries("prefill", 0) == 4
    assert [policy.should_observe_token_queries("decode", i) for i in range(8)] == [
        0, 0, 0, 0, 1, 1, 1, 1,
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
    query_by_layer = {
        name: torch.randn(4, 4, 8) for name in views
    }
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
    policy.observe_token_queries({"layer": torch.arange(3.).view(3, 1, 1)})
    policy.observe_token_queries({"layer": torch.arange(3., 7.).view(4, 1, 1)})
    assert policy._serving_query_history["layer"][:, 0, 0].tolist() == [
        3.0, 4.0, 5.0, 6.0,
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
