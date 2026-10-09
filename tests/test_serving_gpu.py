"""CUDA smoke test for request-local R-KV query observation history."""

import pytest
import torch

from rkv.serving import RKVServing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_observe_token_queries_cuda_bf16_rolling_history_and_buffer_reuse():
    policy = RKVServing.from_serving_config({"window_size": 3})
    sources = {
        "layer0": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.bfloat16)
        .reshape(4, 1, 1),
        "layer1": torch.tensor([11.0, 12.0, 13.0, 14.0], device="cuda", dtype=torch.bfloat16)
        .reshape(4, 1, 1),
    }
    policy.observe_token_queries(sources)

    for source in sources.values():
        source.fill_(99)

    for layer, expected in (("layer0", [2, 3, 4]), ("layer1", [12, 13, 14])):
        stored = policy._serving_query_history[layer]
        assert stored.is_cuda
        assert stored.dtype == torch.bfloat16
        assert stored[:, 0, 0].tolist() == expected

    policy.observe_token_queries({
        "layer1": torch.tensor([[[15.0]]], device="cuda", dtype=torch.bfloat16),
        "layer0": torch.tensor([[[5.0]]], device="cuda", dtype=torch.bfloat16),
    })
    for layer, expected in (("layer0", [3, 4, 5]), ("layer1", [13, 14, 15])):
        stored = policy._serving_query_history[layer]
        assert stored.is_cuda
        assert stored.dtype == torch.bfloat16
        assert stored[:, 0, 0].tolist() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_select_kept_positions_cuda_bf16_matches_legacy_kv():
    torch.manual_seed(71)
    policy = RKVServing.from_serving_config(
        {"budget": 12, "buffer": 8, "window_size": 4}
    )

    class KeyView:
        def __init__(self, keys):
            self.keys = keys

        def get_keys(self):
            return self.keys

        def get_values(self):
            raise AssertionError("Selection must not fetch V")

    keys_by_layer = {
        layer: torch.randn(1, 2, 24, 8, device="cuda", dtype=torch.bfloat16)
        for layer in ("layer0", "layer1")
    }
    queries_by_layer = {
        layer: torch.randn(4, 4, 8, device="cuda", dtype=torch.bfloat16)
        for layer in keys_by_layer
    }
    policy.observe_token_queries(queries_by_layer)
    result = policy.select_kept_token_positions(
        {layer: KeyView(keys) for layer, keys in keys_by_layer.items()}
    )

    for layer, keys in keys_by_layer.items():
        positions = result[layer]
        assert positions.shape == (2, 12)
        assert positions.is_cuda and positions.dtype == torch.long
        values = torch.randn_like(keys)
        gather = positions[None, :, :, None].expand(1, 2, 12, 8)
        reference_k, reference_v = policy.update_kv(
            keys, queries_by_layer[layer].permute(1, 0, 2).unsqueeze(0), values
        )
        assert torch.equal(keys.gather(2, gather), reference_k)
        assert torch.equal(values.gather(2, gather), reference_v)

    assert policy._serving_query_history == {}
