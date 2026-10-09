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
