import pytest
import torch

from rkv.selection import aggregate_scores


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aggregate_scores_matches_cpu_sdk_rule(dtype):
    layer_scores = [
        torch.tensor([[[1., 2., 3.], [3., 4., 5.]]]),
        torch.tensor([[[4., 2., 6.], [2., 6., 8.]]]),
    ]
    layer_scores = [scores.to(dtype) for scores in layer_scores]
    # Mean over heads within each layer, then sum over layers.
    expected = torch.tensor([5., 7., 11.], dtype=dtype)
    actual = aggregate_scores(layer_scores)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)


def test_aggregate_scores_supports_different_head_counts():
    layers = [
        torch.tensor([[[1., 3.], [3., 5.]]]),
        torch.tensor([[[2., 4.]]]),
    ]
    assert torch.equal(aggregate_scores(layers), torch.tensor([4., 8.]))


@pytest.mark.parametrize("layers", [
    [],
    [torch.ones(2, 2, 3)],
    [torch.ones(1, 2, 3), torch.ones(1, 2, 4)],
])
def test_aggregate_scores_rejects_invalid_inputs(layers):
    with pytest.raises(ValueError):
        aggregate_scores(layers)
