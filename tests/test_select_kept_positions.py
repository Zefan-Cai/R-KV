import pytest
import torch

from rkv.selection import select_kept_positions


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_select_kept_positions_matches_notebook(dtype):
    scores = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.2], dtype=dtype)
    selected = select_kept_positions(scores, keep_past=2, window_size=3)
    assert selected.dtype == torch.long
    assert selected.device == scores.device
    assert torch.equal(selected, torch.tensor([1, 3, 5, 6, 7]))


def test_select_kept_positions_zero_past_budget():
    scores = torch.tensor([1., 2., 3.])
    assert torch.equal(
        select_kept_positions(scores, keep_past=0, window_size=2),
        torch.tensor([3, 4]),
    )


def test_select_kept_positions_all_past_tokens():
    scores = torch.tensor([3., 1., 2.])
    assert torch.equal(
        select_kept_positions(scores, keep_past=3, window_size=2),
        torch.tensor([0, 1, 2, 3, 4]),
    )
