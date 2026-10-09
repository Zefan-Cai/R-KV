import pytest
import torch
import torch.nn.functional as F

from rkv import R1KV
from rkv.utils import cal_similarity, compute_attention_scores


@pytest.mark.parametrize("record_indices", [False, True])
def test_scoring_refactor_preserves_legacy_update_kv(record_indices):
    torch.manual_seed(1)
    policy = R1KV(
        budget=12, window_size=4, kernel_size=7,
        record_kept_token_indices=record_indices,
    )
    keys = torch.randn(1, 2, 24, 8)
    queries = torch.randn(1, 4, 4, 8)
    values = torch.randn_like(keys)

    attention = compute_attention_scores(queries, keys)
    importance = (
        F.softmax(attention[:, :, -4:, :-4], dim=-1, dtype=torch.float32)
        .mean(dim=-2)
        .to(queries.dtype)
    )
    importance = F.max_pool1d(importance, kernel_size=7, padding=3, stride=1)
    redundancy = cal_similarity(
        keys, retain_ratio=policy.retain_ratio,
        retain_direction=policy.retain_direction,
    )[:, :, :-4]
    expected_scores = (
        importance * policy.mix_lambda
        - redundancy * (1 - policy.mix_lambda)
    )
    actual_scores, actual_attention = policy._compute_scores(queries, keys)
    assert torch.equal(actual_scores, expected_scores)
    assert torch.equal(actual_attention, attention)

    kept = expected_scores.topk(8, dim=-1).indices
    gather_idx = kept.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    expected_keys = torch.cat(
        [keys[:, :, :-4, :].gather(2, gather_idx), keys[:, :, -4:, :]], dim=2
    )
    expected_values = torch.cat(
        [values[:, :, :-4, :].gather(2, gather_idx), values[:, :, -4:, :]], dim=2
    )
    actual_keys, actual_values = policy.update_kv(keys, queries, values)
    assert torch.equal(actual_keys, expected_keys)
    assert torch.equal(actual_values, expected_values)
