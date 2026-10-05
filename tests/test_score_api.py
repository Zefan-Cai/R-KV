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
