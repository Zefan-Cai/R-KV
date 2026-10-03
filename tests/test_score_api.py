import torch

from rkv import R1KV


def test_compute_scores_matches_update_kv_selection():
    torch.manual_seed(0)
    budget = 48
    window = 8
    keys = torch.randn(1, 2, 64, 16)
    queries = torch.randn(1, 4, window, 16)
    values = torch.randn_like(keys)

    policy = R1KV(
        budget=budget,
        window_size=window,
        record_kept_token_indices=True,
    )

    scores = policy.compute_scores(keys, queries)
    expected = scores.topk(budget - window, dim=-1).indices.squeeze(0).cpu()

    compressed_k, compressed_v = policy.update_kv(keys, queries, values)

    assert torch.equal(policy.kept_token_indices[-1][:, : budget - window], expected)
    assert compressed_k.shape[-2] == budget
    assert compressed_v.shape[-2] == budget
