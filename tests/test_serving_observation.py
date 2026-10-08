import torch

from rkv import R1KV


def _policy(*, budget=12, window=4, buffer=8):
    return R1KV(
        budget=budget, window_size=window, kernel_size=7,
        mix_lambda=0.1, retain_ratio=0.1,
        retain_direction="last", buffer=buffer,
    )


def test_query_history_is_owned_and_ordered_inside_rkv():
    policy = _policy(window=4)
    for step in range(6):
        query = torch.full((1, 2, 8), float(step))
        policy.observe_query({"layer": query})

    actual = policy._serving_query_windows()[0]
    assert actual.shape == (1, 2, 4, 8)
    assert torch.equal(
        actual[0, 0, :, 0],
        torch.tensor([2.0, 3.0, 4.0, 5.0]),
    )


def test_should_observe_query_tracks_only_rkv_scoring_window():
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


def test_should_compact_owns_rkv_trigger_and_readiness():
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
    }

    assert not policy.should_compact(num_decoded_tokens=128, **common)
    for _ in range(policy.window_size - 1):
        policy.observe_query({"layer": torch.randn(1, 4, 8)})

    # Planning happens before the forward; the boundary step will contribute
    # the final observation needed by the R-KV window.
    assert not policy.should_compact(num_decoded_tokens=127, **common)
    assert policy.should_compact(num_decoded_tokens=128, **common)
    assert not policy.should_compact(
        resident_len=383,
        num_decoded_tokens=128,
        num_new_tokens=1,
        is_genuine_decode=True,
    )
    assert not policy.should_compact(
        resident_len=384,
        num_decoded_tokens=128,
        num_new_tokens=1,
        is_genuine_decode=False,
    )


def test_existing_positional_constructor_binding_is_unchanged():
    policy = R1KV(128, 8, 7, 0.07, 0.1, "last", True)
    assert policy.record_kept_token_indices is True
    assert policy.buffer == 128
