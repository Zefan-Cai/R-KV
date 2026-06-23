"""CPU-only unit test for the ported R-KV algorithm.

Runs against synthetic key / value / query tensors and verifies output
shapes, that the trailing window is preserved verbatim, and that the
pre-trigger path is a no-op. No GPU required.
"""

from __future__ import annotations

import torch

from minisgl.compress import R1KV, RKVCompressor, RKVConfig


def _make_kv(batch: int, kv_heads: int, cache_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    keys = torch.randn(batch, kv_heads, cache_len, head_dim, dtype=torch.float32)
    values = torch.randn(batch, kv_heads, cache_len, head_dim, dtype=torch.float32)
    return keys, values


def test_update_kv_skips_when_within_budget() -> None:
    rkv = R1KV(budget=128, window_size=8)
    keys, values = _make_kv(1, 4, 64, 16)
    q = torch.randn(1, 16, 8, 16, dtype=torch.float32)  # GQA group of 4
    new_keys, new_values = rkv.update_kv(keys, q, values)
    assert torch.equal(new_keys, keys)
    assert torch.equal(new_values, values)


def test_update_kv_compresses_to_budget() -> None:
    budget = 64
    window = 8
    rkv = R1KV(budget=budget, window_size=window)
    kv_heads = 4
    cache_len = 256
    head_dim = 32
    keys, values = _make_kv(1, kv_heads, cache_len, head_dim)
    # Query group of 4 -> 16 query heads with GQA factor 4.
    q = torch.randn(1, kv_heads * 4, window, head_dim, dtype=torch.float32)

    new_keys, new_values = rkv.update_kv(keys, q, values)

    # Output should be exactly `budget` tokens long.
    assert new_keys.shape == (1, kv_heads, budget, head_dim)
    assert new_values.shape == (1, kv_heads, budget, head_dim)

    # The trailing `window` tokens are preserved verbatim per R-KV's
    # local-window guarantee.
    assert torch.equal(new_keys[:, :, -window:, :], keys[:, :, -window:, :])
    assert torch.equal(new_values[:, :, -window:, :], values[:, :, -window:, :])


def test_compressor_disabled_returns_none() -> None:
    cfg = RKVConfig(enabled=False)
    compressor = RKVCompressor(config=cfg, num_layers=4)
    assert not compressor.is_enabled
    assert compressor.r1kv is None


def test_compressor_drop_request_clears_all_layers() -> None:
    cfg = RKVConfig(enabled=True, budget=64, window_size=8)
    compressor = RKVCompressor(config=cfg, num_layers=4)
    # Manually seed the per-(uid, layer_id) buffer.
    for layer_id in range(4):
        compressor._query_buffer[(42, layer_id)] = torch.zeros(1)
    assert len(compressor._query_buffer) == 4
    compressor.drop_request(uid=42)
    assert len(compressor._query_buffer) == 0


def test_drain_pending_free_slots_returns_and_clears() -> None:
    cfg = RKVConfig(enabled=True, budget=64, window_size=8)
    compressor = RKVCompressor(config=cfg, num_layers=4)
    # First drain returns None when nothing pending.
    assert compressor.drain_pending_free_slots() is None
    # Enqueue two slot ranges as the last layer would.
    compressor._pending_free_slots.append(torch.tensor([10, 11, 12], dtype=torch.int32))
    compressor._pending_free_slots.append(torch.tensor([20, 21], dtype=torch.int32))
    drained = compressor.drain_pending_free_slots()
    assert drained is not None
    assert drained.tolist() == [10, 11, 12, 20, 21]
    # Drain is destructive.
    assert compressor.drain_pending_free_slots() is None


if __name__ == "__main__":
    test_update_kv_skips_when_within_budget()
    test_update_kv_compresses_to_budget()
    test_compressor_disabled_returns_none()
    test_compressor_drop_request_clears_all_layers()
    test_drain_pending_free_slots_returns_and_clears()
    print("all tests passed")
