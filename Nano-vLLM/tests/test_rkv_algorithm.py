"""CPU-only unit test for the Nano-vLLM R-KV algorithm.

Loads `nanovllm/layers/rkv.py` directly to avoid pulling in
`nanovllm/__init__.py` and its CUDA-dependent runtime. Verifies the
algorithm matches the Mini-SGLang port bit-for-bit for the same seeded
inputs.
"""

from __future__ import annotations

import importlib.util
import pathlib

import torch


def _load_rkv(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("nano_rkv", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = pathlib.Path(__file__).resolve().parent
NANO_RKV = _load_rkv(HERE.parent / "nanovllm" / "layers" / "rkv.py")


def _make_kv(batch: int, kv_heads: int, cache_len: int, head_dim: int):
    torch.manual_seed(0)
    keys = torch.randn(batch, kv_heads, cache_len, head_dim, dtype=torch.float32)
    values = torch.randn(batch, kv_heads, cache_len, head_dim, dtype=torch.float32)
    return keys, values


def test_update_kv_skips_when_within_budget() -> None:
    rkv = NANO_RKV.R1KV(budget=128, window_size=8)
    keys, values = _make_kv(1, 4, 64, 16)
    q = torch.randn(1, 16, 8, 16, dtype=torch.float32)
    new_keys, new_values = rkv.update_kv(keys, q, values)
    assert torch.equal(new_keys, keys)
    assert torch.equal(new_values, values)


def test_update_kv_compresses_to_budget() -> None:
    budget = 64
    window = 8
    rkv = NANO_RKV.R1KV(budget=budget, window_size=window)
    keys, values = _make_kv(1, 4, 256, 32)
    q = torch.randn(1, 16, window, 32, dtype=torch.float32)

    new_keys, new_values = rkv.update_kv(keys, q, values)

    assert new_keys.shape == (1, 4, budget, 32)
    assert new_values.shape == (1, 4, budget, 32)
    # Trailing window preserved verbatim.
    assert torch.equal(new_keys[:, :, -window:, :], keys[:, :, -window:, :])
    assert torch.equal(new_values[:, :, -window:, :], values[:, :, -window:, :])


def test_matches_mini_sglang_port() -> None:
    """Cross-validate Nano-vLLM and Mini-SGLang R1KV ports give the
    same output for the same seeded inputs.
    """
    mini_path = (
        HERE.parent.parent / "Mini-SGLang" / "python" / "minisgl" / "compress" / "rkv.py"
    )
    if not mini_path.exists():
        # When tested outside the R-KV monorepo this cross check is skipped.
        return
    MINI = _load_rkv(mini_path)

    cfg = dict(
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
    )
    nano_r = NANO_RKV.R1KV(**cfg)
    mini_r = MINI.R1KV(**cfg)

    torch.manual_seed(123)
    keys = torch.randn(1, 4, 512, 32)
    values = torch.randn(1, 4, 512, 32)
    q = torch.randn(1, 16, 8, 32)

    nk, nv = nano_r.update_kv(keys, q, values)
    mk, mv = mini_r.update_kv(keys, q, values)
    assert torch.equal(nk, mk)
    assert torch.equal(nv, mv)


if __name__ == "__main__":
    test_update_kv_skips_when_within_budget()
    test_update_kv_compresses_to_budget()
    test_matches_mini_sglang_port()
    print("all tests passed")
