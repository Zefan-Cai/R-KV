"""CPU-only contract tests for the FA3 attention adapter.

The real kernels remain GPU-only. These tests stub the base engine and FA3
entry points to exercise ragged prefill metadata, non-contiguous active-row
mapping, the no-``cache_batch_idx`` fallback, and fail-fast dependency checks.

Run: python FlashInfer/tests/test_fa3_engine.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_HERE, "..", "rkv"))
_PKG_NAME = "_rkv_fa3_contract"


def _load_module() -> types.ModuleType:
    package = types.ModuleType(_PKG_NAME)
    package.__path__ = [_PKG_DIR]
    sys.modules[_PKG_NAME] = package

    engine = types.ModuleType(f"{_PKG_NAME}.engine")

    class FlashInferEngine:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("base engine must not initialize in CPU contract tests")

    engine.FlashInferEngine = FlashInferEngine
    sys.modules[engine.__name__] = engine

    name = f"{_PKG_NAME}.engine_fa3"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_PKG_DIR, "engine_fa3.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_fa3_module = _load_module()
FA3Engine = _fa3_module.FA3Engine


class _FakeFA3:
    def __init__(self) -> None:
        self.prefill_call = None
        self.decode_call = None

    def flash_attn_varlen_func(self, q, k, v, **kwargs):
        self.prefill_call = (q, k, v, kwargs)
        return q + 1, torch.empty(0)

    def flash_attn_with_kvcache(
        self, q, k_cache, v_cache, *, cache_seqlens, causal, cache_batch_idx=None
    ):
        self.decode_call = {
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "cache_seqlens": cache_seqlens,
            "cache_batch_idx": cache_batch_idx,
            "causal": causal,
        }
        return q + 2


def _engine(*, max_batch_size: int = 4, has_batch_idx: bool = True) -> object:
    engine = FA3Engine.__new__(FA3Engine)
    engine.max_batch_size = max_batch_size
    engine.region_len = 5
    engine.device = torch.device("cpu")
    engine.model = types.SimpleNamespace(num_kv_heads=2, head_dim=4)
    engine.kv_pool = torch.randn(
        1, max_batch_size * engine.region_len, 2, 1, 2, 4
    )
    engine._fa3 = _FakeFA3()
    engine._fa3_has_batch_idx = has_batch_idx
    engine._fa3_pad_seqlens = torch.ones(max_batch_size, dtype=torch.int32)
    return engine


def test_prefill_ragged_metadata() -> None:
    engine = _engine()
    engine._plan_prefill([0, 2, 5], [2, 3])
    q = torch.randn(5, 4, 4)
    k = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    out = engine._run_prefill_attention(q, k, v)

    assert torch.equal(out, q + 1)
    _, _, _, kwargs = engine._fa3.prefill_call
    assert kwargs["cu_seqlens_q"].tolist() == [0, 2, 5]
    assert kwargs["cu_seqlens_k"].tolist() == [0, 2, 5]
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 3
    assert kwargs["causal"] is True


def test_decode_noncontiguous_active_rows() -> None:
    engine = _engine(has_batch_idx=True)
    engine._plan_decode([0, 2], [5, 3])
    q = torch.randn(2, 4, 4)
    out = engine._run_decode_attention(0, q)

    assert torch.equal(out, q + 2)
    call = engine._fa3.decode_call
    assert call["q"].shape == (2, 1, 4, 4)
    assert call["k_cache"].shape == (4, 5, 2, 4)
    assert call["cache_seqlens"].tolist() == [5, 3]
    assert call["cache_batch_idx"].tolist() == [0, 2]
    assert call["causal"] is True


def test_decode_padded_fallback() -> None:
    engine = _engine(has_batch_idx=False)
    engine._plan_decode([1, 3], [4, 2])
    q = torch.randn(2, 4, 4)
    out = engine._run_decode_attention(0, q)

    assert torch.equal(out, q + 2)
    call = engine._fa3.decode_call
    assert call["q"].shape == (4, 1, 4, 4)
    assert torch.count_nonzero(call["q"][[0, 2]]) == 0
    assert call["cache_seqlens"].tolist() == [1, 4, 1, 2]
    assert call["cache_batch_idx"] is None


def test_missing_api_fails_before_base_init() -> None:
    previous = sys.modules.get("flash_attn_interface")
    sys.modules["flash_attn_interface"] = types.ModuleType("flash_attn_interface")
    try:
        try:
            FA3Engine()
        except ImportError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("missing Hopper entry points should raise ImportError")
    finally:
        if previous is None:
            del sys.modules["flash_attn_interface"]
        else:
            sys.modules["flash_attn_interface"] = previous


TESTS = [
    test_prefill_ragged_metadata,
    test_decode_noncontiguous_active_rows,
    test_decode_padded_fallback,
    test_missing_api_fails_before_base_init,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
        except Exception:
            failures += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
        else:
            print(f"[ OK ] {test.__name__}")
    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
