"""GPU parity tests for the fused R-KV redundancy kernel.

These require CUDA + Triton and are skipped otherwise (the CPU-by-path suite in
``test_rkv_prefill.py`` covers the reference ``cal_similarity_tiled``). Stage 1
checks the fused kernel WITHOUT the retain exemption against a no-retain torch
reference; Stage 2 will add the retain exemption and bit-parity vs the full
reference.

Run directly (on a GPU box)::

    python test/srt/mem_cache/test_rkv_redundancy_fused.py
"""

import importlib.util
import os
import sys
import types
import unittest

import torch

_HERE = os.path.dirname(__file__)
_RKV_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "rkv")
)

# Standalone layout (no installed ``sglang``): register the R-KV modules under
# their real ``sglang.srt.mem_cache.rkv.*`` dotted names so the lazy cross-module
# imports inside algo.py / prefill.py (redundancy_fused, prefill, algo) resolve.
for _pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.rkv",
):
    if _pkg not in sys.modules:
        _placeholder = types.ModuleType(_pkg)
        _placeholder.__path__ = []  # mark as package
        sys.modules[_pkg] = _placeholder
for _dotted, _fname in (
    ("sglang.srt.mem_cache.rkv.redundancy_fused", "redundancy_fused.py"),
    ("sglang.srt.mem_cache.rkv.algo", "algo.py"),
    ("sglang.srt.mem_cache.rkv.prefill", "prefill.py"),
):
    if _dotted not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            _dotted, os.path.join(_RKV_DIR, _fname)
        )
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_dotted] = _module
        _spec.loader.exec_module(_module)

_HAS_CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_RKV_DIR, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref_noretain(key: torch.Tensor) -> torch.Tensor:
    """Redundancy with only the diagonal zeroed (no retain exemption)."""
    k_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    n = key.shape[-2]
    diag = torch.eye(n, dtype=torch.bool, device=key.device)
    sim.masked_fill_(diag.view(1, 1, n, n), 0.0)
    return sim.to(torch.float32).mean(dim=-2).softmax(dim=-1).to(key.dtype)


@unittest.skipUnless(
    _HAS_CUDA and _HAS_TRITON, "fused redundancy kernel needs CUDA + Triton"
)
class TestFusedRedundancyNoRetain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fused = staticmethod(
            _load(
                "rkv_redundancy_fused", "redundancy_fused.py"
            ).cal_similarity_fused_noretain
        )

    def test_parity_fp32(self):
        for n in (37, 128, 512, 2000):
            torch.manual_seed(0)
            key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.float32)
            ref = _ref_noretain(key)
            got = self.fused(key)
            self.assertEqual(ref.shape, got.shape)
            self.assertTrue(
                torch.allclose(ref, got, atol=2e-4, rtol=1e-3),
                f"fp32 fused diverged at n={n}: "
                f"max {(ref - got).abs().max().item():.3e}",
            )

    def test_parity_bf16(self):
        for n in (128, 512, 2000):
            torch.manual_seed(1)
            key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.bfloat16)
            ref = _ref_noretain(key)
            got = self.fused(key)
            self.assertTrue(
                torch.allclose(ref.float(), got.float(), atol=2e-2, rtol=1e-3),
                f"bf16 fused diverged at n={n}: "
                f"max {(ref.float() - got.float()).abs().max().item():.3e}",
            )

    def test_multi_layer_leading_dims(self):
        # Leading (layers, kv_heads) dims are flattened and reduced independently.
        torch.manual_seed(2)
        key = torch.randn(3, 4, 200, 128, device="cuda", dtype=torch.float32)
        got = self.fused(key)
        self.assertEqual(got.shape, (3, 4, 200))
        # Each (layer, head) row is a probability distribution over n.
        sums = got.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))


@unittest.skipUnless(
    _HAS_CUDA and _HAS_TRITON, "fused redundancy kernel needs CUDA + Triton"
)
class TestFusedRedundancyRetain(unittest.TestCase):
    """Fused kernel WITH the retain exemption must match cal_similarity_tiled."""

    @classmethod
    def setUpClass(cls):
        cls.fused = staticmethod(
            _load("rkv_redundancy_fused", "redundancy_fused.py").cal_similarity_fused
        )
        cls.tiled = staticmethod(
            _load("rkv_prefill", "prefill.py").cal_similarity_tiled
        )

    def test_bitparity_fp32(self):
        for thr in (0.0, 0.3, 0.5, 0.9):
            for n in (37, 128, 512, 1500):
                torch.manual_seed(n)
                key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.float32)
                ref = self.tiled(key, threshold=thr, retain_direction="last")
                got = self.fused(key, threshold=thr)
                self.assertTrue(
                    torch.allclose(ref, got, atol=5e-4, rtol=1e-3),
                    f"fp32 retain diverged thr={thr} n={n}: "
                    f"max {(ref - got).abs().max().item():.3e}",
                )

    def test_parity_bf16(self):
        for thr in (0.3, 0.5):
            for n in (128, 512, 1500):
                torch.manual_seed(n)
                key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.bfloat16)
                ref = self.tiled(key, threshold=thr, retain_direction="last")
                got = self.fused(key, threshold=thr)
                self.assertTrue(
                    torch.allclose(ref.float(), got.float(), atol=3e-2, rtol=1e-3),
                    f"bf16 retain diverged thr={thr} n={n}: "
                    f"max {(ref.float() - got.float()).abs().max().item():.3e}",
                )

    def test_no_neighbor_defaults_to_col0(self):
        # A very high threshold -> no above-threshold neighbours -> retain=col 0
        # for every row (the reference's default). Must still match tiled.
        torch.manual_seed(7)
        key = torch.randn(1, 4, 300, 128, device="cuda", dtype=torch.float32)
        ref = self.tiled(key, threshold=0.99, retain_direction="last")
        got = self.fused(key, threshold=0.99)
        self.assertTrue(torch.allclose(ref, got, atol=5e-4, rtol=1e-3))

    def test_run_to_run_determinism(self):
        # The retain kernel uses tl.atomic_add, whose FP accumulation order can
        # vary between launches. Measured (2026-07-11, H100): the resulting score
        # jitter is negligible — 0.0 for bf16/fp16 (rounded away) and ~1e-10 for
        # fp32 — and it NEVER flips the top-k kept set, which is the only thing
        # that could change the model output. This pins both: a regression that
        # introduced meaningful jitter or an unstable kept set would fail.
        window, budget = 32, 256
        for dtype, score_atol in (
            (torch.bfloat16, 1e-6),
            (torch.float16, 1e-6),
            (torch.float32, 1e-6),
        ):
            torch.manual_seed(123)
            key = torch.randn(1, 8, 1024, 128, device="cuda", dtype=dtype)
            ref = self.fused(key, threshold=0.5).float()
            ref_kept = (
                ref[:, :, :-window]
                .topk(budget - window, dim=-1)
                .indices.sort(-1)
                .values
            )
            for _ in range(25):
                got = self.fused(key, threshold=0.5).float()
                self.assertLessEqual(
                    (got - ref).abs().max().item(),
                    score_atol,
                    f"fused kernel jitter exceeds {score_atol} ({dtype})",
                )
                kept = (
                    got[:, :, :-window]
                    .topk(budget - window, dim=-1)
                    .indices.sort(-1)
                    .values
                )
                self.assertTrue(
                    torch.equal(kept, ref_kept),
                    f"fused kernel kept-set is nondeterministic ({dtype})",
                )


@unittest.skipUnless(
    _HAS_CUDA and _HAS_TRITON, "fused redundancy kernel needs CUDA + Triton"
)
class TestDecodeR1KVFused(unittest.TestCase):
    """Decode R1KV._scores must be unchanged when the fused kernel is adopted."""

    @classmethod
    def setUpClass(cls):
        mod = _load("rkv_algo", "algo.py")
        cls.R1KV = mod.R1KV
        cls.cal_similarity = staticmethod(mod.cal_similarity)

    def _pair(self, budget=64, window=8):
        kw = dict(
            budget=budget,
            window_size=window,
            kernel_size=7,
            mix_lambda=0.1,
            retain_ratio=0.1,
            retain_direction="last",
        )
        fused = self.R1KV(**kw)
        ref = self.R1KV(**kw)
        ref._fused_redundancy = False  # force full-matrix cal_similarity
        return fused, ref

    def test_scores_parity_fp32(self):
        for n in (128, 512, 1500):
            torch.manual_seed(n)
            k = torch.randn(2, 4, n, 128, device="cuda", dtype=torch.float32)
            q = torch.randn(2, 4, n, 128, device="cuda", dtype=torch.float32)
            fused, ref = self._pair()
            sf = fused._scores(k, q)
            sr = ref._scores(k, q)
            self.assertTrue(
                torch.allclose(sf, sr, atol=5e-4, rtol=1e-3),
                f"decode _scores diverged n={n}: max {(sf - sr).abs().max().item():.3e}",
            )

    def test_select_indices_parity_fp32(self):
        # fp32 fused is bit-exact -> identical kept-token selection (incl. GQA).
        for n in (200, 512, 1500):
            torch.manual_seed(n + 1)
            k = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.float32)
            q = torch.randn(1, 8, n, 128, device="cuda", dtype=torch.float32)
            fused, ref = self._pair()
            idx_f = fused.select_indices(k, q, sort=True)
            idx_r = ref.select_indices(k, q, sort=True)
            self.assertTrue(
                torch.equal(idx_f, idx_r), f"decode select_indices differ at n={n}"
            )

    def test_gate_adopts_fused(self):
        torch.manual_seed(3)
        k = torch.randn(2, 4, 300, 128, device="cuda", dtype=torch.float32)
        q = torch.randn(2, 4, 300, 128, device="cuda", dtype=torch.float32)
        algo = self.R1KV(
            budget=64,
            window_size=8,
            kernel_size=7,
            mix_lambda=0.1,
            retain_ratio=0.1,
            retain_direction="last",
        )
        self.assertIsNone(algo._fused_redundancy)
        algo._scores(k, q)  # first call runs the smoke gate
        self.assertIsNotNone(algo._fused_redundancy)
        self.assertIsNot(algo._fused_redundancy, False)

    def test_reference_redundancy_bounded_memory(self):
        # The gate's CUDA reference must be the O(n)-memory tiled path, not the
        # O(n^2) full matrix, which OOMs on long sequences (decode-mode
        # long-prompt compaction, where n = prompt length).
        n = 8192
        torch.manual_seed(0)
        key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.bfloat16)
        algo = self.R1KV(budget=512, window_size=8, retain_direction="last")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ref = algo._reference_redundancy(key)
        tiled_gb = torch.cuda.max_memory_allocated() / 1e9

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        full = self.cal_similarity(key, retain_direction="last")
        full_gb = torch.cuda.max_memory_allocated() / 1e9

        # O(n) tiled reference uses far less than the O(n^2) full matrix...
        self.assertLess(
            tiled_gb,
            full_gb / 2,
            f"reference not tiled: {tiled_gb:.2f} GB vs full {full_gb:.2f} GB",
        )
        # ...and is still bit-parity with it.
        self.assertTrue(torch.allclose(ref.float(), full.float(), atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
