"""Tests for R-KV **prefill** compression: tiled-similarity parity + A/B diff-test.

Two things are checked without a GPU or the serving stack:

1. **Parity** — :func:`cal_similarity_tiled` (route A's ``O(n)``-memory redundancy)
   reproduces the reference ``algo.cal_similarity`` exactly for
   ``retain_direction='last'``, and route A's one-shot selection matches a
   direct full-matrix R-KV selection.

2. **Diff-test invariants** — route A (one-shot oracle) and route B (buffered)
   both return ``budget`` ascending indices that include the trailing window,
   and the *premature-eviction* metric (tokens A keeps but B drops) is computed
   the way the real-model harness will.

Run directly::

    python test/srt/mem_cache/test_rkv_prefill.py
"""

import importlib.util
import os
import unittest

import torch

_HERE = os.path.dirname(__file__)
_RKV_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "rkv")
)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_RKV_DIR, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_algo = _load("rkv_algo", "algo.py")
_prefill = _load("rkv_prefill", "prefill.py")

R1KV = _algo.R1KV
cal_similarity = _algo.cal_similarity
cal_similarity_tiled = _prefill.cal_similarity_tiled
RKVPrefill = _prefill.RKVPrefill


# --------------------------------------------------------------------------- #
# Diff-test metrics (reused by the real-model capture harness)                 #
# --------------------------------------------------------------------------- #
def diff_metrics(kept_a: torch.Tensor, kept_b: torch.Tensor, n: int) -> dict:
    """Compare two kept-index sets. ``kept_a`` is the oracle (route A).

    Returns Jaccard overlap, the count of tokens A keeps but B drops (premature
    evictions), and the recall of B against A.
    """
    a = set(kept_a.tolist())
    b = set(kept_b.tolist())
    inter = len(a & b)
    union = len(a | b)
    premature = len(a - b)  # A keeps, B dropped -> candidate premature evictions
    return {
        "n": n,
        "budget": len(a),
        "jaccard": inter / union if union else 1.0,
        "recall_b_vs_a": inter / len(a) if a else 1.0,
        "premature_evictions": premature,
    }


def _rand_layers(num_layers, kv_heads, q_heads, n, d, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    keys = [
        torch.randn(kv_heads, n, d, generator=g, dtype=dtype) for _ in range(num_layers)
    ]
    queries = [
        torch.randn(q_heads, n, d, generator=g, dtype=dtype) for _ in range(num_layers)
    ]
    return keys, queries


class TestTiledSimilarityParity(unittest.TestCase):
    def test_matches_reference_last(self):
        torch.manual_seed(0)
        for n in (37, 128, 500):
            key = torch.randn(1, 4, n, 16, dtype=torch.float32)
            ref = cal_similarity(key, retain_direction="last")
            # Force multiple row blocks to exercise the tiling path.
            got = cal_similarity_tiled(key, retain_direction="last", row_block=64)
            self.assertEqual(ref.shape, got.shape)
            self.assertTrue(
                torch.allclose(ref, got, atol=1e-6, rtol=1e-5),
                f"tiled similarity diverged at n={n}: max abs "
                f"{(ref - got).abs().max().item():.3e}",
            )

    def test_single_block_equals_reference(self):
        torch.manual_seed(1)
        key = torch.randn(1, 3, 96, 16, dtype=torch.float32)
        ref = cal_similarity(key, retain_direction="last")
        got = cal_similarity_tiled(key, retain_direction="last", row_block=10_000)
        self.assertTrue(torch.allclose(ref, got, atol=1e-6, rtol=1e-5))


class TestOneShotParity(unittest.TestCase):
    """Route A single-layer selection must match a direct full-matrix R-KV pick."""

    def test_oneshot_matches_algo_select(self):
        torch.manual_seed(2)
        kv_heads, q_heads, n, d = 2, 2, 300, 16
        budget, window = 64, 8
        key = torch.randn(1, kv_heads, n, d, dtype=torch.float32)
        query = torch.randn(1, q_heads, n, d, dtype=torch.float32)

        # Reference: full-matrix R-KV score, head-mean, top past + window.
        algo = R1KV(budget=budget, window_size=window, mix_lambda=0.1, retain_ratio=0.1)
        window_q = query[:, :, -window:, :]
        score = algo._scores(key, window_q).mean(dim=1).squeeze(0)  # (n-window,)
        past = score.topk(budget - window).indices
        win = torch.arange(n - window, n)
        ref = torch.sort(torch.cat([past, win])).values

        # Route A one-shot (tiled).
        pf = RKVPrefill(
            budget=budget, window_size=window, mix_lambda=0.1, retain_ratio=0.1
        )
        got = pf.oneshot_keep([key.squeeze(0)], [window_q.squeeze(0)])

        self.assertTrue(torch.equal(ref, got))


class TestRouteInvariants(unittest.TestCase):
    def setUp(self):
        self.kv_heads, self.q_heads, self.n, self.d = 2, 4, 2000, 16
        self.budget, self.window = 256, 8
        self.keys, self.queries = _rand_layers(
            num_layers=3,
            kv_heads=self.kv_heads,
            q_heads=self.q_heads,
            n=self.n,
            d=self.d,
            seed=3,
        )
        self.pf = RKVPrefill(
            budget=self.budget, window_size=self.window, mix_lambda=0.1
        )
        self.window_q = [q[:, -self.window :, :] for q in self.queries]

    def test_oneshot_shape_and_window(self):
        kept = self.pf.oneshot_keep(self.keys, self.window_q)
        self.assertEqual(kept.numel(), self.budget)
        self.assertTrue(torch.equal(torch.sort(kept).values, kept))  # ascending
        # Trailing window always kept.
        win = set(range(self.n - self.window, self.n))
        self.assertTrue(win.issubset(set(kept.tolist())))

    def test_buffered_shape_and_window(self):
        kept = self.pf.buffered_keep(
            self.keys, self.queries, chunk_size=512, buffer=128
        )
        self.assertEqual(kept.numel(), self.budget)
        self.assertTrue(torch.equal(torch.sort(kept).values, kept))
        win = set(range(self.n - self.window, self.n))
        self.assertTrue(win.issubset(set(kept.tolist())))

    def test_buffered_infinite_buffer_recovers_oneshot(self):
        # buffer >= n means no mid-prefill compaction; the single final
        # compaction scores against the true final window == route A exactly.
        a = self.pf.oneshot_keep(self.keys, self.window_q)
        b = self.pf.buffered_keep(
            self.keys, self.queries, chunk_size=256, buffer=self.n + 1
        )
        self.assertTrue(torch.equal(a, b))

    def test_buffered_no_compaction_below_budget(self):
        short_keys = [k[:, : self.budget - 1, :] for k in self.keys]
        short_q = [q[:, : self.budget - 1, :] for q in self.queries]
        kept = self.pf.buffered_keep(short_keys, short_q, chunk_size=64, buffer=32)
        self.assertEqual(kept.numel(), self.budget - 1)

    def test_diff_metric_runs(self):
        a = self.pf.oneshot_keep(self.keys, self.window_q)
        b = self.pf.buffered_keep(self.keys, self.queries, chunk_size=512, buffer=128)
        m = diff_metrics(a, b, self.n)
        self.assertEqual(m["budget"], self.budget)
        self.assertGreaterEqual(m["jaccard"], 0.0)
        self.assertLessEqual(m["jaccard"], 1.0)
        # The trailing window is kept by both, so overlap is never zero.
        self.assertGreaterEqual(m["premature_evictions"], 0)


class TestBatchedScoreParity(unittest.TestCase):
    """``batched_past_score`` over stacked layers == per-layer ``layer_past_score`` sum."""

    def test_batched_equals_per_layer_sum(self):
        torch.manual_seed(3)
        num_layers, kv_heads, q_heads, n, window, d = 4, 2, 8, 96, 8, 16
        budget = 32
        pf = RKVPrefill(budget=budget, window_size=window, mix_lambda=0.1)
        keys = torch.randn(num_layers, kv_heads, n, d, dtype=torch.float32)
        window_q = torch.randn(num_layers, q_heads, window, d, dtype=torch.float32)

        # Per-layer reference (bsz=1 loop, summed over layers).
        ref = None
        for layer in range(num_layers):
            s = pf.layer_past_score(
                keys[layer : layer + 1], window_q[layer : layer + 1]
            )
            ref = s if ref is None else ref + s

        # Batched (num_layers as the batch dim), summed over layers.
        bat = pf.batched_past_score(keys, window_q).sum(dim=0)  # (n-w,)

        self.assertEqual(ref.shape, bat.shape)
        self.assertTrue(
            torch.allclose(ref, bat, atol=1e-5, rtol=1e-5),
            f"batched score diverged: max abs {(ref - bat).abs().max().item():.3e}",
        )
        # The selection (top past tokens) must be identical — this is the gate.
        num_past = budget - window
        self.assertTrue(
            torch.equal(
                torch.sort(ref.topk(num_past).indices).values,
                torch.sort(bat.topk(num_past).indices).values,
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
