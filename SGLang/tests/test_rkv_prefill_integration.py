"""CPU unit tests for the R-KV **prefill** integration layer (RKVPrefillCompressor).

GPU-free; loads the algorithm + integration modules by file path (bypassing
``sglang/__init__.py``). Focuses on the serving-facing behaviour: config
validation, the ``task_type`` gate, kept-index assembly, physical compaction
(KV relocation / free / ``req_to_token`` rewrite / length bookkeeping), and the
one-shot and buffered end-to-end observe->compact paths.

Run directly::

    python test/srt/mem_cache/test_rkv_prefill_integration.py
"""

import importlib.util
import os
import sys
import types
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_RKV_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "rkv")
)


def _load_by_path(dotted_name, file_path):
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.rkv",
):
    if _pkg not in sys.modules:
        _ph = types.ModuleType(_pkg)
        _ph.__path__ = []
        sys.modules[_pkg] = _ph

# prefill_integration.py declares its config as a ``msgspec.Struct`` (repo
# convention, see .claude/rules/no-dataclasses.md). msgspec is a real SGLang
# runtime dependency, but this suite is GPU-free and installs only torch, so it
# may be absent. Provide a tiny stand-in backed by ``dataclasses`` when it is:
# the Struct uses only plain-typed fields with immutable defaults plus a
# ``__post_init__``, all of which dataclasses reproduce exactly. Mirrors the
# ``sglang.*`` placeholder packages above.
try:
    import msgspec  # noqa: F401
except ModuleNotFoundError:
    import dataclasses

    _msgspec_stub = types.ModuleType("msgspec")

    class _Struct:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            dataclasses.dataclass(cls)

    _msgspec_stub.Struct = _Struct
    sys.modules["msgspec"] = _msgspec_stub

_load_by_path("sglang.srt.mem_cache.rkv.algo", os.path.join(_RKV_DIR, "algo.py"))
_load_by_path("sglang.srt.mem_cache.rkv.prefill", os.path.join(_RKV_DIR, "prefill.py"))
_integration = _load_by_path(
    "sglang.srt.mem_cache.rkv.prefill_integration",
    os.path.join(_RKV_DIR, "prefill_integration.py"),
)

RKVPrefillConfig = _integration.RKVPrefillConfig
RKVPrefillRequestState = _integration.RKVPrefillRequestState
RKVPrefillCompressor = _integration.RKVPrefillCompressor


# --------------------------------------------------------------------------- #
# Minimal mocks                                                                #
# --------------------------------------------------------------------------- #
class MockReqToTokenPool:
    def __init__(self, num_reqs, max_ctx, device):
        self.req_to_token = torch.zeros(
            (num_reqs + 1, max_ctx), dtype=torch.int32, device=device
        )


class MockKVPool:
    def __init__(self, num_layers, num_slots, head_num, head_dim, device, dtype):
        self.k_buffers = [
            torch.randn((num_slots, head_num, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_buffers = [
            torch.randn((num_slots, head_num, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    def get_key_buffer(self, layer_id):
        return self.k_buffers[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v_buffers[layer_id]


class MockAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index):
        self.freed.append(free_index.clone())


class _MockReq:
    def __init__(self, origin_len, output_len=0, req_pool_idx=1):
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = list(range(output_len))
        self.kv_committed_len = origin_len + output_len
        self.kv_allocated_len = origin_len + output_len
        self.req_pool_idx = req_pool_idx

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)


class _MockLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


class _MockForwardBatch:
    def __init__(self, req_pool_indices, seq_lens, extend_seq_lens, positions=None):
        self.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int64)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
        self.seq_lens_cpu = self.seq_lens.clone()
        self.extend_seq_lens = torch.tensor(extend_seq_lens, dtype=torch.int64)
        self.extend_seq_lens_cpu = extend_seq_lens
        self.positions = positions


def _make_compressor(
    mode,
    budget,
    window,
    num_layers,
    head_num,
    head_dim,
    num_slots,
    kv_heads=None,
    buffer=0,
):
    device = torch.device("cpu")
    kv_heads = kv_heads or head_num
    cfg = RKVPrefillConfig(
        mode=mode, budget=budget, window_size=window, kernel_size=7, buffer=buffer
    )
    return RKVPrefillCompressor(
        config=cfg,
        req_to_token_pool=MockReqToTokenPool(4, 4096, device),
        token_to_kv_pool=MockKVPool(
            num_layers, num_slots, kv_heads, head_dim, device, torch.float32
        ),
        kv_allocator=MockAllocator(),
        start_layer=0,
        end_layer=num_layers,
        device=device,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
class TestConfig(unittest.TestCase):
    def test_defaults_and_validation(self):
        cfg = RKVPrefillConfig()
        self.assertEqual(cfg.mode, "oneshot")
        with self.assertRaises(ValueError):
            RKVPrefillConfig(mode="nope")
        with self.assertRaises(ValueError):
            RKVPrefillConfig(budget=8, window_size=8)

    def test_even_kernel_and_bad_dims_rejected(self):
        with self.assertRaises(ValueError):
            RKVPrefillConfig(kernel_size=8)  # even kernel breaks R1KV._scores
        with self.assertRaises(ValueError):
            RKVPrefillConfig(window_size=0)
        with self.assertRaises(ValueError):
            RKVPrefillConfig(row_block=0)


class TestUnconditionalCompression(unittest.TestCase):
    def test_always_wants_compression(self):
        # R-KV prefill is unconditional: every request is compressed when enabled.
        f = RKVPrefillCompressor.request_wants_compression
        self.assertTrue(f(_MockReq(10)))
        self.assertTrue(f(_MockReq(500, output_len=20)))


class TestAssembleKept(unittest.TestCase):
    def test_top_past_plus_window_sorted(self):
        comp = _make_compressor(
            "oneshot",
            budget=4,
            window=2,
            num_layers=1,
            head_num=1,
            head_dim=4,
            num_slots=64,
        )
        score = torch.tensor([0.1, 9.0, 0.2, 8.0])  # past [0,4)
        kept = comp._assemble_kept(score, seq_len=6)
        self.assertEqual(sorted(kept.tolist()), [1, 3, 4, 5])


class TestOneShotEndToEnd(unittest.TestCase):
    def test_single_chunk_prefill_compacts(self):
        budget, window, n = 8, 2, 20
        num_layers, kv_heads, q_heads, head_dim = 3, 2, 2, 8
        comp = _make_compressor(
            "oneshot", budget, window, num_layers, kv_heads, head_dim, num_slots=64
        )
        idx = 1
        slots = torch.arange(30, 30 + n, dtype=torch.int32)
        comp.req_to_token_pool.req_to_token[idx, :n] = slots

        req = _MockReq(origin_len=n, req_pool_idx=idx)
        comp.on_request_begin(req)

        fb = _MockForwardBatch([idx], [n], [n])
        q = torch.randn(n, q_heads, head_dim)
        for layer_id in range(num_layers):
            comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        comp.maybe_compact(fb)

        state = comp.states[idx]
        self.assertTrue(state.compressed)
        self.assertEqual(req.kv_committed_len, budget)
        self.assertEqual(comp.take_pending_length_updates()[idx][0], budget)
        # freed exactly n - budget slots; tail of req_to_token cleared.
        self.assertEqual(int(comp.kv_allocator.freed[0].numel()), n - budget)
        r2t = comp.req_to_token_pool.req_to_token
        self.assertTrue(bool((r2t[idx, :budget] != 0).all()))
        self.assertTrue(bool((r2t[idx, budget:n] == 0).all()))

    def test_reprefill_after_retract_frees_regenerated_slots(self):
        # A retracted request keeps its output_ids and re-prefills
        # origin_input_ids + output_ids, allocating (origin + output) KV slots.
        # Compaction must free the FULL physical tail, not orphan the
        # regenerated output slots (KV-pool leak regression).
        budget, window, origin, output = 8, 2, 20, 12
        n = origin + output  # physical prefill length after re-prefill
        num_layers, kv_heads, q_heads, head_dim = 3, 2, 2, 8
        comp = _make_compressor(
            "oneshot", budget, window, num_layers, kv_heads, head_dim, num_slots=64
        )
        idx = 1
        slots = torch.arange(30, 30 + n, dtype=torch.int32)
        comp.req_to_token_pool.req_to_token[idx, :n] = slots

        req = _MockReq(origin_len=origin, output_len=output, req_pool_idx=idx)
        comp.on_request_begin(req)
        # prompt_len tracks the full physical length (origin + output).
        self.assertEqual(comp.states[idx].prompt_len, n)

        fb = _MockForwardBatch([idx], [n], [n])
        q = torch.randn(n, q_heads, head_dim)
        for layer_id in range(num_layers):
            comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        comp.maybe_compact(fb)

        # Frees the full tail (n - budget); no output slot left orphaned.
        self.assertEqual(int(comp.kv_allocator.freed[0].numel()), n - budget)
        r2t = comp.req_to_token_pool.req_to_token
        self.assertTrue(bool((r2t[idx, budget:n] == 0).all()))
        self.assertEqual(req.kv_committed_len, budget)
        self.assertEqual(req.kv_allocated_len, budget)

    def test_below_budget_no_compaction(self):
        budget, window, n = 32, 2, 20  # n < budget
        comp = _make_compressor("oneshot", budget, window, 2, 2, 8, num_slots=64)
        idx = 1
        comp.req_to_token_pool.req_to_token[idx, :n] = torch.arange(
            5, 5 + n, dtype=torch.int32
        )
        req = _MockReq(origin_len=n, req_pool_idx=idx)
        comp.on_request_begin(req)
        fb = _MockForwardBatch([idx], [n], [n])
        q = torch.randn(n, 2, 8)
        for layer_id in range(2):
            comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        comp.maybe_compact(fb)
        self.assertFalse(comp.states[idx].compressed)
        self.assertEqual(len(comp.kv_allocator.freed), 0)


class TestBufferedCompaction(unittest.TestCase):
    def test_triggers_above_budget_plus_buffer(self):
        budget, window, buffer, n = 8, 2, 4, 20  # n > budget + buffer
        num_layers, kv_heads, q_heads, head_dim = 2, 2, 2, 8
        comp = _make_compressor(
            "buffered",
            budget,
            window,
            num_layers,
            kv_heads,
            head_dim,
            num_slots=64,
            buffer=buffer,
        )
        idx = 1
        comp.req_to_token_pool.req_to_token[idx, :n] = torch.arange(
            40, 40 + n, dtype=torch.int32
        )
        req = _MockReq(origin_len=n, req_pool_idx=idx)
        comp.on_request_begin(req)

        fb = _MockForwardBatch([idx], [n], [n])
        q = torch.randn(n, q_heads, head_dim)
        for layer_id in range(num_layers):
            comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        # Logical compaction already shrank kept_orig to budget by the last layer.
        state = comp.states[idx]
        self.assertIsNotNone(state.kept_orig)
        self.assertEqual(state.kept_orig.numel(), budget)
        # End-of-prefill physical compaction (final chunk: seq_len == prompt_len).
        comp.maybe_compact(fb)
        self.assertTrue(
            state.compressed
        )  # buffered latches after the one physical pass
        self.assertEqual(req.kv_committed_len, budget)
        self.assertEqual(int(comp.kv_allocator.freed[0].numel()), n - budget)

    def test_final_forced_compaction_within_buffer(self):
        # budget < n <= budget + buffer: no mid-prefill logical compaction, but
        # the end-of-prefill forced pass still shrinks to budget (route B always
        # ends at budget, matching the pure algorithm).
        budget, window, buffer, n = 8, 2, 20, 20  # n <= budget + buffer
        comp = _make_compressor(
            "buffered", budget, window, 2, 2, 8, num_slots=64, buffer=buffer
        )
        idx = 1
        comp.req_to_token_pool.req_to_token[idx, :n] = torch.arange(
            40, 40 + n, dtype=torch.int32
        )
        req = _MockReq(origin_len=n, req_pool_idx=idx)
        comp.on_request_begin(req)
        fb = _MockForwardBatch([idx], [n], [n])
        q = torch.randn(n, 2, 8)
        for layer_id in range(2):
            comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        # No mid-prefill compaction: kept_orig still holds all n tokens.
        self.assertEqual(comp.states[idx].kept_orig.numel(), n)
        comp.maybe_compact(fb)
        self.assertTrue(comp.states[idx].compressed)
        self.assertEqual(int(comp.kv_allocator.freed[0].numel()), n - budget)

    def test_below_budget_no_compaction(self):
        budget, window, buffer, n = 32, 2, 8, 20  # n < budget
        comp = _make_compressor(
            "buffered", budget, window, 2, 2, 8, num_slots=64, buffer=buffer
        )
        idx = 1
        comp.req_to_token_pool.req_to_token[idx, :n] = torch.arange(
            40, 40 + n, dtype=torch.int32
        )
        req = _MockReq(origin_len=n, req_pool_idx=idx)
        comp.on_request_begin(req)
        fb = _MockForwardBatch([idx], [n], [n])
        q = torch.randn(n, 2, 8)
        for layer_id in range(2):
            comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        comp.maybe_compact(fb)
        self.assertFalse(comp.states[idx].compressed)
        self.assertEqual(len(comp.kv_allocator.freed), 0)

    def test_multichunk_kept_orig_evolves(self):
        # Two chunks; buffer small so a logical compaction fires after chunk 1.
        budget, window, buffer = 8, 2, 2
        num_layers, kv_heads, q_heads, head_dim = 2, 2, 2, 8
        comp = _make_compressor(
            "buffered",
            budget,
            window,
            num_layers,
            kv_heads,
            head_dim,
            num_slots=128,
            buffer=buffer,
        )
        idx = 1
        n = 24
        comp.req_to_token_pool.req_to_token[idx, :n] = torch.arange(
            50, 50 + n, dtype=torch.int32
        )
        req = _MockReq(origin_len=n, req_pool_idx=idx)
        comp.on_request_begin(req)

        # chunk 1: tokens [0, 12), seq_len=12
        q1 = torch.randn(12, q_heads, head_dim)
        fb1 = _MockForwardBatch([idx], [12], [12])
        for layer_id in range(num_layers):
            comp.observe_prefill_layer(q1, None, None, _MockLayer(layer_id), fb1)
        # 12 > budget+buffer(10) -> logically compacted to budget after chunk 1.
        self.assertEqual(comp.states[idx].kept_orig.numel(), budget)

        # chunk 2: tokens [12, 24), seq_len=24 (final)
        q2 = torch.randn(12, q_heads, head_dim)
        fb2 = _MockForwardBatch([idx], [24], [12])
        for layer_id in range(num_layers):
            comp.observe_prefill_layer(q2, None, None, _MockLayer(layer_id), fb2)
        comp.maybe_compact(fb2)
        state = comp.states[idx]
        self.assertTrue(state.compressed)
        self.assertEqual(req.kv_committed_len, budget)
        self.assertEqual(int(comp.kv_allocator.freed[0].numel()), n - budget)


class TestDecodePositions(unittest.TestCase):
    def test_override_uses_logical(self):
        comp = _make_compressor("oneshot", 8, 2, 1, 2, 8, num_slots=64)
        idx = 1
        req = _MockReq(origin_len=100, output_len=5, req_pool_idx=idx)
        comp.on_request_begin(req)
        positions = torch.tensor([0], dtype=torch.int64)
        fb = _MockForwardBatch([idx], [8], [0], positions=positions)
        comp.override_decode_positions(fb)
        # logical_position - 1 = 100 + 5 - 1 = 104
        self.assertEqual(int(fb.positions[0]), 104)


class TestPrefillPendingLifecycle(unittest.TestCase):
    """Issue 1 (prefill): a request that compacts at prefill end and finishes on
    the same step (the normal prefill-finish path) must not leak its pending
    length update to the next request reusing the slot."""

    def _comp(self):
        return _make_compressor("oneshot", 8, 2, 3, 2, 8, num_slots=64)

    def _compact(self, comp, idx, n=20):
        comp.req_to_token_pool.req_to_token[idx, :n] = torch.arange(
            30, 30 + n, dtype=torch.int32
        )
        req = _MockReq(origin_len=n, req_pool_idx=idx)
        comp.on_request_begin(req)
        fb = _MockForwardBatch([idx], [n], [n])
        q = torch.randn(n, 2, 8)
        for layer_id in range(comp.num_layers):
            comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        comp.maybe_compact(fb)
        return req

    def test_pending_update_carries_owner(self):
        comp = self._comp()
        req = self._compact(comp, 1)
        new_len, owner = comp.pending_length_updates[1]
        self.assertEqual(new_len, 8)
        self.assertIs(owner, req)

    def test_finish_clears_pending(self):
        comp = self._comp()
        req = self._compact(comp, 1)
        self.assertIn(1, comp.pending_length_updates)
        comp.on_request_end(req)
        self.assertNotIn(1, comp.pending_length_updates)
        self.assertNotIn(1, comp.states)

    def test_retract_clears_pending(self):
        comp = self._comp()
        req = self._compact(comp, 1)
        self.assertIn(1, comp.pending_length_updates)
        comp.on_request_retract(req)
        self.assertNotIn(1, comp.pending_length_updates)
        self.assertNotIn(1, comp.states)

    def test_slot_reuse_after_finish_no_stale(self):
        comp = self._comp()
        req_a = self._compact(comp, 1)
        comp.on_request_end(req_a)
        req_b = _MockReq(origin_len=5, req_pool_idx=1)
        comp.on_request_begin(req_b)  # B reuses slot 1
        self.assertEqual(comp.take_pending_length_updates(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
