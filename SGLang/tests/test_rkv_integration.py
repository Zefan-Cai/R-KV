"""CPU unit tests for the R-KV integration layer's physical compaction.

These tests are GPU-free and do not require an installed ``sglang`` package.
The pure algorithm module and the integration module are loaded directly by
file path (the integration module's absolute ``from sglang...algo import R1KV``
is satisfied by pre-registering the algo module under its real dotted name in
``sys.modules``), so we bypass ``sglang/__init__.py`` and the serving stack.

Run directly::

    python test/srt/mem_cache/test_rkv_integration.py

or under a test runner (unittest / pytest).
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


def _load_by_path(dotted_name: str, file_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


# Register placeholder parent packages so integration.py's absolute import of
# ``sglang.srt.mem_cache.rkv.algo`` resolves to the module we load by path.
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

_load_by_path("sglang.srt.mem_cache.rkv.algo", os.path.join(_RKV_DIR, "algo.py"))
_integration = _load_by_path(
    "sglang.srt.mem_cache.rkv.integration", os.path.join(_RKV_DIR, "integration.py")
)

RKVConfig = _integration.RKVConfig
RKVRequestState = _integration.RKVRequestState
RKVCompressor = _integration.RKVCompressor
rkv_runtime_support_error = _integration.rkv_runtime_support_error


# --------------------------------------------------------------------------- #
# Minimal mocks for the SGLang pools / allocator                              #
# --------------------------------------------------------------------------- #
class MockReqToTokenPool:
    def __init__(self, num_reqs: int, max_ctx: int, device: torch.device):
        # +1 padding row at index 0, mirroring the real ReqToTokenPool.
        self.req_to_token = torch.zeros(
            (num_reqs + 1, max_ctx), dtype=torch.int32, device=device
        )


class MockKVPool:
    """Per-layer NHD buffers: (num_slots, head_num, head_dim)."""

    def __init__(self, num_layers, num_slots, head_num, head_dim, device, dtype):
        self.k_buffers = [
            torch.zeros((num_slots, head_num, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_buffers = [
            torch.zeros((num_slots, head_num, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffers[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffers[layer_id]


class MockAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index: torch.Tensor) -> None:
        self.freed.append(free_index.clone())


class _MockReq:
    """Duck-typed stand-in for schedule_batch.Req."""

    def __init__(self, origin_len: int, output_len: int, req_pool_idx: int = 1):
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = list(range(output_len))
        self.kv_committed_len = origin_len + output_len
        self.kv_allocated_len = origin_len + output_len
        self.req_pool_idx = req_pool_idx


def _prepare_and_commit(comp, state, seq_len, kept):
    """Two-phase compaction convenience for tests: prepare (relocate K/V) then
    commit (free tail + bookkeeping), mirroring maybe_compact + the scheduler's
    commit_compactions()."""
    comp._commit_compaction(comp._prepare_compaction(state, seq_len, kept))


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
class TestAssembleKept(unittest.TestCase):
    def _compressor(self, budget, window):
        cfg = RKVConfig(budget=budget, window_size=window, buffer_size=4)
        return RKVCompressor(
            config=cfg,
            req_to_token_pool=MockReqToTokenPool(1, 64, torch.device("cpu")),
            token_to_kv_pool=MockKVPool(
                1, 64, 1, 4, torch.device("cpu"), torch.float32
            ),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=1,
            device=torch.device("cpu"),
            q_head_num=1,
            head_dim=4,
            q_dtype=torch.float32,
        )

    def test_kept_is_top_past_plus_window_sorted(self):
        budget, window, seq_len = 4, 2, 6
        comp = self._compressor(budget, window)
        # past tokens are indices [0, seq_len - window) = [0, 4).
        # scores favour indices 1 and 3.
        score_accum = torch.tensor([0.1, 9.0, 0.2, 8.0])
        kept = comp._assemble_kept(score_accum, seq_len)

        self.assertEqual(kept.numel(), budget)
        # ascending
        self.assertTrue(torch.equal(kept, torch.sort(kept).values))
        # window tokens [4, 5] always kept
        self.assertIn(4, kept.tolist())
        self.assertIn(5, kept.tolist())
        # top-2 past = indices 1 and 3
        self.assertIn(1, kept.tolist())
        self.assertIn(3, kept.tolist())
        self.assertEqual(sorted(kept.tolist()), [1, 3, 4, 5])


class TestCompactRequest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_layers = 2
        self.head_num = 2
        self.head_dim = 4
        self.budget = 4
        self.window = 2
        self.seq_len = 6
        self.req_pool_idx = 1

        self.num_slots = 32
        self.r2t_pool = MockReqToTokenPool(4, 64, self.device)
        self.kv_pool = MockKVPool(
            self.num_layers,
            self.num_slots,
            self.head_num,
            self.head_dim,
            self.device,
            self.dtype,
        )
        self.alloc = MockAllocator()

        cfg = RKVConfig(budget=self.budget, window_size=self.window, buffer_size=4)
        self.comp = RKVCompressor(
            config=cfg,
            req_to_token_pool=self.r2t_pool,
            token_to_kv_pool=self.kv_pool,
            kv_allocator=self.alloc,
            start_layer=0,
            end_layer=self.num_layers,
            device=self.device,
            q_head_num=self.head_num,
            head_dim=self.head_dim,
            q_dtype=self.dtype,
        )

        # Physical slots this request occupies, in temporal order.
        self.slots = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        self.r2t_pool.req_to_token[self.req_pool_idx, : self.seq_len] = self.slots

        # Fill each slot with a unique, recognisable pattern per layer so we can
        # verify relocation: value = slot * 100 + layer + 0.01 * (head + dim).
        for layer in range(self.num_layers):
            kb = self.kv_pool.get_key_buffer(layer)
            vb = self.kv_pool.get_value_buffer(layer)
            for s in self.slots.tolist():
                base = s * 100 + layer
                pattern = base + 0.01 * (
                    torch.arange(self.head_num).view(-1, 1)
                    + torch.arange(self.head_dim).view(1, -1)
                )
                kb[s] = pattern
                vb[s] = pattern + 0.5  # distinguish V from K

        self.state = RKVRequestState(req_pool_idx=self.req_pool_idx)
        # Register the state so the commit-phase identity guard resolves it
        # (the real flow registers via on_request_begin).
        self.comp.states[self.req_pool_idx] = self.state

    def test_relocation_free_and_rewrite(self):
        # Keep past tokens {0, 2} plus the window {4, 5}. Ascending.
        kept_local = torch.tensor([0, 2, 4, 5])
        src = self.slots[kept_local]  # [10, 12, 14, 15]
        dst = self.slots[: self.budget]  # [10, 11, 12, 13]

        # Snapshot the kept slots' KV BEFORE compaction (from src slots).
        expected_k = [
            self.kv_pool.get_key_buffer(l)[src].clone() for l in range(self.num_layers)
        ]
        expected_v = [
            self.kv_pool.get_value_buffer(l)[src].clone()
            for l in range(self.num_layers)
        ]

        _prepare_and_commit(self.comp, self.state, self.seq_len, kept_local)

        # (1) Front `budget` dst slots now hold the kept KV, in temporal order.
        for l in range(self.num_layers):
            self.assertTrue(
                torch.equal(self.kv_pool.get_key_buffer(l)[dst], expected_k[l]),
                f"K relocation mismatch at layer {l}",
            )
            self.assertTrue(
                torch.equal(self.kv_pool.get_value_buffer(l)[dst], expected_v[l]),
                f"V relocation mismatch at layer {l}",
            )

        # (2) Tail slots were freed exactly once with [14, 15].
        self.assertEqual(len(self.alloc.freed), 1)
        self.assertEqual(sorted(self.alloc.freed[0].tolist()), [14, 15])

        # (3) req_to_token: front budget unchanged (same physical slots),
        #     tail cleared to 0.
        row = self.r2t_pool.req_to_token[self.req_pool_idx]
        self.assertEqual(row[: self.budget].tolist(), dst.tolist())
        self.assertEqual(row[self.budget : self.seq_len].tolist(), [0, 0])

        # (4) Trigger counter reset; position offset NOT rewound.
        self.assertEqual(self.state.steps_since_compact, 0)

    def test_overlapping_src_dst_no_corruption(self):
        # Keep tokens whose src slots overlap dst heavily to stress the
        # clone-before-write path: keep {1, 3} + window {4, 5}.
        kept_local = torch.tensor([1, 3, 4, 5])
        src = self.slots[kept_local]  # [11, 13, 14, 15]
        dst = self.slots[: self.budget]  # [10, 11, 12, 13]  (overlaps src)

        expected_k = [
            self.kv_pool.get_key_buffer(l)[src].clone() for l in range(self.num_layers)
        ]

        _prepare_and_commit(self.comp, self.state, self.seq_len, kept_local)

        for l in range(self.num_layers):
            self.assertTrue(
                torch.equal(self.kv_pool.get_key_buffer(l)[dst], expected_k[l]),
                f"overlap corruption at layer {l}",
            )

    def test_length_bookkeeping_and_pending(self):
        # Attach a request whose logical length (120) far exceeds budget (4).
        req = _MockReq(origin_len=20, output_len=100, req_pool_idx=self.req_pool_idx)
        self.state.req = req
        kept_local = torch.tensor([0, 2, 4, 5])

        _prepare_and_commit(self.comp, self.state, self.seq_len, kept_local)

        # Physical length shrunk to budget on the request.
        self.assertEqual(req.kv_committed_len, self.budget)
        self.assertEqual(req.kv_allocated_len, self.budget)
        # Pending physical-length update exposed for the scheduler (with the
        # owning request for identity validation), drained once.
        self.assertEqual(
            self.comp.pending_length_updates[self.req_pool_idx][0], self.budget
        )
        self.assertIs(self.comp.pending_length_updates[self.req_pool_idx][1], req)
        drained = self.comp.take_pending_length_updates()
        self.assertEqual(drained[self.req_pool_idx][0], self.budget)
        self.assertIs(drained[self.req_pool_idx][1], req)
        self.assertEqual(self.comp.pending_length_updates, {})


class TestLogicalPosition(unittest.TestCase):
    def test_logical_position_ignores_eviction(self):
        # 20 prompt + 100 generated = 120 tokens seen; physical KV may be much
        # smaller after compaction, but the rotary position must not rewind.
        req = _MockReq(origin_len=20, output_len=100)
        self.assertEqual(RKVCompressor.logical_position(req), 120)


class TestBatchObserve(unittest.TestCase):
    """batch >= 2 per-request triggering (method A).

    Two requests share a decode batch; only the one whose own KV length reaches
    ``min_seq_len`` (and after ``buffer_size`` steps) may arm a compaction. The
    short request must never arm, even while the long one does.
    """

    def _build(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_layers = 2
        self.head_num = 2
        self.head_dim = 4
        self.window = 2
        self.budget = 4
        self.min_seq_len = 6
        self.buffer_size = 3

        self.r2t_pool = MockReqToTokenPool(4, 64, self.device)
        self.kv_pool = MockKVPool(
            self.num_layers, 64, self.head_num, self.head_dim, self.device, self.dtype
        )
        cfg = RKVConfig(
            budget=self.budget,
            window_size=self.window,
            buffer_size=self.buffer_size,
            min_seq_len=self.min_seq_len,
        )
        comp = RKVCompressor(
            config=cfg,
            req_to_token_pool=self.r2t_pool,
            token_to_kv_pool=self.kv_pool,
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=self.num_layers,
            device=self.device,
            q_head_num=self.head_num,
            head_dim=self.head_dim,
            q_dtype=self.dtype,
        )

        # req A (idx 1): long enough to be eligible; req B (idx 2): short.
        self.long_idx, self.long_len = 1, 6
        self.short_idx, self.short_len = 2, 3
        comp.on_request_begin(_MockReq(20, 0, req_pool_idx=self.long_idx))
        comp.on_request_begin(_MockReq(3, 0, req_pool_idx=self.short_idx))

        # Physical slots so _layer_score can gather real KV for the long req.
        self.r2t_pool.req_to_token[self.long_idx, : self.long_len] = torch.arange(
            10, 10 + self.long_len, dtype=torch.int32
        )
        self.r2t_pool.req_to_token[self.short_idx, : self.short_len] = torch.arange(
            30, 30 + self.short_len, dtype=torch.int32
        )
        for layer in range(self.num_layers):
            self.kv_pool.get_key_buffer(layer).normal_()

        self.comp = comp

    def _forward_batch(self):
        return types.SimpleNamespace(
            # req_pool_indices is int64 in the real runtime (schedule_batch and
            # every cuda-graph runner allocate it as torch.int64); match that so
            # the in-graph collect_decode_query index arithmetic is a long index.
            req_pool_indices=torch.tensor(
                [self.long_idx, self.short_idx], dtype=torch.int64
            ),
            seq_lens=torch.tensor([self.long_len, self.short_len], dtype=torch.int32),
            seq_lens_cpu=torch.tensor(
                [self.long_len, self.short_len], dtype=torch.int32
            ),
        )

    def test_only_eligible_request_arms(self):
        self._build()
        torch.manual_seed(0)
        fb = self._forward_batch()

        # Feed buffer_size decode steps; each step refreshes the write slot and
        # collects every layer's query into the in-graph rolling buffer.
        for _ in range(self.buffer_size):
            self.comp.begin_decode_step(fb)
            for layer_idx in range(self.num_layers):
                layer = types.SimpleNamespace(layer_id=layer_idx)
                q = torch.randn(2, self.head_num, self.head_dim)
                self.comp.collect_decode_query(q, layer, fb)

        # Long request armed; short request did not.
        self.assertIn(self.long_idx, self.comp._armed)
        self.assertNotIn(self.short_idx, self.comp._armed)

        # begin_decode_step advanced only the eligible (long) request; the short
        # request's seq_len stays below min_seq_len so its clock never starts.
        self.assertEqual(
            self.comp.states[self.long_idx].steps_since_compact, self.buffer_size
        )
        self.assertEqual(self.comp.states[self.short_idx].steps_since_compact, 0)

        # Scoring reads the observation window from the rolling buffer, exactly
        # as maybe_compact does. Batched scoring must match the per-layer
        # reference (the invariant the A/B gate relies on).
        long_state = self.comp.states[self.long_idx]
        long_state.window_q = self.comp._read_window_rolling(self.long_idx)
        ref = self.comp._reference_scores(long_state, self.long_len)
        bat = self.comp._batched_scores(long_state, self.long_len)
        self.assertEqual(ref.shape, (self.long_len - self.window,))
        self.assertEqual(bat.shape, (self.long_len - self.window,))
        self.assertTrue(torch.allclose(ref, bat, atol=1e-4))

    def test_eager_gating_schedule(self):
        # begin_decode_step forces EAGER only on the compaction step now: the
        # window queries are collected in-graph, so the observation-window steps
        # replay the captured graph. window=2, buffer=3 -> compaction (and the
        # only eager step) at steps_since_compact==3.
        self._build()
        fb = self._forward_batch()
        needs = [self.comp.begin_decode_step(fb) for _ in range(self.buffer_size)]
        self.assertEqual(needs, [False, False, True])
        # Last step is the compaction step: the long request armed.
        self.assertIn(self.long_idx, self.comp._armed)
        # The short request (below min_seq_len) never arms or forces eager.
        self.assertNotIn(self.short_idx, self.comp._armed)
        self.assertEqual(self.comp.states[self.short_idx].steps_since_compact, 0)

    def test_maybe_compact_uses_per_request_seq_len(self):
        self._build()
        torch.manual_seed(1)
        fb = self._forward_batch()
        for _ in range(self.buffer_size):
            self.comp.begin_decode_step(fb)
            for layer_idx in range(self.num_layers):
                layer = types.SimpleNamespace(layer_id=layer_idx)
                q = torch.randn(2, self.head_num, self.head_dim)
                self.comp.collect_decode_query(q, layer, fb)

        self.comp.maybe_compact(fb)
        self.comp.commit_compactions()

        # Only the long request was compacted, shrunk to budget.
        self.assertEqual(
            self.comp.pending_length_updates[self.long_idx][0], self.budget
        )
        self.assertNotIn(self.short_idx, self.comp.pending_length_updates)
        self.assertEqual(len(self.comp._armed), 0)


class TestLifecycle(unittest.TestCase):
    def _compressor(self):
        return RKVCompressor(
            config=RKVConfig(budget=64, window_size=8),
            req_to_token_pool=MockReqToTokenPool(4, 128, torch.device("cpu")),
            token_to_kv_pool=MockKVPool(
                2, 128, 2, 4, torch.device("cpu"), torch.float32
            ),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=2,
            device=torch.device("cpu"),
            q_head_num=2,
            head_dim=4,
            q_dtype=torch.float32,
        )

    def test_begin_registers_and_end_clears_state(self):
        comp = self._compressor()
        req = _MockReq(origin_len=10, output_len=0, req_pool_idx=2)
        comp.on_request_begin(req)
        self.assertIn(2, comp.states)
        self.assertIs(comp.states[2].req, req)
        comp.on_request_end(req)
        self.assertNotIn(2, comp.states)
        self.assertEqual(len(comp.states), 0)

    def test_end_is_idempotent(self):
        comp = self._compressor()
        req = _MockReq(origin_len=10, output_len=0, req_pool_idx=2)
        comp.on_request_end(req)  # no state yet -> must not raise
        comp.on_request_begin(req)
        comp.on_request_end(req)
        comp.on_request_end(req)  # double end -> must not raise
        self.assertEqual(len(comp.states), 0)


class TestCompactInvariants(unittest.TestCase):
    """A bad kept set must fail fast BEFORE any KV buffer is mutated or freed,
    so a scoring/selection bug can never leave a request half-relocated or
    double-free a physical slot (crash-consistency)."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.num_layers = 2
        self.head_num = 2
        self.head_dim = 4
        self.budget = 4
        self.window = 2
        self.seq_len = 6
        self.req_pool_idx = 1
        self.num_slots = 32

        self.r2t_pool = MockReqToTokenPool(4, 64, self.device)
        self.kv_pool = MockKVPool(
            self.num_layers,
            self.num_slots,
            self.head_num,
            self.head_dim,
            self.device,
            torch.float32,
        )
        self.alloc = MockAllocator()
        self.comp = RKVCompressor(
            config=RKVConfig(
                budget=self.budget, window_size=self.window, buffer_size=4
            ),
            req_to_token_pool=self.r2t_pool,
            token_to_kv_pool=self.kv_pool,
            kv_allocator=self.alloc,
            start_layer=0,
            end_layer=self.num_layers,
            device=self.device,
            q_head_num=self.head_num,
            head_dim=self.head_dim,
            q_dtype=torch.float32,
        )
        self.slots = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        self.r2t_pool.req_to_token[self.req_pool_idx, : self.seq_len] = self.slots
        self._k_snapshot = [
            self.kv_pool.get_key_buffer(l).clone() for l in range(self.num_layers)
        ]
        self.state = RKVRequestState(req_pool_idx=self.req_pool_idx)
        # Register the state so the commit-phase identity guard resolves it.
        self.comp.states[self.req_pool_idx] = self.state

    def _assert_no_mutation(self):
        # No slot was freed and no KV buffer changed.
        self.assertEqual(len(self.alloc.freed), 0)
        for l in range(self.num_layers):
            self.assertTrue(
                torch.equal(self.kv_pool.get_key_buffer(l), self._k_snapshot[l])
            )

    def test_wrong_length_raises(self):
        with self.assertRaises(RuntimeError):
            self.comp._prepare_compaction(
                self.state, self.seq_len, torch.tensor([0, 2, 5])  # 3 != budget 4
            )
        self._assert_no_mutation()

    def test_non_ascending_raises(self):
        with self.assertRaises(RuntimeError):
            # descending / unsorted kept indices
            self.comp._prepare_compaction(
                self.state, self.seq_len, torch.tensor([3, 1, 4, 5])
            )
        self._assert_no_mutation()

    def test_duplicate_index_raises(self):
        with self.assertRaises(RuntimeError):
            # 4 appears twice -> not strictly ascending AND duplicate slot
            self.comp._prepare_compaction(
                self.state, self.seq_len, torch.tensor([1, 4, 4, 5])
            )
        self._assert_no_mutation()

    def test_out_of_range_raises(self):
        with self.assertRaises(RuntimeError):
            # index 6 >= seq_len 6
            self.comp._prepare_compaction(
                self.state, self.seq_len, torch.tensor([0, 2, 4, 6])
            )
        self._assert_no_mutation()

    def test_duplicate_physical_slot_raises(self):
        # Corrupt req_to_token: slot 10 appears twice. A valid ascending kept set
        # keeps only ONE copy, so src is unique, but the freed tail would then
        # contain slot 10 which req_to_token[:budget] still references (a
        # use-after-free). The full-table uniqueness guard must catch this before
        # any KV write or free.
        self.r2t_pool.req_to_token[self.req_pool_idx, : self.seq_len] = torch.tensor(
            [10, 11, 12, 10, 14, 15], dtype=torch.int32
        )
        with self.assertRaises(RuntimeError):
            self.comp._prepare_compaction(
                self.state, self.seq_len, torch.tensor([0, 1, 4, 5])
            )
        self._assert_no_mutation()

    def test_valid_kept_still_compacts(self):
        # Sanity: a valid ascending, in-range, budget-sized kept set proceeds.
        _prepare_and_commit(
            self.comp, self.state, self.seq_len, torch.tensor([0, 2, 4, 5])
        )
        self.assertEqual(len(self.alloc.freed), 1)


class TestMultiRequestAndFailure(unittest.TestCase):
    """Two-phase compaction across several requests, and its failure semantics."""

    def _build(self, budget=4, window=2, seq_len=6, allocator=None):
        device = torch.device("cpu")
        self.num_layers, self.head_num, self.head_dim = 2, 2, 4
        self.budget, self.window, self.seq_len = budget, window, seq_len
        r2t = MockReqToTokenPool(8, 64, device)
        kv = MockKVPool(
            self.num_layers, 128, self.head_num, self.head_dim, device, torch.float32
        )
        alloc = allocator if allocator is not None else MockAllocator()
        comp = RKVCompressor(
            config=RKVConfig(budget=budget, window_size=window, buffer_size=4),
            req_to_token_pool=r2t,
            token_to_kv_pool=kv,
            kv_allocator=alloc,
            start_layer=0,
            end_layer=self.num_layers,
            device=device,
            q_head_num=self.head_num,
            head_dim=self.head_dim,
            q_dtype=torch.float32,
        )
        return comp, r2t, alloc

    def _register(self, comp, r2t, idx, i):
        req = _MockReq(origin_len=20, output_len=100, req_pool_idx=idx)
        comp.on_request_begin(req)
        base = 10 + i * 10  # distinct physical slots per request
        r2t.req_to_token[idx, : self.seq_len] = torch.arange(
            base, base + self.seq_len, dtype=torch.int32
        )
        return req, base

    def test_multiple_requests_same_step(self):
        # 3 requests armed the same forward: all prepare, all commit, freed
        # counts + pending lengths correct, queue drained.
        comp, r2t, alloc = self._build()
        reqs = {}
        for i, idx in enumerate((1, 2, 3)):
            req, base = self._register(comp, r2t, idx, i)
            reqs[idx] = (req, base)
            comp._pending_commits.append(
                comp._prepare_compaction(
                    comp.states[idx], self.seq_len, torch.tensor([0, 2, 4, 5])
                )
            )
        self.assertEqual(len(comp._pending_commits), 3)

        comp.commit_compactions()

        self.assertEqual(len(alloc.freed), 3)  # one free per request
        self.assertEqual(len(comp._pending_commits), 0)  # queue drained
        for idx, (req, base) in reqs.items():
            self.assertEqual(comp.pending_length_updates[idx][0], self.budget)
            self.assertEqual(req.kv_committed_len, self.budget)
            # freed = the physical tail [budget, seq_len) of THIS request
            self.assertEqual(
                sorted(
                    s for f in alloc.freed for s in f.tolist() if base <= s < base + 10
                ),
                [base + 4, base + 5],
            )

    def test_compact_then_finish_no_double_free(self):
        # After the commit frees the tail, a same-step finish would release the
        # shrunk head (req_to_token[:budget]). The two freed sets must be disjoint.
        comp, r2t, alloc = self._build()
        req, base = self._register(comp, r2t, 1, 0)
        comp._pending_commits.append(
            comp._prepare_compaction(
                comp.states[1], self.seq_len, torch.tensor([0, 2, 4, 5])
            )
        )
        comp.commit_compactions()

        tail_freed = set(alloc.freed[0].tolist())  # {base+4, base+5}
        head = r2t.req_to_token[1, : req.kv_committed_len].tolist()  # retained head
        self.assertTrue(
            tail_freed.isdisjoint(set(head)), "tail/head overlap => double free"
        )
        # req_to_token tail was cleared to 0.
        self.assertEqual(
            r2t.req_to_token[1, self.budget : self.seq_len].tolist(), [0, 0]
        )

    def test_commit_failure_does_not_recommit(self):
        # A mid-drain allocator failure must not re-commit the already-freed plan
        # on a retry: commit_compactions drains _pending_commits UP FRONT.
        class RaisingAllocator:
            def __init__(self, raise_on):
                self.freed = []
                self._n = 0
                self._raise_on = raise_on

            def free(self, idx):
                self._n += 1
                if self._n == self._raise_on:
                    raise RuntimeError("simulated allocator failure")
                self.freed.append(idx.clone())

        comp, r2t, alloc = self._build(allocator=RaisingAllocator(raise_on=2))
        for i, idx in enumerate((1, 2)):
            self._register(comp, r2t, idx, i)
            comp._pending_commits.append(
                comp._prepare_compaction(
                    comp.states[idx], self.seq_len, torch.tensor([0, 2, 4, 5])
                )
            )
        with self.assertRaises(RuntimeError):
            comp.commit_compactions()
        # First plan committed (freed once); queue drained so a retry is a no-op
        # and cannot re-free the first plan's slots.
        self.assertEqual(len(alloc.freed), 1)
        self.assertEqual(len(comp._pending_commits), 0)
        comp.commit_compactions()  # no-op
        self.assertEqual(len(alloc.freed), 1)

    def test_stale_plan_identity_raises(self):
        # A plan whose request slot was released/reused before commit is an
        # unrecoverable invariant break -> RuntimeError (not a silent skip).
        comp, r2t, alloc = self._build()
        req, _ = self._register(comp, r2t, 1, 0)
        comp._pending_commits.append(
            comp._prepare_compaction(
                comp.states[1], self.seq_len, torch.tensor([0, 2, 4, 5])
            )
        )
        comp.on_request_end(req)  # slot released -> state dropped
        with self.assertRaises(RuntimeError):
            comp.commit_compactions()


class TestRollingQSizeContract(unittest.TestCase):
    """Pin the rolling_q allocation shape/bytes that the KV-pool reservation
    (ModelRunner._reserve_rkv_decode_aux_bytes) is computed from. If the buffer
    layout changes, the reservation formula must change with it — this test
    fails loudly to force that."""

    def test_bytes_match_reservation_formula(self):
        num_reqs, num_layers, window = 5, 3, 8
        q_heads, head_dim = 7, 16
        dtype = torch.bfloat16
        comp = RKVCompressor(
            config=RKVConfig(budget=64, window_size=window),
            req_to_token_pool=MockReqToTokenPool(num_reqs, 128, torch.device("cpu")),
            token_to_kv_pool=MockKVPool(
                num_layers, 128, 2, head_dim, torch.device("cpu"), dtype
            ),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=num_layers,
            device=torch.device("cpu"),
            q_head_num=q_heads,
            head_dim=head_dim,
            q_dtype=dtype,
        )
        # Rows == req_to_token rows == num_reqs + 1 (reserved padding row).
        expected_rows = num_reqs + 1
        self.assertEqual(comp.rolling_q.shape[2], expected_rows)
        actual_bytes = comp.rolling_q.numel() * comp.rolling_q.element_size()
        # Same product the mixin helper computes.
        formula_bytes = (
            num_layers
            * window
            * expected_rows
            * q_heads
            * head_dim
            * torch.finfo(dtype).bits
            // 8
        )
        self.assertEqual(actual_bytes, formula_bytes)


class TestPerRequestCursor(unittest.TestCase):
    """R3: a per-request write cursor keeps a request's observation window
    correctly ordered even when it SKIPS decode steps. The old single global
    cursor advanced on every step regardless of batch membership, so a skipped
    step left a phantom (unwritten) slot BETWEEN a request's real queries."""

    def _build(self, window=3):
        device = torch.device("cpu")
        comp = RKVCompressor(
            config=RKVConfig(
                budget=window + 1, window_size=window, buffer_size=window + 1
            ),
            req_to_token_pool=MockReqToTokenPool(4, 64, device),
            token_to_kv_pool=MockKVPool(1, 64, 1, 1, device, torch.float32),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=1,
            device=device,
            q_head_num=1,
            head_dim=1,
            q_dtype=torch.float32,
        )
        return comp

    @staticmethod
    def _fb(indices):
        idx = torch.tensor(indices, dtype=torch.int64)
        # seq_lens below min_seq_len so nothing arms; only exercise the cursor.
        sl = torch.ones(len(indices), dtype=torch.int32)
        return types.SimpleNamespace(req_pool_indices=idx, seq_lens=sl, seq_lens_cpu=sl)

    def _step(self, comp, indices, tags):
        fb = self._fb(indices)
        comp.begin_decode_step(fb)
        q = torch.tensor(tags, dtype=torch.float32).view(-1, 1, 1)
        comp.collect_decode_query(q, types.SimpleNamespace(layer_id=0), fb)

    def test_skipped_step_keeps_window_order(self):
        comp = self._build(window=3)
        A, B = 1, 2
        comp.on_request_begin(_MockReq(1, 0, req_pool_idx=A))
        comp.on_request_begin(_MockReq(1, 0, req_pool_idx=B))

        # B participates in steps 1 and 3 but is ABSENT from step 2.
        self._step(comp, [A, B], [10.0, 100.0])  # qB1 = 100
        self._step(comp, [A], [11.0])  # B absent (skipped step)
        self._step(comp, [A, B], [12.0, 200.0])  # qB2 = 200

        wB = comp._read_window_rolling(B)[0, :, 0, 0]  # (window,) temporal order
        # B's two real queries must be the most recent two, IN ORDER, with the
        # unwritten slot pushed to the oldest end (not wedged between them).
        self.assertEqual(wB[-1].item(), 200.0)
        self.assertEqual(wB[-2].item(), 100.0)
        # No A query leaked into B's window.
        self.assertNotIn(11.0, wB.tolist())
        self.assertNotIn(12.0, wB.tolist())

        # A participated every step: newest-two in order = [11, 12].
        wA = comp._read_window_rolling(A)[0, :, 0, 0]
        self.assertEqual(wA[-1].item(), 12.0)
        self.assertEqual(wA[-2].item(), 11.0)

    def test_reused_slot_resets_cursor(self):
        comp = self._build(window=3)
        A = 1
        comp.on_request_begin(_MockReq(1, 0, req_pool_idx=A))
        self._step(comp, [A], [1.0])
        self._step(comp, [A], [2.0])
        self.assertEqual(int(comp.step_count_of_req[A].item()), 2)
        # A finishes; a new request reuses the same slot -> counter resets.
        comp.on_request_end(_MockReq(1, 0, req_pool_idx=A))
        comp.on_request_begin(_MockReq(1, 0, req_pool_idx=A))
        self.assertEqual(int(comp.step_count_of_req[A].item()), 0)

    def test_reused_slot_window_has_no_stale_q(self):
        # Request A fills its whole observation window, finishes, then request B
        # reuses the SAME req_pool_idx. After B has run window_size steps its
        # window must contain ONLY B's queries in order (no stale A data) — the
        # invariant buffer_size >= window_size relies on, verified directly.
        window = 3
        comp = self._build(window=window)
        X = 1  # shared req_pool_idx
        comp.on_request_begin(_MockReq(1, 0, req_pool_idx=X))
        for tag in (100.0, 101.0, 102.0):  # fill A's window
            self._step(comp, [X], [tag])
        wA = comp._read_window_rolling(X)[0, :, 0, 0]
        self.assertEqual(wA.tolist(), [100.0, 101.0, 102.0])

        comp.on_request_end(_MockReq(1, 0, req_pool_idx=X))
        comp.on_request_begin(_MockReq(1, 0, req_pool_idx=X))  # B reuses slot X
        for tag in (200.0, 201.0, 202.0):  # B's window_size fresh writes
            self._step(comp, [X], [tag])

        wB = comp._read_window_rolling(X)[0, :, 0, 0]
        # Only B's queries, in temporal order; no A residue.
        self.assertEqual(wB.tolist(), [200.0, 201.0, 202.0])
        for stale in (100.0, 101.0, 102.0):
            self.assertNotIn(stale, wB.tolist())


class _MockTPGroup:
    """Duck-typed stand-in for a ``GroupCoordinator`` attention-TP group.

    ``all_reduce`` here simulates the collective on a SINGLE process: the
    default ``fn`` (``x * world_size``) models the physically-correct case where
    every rank contributed the identical tensor (so the SUM is ``world_size``
    times one rank's value). A custom ``fn`` lets a test simulate ranks that
    DISAGREE, to prove the consistency check fires.
    """

    def __init__(self, world_size, fn=None):
        self.world_size = world_size
        self._fn = fn if fn is not None else (lambda x: x * world_size)
        self.n_all_reduce = 0

    def all_reduce(self, x):
        self.n_all_reduce += 1
        return self._fn(x)


class TestTensorParallelScore(unittest.TestCase):
    """TP: the per-token eviction score is SUMMED across the attention-TP group
    before top-k so every rank keeps the identical tokens (which keeps the
    replicated ``req_to_token`` consistent). Validated on CPU with a mock group:
    uniform scaling is top-k invariant, and a divergent reduce raises loudly."""

    def _compressor(self, group):
        return RKVCompressor(
            config=RKVConfig(budget=4, window_size=2, buffer_size=4),
            req_to_token_pool=MockReqToTokenPool(4, 64, torch.device("cpu")),
            token_to_kv_pool=MockKVPool(
                1, 64, 1, 4, torch.device("cpu"), torch.float32
            ),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=1,
            device=torch.device("cpu"),
            q_head_num=1,
            head_dim=4,
            q_dtype=torch.float32,
            attn_tp_group=group,
        )

    def test_reduce_noop_without_group(self):
        # tp==1 (no group): the reduce returns the score object unchanged.
        comp = self._compressor(None)
        self.assertEqual(comp.attn_tp_size, 1)
        score = torch.tensor([0.1, 9.0, 0.2, 8.0])
        self.assertIs(comp._reduce_score_across_tp(score), score)

    def test_reduce_sums_and_preserves_topk(self):
        # tp==2, ranks identical -> reduced score = 2 x local; top-k unchanged.
        g = _MockTPGroup(2)
        comp = self._compressor(g)
        self.assertEqual(comp.attn_tp_size, 2)
        score = torch.tensor([0.1, 9.0, 0.2, 8.0])
        reduced = comp._reduce_score_across_tp(score)
        self.assertEqual(g.n_all_reduce, 1)
        self.assertTrue(torch.allclose(reduced, score.float() * 2))
        # The kept set derived from the summed score must equal the single-rank
        # kept set: positive uniform scaling cannot change which tokens win.
        seq_len = 6  # past pool = 4 tokens, trailing window = 2 -> [4, 5]
        kept_tp = comp._assemble_kept(reduced, seq_len)
        kept_plain = comp._assemble_kept(score, seq_len)
        self.assertTrue(torch.equal(kept_tp, kept_plain))

    def test_reduce_casts_to_fp32(self):
        # Score all-reduce is done in fp32 so ties break identically per rank.
        g = _MockTPGroup(2)
        comp = self._compressor(g)
        score = torch.tensor([0.1, 9.0, 0.2, 8.0], dtype=torch.float16)
        reduced = comp._reduce_score_across_tp(score)
        self.assertEqual(reduced.dtype, torch.float32)

    def test_consistency_check_passes_when_ranks_agree(self):
        # Identical ranks -> SUM == local * world_size -> no raise.
        g = _MockTPGroup(4)
        comp = self._compressor(g)
        comp._check_kept_consistent_across_tp(torch.tensor([1, 3, 4, 5]))
        self.assertEqual(g.n_all_reduce, 1)

    def test_consistency_check_raises_on_divergence(self):
        # Ranks disagree: the mocked SUM != local * world_size -> loud failure.
        g = _MockTPGroup(2, fn=lambda x: x + 1.0)
        comp = self._compressor(g)
        with self.assertRaises(RuntimeError):
            comp._check_kept_consistent_across_tp(torch.tensor([1, 3, 4, 5]))

    def test_consistency_check_noop_without_group(self):
        # tp==1: no collective is issued at all.
        comp = self._compressor(None)
        comp._check_kept_consistent_across_tp(torch.tensor([1, 3, 4, 5]))  # no raise

    def test_consistency_check_fire_schedule(self):
        # Default (no SGLANG_RKV_TP_CHECK): self-validates only the first few
        # compactions, then stops issuing the extra collective.
        g = _MockTPGroup(2)
        comp = self._compressor(g)
        budget = comp._tp_check_remaining
        self.assertGreater(budget, 0)
        for _ in range(budget):
            comp._check_kept_consistent_across_tp(torch.tensor([1, 3, 4, 5]))
        fired = g.n_all_reduce
        self.assertEqual(fired, budget)
        # Past the startup budget the check no longer issues a collective.
        for _ in range(3):
            comp._check_kept_consistent_across_tp(torch.tensor([1, 3, 4, 5]))
        self.assertEqual(g.n_all_reduce, fired)


class TestPendingUpdateLifecycle(unittest.TestCase):
    """Issue 1 regressions: no pending physical-length update may survive a
    finish/retract and be applied to the next request that reuses the slot."""

    def _build(self):
        device = torch.device("cpu")
        r2t = MockReqToTokenPool(8, 64, device)
        kv = MockKVPool(2, 128, 2, 4, device, torch.float32)
        comp = RKVCompressor(
            config=RKVConfig(budget=4, window_size=2, buffer_size=4),
            req_to_token_pool=r2t,
            token_to_kv_pool=kv,
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=2,
            device=device,
            q_head_num=2,
            head_dim=4,
            q_dtype=torch.float32,
        )
        return comp, r2t

    def _compact(self, comp, r2t, idx):
        """Register a request at ``idx``, prepare + commit one compaction (which
        publishes a pending length update), and return the request."""
        req = _MockReq(origin_len=20, output_len=100, req_pool_idx=idx)
        comp.on_request_begin(req)
        r2t.req_to_token[idx, :6] = torch.arange(10, 16, dtype=torch.int32)
        comp._pending_commits.append(
            comp._prepare_compaction(comp.states[idx], 6, torch.tensor([0, 2, 4, 5]))
        )
        comp.commit_compactions()
        return req

    def test_pending_update_carries_owner_identity(self):
        comp, r2t = self._build()
        req = self._compact(comp, r2t, 1)
        new_len, owner = comp.pending_length_updates[1]
        self.assertEqual(new_len, 4)
        self.assertIs(owner, req)  # identity attached for scheduler validation

    def test_finish_clears_pending_update(self):
        comp, r2t = self._build()
        req = self._compact(comp, r2t, 1)
        self.assertIn(1, comp.pending_length_updates)  # commit published it
        comp.on_request_end(req)  # compact + finish on the same step
        self.assertNotIn(1, comp.pending_length_updates)  # finish cleared it
        self.assertNotIn(1, comp.states)

    def test_retract_clears_pending_update(self):
        comp, r2t = self._build()
        req = self._compact(comp, r2t, 1)
        self.assertIn(1, comp.pending_length_updates)
        comp.on_request_retract(req)
        self.assertNotIn(1, comp.pending_length_updates)
        self.assertNotIn(1, comp.states)

    def test_slot_reuse_after_finish_leaves_no_stale_update(self):
        # A compacts at idx 1 and finishes as the last request; the running
        # batch goes empty so the scheduler never drained the update. B then
        # reuses idx 1. The drain must not carry A's stale budget onto B.
        comp, r2t = self._build()
        req_a = self._compact(comp, r2t, 1)
        comp.on_request_end(req_a)
        req_b = _MockReq(origin_len=7, output_len=0, req_pool_idx=1)
        comp.on_request_begin(req_b)  # B reuses slot 1
        self.assertEqual(comp.take_pending_length_updates(), {})

    def test_stale_update_owner_mismatches_reused_request(self):
        # If a clear were ever missed, the update's owner is still request A, not
        # the request B that reuses the slot -- so the scheduler identity guard
        # (owner is req) skips it. Pin that the owner is NOT the reused request.
        comp, r2t = self._build()
        req_a = self._compact(comp, r2t, 1)
        _, owner = comp.pending_length_updates[1]
        req_b = _MockReq(origin_len=7, output_len=0, req_pool_idx=1)
        comp.on_request_begin(req_b)
        self.assertIs(owner, req_a)
        self.assertIsNot(owner, req_b)


class TestRuntimeSupportGate(unittest.TestCase):
    """Issue 2: the late (post-resolution) gate hard-fails any runtime the R-KV
    observation/compaction hooks are not wired for."""

    def _reason(self, **over):
        kw = dict(
            mode="decode",
            prefill_backend="flashinfer",
            decode_backend="flashinfer",
            use_mla=False,
            is_hybrid_swa=False,
            spec_enabled=False,
            page_size=1,
        )
        kw.update(over)
        return rkv_runtime_support_error(**kw)

    def test_flashinfer_mha_supported(self):
        self.assertIsNone(self._reason())
        self.assertIsNone(self._reason(mode="prefill", page_size=None))

    def test_non_flashinfer_backend_rejected(self):
        self.assertIsNotNone(self._reason(decode_backend="fa3"))
        self.assertIsNotNone(self._reason(prefill_backend="triton"))
        self.assertIsNotNone(self._reason(decode_backend=None))

    def test_mla_rejected(self):
        self.assertIsNotNone(self._reason(use_mla=True))

    def test_hybrid_swa_rejected(self):
        self.assertIsNotNone(self._reason(is_hybrid_swa=True))

    def test_speculative_rejected(self):
        self.assertIsNotNone(self._reason(spec_enabled=True))

    def test_page_size_gt_one_rejected(self):
        self.assertIsNotNone(self._reason(page_size=16))
        self.assertIsNone(self._reason(page_size=1))
        self.assertIsNone(self._reason(page_size=None))


class TestServingConfigValidation(unittest.TestCase):
    """Issue 3: config-level guards + serving restricted to the bounded path."""

    def _serve(self, **cfg):
        return RKVCompressor(
            config=RKVConfig(**cfg),
            req_to_token_pool=MockReqToTokenPool(4, 128, torch.device("cpu")),
            token_to_kv_pool=MockKVPool(
                2, 128, 2, 4, torch.device("cpu"), torch.float32
            ),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=2,
            device=torch.device("cpu"),
            q_head_num=2,
            head_dim=4,
            q_dtype=torch.float32,
        )

    def test_even_kernel_size_rejected(self):
        with self.assertRaises(ValueError):
            RKVConfig(budget=64, window_size=8, kernel_size=8)

    def test_nonpositive_window_rejected(self):
        with self.assertRaises(ValueError):
            RKVConfig(budget=64, window_size=0)

    def test_unknown_retain_direction_rejected(self):
        with self.assertRaises(ValueError):
            RKVConfig(budget=64, window_size=8, retain_direction="middle")

    def test_odd_kernel_and_known_retain_accepted(self):
        RKVConfig(budget=64, window_size=8, kernel_size=7, retain_direction="first")

    def test_serving_rejects_non_last_retain(self):
        # The algorithm accepts other directions offline, but the SERVED
        # compressor is restricted to the memory-bounded retain_direction="last"
        # path (others build an unbounded kv_heads x n x n matrix).
        with self.assertRaises(ValueError):
            self._serve(budget=64, window_size=8, retain_direction="first")

    def test_serving_accepts_last_retain(self):
        comp = self._serve(budget=64, window_size=8, retain_direction="last")
        self.assertEqual(comp.config.retain_direction, "last")


if __name__ == "__main__":
    unittest.main(verbosity=2)
