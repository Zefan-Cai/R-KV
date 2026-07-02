"""CPU unit tests for the R-KV integration layer's physical compaction.

These tests are GPU-free and do not require an installed ``sglang`` package.
The pure algorithm module and the integration module are loaded directly by
file path (the integration module's absolute ``from sglang...algo import R1KV``
is satisfied by pre-registering the algo module under its real dotted name in
``sys.modules``), so we bypass ``sglang/__init__.py`` and the serving stack.

Run directly::

    python tests/test_rkv_integration.py

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

_load_by_path(
    "sglang.srt.mem_cache.rkv.algo", os.path.join(_RKV_DIR, "algo.py")
)
_integration = _load_by_path(
    "sglang.srt.mem_cache.rkv.integration", os.path.join(_RKV_DIR, "integration.py")
)

RKVConfig = _integration.RKVConfig
RKVRequestState = _integration.RKVRequestState
RKVCompressor = _integration.RKVCompressor


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


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
class TestAssembleKept(unittest.TestCase):
    def _compressor(self, budget, window):
        cfg = RKVConfig(budget=budget, window_size=window, buffer_size=4)
        return RKVCompressor(
            config=cfg,
            req_to_token_pool=MockReqToTokenPool(1, 64, torch.device("cpu")),
            token_to_kv_pool=MockKVPool(1, 64, 1, 4, torch.device("cpu"), torch.float32),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=1,
            device=torch.device("cpu"),
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

        self.state = RKVRequestState(
            req_pool_idx=self.req_pool_idx,
            num_layers=self.num_layers,
            window_size=self.window,
            device=self.device,
            dtype=self.dtype,
        )

    def test_relocation_free_and_rewrite(self):
        # Keep past tokens {0, 2} plus the window {4, 5}. Ascending.
        kept_local = torch.tensor([0, 2, 4, 5])
        src = self.slots[kept_local]              # [10, 12, 14, 15]
        dst = self.slots[: self.budget]           # [10, 11, 12, 13]

        # Snapshot the kept slots' KV BEFORE compaction (from src slots).
        expected_k = [
            self.kv_pool.get_key_buffer(l)[src].clone()
            for l in range(self.num_layers)
        ]
        expected_v = [
            self.kv_pool.get_value_buffer(l)[src].clone()
            for l in range(self.num_layers)
        ]

        self.comp._compact_request(self.state, self.seq_len, kept_local)

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
        src = self.slots[kept_local]              # [11, 13, 14, 15]
        dst = self.slots[: self.budget]           # [10, 11, 12, 13]  (overlaps src)

        expected_k = [
            self.kv_pool.get_key_buffer(l)[src].clone()
            for l in range(self.num_layers)
        ]

        self.comp._compact_request(self.state, self.seq_len, kept_local)

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

        self.comp._compact_request(self.state, self.seq_len, kept_local)

        # Physical length shrunk to budget on the request.
        self.assertEqual(req.kv_committed_len, self.budget)
        self.assertEqual(req.kv_allocated_len, self.budget)
        # Pending physical-length update exposed for the scheduler, drained once.
        self.assertEqual(
            self.comp.pending_length_updates[self.req_pool_idx], self.budget
        )
        drained = self.comp.take_pending_length_updates()
        self.assertEqual(drained, {self.req_pool_idx: self.budget})
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
            req_pool_indices=torch.tensor(
                [self.long_idx, self.short_idx], dtype=torch.int32
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

        # Feed buffer_size decode steps; each step touches all layers.
        for _ in range(self.buffer_size):
            for layer_idx in range(self.num_layers):
                layer = types.SimpleNamespace(layer_id=layer_idx)
                q = torch.randn(2, self.head_num, self.head_dim)
                self.comp.observe_decode_layer(q, None, None, layer, fb)

        # Long request armed; short request did not.
        self.assertIn(self.long_idx, self.comp._armed)
        self.assertNotIn(self.short_idx, self.comp._armed)

        # begin_step advanced both requests once per step (per-request state).
        self.assertEqual(
            self.comp.states[self.long_idx].steps_since_compact, self.buffer_size
        )
        self.assertEqual(
            self.comp.states[self.short_idx].steps_since_compact, self.buffer_size
        )

        # Armed request accumulated a per-past-token score of the right shape.
        accum = self.comp.states[self.long_idx].score_accum
        self.assertIsNotNone(accum)
        self.assertEqual(accum.shape, (self.long_len - self.window,))
        self.assertIsNone(self.comp.states[self.short_idx].score_accum)

    def test_maybe_compact_uses_per_request_seq_len(self):
        self._build()
        torch.manual_seed(1)
        fb = self._forward_batch()
        for _ in range(self.buffer_size):
            for layer_idx in range(self.num_layers):
                layer = types.SimpleNamespace(layer_id=layer_idx)
                q = torch.randn(2, self.head_num, self.head_dim)
                self.comp.observe_decode_layer(q, None, None, layer, fb)

        self.comp.maybe_compact(fb)

        # Only the long request was compacted, shrunk to budget.
        self.assertEqual(
            self.comp.pending_length_updates.get(self.long_idx), self.budget
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
