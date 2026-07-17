"""Integration tests for the R-KV paged-cache compaction (vLLM port).

These exercise ``RKVCompressor`` against a *fake paged KV cache* -- the block
gather/scatter, in-place relocation, cross-layer application, per-request
isolation, and the config validation -- none of which the pure-algorithm tests
in ``test_rkv_algo.py`` cover. Like that suite this is GPU-free and does not
require the vLLM serving stack: ``rkv/integration.py`` is loaded by file path
with a stub ``vllm.rkv.algo`` package so its ``from vllm.rkv.algo import R1KV``
resolves to the source ``rkv/algo.py``.

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

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_BASE, rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub the ``vllm`` / ``vllm.rkv`` packages so integration.py's
# ``from vllm.rkv.algo import R1KV`` resolves to the *source* algo module,
# independent of any installed vLLM. (integration.py's only other vllm import,
# ``vllm.distributed.parallel_state``, is lazy + guarded, so it stays a no-op
# here and the TP all-reduce is skipped.)
sys.modules.setdefault("vllm", types.ModuleType("vllm"))
sys.modules.setdefault("vllm.rkv", types.ModuleType("vllm.rkv"))
sys.modules["vllm"].__path__ = []
sys.modules["vllm.rkv"].__path__ = []
_load("vllm.rkv.algo", "rkv/algo.py")
_integration = _load("vllm.rkv.integration", "rkv/integration.py")
RKVConfig = _integration.RKVConfig
RKVCompressor = _integration.RKVCompressor


def make_paged_cache(num_blocks, block_size, kv_heads, head_dim, base=0.0, sign=1.0):
    """Paged cache whose every element at physical ``slot`` equals
    ``sign * (slot + base)`` -- so a relocated entry is trivially identifiable."""
    slots = torch.arange(num_blocks * block_size, dtype=torch.float32).reshape(
        num_blocks, block_size
    )
    return (sign * (slots + base)).view(num_blocks, block_size, 1, 1).expand(
        num_blocks, block_size, kv_heads, head_dim
    ).contiguous()


def slot_val(cache, slot):
    bs = cache.size(1)
    return cache[slot // bs, slot % bs, 0, 0].item()


class TestPagedCompaction(unittest.TestCase):
    def test_noncontiguous_blocks_and_overlapping_relocation(self):
        """Reference-mode compaction over non-contiguous blocks, with source
        and destination slot ranges overlapping (the alias-safety case)."""
        block_size, kv_heads, head_dim = 4, 2, 4
        cfg = RKVConfig(budget=6, buffer_size=2, window_size=2, score_mode="reference")
        comp = RKVCompressor(cfg)

        # One request, physical KV length 10, over non-contiguous blocks 7,2,11.
        occupied = torch.tensor(
            [28, 29, 30, 31, 8, 9, 10, 11, 44, 45], dtype=torch.long
        )
        seq_len = occupied.numel()
        num_blocks = 12

        # Two layers with distinct fills, to prove the *one* decision is applied
        # to *every* layer.
        k0 = make_paged_cache(num_blocks, block_size, kv_heads, head_dim, sign=1.0)
        v0 = make_paged_cache(num_blocks, block_size, kv_heads, head_dim, sign=-1.0)
        k1 = make_paged_cache(num_blocks, block_size, kv_heads, head_dim, base=1000.0)
        v1 = make_paged_cache(
            num_blocks, block_size, kv_heads, head_dim, base=1000.0, sign=-1.0
        )
        layer_caches = [(k0, v0, {}), (k1, v1, {})]

        # Reference-mode score: control the kept set exactly. Highest at past
        # indices 0,2,4,6 -> those are the top `budget - window` = 4 past tokens.
        score = torch.tensor([10.0, 1, 9, 1, 8, 1, 7, 1])  # (seq_len - window,)
        num_dropped = [0]
        comp.compact_step(
            num_reqs=1,
            seq_lens=torch.tensor([seq_len]),
            occupied_slot_mapping=occupied,
            should_compress=(True,),
            score_acc=[score],
            layer_caches=layer_caches,
            num_dropped_tokens_list=num_dropped,
        )

        # Expected kept (sorted): past [0,2,4,6] + window [8,9].
        kept = [0, 2, 4, 6, 8, 9]
        src = [occupied[i].item() for i in kept]          # [28,30,8,10,44,45]
        dst = [occupied[i].item() for i in range(6)]       # [28,29,30,31,8,9]

        self.assertEqual(num_dropped[0], seq_len - cfg.budget)  # 4
        for d, s in zip(dst, src):
            self.assertEqual(slot_val(k0, d), float(s), f"k0 dst {d}")
            self.assertEqual(slot_val(v0, d), float(-s), f"v0 dst {d}")
            self.assertEqual(slot_val(k1, d), float(s + 1000), f"k1 dst {d}")
            self.assertEqual(slot_val(v1, d), float(-(s + 1000)), f"v1 dst {d}")

    def test_unarmed_request_is_untouched(self):
        """With two same-length requests but only one armed, the other request's
        paged KV must be byte-for-byte unchanged, and its dropped count 0."""
        block_size, kv_heads, head_dim = 4, 1, 2
        cfg = RKVConfig(budget=6, buffer_size=2, window_size=2, score_mode="reference")
        comp = RKVCompressor(cfg)

        occ_a = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
        occ_b = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27, 28, 29], dtype=torch.long)
        occupied = torch.cat([occ_a, occ_b])
        num_blocks = 8

        k = make_paged_cache(num_blocks, block_size, kv_heads, head_dim)
        v = make_paged_cache(num_blocks, block_size, kv_heads, head_dim, sign=-1.0)
        k_ref = k.clone()

        score_a = torch.tensor([9.0, 1, 8, 1, 7, 1, 6, 1])
        num_dropped = [0, 0]
        comp.compact_step(
            num_reqs=2,
            seq_lens=torch.tensor([10, 10]),
            occupied_slot_mapping=occupied,
            should_compress=(True, False),   # only request A armed
            score_acc=[score_a, None],
            layer_caches=[(k, v, {})],
            num_dropped_tokens_list=num_dropped,
        )

        self.assertEqual(num_dropped[0], 4)
        self.assertEqual(num_dropped[1], 0)  # unarmed request untouched
        # Request B's blocks (slots 20-29) are unchanged.
        for slot in occ_b.tolist():
            self.assertEqual(slot_val(k, slot), slot_val(k_ref, slot))

    def test_batched_mode_end_to_end_keeps_window(self):
        """Full default path (record_query ring + observe_layer scoring +
        compact_step) on a tiny paged cache. The trailing observation window is
        always kept, so it must land in the leading budget's tail slots."""
        block_size, kv_heads, head_dim = 4, 2, 4
        budget, buffer_size, window = 4, 2, 2
        cfg = RKVConfig(
            budget=budget, buffer_size=buffer_size, window_size=window,
            kernel_size=1, score_mode="batched",
        )
        comp = RKVCompressor(cfg)

        occupied = torch.tensor([12, 13, 14, 15, 4, 5], dtype=torch.long)  # blocks 3,1
        seq_len = occupied.numel()  # == budget + buffer, so it compacts
        num_blocks = 4
        k = make_paged_cache(num_blocks, block_size, kv_heads, head_dim)
        v = make_paged_cache(num_blocks, block_size, kv_heads, head_dim, sign=-1.0)

        req_ids = ["r0"]
        query = torch.randn(1, kv_heads, head_dim)  # one decode query row
        query_start_loc = torch.tensor([0, 1], dtype=torch.long)

        attn_md = types.SimpleNamespace(
            num_reqs=1,
            should_compress_list=(True,),
            occupied_slot_mapping=occupied,
            seq_lens=torch.tensor([seq_len]),
            query_start_loc=query_start_loc,
            rkv_score_acc=[None],
            rkv_layer_caches=[],
            rkv_req_ids=req_ids,
            rkv_qplan=comp.plan_qwrite(req_ids, torch.device("cpu")),
        )
        comp.record_query(query, attn_md)
        comp.observe_layer(query, k, v, attn_md)

        num_dropped = [0]
        comp.compact_step(
            num_reqs=1,
            seq_lens=attn_md.seq_lens,
            occupied_slot_mapping=occupied,
            should_compress=(True,),
            score_acc=attn_md.rkv_score_acc,
            layer_caches=attn_md.rkv_layer_caches,
            num_dropped_tokens_list=num_dropped,
        )

        self.assertEqual(num_dropped[0], seq_len - budget)  # 2
        # The trailing `window` tokens (occupied[seq_len-window:]) are always
        # kept and, being the largest indices, sort to the tail of the kept set,
        # so they relocate to occupied[budget-window:budget].
        for j in range(window):
            dst = occupied[budget - window + j].item()
            src = occupied[seq_len - window + j].item()
            self.assertEqual(slot_val(k, dst), float(src))
            self.assertEqual(slot_val(v, dst), float(-src))


class TestConfigValidation(unittest.TestCase):
    def test_disabled_config_never_raises(self):
        # budget/buffer == 0 => pure no-op; must tolerate otherwise-invalid values.
        RKVConfig(budget=0, buffer_size=0)
        RKVConfig(budget=0, buffer_size=0, kernel_size=6, window_size=0)

    def test_invalid_enabled_configs_raise(self):
        bad = [
            dict(budget=6, buffer_size=64, window_size=8),   # budget <= window
            dict(budget=64, buffer_size=4, window_size=8),   # buffer < window
            dict(budget=64, buffer_size=64, window_size=0),  # window <= 0
            dict(budget=64, buffer_size=64, kernel_size=6),  # even kernel
            dict(budget=64, buffer_size=64, kernel_size=0),  # non-positive kernel
            dict(budget=64, buffer_size=64, mix_lambda=1.5),  # out of [0,1]
            dict(budget=64, buffer_size=64, retain_ratio=0.0),  # out of (0,1]
            dict(budget=64, buffer_size=64, score_mode="bogus"),
            dict(budget=64, buffer_size=64, score_chunk_bytes=0),  # non-positive cap
        ]
        for kwargs in bad:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                RKVConfig(**kwargs)

    def test_valid_enabled_config_ok(self):
        cfg = RKVConfig(budget=256, buffer_size=64, window_size=8, kernel_size=7)
        self.assertTrue(cfg.enabled)


class TestCompactionSafety(unittest.TestCase):
    """Transactional / fail-closed guarantees of compact_step."""

    def _ref_setup(self, num_blocks=8, kv_heads=1, head_dim=2, block_size=4):
        cfg = RKVConfig(budget=6, buffer_size=2, window_size=2, score_mode="reference")
        comp = RKVCompressor(cfg)
        occupied = torch.arange(10, dtype=torch.long)
        k = make_paged_cache(num_blocks, block_size, kv_heads, head_dim)
        v = make_paged_cache(num_blocks, block_size, kv_heads, head_dim, sign=-1.0)
        score = torch.tensor([10.0, 1, 9, 1, 8, 1, 7, 1])  # kept past = 0,2,4,6
        return cfg, comp, occupied, k, v, score

    def test_partial_layer_registration_raises(self):
        """Expecting N layers but seeing N-1 must raise; dropped stays 0."""
        _, comp, occupied, k, v, score = self._ref_setup()
        nd = [0]
        with self.assertRaises(RuntimeError):
            comp.compact_step(
                1, torch.tensor([10]), occupied, (True,), [score],
                [(k, v, {})], nd, expected_layer_count=2,  # only 1 registered
            )
        self.assertEqual(nd[0], 0)  # not committed

    def test_missing_state_when_required_raises(self):
        """A required compaction with no occupied mapping / layer caches raises
        rather than silently skipping (the request may already be capped)."""
        _, comp, occupied, k, v, score = self._ref_setup()
        with self.assertRaises(RuntimeError):
            comp.compact_step(
                1, torch.tensor([10]), None, (True,), [score], [(k, v, {})],
                [0], expected_layer_count=1,
            )
        with self.assertRaises(RuntimeError):
            comp.compact_step(
                1, torch.tensor([10]), occupied, (True,), [score], [],
                [0], expected_layer_count=0,
            )

    def test_missing_window_batched_required_raises(self):
        """Batched mode: an armed required request whose observation window was
        never recorded must raise (not silently skip)."""
        cfg = RKVConfig(budget=4, buffer_size=2, window_size=2, kernel_size=1,
                        score_mode="batched")
        comp = RKVCompressor(cfg)
        occupied = torch.tensor([12, 13, 14, 15, 4, 5], dtype=torch.long)
        k = make_paged_cache(4, 4, 2, 4)
        v = make_paged_cache(4, 4, 2, 4, sign=-1.0)
        nd = [0]
        with self.assertRaises(RuntimeError):
            comp.compact_step(
                1, torch.tensor([6]), occupied, (True,), [None],
                [(k, v, {})], nd, expected_layer_count=1,  # empty window dict
            )
        self.assertEqual(nd[0], 0)

    def test_shared_storage_relocated_once(self):
        """The same KV storage registered as two layers is relocated once, not
        twice (a second in-place gather would read already-moved data)."""
        _, comp, occupied, k, v, score = self._ref_setup()
        nd = [0]
        comp.compact_step(
            1, torch.tensor([10]), occupied, (True,), [score],
            [(k, v, {}), (k, v, {})], nd, expected_layer_count=2,  # shared tensors
        )
        # kept=[0,2,4,6,8,9] -> src=[0,2,4,6,8,9], dst=[0,1,2,3,4,5].
        # Relocated ONCE: k[1]==2, k[2]==4. Relocated twice it would be 4, 8.
        self.assertEqual(slot_val(k, 1), 2.0)
        self.assertEqual(slot_val(k, 2), 4.0)
        self.assertEqual(nd[0], 4)

    def test_memory_guard_skips_first_but_raises_after_drop(self):
        """A first-compaction request over the score-memory cap is left Full-KV
        (dropped stays 0, no relocation); an already-compacted one raises."""
        cfg = RKVConfig(budget=6, buffer_size=2, window_size=2,
                        score_mode="reference", score_chunk_bytes=1)  # 1-byte cap
        comp = RKVCompressor(cfg)
        occupied = torch.arange(10, dtype=torch.long)
        k = make_paged_cache(8, 4, 1, 2)
        v = make_paged_cache(8, 4, 1, 2, sign=-1.0)
        score = torch.tensor([10.0, 1, 9, 1, 8, 1, 7, 1])

        nd = [0]
        comp.compact_step(
            1, torch.tensor([10]), occupied, (True,), [score], [(k, v, {})],
            nd, expected_layer_count=1, prev_dropped=[0],  # never compacted
        )
        self.assertEqual(nd[0], 0)          # skipped -> Full-KV
        self.assertEqual(slot_val(k, 1), 1.0)  # untouched (no relocation)

        with self.assertRaises(RuntimeError):
            comp.compact_step(
                1, torch.tensor([10]), occupied, (True,), [score], [(k, v, {})],
                [0], expected_layer_count=1, prev_dropped=[3],  # already dropped
            )

    def test_tp_group_mismatch_fails_closed(self):
        """With an expected TP world size > 1, a missing/mismatched group must
        raise rather than degrade to a per-rank-local decision."""
        cfg = RKVConfig(budget=6, buffer_size=2, window_size=2, score_mode="reference")
        comp = RKVCompressor(cfg)
        comp.set_parallel_context(2)  # expect a 2-way TP group

        fake_ps = types.ModuleType("vllm.distributed.parallel_state")
        fake_ps.get_tp_group = lambda: types.SimpleNamespace(world_size=1)
        fake_dist = types.ModuleType("vllm.distributed")
        fake_dist.__path__ = []
        sys.modules["vllm.distributed"] = fake_dist
        sys.modules["vllm.distributed.parallel_state"] = fake_ps
        try:
            with self.assertRaises(RuntimeError):
                comp._tp_group()
        finally:
            del sys.modules["vllm.distributed.parallel_state"]
            del sys.modules["vllm.distributed"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
