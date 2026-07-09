"""CPU-only unit tests for the FlashInfer R-KV port (DESIGN.md §7).

Covers RKVConfig validation, the scatter-based near-duplicate exemption
(regression for the silent advanced-indexing no-op in the RKV-HS prototype),
selection/window invariants, GQA layouts, dtypes, kernel/window edge cases,
and the DESIGN.md §5.3 compressor trigger arithmetic. All rkv modules are
loaded by file path through a synthetic parent package, so nothing imports
flashinfer (CPU-only torch).

Run: python FlashInfer/tests/test_rkv_algo.py
"""

import dataclasses
import importlib.util
import math
import os
import sys
import traceback
import types

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_HERE, "..", "rkv"))


def _load(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Synthetic parent package: lets intra-package imports inside the loaded
# modules resolve without executing rkv/__init__.py (which imports flashinfer).
_pkg = types.ModuleType("rkv")
_pkg.__path__ = [_PKG_DIR]
sys.modules["rkv"] = _pkg

_config = _load("rkv.config", os.path.join(_PKG_DIR, "config.py"))
_algo = _load("rkv.algo", os.path.join(_PKG_DIR, "algo.py"))

RKVConfig = _config.RKVConfig


def _expect_value_error(**kwargs: object) -> None:
    try:
        RKVConfig(**kwargs)
    except ValueError:
        return
    raise AssertionError(f"RKVConfig(**{kwargs}) should have raised ValueError")


def test_config_defaults_and_validation() -> None:
    cfg = RKVConfig()
    assert cfg.budget == 1024
    assert cfg.buffer == 128
    assert cfg.window_size == 8
    assert cfg.kernel_size == 7
    assert cfg.mix_lambda == 0.1
    assert cfg.retain_ratio == 0.1
    assert cfg.retain_direction == "last"
    assert cfg.compress_prefill is True

    try:
        cfg.budget = 2048
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("RKVConfig must be a frozen dataclass")

    _expect_value_error(budget=8, window_size=8)
    _expect_value_error(budget=4, window_size=8)
    _expect_value_error(buffer=4, window_size=8)
    _expect_value_error(buffer=-1)
    _expect_value_error(kernel_size=6)
    _expect_value_error(mix_lambda=0.0)
    _expect_value_error(mix_lambda=-0.5)
    _expect_value_error(mix_lambda=1.5)
    _expect_value_error(retain_direction="middle")
    _expect_value_error(retain_direction="last_percent")

    RKVConfig(mix_lambda=1.0)  # attention-only (SnapKV-style) scoring is legal
    RKVConfig(buffer=8, window_size=8)  # boundary: buffer == window_size
    RKVConfig(retain_direction="first")
    RKVConfig(kernel_size=1)


def test_scatter_exemption_mutates_similarity() -> None:
    """Regression for the RKV-HS advanced-indexing `.zero_()` silent no-op.

    Builds keys where token 4 is a >0.5-cosine near-duplicate of token 1 and
    every other pair is orthogonal, then checks that ``cal_similarity``
    actually zeroes the retained duplicate's contribution before the row mean
    (the buggy no-op variant must produce a different, larger score).
    """
    seq_len, head_dim = 6, 8
    k = torch.zeros(1, 1, seq_len, head_dim)
    for t in range(seq_len):
        k[0, 0, t, t] = 1.0
    k[0, 0, 4] = 0.0
    k[0, 0, 4, 1] = 1.0
    k[0, 0, 4, 7] = 0.1  # cosine(k1, k4) = 1/sqrt(1.01) > 0.5 threshold

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool)
    sim.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)
    assert sim[0, 0, 1, 4] > 0.5

    mask = sim > 0.5
    indices = torch.where(
        mask,
        torch.arange(seq_len).view(1, 1, 1, seq_len),
        torch.zeros_like(mask, dtype=torch.long),
    )
    retain = torch.max(indices, dim=-1)[0]
    assert retain[0, 0].tolist() == [0, 4, 0, 0, 1, 0]

    buggy = sim.clone()  # RKV-HS never applied the exemption
    sim.scatter_(-1, retain.unsqueeze(-1), 0)
    expected = sim.mean(dim=-2).softmax(dim=-1)
    buggy_out = buggy.mean(dim=-2).softmax(dim=-1)

    got = _algo.cal_similarity(k)
    assert torch.equal(got, expected)
    assert not torch.equal(got, buggy_out)
    # Zeroing the retained duplicates must lower their redundancy scores.
    assert got[0, 0, 1] < buggy_out[0, 0, 1]
    assert got[0, 0, 4] < buggy_out[0, 0, 4]


def test_attention_scores_gqa() -> None:
    layouts = [(8, 8), (28, 4), (32, 8), (14, 2)]  # incl. Qwen2.5-Math-7B 28/4
    for i, (q_heads, kv_heads) in enumerate(layouts):
        torch.manual_seed(1234 + i)
        q_len, kv_len, head_dim = 8, 32, 64
        q = torch.randn(2, q_heads, q_len, head_dim)
        k = torch.randn(2, kv_heads, kv_len, head_dim)

        out = _algo.compute_attention_scores(q, k)
        assert out.shape == (2, kv_heads, q_len, kv_len)
        assert out.dtype == q.dtype

        group = q_heads // kv_heads
        full = torch.matmul(
            q, k.repeat_interleave(group, dim=1).transpose(2, 3)
        ) / math.sqrt(head_dim)
        ref = full.view(2, kv_heads, group, q_len, kv_len).max(dim=2).values
        assert torch.allclose(out, ref, atol=1e-5, rtol=0)


def test_selection_keeps_trailing_window() -> None:
    cases = [
        # (bsz, q_heads, kv_heads, seq_len, head_dim, budget, window)
        (1, 8, 8, 256, 64, 128, 8),
        (1, 28, 4, 300, 128, 128, 8),  # Qwen2.5-Math-7B GQA layout
        (2, 16, 2, 200, 64, 96, 16),
    ]
    for i, (bsz, q_heads, kv_heads, seq_len, head_dim, budget, window) in enumerate(cases):
        torch.manual_seed(1234 + i)
        q = torch.randn(bsz, q_heads, seq_len, head_dim)
        k = torch.randn(bsz, kv_heads, seq_len, head_dim)
        v = torch.randn(bsz, kv_heads, seq_len, head_dim)
        r1 = _algo.R1KV(budget=budget, window_size=window)

        idx = r1.select_indices(k, q, sort=True)
        assert idx.shape == (bsz, kv_heads, budget)
        assert bool((idx[..., 1:] > idx[..., :-1]).all())  # ascending + unique
        assert int(idx.min()) >= 0 and int(idx.max()) < seq_len
        window_idx = (
            torch.arange(seq_len - window, seq_len).view(1, 1, -1).expand(bsz, kv_heads, -1)
        )
        assert torch.equal(idx[..., -window:], window_idx)

        kept_k, kept_v = r1.update_kv(k.clone(), q.clone(), v.clone())
        assert kept_k.shape == (bsz, kv_heads, budget, head_dim)
        assert kept_v.shape == (bsz, kv_heads, budget, head_dim)
        assert torch.equal(kept_k[..., -window:, :], k[..., -window:, :])
        assert torch.equal(kept_v[..., -window:, :], v[..., -window:, :])


def test_below_budget_noop() -> None:
    torch.manual_seed(1234)
    q = torch.randn(1, 8, 100, 64)
    k = torch.randn(1, 8, 100, 64)
    v = torch.randn(1, 8, 100, 64)
    r1 = _algo.R1KV(budget=128, window_size=8)
    kept_k, kept_v = r1.update_kv(k, q, v)
    assert torch.equal(kept_k, k)
    assert torch.equal(kept_v, v)
    assert r1.select_indices(k, q) is None

    # At exactly budget tokens compression fires (reference no-op is len < budget).
    torch.manual_seed(1235)
    q = torch.randn(1, 8, 128, 64)
    k = torch.randn(1, 8, 128, 64)
    idx = r1.select_indices(k, q)
    assert idx is not None
    assert idx.shape == (1, 8, 128)


def test_dtypes_bf16_fp32() -> None:
    for i, dtype in enumerate((torch.float32, torch.bfloat16)):
        torch.manual_seed(1234 + i)
        q = torch.randn(1, 28, 300, 128, dtype=dtype)
        k = torch.randn(1, 4, 300, 128, dtype=dtype)
        v = torch.randn(1, 4, 300, 128, dtype=dtype)
        r1 = _algo.R1KV(budget=128, window_size=8)

        kept_k, kept_v = r1.update_kv(k, q, v)
        assert kept_k.dtype == dtype and kept_v.dtype == dtype
        assert kept_k.shape == (1, 4, 128, 128)
        assert bool(torch.isfinite(kept_k.float()).all())

        sim = _algo.cal_similarity(k)
        assert sim.dtype == dtype
        assert sim.shape == (1, 4, 300)

        attn = _algo.compute_attention_scores(q, k)
        assert attn.dtype == dtype
        assert attn.shape == (1, 4, 300, 300)


def test_kernel_window_edges() -> None:
    # kernel_size=1: the max-pool is the identity, so _scores reduces to the raw mix.
    torch.manual_seed(1234)
    q = torch.randn(1, 4, 64, 32)
    k = torch.randn(1, 4, 64, 32)
    r1 = _algo.R1KV(budget=32, window_size=8, kernel_size=1)
    scores = r1._scores(k, q)
    assert scores.shape == (1, 4, 56)
    attn = _algo.compute_attention_scores(q, k)
    attn_sum = (
        F.softmax(attn[:, :, -8:, :-8], dim=-1, dtype=torch.float32)
        .mean(dim=-2)
        .to(q.dtype)
    )
    sim = _algo.cal_similarity(k)[:, :, :-8]
    expected = attn_sum * r1.mix_lambda - sim * (1 - r1.mix_lambda)
    assert torch.equal(scores, expected)

    # Minimal past region: budget = window_size + 1 with the past shorter than
    # the pooling kernel (max_pool1d must still be well-defined via padding).
    torch.manual_seed(1235)
    q = torch.randn(1, 2, 10, 16)
    k = torch.randn(1, 2, 10, 16)
    r1 = _algo.R1KV(budget=9, window_size=8)
    idx = r1.select_indices(k, q, sort=True)
    assert idx.shape == (1, 2, 9)
    assert torch.equal(
        idx[..., -8:], torch.arange(2, 10).view(1, 1, -1).expand(1, 2, -1)
    )

    # window_size=1: only the newest token is unconditionally kept.
    torch.manual_seed(1236)
    q = torch.randn(1, 2, 16, 16)
    k = torch.randn(1, 2, 16, 16)
    v = torch.randn(1, 2, 16, 16)
    r1 = _algo.R1KV(budget=4, window_size=1, kernel_size=3)
    kept_k, kept_v = r1.update_kv(k, q, v)
    assert kept_k.shape == (1, 2, 4, 16)
    assert torch.equal(kept_k[..., -1:, :], k[..., -1:, :])
    assert torch.equal(kept_v[..., -1:, :], v[..., -1:, :])


def _expected_kept_front(
    cfg: RKVConfig,
    kv_pool: torch.Tensor,
    req: int,
    phys: int,
    region: int,
    window_queries: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer expected kept K/V for one request, via the parity-tested algo.

    ``kv_pool``: ``[num_layers, slots, 2, 1, kv_heads, head_dim]`` (DESIGN §5.1);
    ``window_queries``: ``[window, num_layers, batch, q_heads, head_dim]`` in
    temporal push order — exactly the compressor's rolling window content.
    Kept-token order follows the reference ``update_kv`` (top-scoring past
    tokens in score order, then the observation window), which is what
    ``compressor.R1KV.compact`` writes back.
    """
    start = req * region
    expected = []
    for layer in range(kv_pool.shape[0]):
        slots = kv_pool[layer, start : start + phys]
        k = slots[:, 0, 0].permute(1, 0, 2).unsqueeze(0)  # [1, kv_heads, phys, hd]
        v = slots[:, 1, 0].permute(1, 0, 2).unsqueeze(0)
        q = window_queries[-cfg.window_size :, layer, req].permute(1, 0, 2).unsqueeze(0)
        kept_k, kept_v = _algo.update_kv(
            k,
            q,
            v,
            budget=cfg.budget,
            window_size=cfg.window_size,
            kernel_size=cfg.kernel_size,
            mix_lambda=cfg.mix_lambda,
            retain_ratio=cfg.retain_ratio,
            retain_direction=cfg.retain_direction,
        )
        expected.append((kept_k.squeeze(0), kept_v.squeeze(0)))  # [kv_heads, budget, hd]
    return expected


def _drive_compaction(
    comp: object, cfg: RKVConfig, kv_pool: torch.Tensor, phys_len: list[int], region: int
) -> None:
    """Mirror ``engine._compact_request`` for rows where the trigger fires."""
    for row in range(len(phys_len)):
        if not comp.should_compact(phys_len[row]):
            continue
        base = row * region
        phys = phys_len[row]
        for layer in range(kv_pool.shape[0]):
            keys = kv_pool[layer, base : base + phys, 0, 0]
            values = kv_pool[layer, base : base + phys, 1, 0]
            kept_k, kept_v = comp.compact(row, layer, keys, values)
            kv_pool[layer, base : base + cfg.budget, 0, 0] = kept_k
            kv_pool[layer, base : base + cfg.budget, 1, 0] = kept_v
        phys_len[row] = cfg.budget


def test_compressor_trigger_arithmetic() -> None:
    """DESIGN §5.3: fire at phys_len == budget + buffer, collapse to budget.

    Driver surface exercised (pure-torch R1KV in compressor.py, as consumed
    by ``engine.py``):
      R1KV(config, num_layers=, max_batch_size=, num_q_heads=, head_dim=,
           device=, dtype=)
      .begin_step(rows)                 once per decode step (window fill)
      .push_queries(layer, rows, q)     q: [n, q_heads, head_dim], post-RoPE
      .should_compact(phys_len)         per-request trigger check
      .compact(request, layer, k, v)    k/v: [seq_len, kv_heads, head_dim]
      .num_compactions                  per-request counters
    """
    compressor = _load("rkv.compressor", os.path.join(_PKG_DIR, "compressor.py"))

    torch.manual_seed(1234)
    cfg = RKVConfig(budget=16, buffer=8, window_size=8, kernel_size=7)
    num_layers, batch, q_heads, kv_heads, head_dim = 2, 2, 4, 2, 16
    region = cfg.budget + cfg.buffer

    kv_pool = torch.randn(num_layers, batch * region, 2, 1, kv_heads, head_dim)
    comp = compressor.R1KV(
        cfg,
        num_layers=num_layers,
        max_batch_size=batch,
        num_q_heads=q_heads,
        head_dim=head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    rows = torch.arange(batch, dtype=torch.long)
    queries = torch.randn(cfg.window_size, num_layers, batch, q_heads, head_dim)
    for step in range(cfg.window_size):
        comp.begin_step(list(range(batch)))
        for layer in range(num_layers):
            comp.push_queries(layer, rows, queries[step, layer])

    # One slot short of the trigger: nothing may move.
    phys_len = [region - 1, cfg.budget]
    pool_before = kv_pool.clone()
    _drive_compaction(comp, cfg, kv_pool, phys_len, region)
    assert phys_len == [region - 1, cfg.budget]
    assert torch.equal(kv_pool, pool_before)
    assert torch.as_tensor(comp.num_compactions).tolist() == [0, 0]

    # Request 0 reaches budget + buffer: it alone compacts down to budget.
    phys_len[0] += 1
    expected = _expected_kept_front(
        cfg, kv_pool, req=0, phys=region, region=region, window_queries=queries
    )
    _drive_compaction(comp, cfg, kv_pool, phys_len, region)
    assert phys_len == [cfg.budget, cfg.budget]
    assert torch.as_tensor(comp.num_compactions).tolist() == [1, 0]
    for layer, (exp_k, exp_v) in enumerate(expected):
        front = kv_pool[layer, : cfg.budget]
        assert torch.equal(front[:, 0, 0].permute(1, 0, 2), exp_k)
        assert torch.equal(front[:, 1, 0].permute(1, 0, 2), exp_v)
    # Request 1's region is untouched.
    assert torch.equal(kv_pool[:, region : 2 * region], pool_before[:, region : 2 * region])

    # Appending `buffer` more tokens re-arms the trigger exactly once, at the
    # last append and not before.
    for step in range(cfg.buffer):
        slot = phys_len[0]
        comp.begin_step([0])
        for layer in range(num_layers):
            kv_pool[layer, slot] = torch.randn(2, 1, kv_heads, head_dim)
            comp.push_queries(
                layer, torch.tensor([0]), torch.randn(1, q_heads, head_dim)
            )
        phys_len[0] += 1
        _drive_compaction(comp, cfg, kv_pool, phys_len, region)
        fired = 2 if step == cfg.buffer - 1 else 1
        assert torch.as_tensor(comp.num_compactions).tolist() == [fired, 0]
    assert phys_len == [cfg.budget, cfg.budget]


def test_compressor_window_rotation() -> None:
    """Windows must reach scoring in temporal order when compaction fires
    between window-aligned steps (regression for the circular window buffer;
    also holds for any shift-based implementation)."""
    compressor = _load("rkv.compressor", os.path.join(_PKG_DIR, "compressor.py"))

    torch.manual_seed(1234)
    cfg = RKVConfig(budget=16, buffer=8, window_size=8, kernel_size=7)
    num_layers, q_heads, kv_heads, head_dim = 2, 4, 2, 16
    phys = cfg.budget + cfg.buffer

    comp = compressor.R1KV(
        cfg,
        num_layers=num_layers,
        max_batch_size=1,
        num_q_heads=q_heads,
        head_dim=head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    seed = torch.randn(cfg.window_size, num_layers, q_heads, head_dim)
    for layer in range(num_layers):
        comp.seed_window(layer, 0, seed[:, layer])
    # Three decode steps: the write cursor sits mid-window at compaction time,
    # so the temporal window is [seed[3:], push0, push1, push2].
    pushes = torch.randn(3, num_layers, 1, q_heads, head_dim)
    for step in range(3):
        comp.begin_step([0])
        for layer in range(num_layers):
            comp.push_queries(layer, torch.tensor([0]), pushes[step, layer])

    keys = torch.randn(num_layers, phys, kv_heads, head_dim)
    values = torch.randn(num_layers, phys, kv_heads, head_dim)
    for layer in range(num_layers):
        kept_k, kept_v = comp.compact(0, layer, keys[layer], values[layer])
        window = torch.cat([seed[3:, layer], pushes[:, layer, 0]], dim=0)
        exp_k, exp_v = _algo.update_kv(
            keys[layer].transpose(0, 1).unsqueeze(0),
            window.transpose(0, 1).unsqueeze(0),
            values[layer].transpose(0, 1).unsqueeze(0),
            budget=cfg.budget,
            window_size=cfg.window_size,
            kernel_size=cfg.kernel_size,
            mix_lambda=cfg.mix_lambda,
            retain_ratio=cfg.retain_ratio,
            retain_direction=cfg.retain_direction,
        )
        assert torch.equal(kept_k, exp_k.squeeze(0).transpose(0, 1))
        assert torch.equal(kept_v, exp_v.squeeze(0).transpose(0, 1))


def test_compressor_batch_matches_per_layer() -> None:
    """compact_batch must return exactly what per-(layer, request) compact()
    returns — the internal A/B gate guarantees this even on devices where
    batched GEMMs differ, so the contract holds unconditionally."""
    compressor = _load("rkv.compressor", os.path.join(_PKG_DIR, "compressor.py"))

    torch.manual_seed(1234)
    cfg = RKVConfig(budget=16, buffer=8, window_size=8, kernel_size=7)
    num_layers, q_heads, kv_heads, head_dim = 3, 4, 2, 16
    max_batch, rows = 3, [0, 2]  # non-contiguous request rows
    phys = cfg.budget + cfg.buffer

    def build() -> object:
        comp = compressor.R1KV(
            cfg,
            num_layers=num_layers,
            max_batch_size=max_batch,
            num_q_heads=q_heads,
            head_dim=head_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        torch.manual_seed(77)
        seed = torch.randn(cfg.window_size, num_layers, max_batch, q_heads, head_dim)
        for layer in range(num_layers):
            for r in range(max_batch):
                comp.seed_window(layer, r, seed[:, layer, r])
        pushes = torch.randn(3, num_layers, max_batch, q_heads, head_dim)
        for step in range(3):  # cursor mid-window at compaction time
            comp.begin_step(list(range(max_batch)))
            for layer in range(num_layers):
                comp.push_queries(
                    layer, torch.arange(max_batch), pushes[step, layer]
                )
        return comp

    comp_batch = build()
    comp_loop = build()
    keys = torch.randn(num_layers, len(rows), phys, kv_heads, head_dim)
    values = torch.randn(num_layers, len(rows), phys, kv_heads, head_dim)

    kept_k, kept_v = comp_batch.compact_batch(rows, keys.clone(), values.clone())
    assert kept_k.shape == (num_layers, len(rows), cfg.budget, kv_heads, head_dim)
    for layer in range(num_layers):
        for i, row in enumerate(rows):
            exp_k, exp_v = comp_loop.compact(row, layer, keys[layer, i], values[layer, i])
            assert torch.equal(kept_k[layer, i], exp_k)
            assert torch.equal(kept_v[layer, i], exp_v)
    assert torch.as_tensor(comp_batch.num_compactions).tolist() == [1, 0, 1]
    assert torch.as_tensor(comp_loop.num_compactions).tolist() == [1, 0, 1]


TESTS = [
    test_config_defaults_and_validation,
    test_scatter_exemption_mutates_similarity,
    test_attention_scores_gqa,
    test_selection_keeps_trailing_window,
    test_below_budget_noop,
    test_dtypes_bf16_fp32,
    test_kernel_window_edges,
    test_compressor_trigger_arithmetic,
    test_compressor_window_rotation,
    test_compressor_batch_matches_per_layer,
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
