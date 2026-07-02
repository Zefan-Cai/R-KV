# R-KV on Mini-SGLang — Implementation & Wiring

This is the deep-dive companion to [`DESIGN.md`](./DESIGN.md): the exact files
the wiring patch touches, the per-step compression schedule, and how to build the
patched tree.

## 1. Port layout (this directory)

```
Mini-SGLang/
├── README.md                          # port overview + apply steps
├── requirements-rkv.txt               # dependency notes (R-KV adds none)
├── rkv/                               # standalone R-KV package
│   ├── algo.py                        # R1KV: scoring / similarity / update_kv
│   ├── integration.py                 # RKVConfig, RKVCompressor (driver)
│   └── __init__.py
├── patch/rkv-mini-sglang-9a91cfa.patch # wiring diff (9 upstream files)
├── scripts/apply_rkv.sh               # clone pin → drop in rkv/ → apply patch
├── docs/                              # DESIGN.md, IMPLEMENTATION.md
├── benchmark/                         # bench_rkv.py (offline), README.md
└── tests/                             # GPU-free CPU unit test
```

`scripts/apply_rkv.sh` clones `sgl-project/mini-sglang@9a91cfa` into a
git-ignored `mini-sglang-src/`, copies `rkv/*.py` to
`python/minisgl/rkv/`, and applies the wiring patch — so you always see *exactly*
what R-KV changes on top of a known-good upstream commit.

## 2. Wiring patch (9 files)

| File | Change |
| --- | --- |
| `engine/config.py` | `EngineConfig.rkv_*` fields + `rkv_config` cached property → builds `RKVConfig` (`from minisgl.rkv import RKVConfig`) |
| `engine/engine.py` | Instantiate `RKVCompressor` and place it on the global `Context` as `ctx.rkv` when `rkv_enabled=True`; force-disable CUDA-graph capture |
| `core.py` | `ctx.rkv` field + `Req.num_dropped_tokens` / derived `Req.kv_len = device_len - num_dropped_tokens` |
| `layers/attention.py` | `AttentionLayer.forward` calls `update_query_buffer` before attention and `maybe_compress` after (decode only), with the last-layer `num_dropped_tokens` guard |
| `scheduler/cache.py` | Page allocation at the **physical** frontier (`kv_len` / `kv_cached_len`); do not insert compressed pages into the prefix cache |
| `scheduler/scheduler.py` | Drain `drain_pending_free_slots()` back to the cache manager after each forward; call `drop_request(uid)` on finished requests; **force the non-overlap scheduler loop when `rkv_enabled`** (overlap's separate stream races with the in-place compaction) |
| `attention/{fa,fi,trtllm}.py` | Read the physical `kv_len` for `cache_seqlens` / slot rows |

## 3. Per-step compression schedule

`AttentionLayer.forward` (decode step), for each layer:

1. **Before** `attn_backend.forward` — `update_query_buffer(q, batch, layer_id)`
   records the trailing window of this layer's queries. Per-`(uid, layer_id)`
   buffering matters: each layer projects its own `q`, and R-KV scores
   layer-specific K/V against the matching layer's queries.
2. **Inside** `attn_backend.forward` — the backend stores this step's K/V and
   attends over the full (uncompressed) cache; the current step's output is
   unaffected by R-KV.
3. **After** `attn_backend.forward` — `maybe_compress(layer_id, ...)` compacts
   this layer's K/V in place: each request's kept K/V move to the first `budget`
   slots of its page allocation, so the shared `page_table` stays valid.
4. **Last layer only** — bump `req.num_dropped_tokens` by the number of dropped
   slots. `device_len` / `cached_len` stay logical; physical KV length is the
   derived `req.kv_len` (see [`DESIGN.md`](./DESIGN.md) §3).

Freed tail slots are enqueued on `RKVCompressor._pending_free_slots` (dtype
`int32`, matching the cache manager's `free_slots`) and drained by the scheduler
via `drain_pending_free_slots()` → `CacheManager._free` after each forward, so
other requests can reuse them on the next step.

## 4. Constraints

- **Eager mode required.** R-KV mutates `cache_seqlens` per step; CUDA graphs and
  `torch.compile` must be off. `cuda_graph_max_bs=None` means *auto-enable* (not
  disable) — only `cuda_graph_max_bs=0` disables capture. The engine
  force-disables capture whenever `rkv_enabled=True`, since graph replay would
  silently skip the Python compression hooks.
- **Overlap scheduling off.** R-KV mutates the KV cache in place mid-forward and
  frees slots for reuse; the overlap scheduler runs the next step on a **separate
  CUDA stream** and races with those in-place writes → KV corruption (degenerate/
  looping output) or a CUDA illegal-access crash. The scheduler force-disables
  the overlap loop when `rkv_enabled` (mirroring the CUDA-graph force-disable).
  *Found and fixed via GPU validation, 2026-07-02.*
- **`page_size = 1`.** `maybe_compress` slot resolution assumes it (matches
  Mini-SGLang's default); `page_size > 1` needs page-index × `page_size` scaling.
- **No prefix reuse of compressed pages.** Compressed pages no longer represent
  the verbatim prefix — keep `RadixCache` from matching compressed branches.
- **Single GPU first.** TP > 1 needs each rank to compress its own shard and the
  per-request `new_len` list to agree across ranks.

## 5. Build & run

```bash
cd Mini-SGLang
scripts/apply_rkv.sh                    # -> ./mini-sglang-src (patched, pinned)
cd mini-sglang-src && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
# Offline R-KV benchmark (see ../benchmark/README.md):
python3 ../benchmark/bench_rkv.py
```

CPU unit test (no GPU, no install — loads `rkv/` by path):

```bash
python3 tests/test_rkv_algorithm.py
```

## 6. Validation status

- [x] Algorithm port (`rkv/algo.py`) — CPU-tested, bit-for-bit vs the parent-repo
  reference.
- [x] Per-`(uid, layer_id)` driver (`rkv/integration.py`) — CPU-tested
  (disabled-noop, `drop_request`, `drain_pending_free_slots`).
- [x] `EngineConfig.rkv_*` knobs + `ctx.rkv` + `AttentionLayer` wiring — patch
  applies cleanly to the pinned upstream and imports resolve.
- [x] **End-to-end GPU serve run** — validated 2026-07-02 on Qwen2.5-Math-7B
  (FlashInfer, budget 256/512, batched=8): accuracy matches the baseline (7/8),
  compaction fires (140–1120×), no crash. This surfaced and fixed a concurrency
  bug — **overlap scheduling is now force-disabled when R-KV is on** (§4); the
  algorithm / compaction logic itself was already correct (it matched the
  baseline even before the fix under `CUDA_LAUNCH_BLOCKING=1`).
