# R-KV on SGLang v0.5.14 — Design & Porting Notes

This document is the hand-off reference for anyone (human or AI agent) continuing
the R-KV port. It captures the architecture research and design decisions that
are **not** obvious from the code alone. Read this first before touching the
integration layer.

> **Scope: decode-time R-KV.** This doc covers the original *decoding-time* R-KV
> (`--enable-rkv`). R-KV also has a **prefill-time** mode (`--enable-rkv-prefill`,
> code in [`prefill.py`](../rkv/prefill.py) /
> [`prefill_integration.py`](../rkv/prefill_integration.py));
> its design, results and roadmap are in
> [`FINDINGS_AND_ROADMAP.md`](./OPTIMIZATIONS.md).

## 1. Goal

Port **R-KV** (Redundancy-aware KV Cache Compression for reasoning models,
NeurIPS 2025) onto a clean **SGLang v0.5.14** baseline. R-KV is a
**decoding-time** KV-cache compressor: while a reasoning model generates long
chain-of-thought output, it keeps only the *important* and *non-redundant*
tokens, giving large memory savings and throughput gains at near-full accuracy.

**Source of truth for the algorithm** is the R-KV reference repo, file
`rkv/compression/r1_kv.py` (+ `rkv/utils.py`). We port that faithfully rather
than reworking a previous (buggy, slow) integration.

## 2. Where this port comes from

This directory is a **patch-style distribution**: it does not vendor SGLang.
Instead it ships the self-contained R-KV package ([`../rkv/`](../rkv)) plus a
small wiring diff ([`../patch/rkv-sglang-0.5.14.patch`](../patch/rkv-sglang-0.5.14.patch)),
and [`../scripts/apply_rkv.sh`](../scripts/apply_rkv.sh) reproducibly rebuilds a
patched tree:

1. clone upstream SGLang at the **exact pinned commit**
   `49e384ce9d304648e9959666ecb8ce8cd98d0deb` (release/v0.5.14) into `sglang-src/`;
2. copy `rkv/*.py` into `sglang-src/python/sglang/srt/mem_cache/rkv/`;
3. apply the 9-file wiring patch.

The port was developed on a full SGLang fork sitting **directly** on that pinned
commit with zero unrelated upstream drift, so the tree produced by `apply_rkv.sh`
is byte-identical to the development tree (see
[`./REPRODUCE.md`](./REPRODUCE.md) for the equivalence check).

## 3. The R-KV algorithm (what `algo.py` implements)

Given the per-request `key/query/value_states` of shape
`(bsz, heads, seq_len, head_dim)` (GQA: `q_heads` may be a multiple of
`kv_heads`), compression triggers once `kv_cache_len >= budget`:

1. **Importance** — take the last `window_size` "observation" queries, attend
   over all earlier keys, softmax, average across the observation rows, then
   smooth with `max_pool1d(kernel_size)`. → `attn_cache`.
2. **Redundancy** — pairwise key cosine similarity (`cal_similarity`): mask the
   diagonal, threshold, exempt the most-recent similar neighbour per key,
   aggregate the rest into a per-key redundancy distribution.
3. **Joint score** — `score = importance * mix_lambda - redundancy * (1 - mix_lambda)`.
4. **Selection** — keep the top `budget - window_size` past tokens **plus** the
   trailing `window_size` observation tokens (always retained).

Default hyper-parameters: `budget=1024`, `window_size=8`, `kernel_size=7`,
`mix_lambda≈0.1`, `retain_ratio=0.1`, `retain_direction="last"`. Compression is
triggered every `B_buffer=128` newly generated tokens (the staging buffer size).

## 4. Two-layer design

The port is deliberately split so the algorithm can be validated without a GPU
or the full serving stack.

```
┌───────────────────────────────────────────────────────────┐
│  Pure algorithm layer  —  algo.py                          │
│  • R1KV, compute_attention_scores, cal_similarity          │
│  • update_kv(): reference-compatible (returns compacted KV)│
│  • select_indices(): returns ONLY kept token indices       │
│      shape (bsz, kv_heads, budget); None when below budget │
│  • device-agnostic, ZERO sglang deps → runs on GPU in prod,│
│      CPU-testable in isolation                             │
└───────────────────────────────────────────────────────────┘
                         ▲ called by
┌───────────────────────────────────────────────────────────┐
│  Integration layer  —  integration.py  (DONE, see §9)     │
│  • per-request R-KV state: query cache, trigger counter,   │
│      dropped-token count                                   │
│  • read K/V back from the paged pool for a request         │
│  • compact: relocate kept slots, rewrite req_to_token,     │
│      free dropped slots                                    │
│  • hook into FlashInfer forward_decode                     │
└───────────────────────────────────────────────────────────┘
```

`select_indices()` exists because the integration layer must **relocate slots**
in a paged pool, not `torch.cat` new tensors — it needs the surviving indices,
not a rebuilt KV tensor.

## 5. SGLang v0.5.14 architecture (verified findings)

File paths are relative to the repo root.

- **KV pool**: `MHATokenToKVPool` in
  `python/sglang/srt/mem_cache/memory_pool.py` (~L1068). Per-layer buffers
  `k_buffer[layer]` / `v_buffer[layer]`, shape `(size + page, head_num, head_dim)`,
  NHD layout.
- **Request → token map**: `ReqToTokenPool` (memory_pool.py ~L231),
  `req_to_token[req_pool_idx, :seq_len]` gives the pool slot indices for a
  request. So per-request K for a layer is
  `k_buffer[layer_id][req_to_token[req_pool_idx, :seq_len]]`.
- **Attention is fully fused** (FlashInfer / FlashAttention). Attention weights
  are **not** exposed, so R-KV importance needs a **separate scoring pass**
  (recompute `q @ kᵀ` over the observation window) — this is exactly what
  `compute_attention_scores` does.
- **No built-in mid-generation token dropping.** Eviction (`evict_from_tree_cache`
  in `mem_cache/common.py` ~L298) only fires when the allocator runs out of
  space. R-KV compaction must be added explicitly.
- **Per-request state**: `Req` in `managers/schedule_batch.py` —
  `req_pool_idx` (~L787), `kv_committed_len` (~L737), `kv_allocated_len` (~L738).
- **ForwardBatch**: `model_executor/forward_batch_info.py` (~L322) exposes
  `req_pool_indices`, `seq_lens`, `seq_lens_cpu`, `out_cache_loc`.

## 6. Chosen injection point

`FlashInferAttnBackend.forward_decode()` in
`python/sglang/srt/layers/attention/flashinfer_backend.py` (~L1086–1120),
**after** `set_kv_buffer()` (new K/V written) and **before**
`decode_wrapper.forward()`. At that point we have `q`, `k`, `v`, `layer`, and
`forward_batch` in hand. Backend for phase 1 is **FlashInfer only**.

## 7. Known challenges (where the old impl went wrong)

1. **Paged pool vs dense concat** — compression means physically relocating kept
   slots, rewriting `req_to_token`, and freeing dropped slots. Getting this
   wrong corrupts the cache; doing it inefficiently is the main source of
   slowness.
2. **Query cache** — decode produces only 1 query/step, but R-KV needs the last
   `window_size` observation queries. The integration layer must cache them
   per request.
3. **Trigger cadence** — compress every `B_buffer` generated tokens per request,
   not every step; needs a per-request counter.
4. **Position / rotary handling** — rotary is already baked into stored keys, so
   retained keys keep their embeddings. **Resolved via "scheme A":** treat
   `seq_lens` / `kv_committed_len` as the *physical* KV length (drops to `budget`
   at compaction, `+1` per step after), so new tokens append at the physical
   tail and attention reads the right length automatically. Keep rotary
   *logical*: override `ForwardBatch.positions` for R-KV requests with
   `len(origin_input_ids)+len(output_ids)-1` instead of
   `clamp_position(seq_lens)`, so future tokens keep absolute positions
   consistent with the retained keys. Scheduler and ModelRunner share the
   process, so `Req` length fields are updated in place; the batch `seq_lens`
   tensor is updated from `RKVCompressor.take_pending_length_updates()`.
5. **O(budget²) similarity** — `cal_similarity` builds a `budget × budget`
   matrix per layer per trigger. The per-layer scoring is now **batched across
   layers** (one pass instead of `num_layers` GEMMs); the O(budget²) matrix
   itself is still a phase-2 target (chunking / cheaper redundancy estimate).

## 8. `sparsity/` framework — evaluated, not reused (2026-07)

Before writing the integration layer we evaluated the pre-existing
`python/sglang/srt/mem_cache/sparsity/` framework (`SparseCoordinator`,
`BaseSparseAlgorithm`, `BackendAdaptor`; algorithms `quest`, `deepseek_dsa`).
**Decision: do not build R-KV on top of it. Take route A — a standalone
integration layer under `rkv/`, borrowing only its layering and hook naming.**

Findings:

1. **Not wired into the runtime.** `create_sparse_coordinator` /
   `get_sparse_coordinator` (`sparsity/factory.py` L108+) have zero callers;
   `SparseCoordinator.attention_begin/end`, `forward_begin/end` are never called
   from `layers/` or `model_executor/`. The only symbol the runtime imports from
   `sparsity/` is `parse_hisparse_config`, which feeds a *different* subsystem,
   `HiSparseCoordinator` (`managers/hisparse_coordinator.py`), used for DeepSeek
   DSA hierarchical KV offloading — not general R-KV.

2. **Opposite semantics — non-destructive sparse attention, not eviction.**
   `SparseCoordinator` (docstring L74) targets "retrievable algorithms (Quest,
   PQCache, SnapKV) that dynamically *select* important KV entries". The flow
   `attention_begin → _handle_sparse_retrieve → retrieve_topk →
   FlashAttentionAdaptor.adapt_for_attn_metadata` only rewrites `page_table` /
   `cache_seqlens` so the backend reads a subset **this step**; all physical KV
   slots stay allocated and are restored to full next step. There is **no
   `free()` anywhere** in the framework. R-KV instead needs *permanent*
   eviction: drop slots, `free()` them, rewrite `req_to_token`, shorten the
   sequence.

3. **Wrong cadence.** The framework re-selects **every layer, every decode
   step**; R-KV compresses **once every `B_buffer` steps** and the drop is
   irreversible.

4. **Still a skeleton.** `forward_begin/end`, `on_request_end` resource release,
   and `DSABackendAdaptor.adapt_for_attn_metadata` are all `TODO/pass`. The
   hardest part for R-KV (evict + `free` + shorten) simply does not exist here.

What we *do* borrow: the three-layer split (algorithm / coordinator / backend
adaptor), the lifecycle hook names (`on_request_begin/end`,
`attention_begin/end`), and the per-request state-tensor organization of
`RequestTrackers`.

## 9. Status & roadmap

- **[DONE] Phase 1 · step 1** — pure algorithm layer (`algo.py`) + CPU parity
  tests (`test/srt/mem_cache/test_rkv_algo.py`). Verified bit-for-bit against an
  inline reference (4 configs incl. GQA and below-budget). Run:
  `python3 test/srt/mem_cache/test_rkv_algo.py` (no GPU, no PYTHONPATH needed —
  the test loads `algo.py` by file path to bypass the heavy `sglang/__init__.py`).
- **[DONE] Phase 1 · step 2** — integration layer (route A, §8), wired and
  **verified end-to-end** on Qwen2.5-0.5B (FlashInfer, `batch=1`, `page_size=1`,
  decode/prefill CUDA graph + overlap disabled). Compaction fires (e.g. phys
  80→64, frees 16 slots per trigger every `buffer_size` steps) and output stays
  coherent — no garbage or loops, confirming eviction + `free` + `req_to_token`
  rewrite + logical-position/rotary decoupling are correct. `integration.py` =
  RKVConfig / RKVRequestState / RKVCompressor (cross-head mean + cross-layer sum
  reduction, physical compaction, scheme-A length bookkeeping). Hooks: server
  arg `--enable-rkv`; RKVCompressor built in `model_runner.alloc_memory_pool`;
  the FlashInfer backend holds it, binds it onto the real decode batch and
  overrides positions in `init_forward_metadata`, and calls
  `observe_decode_layer` in `forward_decode`; end-of-forward `maybe_compact` in
  model_runner; scheduler `_apply_rkv_pre_decode` (on_request_begin + apply the
  physical-length shrink). CPU unit tests:
  `test/srt/mem_cache/test_rkv_integration.py`.
- **[DONE] Phase 1 · step 3** — multi-request batching (`batch >= 1`, method A:
  per-request triggering) + accuracy validation. `observe_decode_layer` loops
  over every request in the decode batch and each request arms / compacts
  **independently** (state keyed by `req_pool_idx`; `maybe_compact` resolves each
  armed request's own `seq_len`). Validated end-to-end on
  Qwen2.5-Math-7B-Instruct (FlashInfer, `budget=512`, 8 concurrent requests,
  decode `#running-req` up to 8): accuracy is **lossless at budget=512** on GSM8K
  and dozens–hundreds of physical compactions run with no crashes (see
  [`../benchmark/RESULTS.md`](../benchmark/RESULTS.md)).
  (`on_request_end` cleanup is
  **DONE**: wired in `batch_result_processor` beside `hisparse.request_finished`
  at the two real-finished points; verified per-request state clears to 0.)
  Plain data parallelism (`--dp-size N --tp-size 1`) is also validated — each
  rank runs its own R-KV; throughput scales up to 7.8× on 8× H100 (see
  benchmark/RESULTS_dp.md). Tensor parallelism is supported too (cross-rank
  score all-reduce, §11.2; see benchmark/RESULTS_tp.md); only dp-attention
  remains untested.
- **[DONE] Phase 2 (partial)** — **decode CUDA-graph compatibility** (hybrid
  eager/graph path), **batched cross-layer scoring** (one pass instead of
  `num_layers` GEMMs; +80% decode throughput at `buffer_size=16`, and 8× prefill
  scoring on a 2174-token prompt), and **tensor-parallel support** (cross-rank
  all-reduce of per-token scores before top-k, so every rank evicts the identical
  tokens; see IMPLEMENTATION.md §11.2) are shipped.
- **[LATER] Phase 2 (remaining)** — performance: optimize the O(budget²)
  redundancy matrix, reduce host/device syncs, and let the forced-eager window
  steps replay the graph. Then larger-sample accuracy on MATH-500 / AIME-24.
