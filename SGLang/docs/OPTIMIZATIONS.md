# R-KV on SGLang — Optimizations & Production Hardening

This document is the "what makes the port fast and safe" layer. Read
[`DESIGN.md`](./DESIGN.md) (architecture) and [`IMPLEMENTATION.md`](./IMPLEMENTATION.md)
(code map) first; this file collects the **performance optimizations** and the
**production-hardening** work layered on top of the correct-but-naive port, with
the measured effect of each.

There are two R-KV compressors, which share the same paged-pool eviction
machinery but differ in *when* they compress:

| Compressor | File | Compaction timing | Scoring cost |
| --- | --- | --- | --- |
| **decode R-KV** (`--enable-rkv`) | [`../rkv/integration.py`](../rkv/integration.py) | every `buffer_size` decode steps | importance + redundancy |
| **R-KV-prefill** (`--enable-rkv-prefill`) | [`../rkv/prefill_integration.py`](../rkv/prefill_integration.py) | end of prefill (`oneshot`) or mid-prefill (`buffered`) | importance + O(n²) redundancy |

---

## 1. Performance optimizations

### 1.1 In-graph observation + hybrid CUDA-graph decode
Decode R-KV needs the last `window_size` decode queries per request for
importance scoring. Collecting them used to force **every** decode step eager.
Now the observation-query collection runs **inside the captured decode CUDA
graph** (a fixed-address rolling buffer written in-graph, keyed by a per-request
GPU step counter), so only the `window_size` steps ending at a compaction — plus
the compaction step itself — run eager; every other decode step **replays the
captured graph**. Logical rotary positions are restored at `ForwardBatch`
construction so both the eager and the replayed path read correct positions.

*Effect:* decode CUDA graph is now **supported** (the old port was eager-only).
End-to-end on Qwen2.5-Math-7B, budget 512, 8 concurrent: **~424 tok/s eager →
~580 tok/s with the decode graph on (+37%)**, same accuracy within n=20 noise
(see [`REPRODUCE.md`](./REPRODUCE.md)).

### 1.2 Batched cross-layer scoring
Scoring is per-layer (each layer contributes its own `q·kᵀ` importance and
key-similarity redundancy). The naive port ran one GEMM **per layer**
(`num_layers` launches). Scoring is now **batched across layers** in a single
pass.

*Effect:* removes the layer-loop launch overhead — **8× on a short (2174-token)
prompt** (94 ms → 12 ms, launch-bound) and **+80% decode throughput at
`buffer_size=16`**. Note: on long (8–23 K-token) prompts the per-layer O(n²)
redundancy *compute* dominates, so batching is compute-neutral there — that
remaining O(n²) is the top roadmap item (§4).

### 1.3 Fused Triton redundancy kernel (+ startup validation gate)
The O(n²) key cosine-similarity redundancy term has a **fused Triton kernel**
([`../rkv/redundancy_fused.py`](../rkv/redundancy_fused.py)) that computes the
row-blocked similarity → mean → softmax without materializing the full `n×n`
matrix. It is validated against the tiled/reference path (bit-parity gate) and
falls back **permanently** to the reference on any per-call compile/exec failure
(never crashes the server).

`--rkv-fused-validation` controls *when* the gate runs:
- `startup` (default) — validate once at load on a synthetic tensor sized to the
  real model's kv-heads/head-dim/dtype, so **no real request pays the gate**;
- `first-request` — validate lazily on the first real compaction (the old
  behavior; an unlucky long first prompt paid the cost);
- `off` — never use the fused kernel (always the reference path).

*Determinism:* measured run-to-run jitter on H100 = **0.0 for bf16/fp16, ~1e-10
for fp32**, and the top-k kept set never flips.

### 1.4 Host-sync reduction on the hot path
Two GPU→CPU syncs were removed with no behavior change:
- `override_decode_positions` replaced a per-request `.item()` (one device sync
  **per request, every decode step**) with a single `.tolist()` + one batched
  scatter;
- `observe_prefill_layer` computed its host-side batch metadata **once per
  layer**; it now computes it once per forward (at layer 0) and reuses it,
  saving `num_layers − 1` syncs per prefill.

### 1.5 Compression-aware admission — the throughput unlock
A decode R-KV request's steady-state KV footprint is `min(prompt, budget) +
output`, **not** `prompt + output`. The scheduler's `PrefillAdder`
([`schedule_policy.py`](../patch/rkv-sglang-0.5.14.patch)) reserves that smaller,
*constant* compressed ceiling at admission, so many more requests run
concurrently under a fixed KV pool.

*Effect (memory-bound regime):* peak concurrent running-requests **10 → 65
(6.5×)** for R-KV-prefill; the concurrency edge **grows with prompt length**
because the Full-KV footprint scales with the prompt while R-KV's stays constant
(equal-work `--ignore-eos` A/B: even at ~5 K prompt, **1.48× at ~20 K**).

> R-KV wins on throughput **only when memory-bound**. With a large (non-pressured)
> KV pool, Full KV wins because there is no memory pressure to exploit and R-KV
> still pays its scoring cost. Quality is lossless at budget 4096 on a 30B
> summarization judge (Full 75/136 == R-KV oneshot 75/136).

---

## 2. Production hardening

The correct-but-naive port had several safety and accounting gaps that only bite
at scale. These were closed and covered with tests.

### 2.1 Memory accounting — reserve every R-KV allocation
The KV-slot admission accounting only tracked pool tokens, not the R-KV
transients. Both are now reserved in KV-pool sizing
([`model_runner_kv_cache_mixin.py`](../patch/rkv-sglang-0.5.14.patch)):
- the decode **rolling observation-query buffer** (`rolling_q`, one row per
  request-pool slot — ~6.1 GiB on a 7B at `max_running_requests=4096`);
- the transient **compaction workspace** (the batched similarity matrix +
  gathered keys + per-layer relocation clones, capped by the score-chunk bound
  + 25% ≈ 0.6 GiB).

`--rkv-max-active-requests` caps `max_running_requests` when R-KV is on, which
cascades to `req_to_token`, `rolling_q`, and the reservation — e.g. cap 128
shrinks `rolling_q` **6.13 → 0.19 GiB**, returning ~5.9 GiB to the KV pool.

### 2.2 Two-phase compaction (decouples forward compute from allocator state)
Compaction used to call `kv_allocator.free()` from **inside** the forward (on the
forward stream), which is the root of the overlap free-list race. It is now split:
- **prepare** (in the decode forward): relocate surviving K/V to the front
  `budget` slots and queue a commit record — **no allocator mutation**;
- **commit** (in the scheduler, after the forward completes, at a stream-synced
  point, *before* the finished-request release loop): free the evicted tail and
  finalize per-request length bookkeeping.

This removes the cross-layer coupling and makes a compact-and-finish-in-the-same-
step request free its tail exactly once (no double-free). (Overlap is still
disabled for R-KV, but the *blocker* is gone — see §3.)

### 2.3 Per-request rolling-query cursor
The in-graph observation buffer used a single **global** circular cursor advanced
every decode step. If any managed request skipped a step (preemption / future
overlap), the global cursor advanced anyway, wedging a phantom unwritten slot
between a request's real queries and corrupting its importance window. Replaced
with a **per-request** GPU step counter (`step_count_of_req`, keyed by
`req_pool_idx`) so each request's window advances only on steps it participates
in — correct regardless of batch membership.

### 2.4 Un-bypassable invariant guards
Safety-critical checks were converted from `assert` (stripped by `python -O`) to
explicit `RuntimeError` / `ValueError`, and the compaction kept-set and the
**whole** `req_to_token` slot table are validated for 1-to-1 uniqueness **before**
any KV buffer is mutated (a duplicate slot could put one slot in both the freed
tail and the kept head → use-after-free). Added a stale-plan identity guard
(request slot released/reused between prepare and commit) and fail-fast on any
lifecycle desync. In-place relocation means any exception is unrecoverable and
must terminate the worker rather than be silently swallowed.

*GPU-validated:* 40-way near-saturation stress, 202 compactions — guards never
false-fired, no OOM, correct answers.

---

## 3. Known constraints & gotchas

- **Overlap schedule must be OFF** for both R-KV modes. Even with two-phase
  compaction the mode stays gated off (`--disable-overlap-schedule` required):
  re-enabling it needs the compaction free deferred to the scheduler's synced
  default-stream point *and* expensive large-scale race re-validation. The race
  only reproduces at real scale (30B, dp8, async); `CUDA_LAUNCH_BLOCKING=1` and
  small models **hide** it.
- **Prefill CUDA graph must be OFF** for `--enable-rkv-prefill` (prompt-phase
  scoring/compaction are dynamic shapes); it gives ~0% benefit on long prompts
  anyway (compute-bound). Decode CUDA graph is supported for both modes.
- **TP ≥ 2 is not supported** (silently incorrect without a cross-rank score
  all-reduce; hard-blocked at startup). Plain DP (`--dp-size N --tp-size 1`) is
  validated. See [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) §11.
- Use **`--ignore-eos`** for clean equal-work throughput A/B (compression changes
  output length, which otherwise confounds `gen_throughput`).

---

## 4. Optimization roadmap (ROI-ordered)

1. **Kill R-KV-prefill's O(n²) prefill scoring** (highest value — it is why
   R-KV-prefill only ties Full KV at 5 K prompts instead of winning):
   - *cross-layer subsampling* — score K of N layers (compute ÷ N/K); cheapest,
     biggest win — validate keep-set Jaccard + judge accuracy A/B first;
   - *redundancy-term approximation* — local/blocked similarity or clustering
     instead of full pairwise `k_norm @ k_normᵀ`, bounding compute to O(n·w);
   - *fp8 / lower-precision scoring* — scores only need to rank tokens;
   - *fused scoring+relocate sgl-kernel* — hardest, highest ceiling.
2. **True mid-prefill physical KV release** — compress *during* prefill so input
   length can exceed the KV pool (buffered mode does a logical compaction today
   but not physical release; requires decoupling chunked-prefill's processed-len
   invariant).
3. **Async-safe overlap** — defer the compaction `free()` to the scheduler's
   synced point, then re-enable overlap (low priority: ~2–5% at scale on the
   prefill-bound workload).
4. **Decode R-KV throughput** — adaptive compaction frequency, cross-layer score
   *subsampling*, relocate on a separate CUDA stream, and letting the forced-
   eager window steps replay the graph.
5. **Quality robustness** — multi-seed / larger-sample judging (n=136 single-pass
   is ~±2 rows of noise, which flips A-vs-B at some budgets).
