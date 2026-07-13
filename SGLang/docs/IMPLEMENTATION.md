# R-KV Integration Layer — Implementation Notes

This document describes **how** R-KV (Redundancy-aware KV Cache Compression) is
implemented on top of a clean SGLang v0.5.14 baseline. For the *why* (research
findings, rejected alternatives, the `sparsity/` framework evaluation), read
[`DESIGN.md`](./DESIGN.md) first. This file is the practical map of the code:
the components, the per-step data flow, the exact wiring points, and the
decisions baked into them.

> **Scope: decode-time R-KV.** This file documents the *decoding-time* path
> (`--enable-rkv`). The **prefill-time** mode (`--enable-rkv-prefill`) is a
> separate integration
> ([`prefill_integration.py`](../rkv/prefill_integration.py));
> see [`FINDINGS_AND_ROADMAP.md`](./OPTIMIZATIONS.md).

## 1. Overview

R-KV is a **decoding-time** KV-cache compressor. While a model generates a long
output, R-KV periodically evicts the *unimportant* and *redundant* past tokens,
keeping only a fixed `budget` of KV entries per request — freeing GPU memory
while preserving generation quality.

The port is split into two layers:

| Layer | File | Responsibility |
| --- | --- | --- |
| **Algorithm** | [`algo.py`](../rkv/algo.py) | Pure, device-agnostic R-KV scoring & selection. Zero SGLang deps. CPU-testable. |
| **Integration** | [`integration.py`](../rkv/integration.py) | Bridges the algorithm to SGLang's paged KV pool, FlashInfer decode path, and scheduler lifecycle. |

Phase-1 scope: **FlashInfer backend, `page_size=1`, correctness first.**
`batch >= 1` is supported via **per-request triggering** (each request decides
independently when to compress); see §11.

## 2. The core tension (and how we resolve it)

The algorithm is **per-head / per-layer**: `R1KV.select_indices` returns a kept
set of shape `(bsz, kv_heads, budget)` — different heads may keep different
tokens. But SGLang's physical layout is **per-token / global**: one
`req_to_token` slot is shared across *all* layers and *all* heads, so dropping a
token drops it everywhere.

We therefore **reduce** the algorithm's scores into a single global per-token
decision:

- **cross-head:** mean of the per-head joint scores.
- **cross-layer:** sum over every layer's scores (each layer contributes its own
  `q·kᵀ` importance + key-similarity redundancy).

The result is one score vector per request; we keep the top `budget - window`
past tokens plus the trailing `window` observation tokens.

## 3. Integration components (`integration.py`)

- **`RKVConfig`** — `budget`, `window_size`, `kernel_size`, `mix_lambda`,
  `buffer_size` (compress every N generated tokens), `min_seq_len`.
- **`RKVRequestState`** — per-request bookkeeping: trigger counter
  (`steps_since_compact`), a **per-layer ring buffer of the last `window_size`
  decode queries** (queries are per-layer, allocated lazily from the first
  query's shape), the transient cross-layer score accumulator, and a
  back-reference to the owning `Req`.
- **`RKVCompressor`** — the coordinator. Holds `R1KV`, the pools/allocator, and
  the per-request states. Public surface:
  - `on_request_begin(req)` / `on_request_end(req)` — lifecycle.
  - `observe_decode_layer(q, k, v, layer, forward_batch)` — called per layer
    during decode: caches the query into the observation window. (Scoring is
    **not** done here; it is batched across all layers in `maybe_compact`.)
  - `maybe_compact(forward_batch)` — called after the full forward pass; for any
    armed request, scores all layers in **one batched pass** (with an A/B gate
    against the per-layer reference), assembles the global kept set, and
    physically compacts.
  - `override_decode_positions(forward_batch)` — replaces decode positions with
    *logical* positions (see §5).
  - `take_pending_length_updates()` — hands the scheduler the new physical
    lengths to apply to its batch tensors.

## 4. Per-decode-step data flow

```
scheduler.update_running_batch
  └─ _apply_rkv_pre_decode(batch)            # register new reqs; apply previous
        ├─ on_request_begin(req) if new        step's physical-length shrink to
        └─ apply take_pending_length_updates   batch.seq_lens BEFORE +1
  └─ batch.prepare_for_decode()              # seq_lens += 1, alloc out_cache_loc

FlashInferAttnBackend.init_forward_metadata(fb)   # fb = the REAL decode batch
  ├─ forward_batch.rkv_compressor = self.rkv_compressor   # bind onto real batch
  └─ override_decode_positions(fb)                        # logical positions

model.forward → per layer → RadixAttention → FlashInferAttnBackend.forward_decode
  └─ after set_kv_buffer:
       observe_decode_layer(q, k, v, layer, fb)   # cache query into window
                                                   # arm compaction every buffer_size
                                                   # steps once seq_len >= budget

model_runner.forward (after all layers)
  └─ maybe_compact(fb)                          # for armed reqs:
        ├─ score all layers in ONE batched pass (A/B gate vs per-layer ref)
        ├─ assemble kept = top(budget-window) past + trailing window
        └─ _compact_request(...)                # see §6
```

## 5. Key design decision — position / rotary (“scheme A”)

SGLang treats `seq_lens` as **both** the physical KV length **and** the rotary
position source (`positions = clamp_position(seq_lens)`). R-KV breaks that
identity: after eviction the physical KV length shrinks to `budget`, but future
tokens must keep their **original absolute positions** so their rotary stays
consistent with the retained keys (rotary is baked into stored keys).

Resolution:

- **`seq_lens` / `kv_committed_len` / `kv_allocated_len` track the physical KV
  length** (drop to `budget` at compaction, `+1` per step after). This makes
  slot allocation and attention length automatically correct — new tokens append
  at the physical tail, attention reads exactly `budget (+n)` slots.
- **Positions stay logical.** `override_decode_positions` sets
  `positions[i] = len(origin_input_ids) + len(output_ids) - 1` for each R-KV
  request, overriding the physical-length-derived value. The just-sampled token
  is already appended to `output_ids` at forward time, so the count *minus one*
  is the current token's 0-based rotary position; for an un-compacted request
  this equals the baseline `clamp_position(seq_lens) = seq_lens - 1`, so it is
  safe to always apply. (Omitting the `-1` was a bug fixed 2026-07-02 — it
  rotated every R-KV decode token at position+1, leaving a one-slot gap between
  the prompt and the generation; see §8.)

Scheduler and ModelRunner share the process, so `Req` length fields are updated
in place during compaction; the batch-level `seq_lens` tensor is updated by the
scheduler via `take_pending_length_updates()`.

## 6. Physical compaction (`_compact_request`, `page_size=1`)

Given the surviving logical indices `kept` (ascending, length `budget`) for a
request occupying physical slots `slots = req_to_token[idx, :seq_len]`:

1. `src = slots[kept]` (surviving physical slots), `dst = slots[:budget]`
   (target front slots).
2. For **every layer**: read `k/v_buffer[src]`, `.clone()` (avoids src/dst
   overlap corruption), write to `k/v_buffer[dst]`.
3. `free(slots[budget:seq_len])` — return the tail slots to the allocator
   (`page_size=1` ⇒ per-slot free is clean; the paged allocator would otherwise
   free at page granularity and clobber live neighbours).
4. `req_to_token[idx, budget:seq_len] = 0` (clear the tail); the front `budget`
   entries already point at `dst`.
5. Shrink `req.kv_committed_len / kv_allocated_len = budget`; publish the new
   physical length via `pending_length_updates`.

## 7. Wiring points (9 core files)

All changes are additive and applied by [`../patch/rkv-sglang-0.5.14.patch`](../patch/rkv-sglang-0.5.14.patch).

| File | Change |
| --- | --- |
| `server_args.py` | `--enable-rkv` + per-field flags (`--rkv-budget`, `--rkv-window-size`, `--rkv-kernel-size`, `--rkv-mix-lambda`, `--rkv-retain-ratio`, `--rkv-retain-direction`, `--rkv-buffer-size`, `--rkv-min-seq-len`), `--rkv-config` JSON override, `--rkv-max-active-requests`, `--rkv-fused-validation`; prefill mode `--enable-rkv-prefill` + `--rkv-prefill-config`. `_handle_rkv_validation` / `_handle_rkv_prefill_validation` reject unsafe combos at startup |
| `model_executor/model_runner.py` | Build `RKVCompressor` (and the prefill compressor) in `alloc_memory_pool` after the pools exist; call `maybe_compact` after the decode forward pass |
| `model_executor/model_runner_kv_cache_mixin.py` | Reserve the R-KV auxiliary GPU memory (the `rolling_q` observation-query buffer **and** the transient compaction workspace) inside KV-pool sizing so admission accounting is correct; apply the `--rkv-max-active-requests` cap on `max_running_requests` |
| `model_executor/forward_batch_info.py` | Restore **logical** rotary positions (`override_decode_positions`) at `ForwardBatch` construction so both eager and captured-CUDA-graph decode read the correct positions |
| `layers/attention/flashinfer_backend.py` | Backend holds `rkv_compressor` / `rkv_prefill_compressor`; bind onto the real decode batch in `init_forward_metadata`; call `observe_decode_layer` (decode) / `observe_prefill_layer` (prefill) after `set_kv_buffer` |
| `managers/scheduler.py` | Re-bind the compressor in `init_memory_pools` (see §8); `_apply_rkv_pre_decode` before `prepare_for_decode`; `commit_compactions` after the decode forward (two-phase compaction, §6) |
| `managers/scheduler_components/batch_result_processor.py` | Call `on_request_end` at the two real-finished points (beside `hisparse.request_finished`); commit any queued compaction frees before the finished-request release loop |
| `managers/schedule_batch.py` | `on_request_retract` hooks so a retracted request drops its per-request compressor state while `req_pool_idx` is still valid (avoids stale bookkeeping / KV leak on the next prefill) |
| `managers/schedule_policy.py` | **Compression-aware admission**: `PrefillAdder` reserves a request's *constant* compressed KV ceiling (`min(prompt, budget) + output`) instead of `prompt + output`, so many more requests are admitted concurrently |

## 8. Bugs fixed during bring-up

1. **`observe` never fired — compressor bound too early.** The scheduler
   snapshotted `rkv_compressor` in `__init__`, *before* `alloc_memory_pool`
   constructs it → it was `None`. Fix: re-bind in `init_memory_pools`, after the
   pools (and the compressor) exist.
2. **`observe` still saw `rkv=False` — wrong `forward_batch` object.** Binding
   `forward_batch.rkv_compressor` in `model_runner.forward` was useless because
   `_prepare_eager_forward_batch` rebuilds the batch before the model runs. Fix:
   the FlashInfer backend holds the compressor itself and binds it (and overrides
   positions) inside `init_forward_metadata`, which receives the *actual* decode
   batch; `forward_decode` uses `self.rkv_compressor`.
3. **`on_request_end` and frozen dataclass.** `SchedulerBatchResultProcessor` is
   a frozen dataclass built *before* `init_memory_pools` binds the compressor, so
   a snapshot field would capture `None`. Fix: look it up dynamically via
   `model_worker.model_runner.rkv_compressor` at the finished-request points.
4. **Rotary off-by-one in `override_decode_positions` (owner review, 2026-07-02).**
   The override used `len(origin_input_ids) + len(output_ids)`, but the
   just-sampled token is already in `output_ids` at forward time, so every R-KV
   decode token was rotated at `position + 1` (baseline is
   `clamp_position(seq_lens) = seq_lens - 1`). Impact was small because RoPE is
   relative — a uniform shift only leaves a single-position gap between the
   prompt and the generation — so end-to-end output stayed coherent. Fix:
   subtract 1 (see §5).
5. **No startup validation of R-KV-incompatible flags (owner review, 2026-07-02).**
   `--enable-rkv` silently corrupted the KV pool when combined with the radix
   cache, overlap scheduling, `page_size > 1`, or `tp > 1`. Fix:
   `ServerArgs._handle_rkv_validation` now rejects those combos at startup with an
   explicit error. (Decode CUDA graph was **later made compatible** via the
   hybrid eager/graph path and is no longer rejected.)

## 9. Environment (dev-v0.5.14 needs a newer stack than v0.5.3-era wheels)

- `transformers==5.8.1`, `flashinfer{,-cubin}==0.6.12`,
  `sglang-kernel==0.4.4+cu129` (from `https://docs.sglang.ai/whl/cu129/` so the
  torch C10 ABI matches), `torch==2.11.0+cu129`.
- HF **Xet** transfer can hang on large files behind some networks; download
  with `HF_HUB_DISABLE_XET=1` or `curl -L` on the resolved CDN URL.

## 10. Running & validation

Launch (required flags — radix cache off so R-KV can free slots, overlap off,
`page_size=1` for clean slot free). **Decode CUDA graph is supported and left on**
via the hybrid eager/graph path (the `window_size` steps ending at each
compaction, plus the compaction step, run eager; every other decode step replays
the captured graph). Pass `--disable-decode-cuda-graph` only if you want the
fully-eager path:

```bash
PYTHONPATH=$PWD/python HF_HUB_DISABLE_XET=1 python3 -m sglang.launch_server \
  --model-path /data/model/Qwen2.5-0.5B-Instruct \
  --attention-backend flashinfer \
  --disable-radix-cache --disable-overlap-schedule --page-size 1 \
  --enable-rkv --rkv-config '{"budget":64,"window_size":8,"buffer_size":16}' \
  --mem-fraction-static 0.6 --host 127.0.0.1 --port 30000
```

Verified end-to-end on Qwen2.5-0.5B-Instruct: with `budget=64, buffer_size=16`,
a long generation triggers compaction repeatedly (e.g. `phys 80 -> 64, frees 16
slots`), output stays coherent (no garbage, no loops), and `on_request_end`
clears per-request state to 0 after each request.

CPU unit tests (no GPU, no installed `sglang` needed — modules are loaded by
path):

```bash
python3 test/srt/mem_cache/test_rkv_integration.py
```

9 cases: kept-set assembly, slot relocation, overlap safety, physical-length
bookkeeping, logical-position decoupling, request lifecycle
(begin/end/idempotent), and batch >= 2 per-request triggering (arming + compaction).

## 11. Parallelism & batching support

**Current status: `batch >= 1`, `tp >= 1`, plain `dp >= 1`.** Validated for
`batch=1` and `batch > 1`; **tensor parallel (TP) is supported** via a cross-rank
score all-reduce (§11.2); **plain data parallel (DP) is supported** (§11.3). The
only unsupported multi-GPU mode is **DP attention** (`--enable-dp-attention`).

### 11.1 `batch > 1` — supported (method A: per-request triggering)

`observe_decode_layer` loops over every request in the decode batch
(`forward_batch.req_pool_indices`), and each request **independently** arms a
compaction once its own KV length reaches `min_seq_len` and `buffer_size` steps
have elapsed since its last compaction. Per-request arm flags, score
accumulators, query ring buffers, and compaction are all keyed by
`req_pool_idx`; `maybe_compact` resolves each armed request's own physical
`seq_len` via `_seq_len_by_req(forward_batch)`. No cross-rank concern.

> A batch-wide alternative ("compress the whole batch when the longest sequence
> exceeds budget") was considered and rejected: short requests below budget are
> no-ops (`select_indices` returns `None`), so it does the same real work as
> method A while adding wasted checks on short requests and ragged-slot batching
> overhead — no speedup.

### 11.2 Tensor parallel (TP ≥ 2) — supported (cross-rank score all-reduce)

Under TP, each rank holds only a **subset of the KV heads**. R-KV's importance /
redundancy score is reduced with a **mean over heads**, so each rank's *local*
score sees only its own heads. Left uncoordinated this is the dangerous case: it
does not crash, it **corrupts the KV cache**. Each rank would select a
**different** `kept` set, yet KV-slot allocation and `req_to_token` are
synchronized and identical across ranks — so `_compact_request` would rewrite
`req_to_token` differently and `free()` different physical slots per rank, and
the physical KV layout would **silently diverge** (wrong attention outputs,
eventual pool corruption; nothing detects it because each rank is internally
self-consistent).

**Fix (implemented).** The per-token score is **all-reduced (SUM) across the
attention-TP group before `_assemble_kept`**, so every rank tops-k the *same*
global score and evicts the *identical* tokens, keeping the replicated
`req_to_token` consistent. This is correct because the score is a cross-head
**mean** and every softmax/pool inside it is **per-head** (over the sequence
axis) → the cross-head reduction is **linear**. Head sharding is uniform
(`num_kv_heads % tp == 0` for distinct heads, or `tp % num_kv_heads == 0` with
uniform replication), so the all-reduced SUM equals the true global cross-head
mean scaled by a positive constant; `topk` is invariant to positive scaling, so
the kept set is identical on every rank.

Implementation:

- the compressor takes an `attn_tp_group` handle
  (`model_runner.attention_tp_group`, i.e. `get_attention_tp_group()`), stored as
  `self.attn_tp_group` / `self.attn_tp_size`; `None` / `world_size == 1` makes
  the path a no-op, so the single-GPU code is unchanged;
- `_reduce_score_across_tp(score)` does `attn_tp_group.all_reduce(score.float())`
  (fp32 so ties break identically on every rank) on each score right before
  `_assemble_kept`, in both the decode (`maybe_compact`) and prefill paths;
- armed requests are iterated in **sorted `req_pool_idx` order** so every rank
  issues its collectives in the same order (a mismatched order would mis-pair
  tensors or hang);
- `_check_kept_consistent_across_tp(kept)` all-reduces the kept indices and
  raises `RuntimeError` if any rank disagrees — self-validating on the first few
  compactions, or on **every** compaction when `SGLANG_RKV_TP_CHECK=1`.

**Validated (8x H100)** with `SGLANG_RKV_TP_CHECK=1` forcing the consistency
check on every compaction, across all three head-sharding regimes:

| tp | model (Q/KV heads) | local KV heads | phase | result |
| --- | --- | --- | --- | --- |
| 2 | Qwen2.5-Math-7B (28/4) | 2/rank (distinct) | decode | both ranks compact the same `req_pool_idx` at the same step, identical freed count; assertion never fired |
| 4 | Qwen2.5-Math-7B (28/4) | 1/rank (distinct) | decode | all 4 ranks 6 compactions each in lockstep; outputs byte-identical to tp=2 |
| 8 | Qwen3-30B-A3B (32/4) | 4 KV replicated x2 | prefill | all 8 ranks 4 compactions each in lockstep (`1163 -> 512`, freed 651); assertion never fired |

**DP attention** (`--enable-dp-attention`) is still unsupported (§11.3).

### 11.3 Data parallel (DP ≥ 2)

**Plain DP (`--dp-size N --tp-size 1`) — validated.** Under plain DP each rank
serves a **disjoint set of requests** with its own KV pool; requests never cross
ranks. So each rank runs its own R-KV over its own requests — unlike TP there is
**no cross-rank eviction-agreement problem**. Verified on
Qwen2.5-Math-7B (8× H100): every rank compresses independently, accuracy matches
single-GPU, no leaks/crashes, and throughput scales up to **5.1× on 8 GPUs**. See
[`../benchmark/RESULTS_dp.md`](../benchmark/RESULTS_dp.md).

**DP attention (`--enable-dp-attention`) — still unsupported.** This mode makes
attention data-parallel while MoE/FFN stay tensor-parallel. While plain TP is now
supported (§11.2), DP attention's padded/scattered `forward_batch` layout
(per-rank `num_real_reqs`, all-gather of attention inputs) has **not been
tested** against R-KV's `observe` / `override_decode_positions` / `maybe_compact`
hooks, so it is left unsupported.

### Support matrix

| Config | Status | Blocker |
| --- | --- | --- |
| `batch=1, tp=1, dp=1` | ✅ validated | — |
| `batch > 1` (tp=1, dp=1) | ✅ supported | per-request triggering (method A) |
| **TP ≥ 2** | ✅ supported | cross-rank score all-reduce before top-k (§11.2) |
| DP ≥ 2 (plain, `tp=1`) | ✅ validated | per-rank independent R-KV (see benchmark/RESULTS_dp.md) |
| DP attention (`--enable-dp-attention`) | ❌ unsupported | padded forward_batch layout unverified against the hooks |

> **Enforced at startup:** `ServerArgs._handle_rkv_validation` requires
> `--disable-radix-cache` / `--disable-overlap-schedule` / `--page-size 1`, and a
> late, post-resolution gate in `ModelRunner` rejects runtimes the hooks are not
> wired for (non-FlashInfer backend, MLA, hybrid-SWA, speculative decoding).
> **Plain TP and plain DP are allowed and validated** (§11.2 / §11.3);
> DP attention remains unsupported.

## 12. Other limitations / next

- **O(budget²) similarity** in `cal_similarity` — the per-layer scoring GEMMs are
  now **batched across layers** (one pass instead of `num_layers`); the per-token
  O(budget²) redundancy matrix itself is still a phase-2 target (chunking / a
  cheaper redundancy estimate).
- **`on_request_end`** is wired at finish, but state is otherwise only cleared
  lazily on `req_pool_idx` reuse — fine for phase 1.
- Larger-sample accuracy (MATH-500 / AIME-24) and a long-sequence throughput
  test (to show the memory/latency *benefit*, not just the overhead) are the
  next milestones.
