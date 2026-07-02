# R-KV Integration Layer — Implementation Notes

This document describes **how** R-KV (Redundancy-aware KV Cache Compression) is
implemented on top of a clean SGLang v0.5.14 baseline. For the *why* (research
findings, rejected alternatives, the `sparsity/` framework evaluation), read
[`DESIGN.md`](./DESIGN.md) first. This file is the practical map of the code:
the components, the per-step data flow, the exact wiring points, and the
decisions baked into them.

## 1. Overview

R-KV is a **decoding-time** KV-cache compressor. While a model generates a long
output, R-KV periodically evicts the *unimportant* and *redundant* past tokens,
keeping only a fixed `budget` of KV entries per request — freeing GPU memory
while preserving generation quality.

The port is split into two layers:

| Layer | File | Responsibility |
| --- | --- | --- |
| **Algorithm** | [`algo.py`](./algo.py) | Pure, device-agnostic R-KV scoring & selection. Zero SGLang deps. CPU-testable. |
| **Integration** | [`integration.py`](./integration.py) | Bridges the algorithm to SGLang's paged KV pool, FlashInfer decode path, and scheduler lifecycle. |

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
    during decode: caches the query, and (when a compaction is armed) computes
    and accumulates this layer's per-token score.
  - `maybe_compact(forward_batch)` — called after the full forward pass; for any
    armed request, assembles the global kept set and physically compacts.
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
       observe_decode_layer(q, k, v, layer, fb)   # cache query; accumulate score
                                                   # arm compaction every buffer_size
                                                   # steps once seq_len >= budget

model_runner.forward (after all layers)
  └─ maybe_compact(fb)                          # for armed reqs: compact
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
  `positions[i] = len(origin_input_ids) + len(output_ids)` for each R-KV
  request, overriding the physical-length-derived value. For an un-compacted
  request this equals the normal value, so it is safe to always apply.

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

## 7. Wiring points (5 core files)

| File | Change |
| --- | --- |
| `server_args.py` | `--enable-rkv`; per-field flags `--rkv-budget`, `--rkv-window-size`, `--rkv-kernel-size`, `--rkv-mix-lambda`, `--rkv-retain-ratio`, `--rkv-retain-direction`, `--rkv-buffer-size`, `--rkv-min-seq-len`; `--rkv-config '{...}'` JSON overrides the per-field flags |
| `model_executor/model_runner.py` | Build `RKVCompressor` in `alloc_memory_pool` (after pools exist); call `maybe_compact` after the decode forward pass |
| `layers/attention/flashinfer_backend.py` | Backend holds `rkv_compressor`; in `init_forward_metadata` bind it onto the real decode batch + `override_decode_positions`; call `observe_decode_layer` in `forward_decode` after `set_kv_buffer` |
| `managers/scheduler.py` | Re-bind `rkv_compressor` in `init_memory_pools` (see §8); `_apply_rkv_pre_decode` before `prepare_for_decode` |
| `managers/scheduler_components/batch_result_processor.py` | Call `on_request_end` at the two real-finished points (beside `hisparse.request_finished`, **not** at retract points) |

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

## 9. Environment (dev-v0.5.14 needs a newer stack than v0.5.3-era wheels)

- `transformers==5.8.1`, `flashinfer{,-cubin}==0.6.12`,
  `sglang-kernel==0.4.4+cu129` (from `https://docs.sglang.ai/whl/cu129/` so the
  torch C10 ABI matches), `torch==2.11.0+cu129`.
- HF **Xet** transfer can hang on large files behind some networks; download
  with `HF_HUB_DISABLE_XET=1` or `curl -L` on the resolved CDN URL.

## 10. Running & validation

Launch (phase-1 flags are required — R-KV runs only on the **eager decode**
path, so decode CUDA graph must be off; `page_size=1` for clean slot free;
overlap off for simple timing):

```bash
PYTHONPATH=sglang-src/python HF_HUB_DISABLE_XET=1 python3 -m sglang.launch_server \
  --model-path /data/model/Qwen2.5-0.5B-Instruct \
  --attention-backend flashinfer \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph \
  --disable-overlap-schedule --page-size 1 \
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
python3 tests/test_rkv_integration.py
```

9 cases: kept-set assembly, slot relocation, overlap safety, physical-length
bookkeeping, logical-position decoupling, request lifecycle
(begin/end/idempotent), and batch >= 2 per-request triggering (arming + compaction).

## 11. Parallelism & batching support

**Current status: single-GPU, `batch >= 1`** — validated at
`tp_size=1, dp_size=1` for both `batch=1` and `batch > 1`. Tensor parallel (TP)
and data parallel (DP) are **not** supported. The integration layer contains
**no distributed code** (no `tp_group`, no `all_reduce`, no `torch.distributed`).

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

### 11.2 Tensor parallel (TP ≥ 2) — NOT supported, silently incorrect ⚠️

This is the dangerous case: it will not crash, it will **corrupt the KV cache**.

Under TP, each rank holds only a **subset of the KV heads**. R-KV's importance /
redundancy score is reduced with a **mean over heads** — but each rank can only
see *its own* heads. Therefore:

1. each rank computes a **different** per-token score → selects a **different**
   `kept` set;
2. yet KV-slot allocation and `req_to_token` are **synchronized and identical**
   across ranks (every rank gets the same `out_cache_loc` each step, and the
   scheduler drives one logical sequence);
3. so `_compact_request` rewrites `req_to_token` **differently** and calls
   `free()` on **different physical slots** on each rank → the physical KV
   layout **diverges between ranks**.

Once the layout diverges, FlashInfer reads the wrong slots on some ranks and the
allocator's slot accounting no longer matches the (shared) `req_to_token` — i.e.
wrong attention outputs and, eventually, pool corruption. **Nothing detects
this**, because each rank is internally self-consistent; only the *cross-rank*
agreement is broken.

**To support TP**, the per-token score must be **all-reduced across the
attention-TP group** (summing each rank's head contributions) into one global
score *before* `kept` is assembled, so every rank evicts the **exact same**
tokens and keeps `req_to_token` identical. Concretely the compressor would need:

- a handle to the attention-TP process group (e.g. `model_runner.tp_group` /
  `attention_tp_group`);
- an `all_reduce(SUM)` of the accumulated per-token score in `maybe_compact`,
  right before `_assemble_kept`;
- (ideally) an assertion that every rank derived the identical `kept` indices.

None of this exists today, so **`--tp 2` or higher will silently produce wrong
results.** If TP is attempted before this is implemented, it should be hard-
blocked in `server_args` (reject `enable_rkv && tp_size > 1`).

### 11.3 Data parallel / dp-attention (DP ≥ 2) — unverified, no fundamental blocker

Under DP each rank serves a **disjoint set of requests** with its own KV pool;
requests never cross ranks. So "each rank runs its own R-KV over its own
requests" is self-consistent in principle — unlike TP there is **no cross-rank
eviction-agreement problem**. What is missing is only:

1. per-rank `batch > 1` already works (§11.1), but has only been validated in
   the single-rank (`dp=1`) case;
2. the dp-attention `forward_batch` layout (padded/scattered tokens, per-rank
   `num_real_reqs`, all-gather of attention inputs) has **not been tested** with
   R-KV's `observe` / `override_decode_positions` / `maybe_compact` hooks.

Likely extends with modest work, but it is neither implemented nor tested.

### Support matrix

| Config | Status | Blocker |
| --- | --- | --- |
| `batch=1, tp=1, dp=1` | ✅ validated | — |
| `batch > 1` (tp=1, dp=1) | ✅ supported | per-request triggering (method A) |
| **TP ≥ 2** | ❌ **silently incorrect** | **missing cross-rank all-reduce of scores** (fundamental) |
| DP ≥ 2 | ❌ untested | batch loop + dp-attention verification (no fundamental conflict) |

> **Recommendation:** until §11.2 is implemented, treat `--enable-rkv` as
> incompatible with `--tp > 1`, and ideally reject that combination at startup.

## 12. Other limitations / next

- **O(budget²) similarity** in `cal_similarity` — a phase-2 perf target
  (chunking / a cheaper redundancy estimate).
- **No CUDA-graph decode** yet (dynamic eviction can't live in a captured
  graph). Phase-2 would need a graph-compatible compaction scheme.
- **`on_request_end`** is wired at finish, but state is otherwise only cleared
  lazily on `req_pool_idx` reuse — fine for phase 1.
- Larger-sample accuracy (MATH-500 / AIME-24) and a long-sequence throughput
  test (to show the memory/latency *benefit*, not just the overhead) are the
  next milestones.
