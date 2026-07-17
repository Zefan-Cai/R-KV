# R-KV Integration — Implementation Notes (vLLM v0.25.1)

This document maps the R-KV runtime wiring onto vLLM v1. The patch is
**small and additive** — 13 files, ~849 inserted lines — and every hook is gated
so that when `VLLM_V1_R_KV_BUDGET`/`BUFFER` are unset the code is fully inert.
R-KV is wired into vLLM's **V1 GPU model runner**
(`vllm/v1/worker/gpu_model_runner.py`); because v0.25.1 defaults to a newer V2
runner, `VllmConfig.use_v2_model_runner` is gated to select V1 whenever R-KV is
enabled.

## 1. Layers

| Layer | File | Responsibility |
| --- | --- | --- |
| **Algorithm** | [`rkv/algo.py`](../rkv/algo.py) | Pure, device-agnostic R-KV scoring & selection (`R1KV`). No vLLM deps; CPU-testable. |
| **Integration** | [`rkv/integration.py`](../rkv/integration.py) | `RKVConfig` (env-driven, self-validating) + `RKVCompressor` — two-phase cross-layer eviction (`observe_layer` per layer, then one post-forward `compact_step`) against the paged KV cache. |

`rkv/` is copied into the vLLM tree as `vllm/rkv/` by `scripts/apply_rkv.sh`;
the patch only wires the runtime to call into it.

## 2. Per-decode-step data flow

```
Scheduler._update_after_schedule
  └─ set request.should_compress   (every VLLM_V1_R_KV_BUFFER tokens)
Scheduler._make_cached_request_data
  └─ CachedRequestData.{num_dropped_tokens, should_compress}
GPUModelRunner._update_states
  └─ input_batch.{num_dropped_tokens_cpu, should_compress_list}
GPUModelRunner._prepare_inputs → _rkv_prepare_physical
  └─ physical positions + seq_lens + occupied_slot_mapping
GPUModelRunner._build_attention_metadata
  └─ CommonAttentionMetadata.{should_compress_list, num_dropped_tokens_list,
                              occupied_slot_mapping}
FlashAttentionImpl.forward   (every layer, after attention reads this step's KV)
  ├─ RKVCompressor.record_query   (append this step's query to the rolling window)
  └─ RKVCompressor.observe_layer  (sum this layer's cross-head-mean score into
                                 rkv_score_acc; register this layer's KV)
GPUModelRunner.execute_model   (once, after the full forward)
  └─ RKVCompressor.compact_step   (one global cross-layer kept set; evict it from
                                 every registered layer; write num_dropped_tokens_list)
     └─ ModelRunnerOutput.num_dropped_tokens_list
Scheduler.update_from_output
  └─ request.num_dropped_tokens += ...
```

## 3. Wiring points (the 13 patched files)

| File | Change |
| --- | --- |
| `vllm/envs.py` | `VLLM_V1_R_KV_{BUDGET,BUFFER,WINDOW,KERNEL,MIX_LAMBDA,RETAIN_RATIO,SCORE_MODE,ASYNC,FREE_BLOCKS}` env vars (default 0/off). |
| `vllm/config/vllm.py` | select the V1 runner; disable prefix caching; force PIECEWISE cudagraph; gate async scheduling; **fail closed** on unsupported combos (speculative decoding, PP>1, DCP>1, DBO/microbatching, KV connectors, quantized KV, multimodal prefix-LM). |
| `vllm/v1/request.py` | `Request.{num_dropped_tokens, should_compress}`. |
| `vllm/v1/core/sched/output.py` | `CachedRequestData.{num_dropped_tokens, should_compress}` lists. |
| `vllm/v1/core/sched/scheduler.py` | arm `should_compress` on buffer-boundary crossings (gated on `not is_prefill_chunk`, past the prompt); carry the lists; accumulate `num_dropped_tokens`; reset it on preemption. |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.num_dropped_tokens_list`. |
| `vllm/v1/worker/gpu_input_batch.py` | `CachedRequestState.num_dropped_tokens`; `InputBatch.{num_dropped_tokens_cpu, should_compress_list}` + `add_request`/`condense` handling. |
| `vllm/v1/worker/gpu_model_runner.py` | `rkv_enabled`/`rkv_async` + buffers; `_rkv_prepare_physical` (with dropped/physical bounds invariants); `_rkv_validate_kv_cache` (exactly one `FullAttentionSpec` group of exact type on FlashAttention — rejects sliding-window / chunked / non-causal / `head_size_v` mismatch / MLA-or-sink subclasses, and non-`batched` scoring on the serving path); two-phase `compact_step` after the forward with a layer-identity (count + distinct-storage) check; feed `num_dropped_tokens_list` back into `ModelRunnerOutput`. |
| `vllm/v1/attention/backend.py` | optional R-KV fields on `CommonAttentionMetadata` (incl. `rkv_qplan`, `rkv_req_ids`). |
| `vllm/v1/attention/backends/flash_attn.py` | R-KV fields on `FlashAttentionMetadata`; construct `RKVCompressor`; **reject ALiBi and cascade attention** when R-KV is on; call `record_query` + `observe_layer` after `flash_attn_varlen_func`. |
| `vllm/v1/core/kv_cache_manager.py` | cap block reservation at `budget+buffer` and free R-KV-evicted blocks (`VLLM_V1_R_KV_FREE_BLOCKS`, **opt-in / default off**). |
| `vllm/v1/core/kv_cache_coordinator.py` | route `remove_rkv_evicted_blocks` to the single-type manager(s). |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `remove_rkv_evicted_blocks` — free a request's blocks above the kept cap. |

## 4. Physical eviction (`RKVCompressor.observe_layer` + `compact_step`)

Eviction is **two-phase and cross-layer** so a single kept set is applied to
every layer (see [`DESIGN.md`](./DESIGN.md) §2). During the forward, each layer
calls `observe_layer`, which — for every armed request whose physical KV length
≥ `budget + buffer` — scores that layer's past tokens, reduces across KV heads
(**mean**), and sums the result into a shared per-request accumulator; it also
registers the layer's `(key_cache, value_cache)`. After the full forward the
runner calls `compact_step` once, which for each armed request:

1. Selects one global kept set: the top `budget − window_size` past tokens by the
   summed score plus the trailing `window_size` window, sorted to preserve
   temporal order. Under tensor parallelism the summed scores are all-reduced
   across the TP group first, so every rank selects the identical set.
2. Relocates exactly that set into the leading `budget` physical slots
   (`occupied_slot_mapping[kv_start : kv_start + budget]`) of **every**
   registered layer, with one gather + scatter per layer.
3. Records `num_dropped_tokens_list[i] = kv_len − budget`.

The scoring/selection is batched across requests that share a cache length; the
default `batched` score mode defers scoring to `compact_step` (one cross-layer
GEMM), while `reference` scores each layer inside `observe_layer`. Both produce
the same kept set.

## 5. `occupied_slot_mapping`

Built in `_rkv_prepare_physical` only on steps where at least one request is
armed for compression. For each request it enumerates physical positions
`[0, num_kv + num_scheduled)` and maps them through the block table to physical
slot ids (numpy, mirroring how vLLM computes the scheduled-token slot mapping).
This is the array the compressor indexes to read and overwrite each request's
KV in place.

## 6. Review history — verified invariants (read before re-reviewing)

This port has been through several rounds of independent adversarial review.
A few findings **recur** because they look like bugs from a partial read of the
control flow but are already correct or already handled. This section records the
verified invariants and their evidence so they are not re-litigated. Each item
names the exact code to check.

### 6.1 The physical slot mapping is NOT overwritten by a logical one

**Recurring claim (raised more than once):** `_rkv_prepare_physical` computes the
physical slot mapping, but a later *unconditional*
`block_table.compute_slot_mapping(..., self.positions)` re-runs it with
**logical** positions and overwrites it, so after the first compaction new KV is
written to the wrong slots.

**This is a false positive.** In `GPUModelRunner._prepare_inputs` the two
`compute_slot_mapping` calls are the two branches of a single `if/else` — they
are **mutually exclusive**:

```python
if self.rkv_enabled:
    self._rkv_prepare_physical(...)          # physical compute_slot_mapping
else:
    self.seq_lens[:num_reqs] = ...           # logical seq_lens
    self.input_batch.block_table.compute_slot_mapping(
        ..., self.positions[:total_num_scheduled_tokens])   # logical, else-only
```

When R-KV is enabled, only `_rkv_prepare_physical` runs; the logical call lives
in the `else` and never executes. `grep compute_slot_mapping gpu_model_runner.py`
returns **exactly two sites**: the physical one inside `_rkv_prepare_physical`
and this `else`-branch one — there is no third, unconditional call. `self.positions`
intentionally keeps **logical** positions (for RoPE); physical positions live in a
separate `rkv_physical_positions` buffer that feeds the physical slot mapping.

**Empirical corroboration:** heavy-compaction few-shot GSM8K is coherent
(0.88 at `budget=256 buffer=64`, with dozens of physical `[RKV-COMPACT]`
evictions per 100 requests). If new-token KV were written to logical slots, every
token past the first compaction would land outside the physical attention range
and accuracy would collapse to near-zero.

### 6.2 Genuine-decode is detected by phase — mind the off-by-one

The query ring's observation window must advance **only on a genuine new-token
decode step** (not chunked prefill, not a preempted request's recompute). This
is computed by `rkv.integration.is_genuine_decode`, a pure, unit-tested function
the runner calls at the `plan_qwrite` site:

```python
rkv_genuine_decode = (
    (num_scheduled > 0)
    & (num_computed + num_scheduled >= num_tokens)   # reaches the frontier
    & (num_computed >= num_prompt)                   # already past the prompt
)
```

**The subtlety that bit an earlier version:** `num_computed_tokens` **lags
`num_tokens` by one on a normal decode step** — the token sampled last step is
already counted in `num_tokens`, but its KV is computed *this* step. So
`num_computed >= num_tokens` is **never true during decode**, and using it (an
earlier revision did, as did the scheduler's `not (num_computed < num_tokens)`
arming guard) silently marks **every** decode step non-genuine → the window
never fills → compaction is skipped everywhere → the engine quietly runs
Full-KV. Accuracy-only validation *cannot* catch this: at `budget=256
buffer=64`, real R-KV is ~0.88 and Full-KV is ~0.90, so a broken build looks
"fine". **Always confirm compaction is active** via the `[RKV-COMPACT]` /
`[RKV-SKIP]` TRACE lines (`VLLM_V1_R_KV_TRACE=1`) or the reported compaction
count, not accuracy alone. `is_genuine_decode` therefore tests *frontier + past
prompt*, not `num_computed >= num_tokens`. The single post-preemption catch-up
step is indistinguishable by counts and returns True; that is safe because the
ring was released on preemption, so `compact_step` finds an incomplete window
and skips until it refills.

### 6.3 Zero-token batched rows are unreachable on v0.25.1

**Claim:** `record_query`'s `last_idx = query_start_loc[1:num_reqs+1] - 1` can hit
a `num_scheduled_tokens == 0` row (wrong query, or `-1` → `index_select` raises).

**Not reachable on this vLLM.** The runner builds the per-request scheduled count
as `[scheduler_output.num_scheduled_tokens[i] for i in req_ids]` — an
**unconditional per-request dict lookup** over the whole persistent batch. The
scheduler only writes an entry for a request it actually scheduled with
`num_new_tokens > 0`, so a batched request with no scheduled tokens would raise
`KeyError` *upstream* before R-KV runs. Hence every forward row has ≥ 1 token and
`query_start_loc` is strictly increasing.

`record_query` still applies a `clamp_min(0)` to `last_idx`. **This is a
crash-only boundary guard**, not semantic future-proofing: if a future vLLM
allowed zero-token rows, the clamp would stop the raise but the row could still
read a wrong query, and the current `rkv_genuine_decode` mask does **not** yet
include `num_scheduled > 0`, so a caught-up-but-unscheduled row
(`num_computed == num_tokens`) could be marked a genuine decode. **Known minor
hardening gap** (see §6.9); it does not affect correctness under the current
upstream invariant.

### 6.4 "Compaction is transactional" means host-side control flow only

The transactional guarantee is **host-side and all-or-nothing**: once armed, a
step either completes compaction or **raises** (no silent partial skip; an
already-dropped request cannot fall back to Full-KV). It is **not** a CUDA
transaction. The GPU relocation (gather → scatter) and the metadata/count
publication are enqueued in program order on the **same stream**, so they execute
in order, but nothing waits on kernel completion — a *deferred* async CUDA error
surfaces on the next sync, by which point the worker is already unrecoverable, so
there is nothing to roll back. The docs and comments say "same-stream ordered
enqueue", not "transactional relocation".

### 6.5 `FREE_BLOCKS=1` is a known, opt-in, not-production-supported config

`VLLM_V1_R_KV_FREE_BLOCKS` is **opt-in and defaults off**. When off (the default
and the validated path) evicted blocks are not returned to the allocator, so
there is no cross-request block reuse and no ABA concern. When on, the scheduler
frees a request's tail blocks below the runner's cap **without** a
scheduler↔worker block-table version handshake, so a freed-then-reallocated block
could still be named by a stale worker row. This is a **known unsupported
configuration**, not an unhandled default-path bug. The fix (per-request
block-table generation/version + worker ACK before a freed block is reused, or
generation-tagged block handles) is a documented roadmap item. Do **not** re-file
this as a default-path P0.

### 6.6 `O(seq_len²)` score memory is guarded; sequence tiling is roadmap

The redundancy matrix is `~kv_heads × seq_len²`. This is bounded by a
memory-admission check with a **2× safety margin**, plus oversized-request
handling (a request whose *first* compaction would exceed the cap is left Full-KV;
an already-compacted request that no longer fits raises). This is a **scalability
limitation** with a defense in place, not a correctness bug. True sequence-dim
(blockwise/streaming) tiling that never materializes the `L × L` matrix is a
documented roadmap item.

### 6.7 Fail-closed support matrix (startup `ValueError`)

R-KV is validated only on a narrow configuration and **raises at startup** on
anything else, rather than silently running Full-KV or corrupting KV. Reviewers
should treat these as *rejected*, not as latent bugs: speculative decoding,
`PP>1`, `DCP>1`, quantized/FP8 KV, DBO/microbatching, KV connectors, multimodal
prefix-LM, more than one KV cache group, a non-exact-`FullAttentionSpec` group
(MLA / sink subclasses, sliding-window, chunked, non-causal, `head_size_v`
mismatch), ALiBi, cascade attention, cross-layer KV sharing, non-FlashAttention
backends, and non-`batched` scoring on the serving path. See
[`OPTIMIZATIONS.md`](./OPTIMIZATIONS.md) §"Known limitations".

### 6.8 What the validation numbers prove — and do not

- Few-shot GSM8K **91/100** (TP=1) is **end-to-end smoke evidence** that
  compaction stays active and shows no obvious regression. It is **not** a
  bit-level correctness proof: the gap to Full-KV (~94) is small and subject to
  sampling/model noise, and a partially wrong mask could still fire compaction.
  Treat it as a canary, not a certificate.
- The `FREE_BLOCKS=1`, 200-request, memory-pressured preemption run finishing
  **without crashing** is **stress evidence**, not a proof of ABA safety for
  block reuse (see §6.5).

### 6.9 Known minor follow-ups (acknowledged, deferred — not missed)

- Make the zero-token guard **semantic**, not just crash-safe: fold
  `num_scheduled_tokens > 0` into `rkv_genuine_decode` (and/or `assert
  np.all(num_scheduled_tokens > 0)` to stay fail-closed), and gather/scatter only
  `has_query` rows in `record_query`. Currently unnecessary (§6.3) but the
  correct future-proofing if the upstream invariant ever changes.
- Add a **direct unit test of the phase predicate derivation** — **done**.
  `rkv.integration.is_genuine_decode` is a pure function and
  `test_is_genuine_decode_predicate` covers decode / single-token recompute /
  single-token prefill chunk / multi-token prefill / unscheduled / catch-up. (It
  is what would have caught the off-by-one in §6.2.)
