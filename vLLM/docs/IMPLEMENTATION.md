# R-KV Integration — Implementation Notes (vLLM v0.25.1)

This document maps the R-KV runtime wiring onto vLLM v1. The patch is
**small and additive** — 13 files, ~803 inserted lines — and every hook is gated
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
| `vllm/config/vllm.py` | select the V1 runner; disable prefix caching; force PIECEWISE cudagraph; gate async scheduling; **fail closed** on unsupported combos (speculative decoding, PP>1, DCP>1, DBO/microbatching, KV connectors, quantized KV). |
| `vllm/v1/request.py` | `Request.{num_dropped_tokens, should_compress}`. |
| `vllm/v1/core/sched/output.py` | `CachedRequestData.{num_dropped_tokens, should_compress}` lists. |
| `vllm/v1/core/sched/scheduler.py` | arm `should_compress` on buffer-boundary crossings (**never while a preempted request is recomputing**); carry the lists; accumulate `num_dropped_tokens`; reset it on preemption. |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.num_dropped_tokens_list`. |
| `vllm/v1/worker/gpu_input_batch.py` | `CachedRequestState.num_dropped_tokens`; `InputBatch.{num_dropped_tokens_cpu, should_compress_list}` + `add_request`/`condense` handling. |
| `vllm/v1/worker/gpu_model_runner.py` | `rkv_enabled`/`rkv_async` + buffers; `_rkv_prepare_physical` (with dropped/physical bounds invariants); `_rkv_validate_kv_cache` (exactly one `FullAttentionSpec` group of exact type on FlashAttention — rejects sliding-window / chunked / non-causal / `head_size_v` mismatch / MLA-or-sink subclasses); two-phase `compact_step` after the forward with a layer-identity (count + distinct-storage) check; feed `num_dropped_tokens_list` back into `ModelRunnerOutput`. |
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
