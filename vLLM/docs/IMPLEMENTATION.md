# R-KV Integration — Implementation Notes (vLLM v0.25.1)

This document maps the R-KV runtime wiring onto vLLM v1. The patch is
**small and additive** — 13 files, ~522 inserted lines — and every hook is gated
so that when `VLLM_V1_R_KV_BUDGET`/`BUFFER` are unset the code is fully inert.
R-KV is wired into vLLM's **V1 GPU model runner**
(`vllm/v1/worker/gpu_model_runner.py`); because v0.25.1 defaults to a newer V2
runner, `VllmConfig.use_v2_model_runner` is gated to select V1 whenever R-KV is
enabled.

## 1. Layers

| Layer | File | Responsibility |
| --- | --- | --- |
| **Algorithm** | [`rkv/algo.py`](../rkv/algo.py) | Pure, device-agnostic R-KV scoring & selection (`R1KV`). No vLLM deps; CPU-testable. |
| **Integration** | [`rkv/integration.py`](../rkv/integration.py) | `RKVConfig` (env-driven) + `RKVCompressor.compact_batch` — per-request physical eviction against the paged KV cache. |

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
FlashAttentionImpl.forward
  └─ RKVCompressor.compact_batch  (after attention reads this step's KV)
     └─ writes num_dropped_tokens_list in place
GPUModelRunner (execute_model)
  └─ ModelRunnerOutput.num_dropped_tokens_list
Scheduler.update_from_output
  └─ request.num_dropped_tokens += ...
```

## 3. Wiring points (the 10 patched files)

| File | Change |
| --- | --- |
| `vllm/envs.py` | `VLLM_V1_R_KV_{BUDGET,BUFFER,WINDOW,KERNEL}` env vars (default 0 = off). |
| `vllm/config/vllm.py` | force the V1 model runner when R-KV is enabled (`use_v2_model_runner`). |
| `vllm/v1/request.py` | `Request.{num_dropped_tokens, should_compress}`. |
| `vllm/v1/core/sched/output.py` | `CachedRequestData.{num_dropped_tokens, should_compress}` lists. |
| `vllm/v1/core/sched/scheduler.py` | arm `should_compress` on buffer-boundary crossings; carry the lists; accumulate `num_dropped_tokens` from the model output. |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.num_dropped_tokens_list`. |
| `vllm/v1/worker/gpu_input_batch.py` | `CachedRequestState.num_dropped_tokens`; `InputBatch.{num_dropped_tokens_cpu, should_compress_list}` + `add_request`/`condense` handling. |
| `vllm/v1/worker/gpu_model_runner.py` | `rkv_enabled` flag + buffers; `_rkv_prepare_physical` helper; gated call in `_prepare_inputs`; populate `CommonAttentionMetadata`; feed `num_dropped_tokens_list` back into `ModelRunnerOutput`. |
| `vllm/v1/attention/backend.py` | optional R-KV fields on `CommonAttentionMetadata`. |
| `vllm/v1/attention/backends/flash_attn.py` | R-KV fields on `FlashAttentionMetadata`; construct `RKVCompressor`; call `compact_batch` after `flash_attn_varlen_func`. |

## 4. Physical eviction (`RKVCompressor.compact_batch`)

For each armed request whose physical KV length ≥ `budget + buffer`:

1. Gather the request's full KV from `key_cache` / `value_cache` using
   `occupied_slot_mapping[kv_start:kv_end]`.
2. Score + select with `R1KV.update_kv` (importance − redundancy, keep
   `budget`).
3. Write the survivors back into the leading physical slots
   `occupied_slot_mapping[kv_start : kv_start + budget]`.
4. Record `num_dropped_tokens_list[i] = kv_len − budget`.

Because the same `num_dropped_tokens_list` object is shared across a group's
layers, the first layer sets it and later layers (which drop the same count)
leave it unchanged — matching the reference behaviour.

## 5. `occupied_slot_mapping`

Built in `_rkv_prepare_physical` only on steps where at least one request is
armed for compression. For each request it enumerates physical positions
`[0, num_kv + num_scheduled)` and maps them through the block table to physical
slot ids (numpy, mirroring how vLLM computes the scheduled-token slot mapping).
This is the array the compressor indexes to read and overwrite each request's
KV in place.
