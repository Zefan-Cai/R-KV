# FA3 drop-in backend validation (H100)

- Date: 2026-07-09
- Pluto job: `rkv-fi-bench-1` (`video-world-models`), 2× NVIDIA H100 80GB HBM3;
  all measurements used GPU 0 only.
- Model: DeepSeek-R1-Distill-Qwen-1.5B, bf16.
- Stack: Python 3.12, torch 2.10.0+cu128, FlashInfer 0.6.12,
  `flash-attn-3==3.0.0` built from Dao-AILab/flash-attention commit
  `5835c733e7e9c07606b045255768e8a7e9e851bd` (`hopper/`).
- R-KV implementation: commit `867512b1` plus the parity probe and raw
  artifacts stored beside this file.

## Correctness

- Four health smokes passed: FA3 / FlashInfer × R-KV / FullKV.
- `fa3_parity.log` is a teacher-forced comparison with ragged prompts and the
  non-contiguous active set `[0, 2]`:
  - FullKV: 99/99 argmax agreement, worst logit max-delta 0.2188.
  - R-KV: 99/99 argmax agreement, worst logit max-delta 0.5000; compactions
    `[2, 0, 3]` prove post-compaction decode was covered.
- CPU contracts separately cover the older-FA3 padded fallback that is used
  when `cache_batch_idx` is unavailable.

## Throughput

Both runs use prompt 512, forced decode 2048, batch 16, two trials per cell.
The primary run executed FlashInfer then FA3; the reverse run executed FA3
then FlashInfer. The aggregate is the mean of all four trials.

| mode | primary FI / FA3 tok/s | reverse FI / FA3 tok/s | 4-trial FI / FA3 tok/s | FA3 delta |
|---|---:|---:|---:|---:|
| FullKV | 1365.6 / 1188.3 | 1365.7 / 1145.8 | 1365.6 / 1167.0 | -14.5% |
| R-KV 1024 | 1352.5 / 1134.6 | 1305.0 / 1128.2 | 1328.8 / 1131.4 | -14.9% |

Peak `torch.cuda.max_memory_allocated` is backend-identical: 5.063 GiB
(FullKV) and 5.447 GiB (R-KV). The drop-in path intentionally shares the
same interleaved KV pool and base FlashInfer workspace; these numbers are not
a native planar-cache FA3 memory comparison.

## Isolation experiments

The raw microbench logs explain the end-to-end result:

- Planar vs shared interleaved FA3 K/V views differ by at most 1.1%, so the
  zero-copy stride is not the throughput cause (`fa3_stride_microbench.log`).
- At this model's decode shape (B=16, QH=12, KVH=2, D=128), the steady-state
  per-layer call is about 0.024 ms for FlashInfer and 0.088 ms for FA3, or
  3.6-3.7× slower (`attention_kernel_microbench.log`). Across the model's
  layers this accounts for the observed end-to-end gap.
- Sweeping `pack_gqa` and `num_splits` did not materially close the gap
  (`fa3_tuning_microbench.log`).
- Reusing FA3 scheduler metadata saved 3-7% in the isolated kernel call, but
  generating that metadata every decode step erased the gain end to end. The
  interleaved paired run remained -16.2%; the experiment was rejected and is
  preserved under `scheduler_experiment/`, not shipped in the engine.

## Artifact map

- `primary/`: four strict smoke logs and the first four A/B benchmark logs.
- `reverse/`: reverse-order A/B benchmark logs.
- `scheduler_experiment/`: rejected scheduler-metadata end-to-end run.
- `microbench/`: stride, kernel, tuning, scheduler, and memory probes.
- `tooling/`: the original chain plus strict retry/verifier and ordered A/B
  runners used to create these logs.
- `fa3_retry.log`: strict aggregate status (`FA3_RETRY_DONE rc=0`).
- `fa3_parity.log`: teacher-forced parity result.
- `fa3_chain.log` / `fa3_verify.log`: provenance for the initial automation
  failure. `setup.py install` completed, but its suppressed-error immediate
  import check returned non-zero and stopped before validation; a direct
  import and the strict verifier passed seconds later. The strict retry is the
  authoritative result.
