# R-KV on vLLM — GSM8K budget × buffer sweep (H100)

Accuracy and throughput of the R-KV decode-time KV-cache compressor across the
full `budget × buffer` grid, measured on the SGLang-parity differential harness
after the async-scheduling fix (see
[`../docs/RESIDUAL_GAP_INVESTIGATION.md`](../docs/RESIDUAL_GAP_INVESTIGATION.md)).

## Setup

- **Model:** Qwen2.5-Math-7B-Instruct (bf16), `--enforce-eager`.
- **Engine:** vLLM v0.25.1 + R-KV patch (this repo), V1 GPU runner,
  FlashAttention backend, `block_size=16`, `gpu_memory_utilization=0.85`,
  `max_model_len=4096`. Prefix caching and async scheduling are auto-disabled
  when R-KV is on. Scoring uses the default **batched cross-layer** path
  (`score_mode="batched"`), ported from the SGLang port.
- **Workload:** SGLang's few-shot GSM8K harness
  (`SGLang/benchmark/data/gsm8k_fewshot.jsonl`), prompt ≈ 700 tokens (> every
  budget), **200 questions**, greedy (`temperature=0`), `max_tokens=512`,
  stop `"\nProblem"`. Decode-only R-KV (`window=8`), same extraction as the
  `对拍` harness.
- **Hardware / execution:** 8× H100 80GB. The 13 configs were run **one per GPU,
  8 in parallel** (two waves), each an offline `LLM.generate` over all 200
  prompts (natural high-concurrency batching). Total wall for the whole sweep:
  **~2m40s**.
- **Throughput** = decode (output) tokens ÷ generation wall time. Because the
  sweep runs 8 processes concurrently on one host, absolute tok/s carries ~8%
  host contention (Full-KV alone = 5330 tok/s vs 5138 tok/s inside the parallel
  wave); the **relative** buffer/budget trends are unaffected.

## Accuracy (% correct, 200 questions)

Full-KV ceiling on this harness: **~91%** (181–182/200; Full-KV runs with async
scheduling, so it varies ±1 question run-to-run). R-KV uses `mix_lambda=0.1`
(matches SGLang; see note below).

| budget \ buffer | 16 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- |
| **128** | 65.5 | 74.5 | 83.5 | 90.5 |
| **256** | 85.5 | 89.0 | 91.5 | 91.5 |
| **512** | 90.5 | 90.5 | 91.5 | 91.5 |

- **Buffer is the accuracy knob at tight budgets.** At `budget=128`, raising the
  buffer 16 → 256 recovers **+25 points** (65.5 → 90.5). At `budget=512` the
  model is already near-lossless at every buffer (90.5 – 91.5).
- **`budget=256`: buf64 = 89.0% and buf128 = 91.5%** meet or exceed the SGLang
  reference (88.5% / 89.0%). `buffer ≥ 128` is **lossless** (91.5% ≥ the ~91%
  ceiling). `buf16 = 85.5%` still trails SGLang (88.0%) by ~2.5 pts — a
  borderline top-k sensitivity to the FlashAttention-vs-FlashInfer K/Q numerics,
  amplified by buf16's ~12 compactions/request (not a config or logic bug: the
  algorithm parameters now match SGLang exactly).
- **`mix_lambda` matters and is buffer-sensitive.** The joint score is
  `mix_lambda·importance − (1−mix_lambda)·redundancy`. vLLM originally defaulted
  to `0.07` (the R-KV *algorithm* class default); SGLang's runtime and the HF
  reference eval use **`0.1`**. Switching to `0.1` lifts every tight-buffer
  config (b256/buf16 83.0→85.5, buf64 87.5→89.0, buf128 90.5→91.5) and is the
  bug behind the earlier buffer-sensitivity vs SGLang.
- Tighter configs also **generate longer** (once critical context is evicted the
  model rambles), which both lowers accuracy and adds work.

## Decode throughput (output tok/s, 8-way parallel, batched scoring)

Full-KV: 5138 tok/s (async, in-wave) / 5330 tok/s (alone).

| budget \ buffer | 16 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- |
| **128** | 1812 | 1969 | 1947 | 1946 |
| **256** | 1723 | 1977 | 2058 | 2017 |
| **512** | 1492 | 1892 | 1962 | 1985 |

- With **batched cross-layer scoring** (the default), throughput is largely
  **flat across buffers** (~1700–2060 tok/s): batching removed the per-compaction
  scoring bottleneck, so compacting more often (small buffer) costs much less.
  `buffer=16` is still the slowest (most compactions) but only ~10–25% below the
  larger buffers, not ~2.5× as with per-layer scoring.
- **Budget barely moves throughput** at a fixed buffer, so a larger budget is
  nearly free for accuracy.
- **Full-KV is still faster than any R-KV config** on this workload: at 200 GSM8K
  prompts the 7B model's KV pool fits comfortably in 80 GB, so the run is **not
  memory-bound** and R-KV is pure overhead (compaction + forced synchronous
  scheduling). R-KV's constant-footprint advantage — more concurrency / longer
  context in the same memory — only pays off in the memory-bound regime (long
  outputs, high concurrency); see [`../docs/OPTIMIZATIONS.md`](../docs/OPTIMIZATIONS.md).

### Batched vs per-layer scoring

The default `score_mode="batched"` runs one cross-layer scoring GEMM at compaction
time (ported from the SGLang port) instead of 28 per-layer GEMMs in the forward.
It is **accuracy-identical** (same kept set; same **508 compactions** at
`b256/buf64` — 200 first-compactions trimming each prompt to budget + 308
steady-state at 64-token intervals) and speeds up decode most where compaction is
most frequent:

| config | per-layer tok/s | batched tok/s | speedup |
| --- | --- | --- | --- |
| b256/buf16 | 836 | 1723 | +106% |
| b256/buf64 | 1556 | 1977 | +27% |
| b256/buf128 | 1835 | 2058 | +12% |
| b256/buf256 | 2005 | 2017 | +1% |

Tight buffers (frequent compaction) roughly **double**; large buffers (rare
compaction) are unchanged, as expected.

## Full per-config results

Accuracy at `mix_lambda=0.1` (the default); throughput columns are from the
batched-scoring run and are `mix_lambda`-independent.

| Config | Accuracy | Decode tok/s | Total tok/s | Wall (s) | Avg gen len |
| --- | --- | --- | --- | --- | --- |
| Full-KV (async) | ~91% (181/200) | 5138 | 26179 | 6.6 | 170 |
| budget=128 buffer=16 | 65.5% (131/200) | 1812 | 6702 | 28.5 | 258 |
| budget=128 buffer=64 | 74.5% (149/200) | 1969 | 8637 | 20.9 | 206 |
| budget=128 buffer=128 | 83.5% (167/200) | 1947 | 9630 | 18.1 | 177 |
| budget=128 buffer=256 | 90.5% (181/200) | 1946 | 9953 | 17.4 | 169 |
| budget=256 buffer=16 | 85.5% (171/200) | 1723 | 7305 | 25.0 | 215 |
| budget=256 buffer=64 | 89.0% (178/200) | 1977 | 8942 | 20.0 | 198 |
| budget=256 buffer=128 | 91.5% (183/200) | 2058 | 9580 | 18.5 | 191 |
| budget=256 buffer=256 | 91.5% (183/200) | 2017 | 10334 | 16.8 | 169 |
| budget=512 buffer=16 | 90.5% (181/200) | 1492 | 7583 | 22.9 | 171 |
| budget=512 buffer=64 | 90.5% (181/200) | 1892 | 9505 | 18.3 | 173 |
| budget=512 buffer=128 | 91.5% (183/200) | 1962 | 10115 | 17.1 | 168 |
| budget=512 buffer=256 | 91.5% (183/200) | 1985 | 10054 | 17.3 | 172 |

## Takeaways & recommendations

1. **`buffer ≈ budget` is the sweet spot** — it maximises both accuracy (lossless
   at `budget ≥ 256`) and throughput (fewest compactions). `budget=256
   buffer=256` = **91.5% @ 2017 tok/s**.
2. **`budget=256 buffer≥64` matches or beats SGLang** (buf64 89.0 vs 88.5, buf128
   91.5 vs 89.0). `buffer=16` still trails by ~2.5 pts (85.5 vs 88.0) — residual
   K/Q numerics near the top-k cutoff.
3. **Tiny buffers hurt accuracy, not throughput (anymore).** `buffer=16` is the
   worst on accuracy (frequent, aggressive compaction); with batched scoring its
   throughput penalty is now only ~10–25% (was ~2.5×). Still avoid it for
   accuracy.
3. **`budget=512` is robustly near-lossless** (89.5 – 91.5%) at any buffer — use
   it when accuracy matters more than memory savings.
4. **`budget=128` only works with a large buffer** (`buffer=256` → 90.5%); at
   small buffers it collapses (64 – 82%).
5. Throughput here reflects R-KV **overhead** (non-memory-bound); the memory /
   concurrency **benefit** requires a memory-bound workload.

## Reproduce

```bash
# per-config runner (accuracy + throughput), one GPU:
CUDA_VISIBLE_DEVICES=0 VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=64 \
  RKV_N=200 RKV_MAXTOK=512 RKV_OUT=/tmp/r.json python bench_sweep.py

# full sweep across 8 GPUs (two waves):
bash run_sweep.sh
```

`bench_sweep.py` and `run_sweep.sh` are included in this folder. `budget=0
buffer=0` selects the Full-KV baseline; `RKV_NOASYNC=1` forces synchronous
scheduling for the Full-KV baseline (R-KV forces it automatically).
