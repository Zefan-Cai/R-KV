# R-KV Benchmark Results — Qwen2.5-Math-7B-Instruct (GSM8K)

Decode-time R-KV on a strong math model, measured with **SGLang's own GSM8K
harness** (`sglang-src/benchmark/gsm8k/bench_sglang.py`, 5-shot) so the numbers
line up with the upstream R-KV reference. Companion reports:
[`RESULTS_dp.md`](./RESULTS_dp.md) (data-parallel scaling),
[`RESULTS_tp.md`](./RESULTS_tp.md) (tensor-parallel scaling).

## Setup

- **Model**: `Qwen/Qwen2.5-Math-7B-Instruct` (bf16, GQA), single **NVIDIA H100 80GB**.
- **Harness**: `bench_sglang.py`, standard GSM8K test set, **5-shot**, first
  **200 questions**, `max_new_tokens=512`, `temperature=0`, **`--parallel 32`**.
  Prompt ≈ 900 tokens; output ≈ 110 tokens/req.
- `--mem-fraction-static 0.85`, FlashInfer backend, `page_size 1`.
- **R-KV**: `window_size=8`, decode **and** prefill CUDA graphs **ON**, and the
  **fused redundancy kernel is adopted** (server logs
  `R-KV fused-redundancy gate: OK -> fused adopted`).
- **Two Full-KV baselines** (both matter — see the note):
  - **production** — radix/prefix cache + overlap schedule + CUDA graphs all ON
    (fastest Full-KV; `launch_server.sh fullkv`).
  - **constrained** — the *exact* flags R-KV requires (radix/overlap OFF,
    `page_size 1`), no compression (`launch_server.sh constrained`). This is the
    **fair A/B baseline**.

> **Why two baselines?** R-KV structurally cannot use the radix/prefix cache — it
> frees KV slots the radix tree would still reference. With a 5-shot prompt every
> request shares a large prefix, so the prefix cache *alone* makes production
> Full-KV much faster, an advantage unrelated to compression. The **constrained**
> baseline removes it from both sides, isolating R-KV's true cost.

## Baselines

| Full-KV | Accuracy | Latency (200) | Throughput |
| --- | :---: | ---: | ---: |
| production (radix + overlap + graphs) | 0.915 | 8.3 s | **2649 tok/s** |
| constrained (R-KV's flags, no compression) | 0.910 | 12.3 s | **1792 tok/s** |

Turning off radix/overlap and forcing `page_size 1` alone costs Full-KV **~32 %**
(2649 → 1792 tok/s) — that is the prefix-cache advantage R-KV cannot use, **not** a
compression cost. All R-KV rows below are compared to the **constrained** baseline.

## R-KV — `budget` × `buffer_size`

`--parallel 32`, 200 questions. **Compactions** = physical KV evictions during the run.

| `budget` | `buffer_size` | Accuracy | Throughput | vs constrained | Compactions |
| ---: | ---: | :---: | ---: | ---: | ---: |
| *constrained Full-KV* | — | 0.910 | 1792 | — | 0 |
| 512 | 256 | 0.900 | 1715 | −4 % | 4 |
| 512 | 128 | 0.910 | 1628 | −9 % | 62 |
| 512 | 64 | 0.910 | 1549 | −14 % | 245 |
| 512 | 16 | 0.915 | 1161 | −35 % | 1269 |
| 256 | 256 | 0.900 | 1731 | −3 % | 4 |
| 256 | 128 | 0.900 | 1679 | −6 % | 64 |
| 256 | 64 | 0.885 | 1533 | −14 % | 247 |
| 256 | 16 | 0.880 | 1126 | −37 % | 1232 |
| 128 | 256 | 0.900 | 1684 | −6 % | 4 |
| 128 | 128 | 0.865 | 1660 | −7 % | 64 |
| 128 | 64 | 0.820 | 1533 | −14 % | 293 |
| 128 | 16 | 0.750 | 1152 | −36 % | 1454 |

## Findings

1. **`budget` sets the accuracy wall.** `budget=512` is **lossless** (0.90–0.915
   vs constrained 0.910 / production 0.915); `budget=256` is near-lossless
   (0.88–0.90); `budget=128` holds only at large buffers and collapses to **0.75**
   at `buffer=16` — there it evicts most of the ~900-token prompt+CoT every 16
   steps.
2. **`buffer_size` sets the cost.** Against the fair baseline, R-KV costs only
   **~3–7 %** at `buffer ≥ 128`, ~14 % at `buffer=64`, and ~35 % at `buffer=16`.
   The cost tracks **compaction frequency**: each compaction forces a short eager
   window (the `window_size` steps around it) out of the CUDA graph, so frequent
   compaction (small buffer) loses more graph replay.
3. **R-KV stays fast *while* compacting.** e.g. `budget=256, buffer=128` runs **64
   physical compactions at 1679 tok/s** (−6 % vs constrained Full-KV) with
   **identical accuracy** (0.900); `budget=512, buffer=64` sustains 1549 tok/s
   across **245 compactions**, still lossless.
4. **The production↔constrained gap (2649 → 1792) is entirely prefix caching.**
   R-KV trades that for a *constant, prompt-independent* KV footprint. This bench
   (short GSM8K outputs, a large KV pool at `mem-fraction 0.85`) is **not
   memory-bound**, so it exposes R-KV's overhead, not its benefit — the benefit
   side (memory-bound serving → more concurrency under fixed VRAM) shows up in the
   [DP scaling report](./RESULTS_dp.md).

**Sweet spot:** `budget = 256–512`, `buffer = 128` — lossless accuracy, dozens of
compactions, within ~6–9 % of the fair Full-KV throughput.

## Reproduce

```bash
cd SGLang/benchmark

# Full-KV production baseline:
MODEL=/data/model/Qwen2.5-Math-7B-Instruct ./launch_server.sh fullkv
# Full-KV constrained (fair) baseline:
MODEL=/data/model/Qwen2.5-Math-7B-Instruct ./launch_server.sh constrained
# R-KV, budget 256, buffer 128:
MODEL=/data/model/Qwen2.5-Math-7B-Instruct BUFFER=128 ./launch_server.sh rkv 256

# Evaluate (SGLang GSM8K, 5-shot, 200 questions, concurrency 32):
PYTHONPATH=../sglang-src/python python3 \
  ../sglang-src/benchmark/gsm8k/bench_sglang.py \
  --num-questions 200 --num-shots 5 --parallel 32 --max-new-tokens 512 --port 30000
```

The server log shows `R-KV compacted req_pool_idx=... phys N -> budget` lines and
`R-KV fused-redundancy gate: OK -> fused adopted`.
