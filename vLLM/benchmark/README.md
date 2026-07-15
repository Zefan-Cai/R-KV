# Benchmarking R-KV on vLLM

> **Note:** this port is pending GPU validation (see
> [`../docs/OPTIMIZATIONS.md`](../docs/OPTIMIZATIONS.md)). The commands below are
> the intended A/B harness; run them once the serving path is validated on a
> GPU.

R-KV is a drop-in serving change gated by environment variables, so any of
vLLM's standard benchmarks work — just launch the server twice (Full-KV vs
R-KV) and compare.

## Fair A/B setup

Launch **Full-KV** (baseline) and **R-KV** with identical flags except the R-KV
env vars, and use `--enforce-eager` for both so the comparison is apples-to-apples:

```bash
# Full-KV baseline
vllm serve Qwen/Qwen2.5-Math-7B-Instruct --enforce-eager --port 8000

# R-KV (budget=512, compress every 64 tokens)
VLLM_V1_R_KV_BUDGET=512 VLLM_V1_R_KV_BUFFER=64 \
  vllm serve Qwen/Qwen2.5-Math-7B-Instruct --enforce-eager --port 8001
```

## Accuracy (GSM8K / MATH)

Use vLLM's `benchmarks/` accuracy harness or `lm-eval-harness` against each
endpoint. Expect R-KV at `budget=512` to be **near-lossless** vs Full-KV, and
accuracy to degrade only at small budgets — mirroring the SGLang results in
[`../../SGLang/benchmark/RESULTS.md`](../../SGLang/benchmark/RESULTS.md).

## Throughput / memory

Use `vllm bench serve` (or `benchmarks/benchmark_serving.py`) against each
endpoint. R-KV keeps a **constant** per-request KV footprint (`budget`), so its
advantage grows in the **memory-bound** regime (long outputs, many concurrent
requests); in a non-memory-bound setting it is pure overhead. See the SGLang
port's findings for the expected shape of these curves.

## Sweeps to run

| Axis | Suggested values |
| --- | --- |
| `VLLM_V1_R_KV_BUDGET` | 128, 256, 512, 1024 |
| `VLLM_V1_R_KV_BUFFER` | 32, 64, 128 |
| concurrency | 1, 8, 32 |
