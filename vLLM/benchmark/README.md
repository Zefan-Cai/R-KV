# Benchmarking R-KV on vLLM

> **Validated on 8× H100** (Qwen2.5-Math-7B, GSM8K). Measured accuracy and
> throughput across the full `budget × buffer` grid are in
> [`RESULTS_H100.md`](RESULTS_H100.md); the debugging story behind the numbers is
> in [`../docs/RESIDUAL_GAP_INVESTIGATION.md`](../docs/RESIDUAL_GAP_INVESTIGATION.md).

## Quick sweep (this folder)

`bench_sweep.py` runs one `(budget, buffer)` config offline and reports accuracy +
throughput; `run_sweep.sh` fans the whole grid out one-config-per-GPU across 8
GPUs (two waves, ~2m40s total):

```bash
bash run_sweep.sh                 # 13 configs (Full-KV + budget{128,256,512}×buffer{16,64,128,256})
SWEEP_N=100 bash run_sweep.sh     # faster, 100 questions

# single config on one GPU:
CUDA_VISIBLE_DEVICES=0 VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=64 \
  RKV_N=200 RKV_MAXTOK=512 RKV_OUT=/tmp/r.json python bench_sweep.py
```

R-KV is a drop-in serving change gated by environment variables, so any of
vLLM's standard benchmarks also work — just launch the server twice (Full-KV vs
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

| Axis | Values swept in [`RESULTS_H100.md`](RESULTS_H100.md) |
| --- | --- |
| `VLLM_V1_R_KV_BUDGET` | 128, 256, 512 |
| `VLLM_V1_R_KV_BUFFER` | 16, 64, 128, 256 |
| concurrency | offline batched (all 200 prompts in flight) |
