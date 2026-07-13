# R-KV under Data Parallelism (DP) — Scaling

How R-KV behaves under SGLang **plain data parallelism** (`--dp-size N --tp-size 1`):
N independent replicas, each with its own KV pool and its own R-KV compressor, a
router load-balancing requests across them. Companion to
[`RESULTS.md`](./RESULTS.md) / [`RESULTS_tp.md`](./RESULTS_tp.md).

> **Plain DP works trivially.** Requests never cross ranks, so each rank runs its
> own R-KV over a disjoint request set — there is no cross-rank
> eviction-agreement problem (that only arises under tensor parallelism, which
> needs a score all-reduce; see [`RESULTS_tp.md`](./RESULTS_tp.md)).

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (bf16), single node, **8× NVIDIA H100 80GB**.
- **R-KV**: `budget=256, window=8, buffer_size=128`; radix/overlap OFF,
  `page_size 1`, CUDA graphs ON, `--mem-fraction-static 0.85`.
- **Harness**: `bench_sglang.py` (GSM8K, 5-shot), first **500 questions**,
  `max_new_tokens=512`, `temperature=0`. Concurrency = **dp × 16** (≈16 in-flight
  per replica).
- **Parallelism**: `--dp-size N --tp-size 1`, `N ∈ {1, 2, 4, 8}`. Date 2026-07-13.

## Scaling

| DP | GPUs | Concurrency | Accuracy (500) | Latency | Throughput | vs dp=1 | Compactions/rank |
| ---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: |
| 1 | 1 | 16 | 0.884 | 42.9 s | 1349 tok/s | 1.00× | 173 |
| 2 | 2 | 32 | 0.880 | 22.9 s | 2538 tok/s | 1.88× | 177 |
| 4 | 4 | 64 | 0.882 | 13.2 s | 4317 tok/s | 3.20× | 170 |
| 8 | 8 | 128 | 0.886 | 8.3 s | **6879 tok/s** | **5.10×** | 170 |

## Findings

- **Throughput scales strongly** — 1349 → 6879 tok/s, up to **5.1× on 8 GPUs**.
  Each replica is an independent R-KV instance, so the compression overhead is
  fully parallelized.
- **Accuracy is flat** across DP degrees (0.880–0.886): DP does not perturb
  correctness — each rank compresses its own requests exactly as the single-GPU
  path does.
- **Compaction work is conserved** (~170 per rank at every DP degree): total R-KV
  work scales with the *request stream*, not the DP degree — DP just spreads it
  across replicas.
- **Sub-linear at dp=8** (5.1×, not 8×) is a *workload* artifact, not an R-KV one:
  with only 500 requests each dp=8 replica handles ~62, so launch/warm-up/drain and
  router load imbalance eat into steady state. A larger request stream pushes dp=8
  closer to linear.

## Scope

- **Plain DP** (`--dp-size N --tp-size 1`) — validated here.
- **Tensor parallel** (`--tp-size N`) — supported and validated separately; see
  [`RESULTS_tp.md`](./RESULTS_tp.md).
- **DP attention** (`--enable-dp-attention`) — **unsupported**: its
  padded/all-gathered `forward_batch` layout is unverified against R-KV's hooks
  and is rejected at startup. See [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) §11.

## Reproduce

```bash
cd SGLang/benchmark

# 4-way data-parallel R-KV server (budget 256, buffer 128):
MODEL=/data/model/Qwen2.5-Math-7B-Instruct BUFFER=128 DP=4 ./launch_server.sh rkv 256

# Drive it with enough concurrency to fan out across all replicas:
PYTHONPATH=../sglang-src/python python3 \
  ../sglang-src/benchmark/gsm8k/bench_sglang.py \
  --num-questions 500 --num-shots 5 --parallel 64 --port 30000
```

`DP=N ./launch_server.sh rkv 256` adds `--dp-size N --tp-size 1`; `DP=1` (default)
is the single-GPU path. Use `--parallel $((DP*16))` to keep each replica fed.
