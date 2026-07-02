# R-KV under Data Parallelism (DP) — Correctness & Throughput

How R-KV behaves under SGLang **plain data parallelism** (`--dp-size N`, `tp=1`):
N independent replicas, each with its own KV pool and its own R-KV compressor,
with a router load-balancing requests across them. Companion to
[`RESULTS.md`](./RESULTS.md) / [`RESULTS_math7b.md`](./RESULTS_math7b.md).

> **Verdict: plain DP works.** Each rank runs its own R-KV over a disjoint set of
> requests — there is no cross-rank eviction-agreement problem (unlike TP, which
> is still unsupported). Accuracy is unchanged vs single-GPU and throughput
> scales strongly with the number of replicas.

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (bf16), single node, **8× NVIDIA H100 80GB**.
- **R-KV**: `budget=512, window_size=8, buffer_size=16`; flags
  `--disable-radix-cache --disable-decode-cuda-graph --disable-prefill-cuda-graph
  --disable-overlap-schedule --page-size 1`, `--mem-fraction-static 0.85`.
- **Parallelism**: `--dp-size N --tp-size 1` (plain DP; R-KV rejects `tp>1`).
- **Dataset**: GSM8K-style few-shot MATH harness, `temperature=0`,
  `max_new_tokens=512`, numeric judging (`eval.py`).
- Date: 2026-07-02, post rotary off-by-one fix.

## 1. Correctness — dp=2, first 20 items

Validated against the five signals we use to distinguish "DP works" from "silent
KV corruption":

| Signal | Result |
| --- | --- |
| Launches, no crash | ✅ both `DP0` and `DP1` initialize + warm up |
| Accuracy matches single-GPU | ✅ serial dp=2 = **19/20 (95%)**, identical to single-GPU |
| Both ranks actually compress | ✅ `R-KV compacted` logged **independently on DP0 and DP1** |
| No leak / crash at idle | ✅ `/health` OK, `token usage: 0.00` after drain, no leak assertion |
| Output coherent (not garbage) | ✅ `avg_tokens ≈ 180 == baseline`; the one miss is a coherent wrong answer, not garbage |

At `concurrency=16` the same dp=2 server scored 18/20 — a single item flipping vs
the serial 19/20, which is ordinary batched-decode numerical noise (it happens on
a single GPU too), not a DP defect. Physical KV length stayed pinned near `budget`
(`#token ≈ 512`), confirming compaction fires on both ranks.

## 2. Throughput scaling — dp ∈ {1, 2, 4, 8}

128 items, `concurrency = dp × 16` (≈16 in-flight per replica), `budget=512`.

| DP | GPUs | Concurrency | Accuracy (128) | avg tok | Wall | Throughput | vs dp=1 | Per-rank |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: | ---: |
| 1 | 1 | 16 | 112/128 (87.5%) | 173 | 57.7s | 385.0 tok/s | 1.00× | 385 |
| 2 | 2 | 32 | 113/128 (88.3%) | 171 | 30.9s | 709.1 tok/s | 1.84× | 355 |
| 4 | 4 | 64 | 114/128 (89.1%) | 172 | 19.6s | 1125.8 tok/s | 2.92× | 281 |
| 8 | 8 | 128 | 115/128 (89.8%) | 175 | 11.2s | **1997.1 tok/s** | **5.19×** | 250 |

(Accuracy ~88% here vs 95% on the first-20 subset simply because n=128 spans
harder items; it is **stable across all DP degrees**, which is the point.)

**Findings:**

- **Throughput scales strongly**: 385 → 709 → 1126 → 1997 tok/s, i.e. up to
  **5.2× on 8 GPUs**. Each added replica is an independent R-KV instance, so the
  compression overhead is fully parallelized.
- **Scaling is sub-linear at high DP** (dp=8 = 5.2×, not 8×). This is a
  *workload* artifact, not an R-KV one: with only 128 requests each dp=8 replica
  handles ~16, so launch/warm-up/drain and router load imbalance (per-rank
  compactions ranged 111–196) eat into the steady-state. A larger request stream
  would push dp=8 closer to linear.
- **Total compaction work is conserved**: every configuration ran ~1300 physical
  compactions in total, just spread across ranks — dp does not change how much
  R-KV work happens, only how it is parallelized.
- **Every rank compresses, no errors/leaks** in any run; `avg_tokens` stayed
  ~171–175 (coherent output, no runaway generations).

## Scope / caveats

- This is **plain data parallelism** (`--dp-size N --tp-size 1`). It works because
  requests never cross ranks and each rank's R-KV is self-contained.
- **DP attention** (`--enable-dp-attention`, which implies `tp>1`) is **not**
  tested and is currently blocked by the `tp>1` startup guard; its
  padded/gathered `forward_batch` layout has not been validated against R-KV's
  hooks. See [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) §11.
- **Tensor parallelism (`tp>1`) remains unsupported** (would silently corrupt the
  KV cache without a cross-rank score all-reduce) and is rejected at startup.

## Reproduce

```bash
cd SGLang/benchmark
./prepare_data.sh

# Launch a 4-way data-parallel R-KV server (budget=512):
MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 DP=4 ./launch_server.sh rkv 512

# Drive it with enough concurrency to fan out across all replicas:
python3 eval.py --n 128 --concurrency 64 --label rkv7b_b512_dp4
```

`DP=N` on `launch_server.sh` adds `--dp-size N --tp-size 1`; `DP=1` (default) is
the single-GPU path.
