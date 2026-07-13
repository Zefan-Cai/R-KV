# R-KV under Tensor Parallelism (TP) — Scaling & Correctness

How R-KV behaves under SGLang **tensor parallelism** (`--tp-size N`): one model
sharded across N GPUs, each rank holding a subset of the KV heads. Companion to
[`RESULTS.md`](./RESULTS.md) / [`RESULTS_dp.md`](./RESULTS_dp.md).

> **TP is the subtle case.** Each rank scores only its *local* KV heads, so an
> uncoordinated R-KV would let ranks pick **different** tokens to evict and
> silently corrupt the (replicated) `req_to_token`. R-KV **all-reduces the
> per-token eviction score across the attention-TP group before top-k**, so every
> rank evicts the *identical* tokens. See
> [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) §11.2.

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (28 Q / 4 KV heads, bf16), **8× NVIDIA H100 80GB**.
- **R-KV**: `budget=256, window=8, buffer_size=128`; radix/overlap OFF,
  `page_size 1`, CUDA graphs ON, `--mem-fraction-static 0.85`.
- **Harness**: `bench_sglang.py` (GSM8K, 5-shot), first **500 questions**,
  `max_new_tokens=512`, `temperature=0`, **`--parallel 16` (fixed)** — TP shards
  *one* model, so concurrency is held constant to isolate the sharding effect.
- **Parallelism**: `--tp-size N`, `N ∈ {1, 2, 4}`. Local KV heads/rank: tp2 → 2,
  tp4 → 1 (distinct heads). Date 2026-07-13.

## Scaling

| TP | GPUs | Accuracy (500) | Latency | Throughput | vs tp=1 | Compactions/rank |
| ---: | ---: | :---: | ---: | ---: | :---: | ---: |
| 1 | 1 | 0.884 | 42.9 s | 1349 tok/s | 1.00× | 173 |
| 2 | 2 | 0.890 | 33.9 s | 1699 tok/s | 1.26× | 174 |
| 4 | 4 | 0.882 | 27.5 s | 2106 tok/s | 1.56× | 175 |

*(Compactions are **per rank**; every rank performs the same logical compaction in
lockstep — see Correctness below.)*

## Findings

- **Throughput scales sub-linearly** with TP (1.26× at tp=2, 1.56× at tp=4) at
  fixed concurrency. That is expected: TP shards one model, reducing per-GPU
  compute and KV memory and cutting latency, but — unlike DP — it does **not**
  multiply the number of independent request streams. The gain comes from faster
  per-token compute across the shard, tempered by cross-rank communication.
- **Accuracy is flat** (0.882–0.890) across TP degrees: the cross-rank score
  all-reduce keeps every rank's eviction decision identical, so TP output matches
  the single-GPU path within judge noise.
- **Per-rank compaction count is conserved** (~173–175, == the tp=1 count): TP
  replicates the *same* logical compactions on every rank rather than adding work.

## Correctness — every rank evicts identical tokens

With 2 (or 4) ranks each holding a KV-head shard, the server log shows **all ranks
compacting the same `req_pool_idx` at the same step with the same freed count**:

```
[TP0] R-KV compacted req_pool_idx=192: phys 827 -> 256 slots (freed 571)
[TP1] R-KV compacted req_pool_idx=192: phys 827 -> 256 slots (freed 571)
[TP0] R-KV compacted req_pool_idx=193: phys 834 -> 256 slots (freed 578)
[TP1] R-KV compacted req_pool_idx=193: phys 834 -> 256 slots (freed 578)
```

**Why the all-reduce is required and correct.** Under TP each rank sees only its
KV heads; R-KV's score is a cross-head **mean**, so a rank's *local* score covers
only its shard. Left uncoordinated, ranks would top-k different tokens and
`_compact_request` would `free()` different physical slots per rank — silent KV
corruption (nothing crashes; each rank is internally self-consistent). R-KV
**sums** the per-token score across the attention-TP group before top-k; because
every softmax/pool inside the score is per-head and the cross-head reduction is
**linear**, the all-reduced sum equals the true global score up to a positive
constant, and `top-k` is invariant to positive scaling — so the kept set is
**identical on every rank**. A self-check (`SGLANG_RKV_TP_CHECK=1`) all-reduces the
kept indices and raises `RuntimeError` if any rank disagrees.

## Scope

- **Tensor parallel** (`--tp-size N`) — validated here (`N ∈ {2, 4}`).
- **Plain data parallel** (`--dp-size N`) — validated separately; see
  [`RESULTS_dp.md`](./RESULTS_dp.md).
- **DP attention** (`--enable-dp-attention`) — **unsupported** (rejected at startup).

## Reproduce

```bash
cd SGLang/benchmark

# 4-way tensor-parallel R-KV server (budget 256, buffer 128):
MODEL=/data/model/Qwen2.5-Math-7B-Instruct BUFFER=128 TP=4 ./launch_server.sh rkv 256
# Force the cross-rank kept-set consistency check on EVERY compaction:
#   SGLANG_RKV_TP_CHECK=1 MODEL=... BUFFER=128 TP=4 ./launch_server.sh rkv 256

PYTHONPATH=../sglang-src/python python3 \
  ../sglang-src/benchmark/gsm8k/bench_sglang.py \
  --num-questions 500 --num-shots 5 --parallel 16 --port 30000
```

`TP=N ./launch_server.sh rkv 256` adds `--tp-size N`.
