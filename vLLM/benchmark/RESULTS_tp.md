# R-KV under Tensor Parallelism (TP) — Scaling & Correctness

How R-KV behaves under vLLM **tensor parallelism** (`--tensor-parallel-size N`):
one model sharded across N GPUs, each rank holding a subset of the KV heads.
Companion to [`RESULTS.md`](./RESULTS.md) / [`RESULTS_dp.md`](./RESULTS_dp.md).

> **TP is the subtle case.** Each rank scores only its *local* KV heads, so an
> uncoordinated R-KV would let ranks pick **different** tokens to evict and
> silently corrupt the (replicated) block table. R-KV **all-reduces the per-token
> eviction score across the tensor-parallel group before top-k**, so every rank
> evicts the *identical* tokens, guarded by a min/max readiness fingerprint that
> makes divergent ranks fail together instead of deadlocking. See
> [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) and
> [`../docs/OPTIMIZATIONS.md`](../docs/OPTIMIZATIONS.md).

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (28 Q / 4 KV heads, bf16), **8× NVIDIA
  H100 80GB** (this run uses up to 4).
- **R-KV**: `budget=256, window=8, buffer_size=128`; default PIECEWISE cudagraph,
  `gpu_memory_utilization=0.85`, `block_size=16`.
- **Harness**: [`eval.py`](./eval.py) (few-shot GSM8K, `data/gsm8k_fewshot.jsonl`),
  first **500 questions**, `max_tokens=512`, `temperature=0`, **offline batched**
  (all 500 prompts in flight). TP shards *one* model, so the request stream is
  held constant to isolate the sharding effect.
- **Parallelism**: `--tp N`, `N ∈ {1, 2, 4}`. Local KV heads/rank: tp2 → 2,
  tp4 → 1 (distinct heads).

## Scaling

| TP | GPUs | Accuracy (500) | Throughput | vs tp=1 | Compactions/rank |
| ---: | ---: | :---: | ---: | :---: | ---: |
| 1 | 1 | 0.902 (451/500) | 5387 tok/s | 1.00× | 438 |
| 2 | 2 | 0.906 (453/500) | 7844 tok/s | 1.46× | 442 |
| 4 | 4 | 0.906 (453/500) | 9769 tok/s | 1.81× | 450 |

*(Compactions are **per rank**; every rank performs the same logical compaction in
lockstep — see Correctness below. The reported count is the max across ranks.)*

## Findings

- **Throughput scales sub-linearly** with TP (1.46× at tp=2, 1.81× at tp=4) on a
  fixed request stream. That is expected: TP shards one model, reducing per-GPU
  compute and KV memory and cutting latency, but — unlike DP — it does **not**
  multiply the number of independent request streams. The gain comes from faster
  per-token compute across the shard, tempered by cross-rank communication (and,
  for R-KV, the per-compaction score all-reduce).
- **Accuracy is flat** (0.902–0.906) across TP degrees: the cross-rank score
  all-reduce keeps every rank's eviction decision identical, so TP output matches
  the single-GPU path within judge noise.
- **Per-rank compaction count is conserved** (~438–450, ≈ the tp=1 count): TP
  replicates the *same* logical compactions on every rank rather than adding work.

## Correctness — every rank evicts identical tokens

With 2 (or 4) ranks each holding a KV-head shard, R-KV's score is a cross-head
**mean**, so a rank's *local* score covers only its shard. Left uncoordinated,
ranks would top-k different tokens and free different physical slots per rank —
silent KV corruption (nothing crashes; each rank is internally self-consistent).
R-KV **sums** the per-token score across the TP group before top-k; because every
softmax/pool inside the score is per-head and the cross-head reduction is
**linear**, the all-reduced sum equals the true global score up to a positive
constant, and `top-k` is invariant to positive scaling — so the kept set is
**identical on every rank**. A readiness handshake all-reduces a
collision-resistant fingerprint of the compaction plan (min **and** max, folding
each row's request-id) and raises on any disagreement rather than letting some
ranks enter the score all-reduce while others skip (which would hang NCCL).

## Scope

- **Tensor parallel** (`--tp N`) — validated here (`N ∈ {2, 4}`).
- **Data parallel** (`--data-parallel-size N`) — validated separately; see
  [`RESULTS_dp.md`](./RESULTS_dp.md).
- **Pipeline parallel / DCP / DP-attention** — **unsupported** (rejected at
  startup; see the fail-closed matrix in the docs).

## Reproduce

```bash
cd vLLM/benchmark   # uses the .venv-rkv interpreter that has vllm installed

# 4-way tensor-parallel R-KV (budget 256, buffer 128), 500 questions:
VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
  python eval.py --n 500 --tp 4 --label rkv_tp4

# Or a real TP server with the identical knobs:
TP=4 BUFFER=128 ./launch_server.sh rkv 256
```
