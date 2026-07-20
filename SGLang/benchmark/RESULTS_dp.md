# R-KV under Data Parallelism (DP) — Scaling

How R-KV behaves under SGLang **plain data parallelism**: N independent replicas,
each with its own KV pool and its own R-KV compressor, over a disjoint slice of
the request stream. Companion to [`RESULTS.md`](./RESULTS.md) /
[`RESULTS_tp.md`](./RESULTS_tp.md).

> **Plain DP works trivially.** Requests never cross replicas, so each replica
> runs its own R-KV over a disjoint request set — there is no cross-rank
> eviction-agreement problem (that only arises under tensor parallelism, which
> needs a score all-reduce; see [`RESULTS_tp.md`](./RESULTS_tp.md)).

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (bf16), single node, **8× NVIDIA H100 80GB**.
- **R-KV**: `budget=256, window=8, buffer_size=128`; radix/overlap OFF,
  `page_size 1`, CUDA graphs ON, `--mem-fraction-static 0.85`.
- **Harness**: [`eval.py`](./eval.py) (few-shot GSM8K, `data/gsm8k_fewshot.jsonl`),
  `max_new_tokens=512`, `temperature=0`. To keep each replica equally loaded (so
  this measures *scaling*, not amortization), every replica pins **one GPU** and
  processes a **fixed 125 questions** of its own (via `eval.py --offset`), all 125
  in flight (`--concurrency 125`), so the total stream grows with DP (125 →
  1000). Aggregate throughput = Σ decode tokens ÷ max wall (replicas run
  concurrently). This mirrors the vLLM DP report
  ([`../../vLLM/benchmark/RESULTS_dp.md`](../../vLLM/benchmark/RESULTS_dp.md)) for
  an apples-to-apples comparison.
- **Parallelism**: N independent replicas × `--tp-size 1`, `N ∈ {1, 2, 4, 8}`.
  Date 2026-07-20.

## Scaling

| DP | GPUs | Total Q | Accuracy | Throughput | vs dp=1 | Compactions/rank |
| ---: | ---: | ---: | :---: | ---: | :---: | ---: |
| 1 | 1 | 125 | 0.904 (113/125) | 3159 tok/s | 1.00× | 100 |
| 2 | 2 | 250 | 0.900 (225/250) | 6528 tok/s | 2.07× | ~110 |
| 4 | 4 | 500 | 0.908 (454/500) | 12328 tok/s | 3.90× | ~100 |
| 8 | 8 | 1000 | 0.916 (916/1000) | **24544 tok/s** | **7.77×** | ~100 |

## Findings

- **Throughput scales near-linearly** — 3159 → 24544 tok/s, **7.77× on 8 GPUs**.
  Each replica is an independent R-KV instance over its own request slice, so the
  compression work is fully parallelized with no cross-replica coordination.
- **Accuracy is flat** across DP degrees (0.900–0.916): DP does not perturb
  correctness — each replica compresses its own requests exactly as the
  single-GPU path does.
- **Per-replica compaction count is constant** (~100 per rank at every DP degree,
  since each replica handles the same 125-question load; the small per-rank
  spread — 86–119 — just reflects generation-length variance across slices):
  R-KV work is set by the *request stream per replica*, and DP simply adds more
  independent replicas.
- **Fixed per-replica load is what makes this near-linear.** An earlier version of
  this report split a *fixed 500-question total* across the replicas, so at dp=8
  each replica saw only ~62 questions and launch/warm-up/drain dominated its short
  steady state — understating dp=8 as 5.1×. Loading each replica with a full
  125-question slice (total grows with DP) removes that amortization artifact and
  exposes R-KV's true DP scaling.
- Because plain DP is embarrassingly parallel for R-KV (no score all-reduce,
  unlike TP), it is the scaling path that most directly multiplies R-KV's
  constant-KV-footprint benefit across GPUs.

## Scope

- **Plain DP** (independent replicas, `--tp-size 1`) — validated here.
- **Tensor parallel** (`--tp-size N`) — supported and validated separately; see
  [`RESULTS_tp.md`](./RESULTS_tp.md).
- **DP attention** (`--enable-dp-attention`) — **unsupported**: its
  padded/all-gathered `forward_batch` layout is unverified against R-KV's hooks
  and is rejected at startup. See [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) §11.

## Reproduce

```bash
cd SGLang/benchmark

# 8 independent R-KV replicas (budget 256, buffer 128), 125 questions each,
# one GPU per replica (offset shards the stream); each replica is its own server:
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i PORT=$((30000+i)) BUFFER=128 \
    MODEL=/data/model/Qwen2.5-Math-7B-Instruct ./launch_server.sh rkv 256 \
    >/tmp/sgl_dp8_srv$i.log 2>&1 &
done
# once all 8 answer /health_generate, drive each with its own 125-question slice:
for i in $(seq 0 7); do
  PYTHONPATH=../sglang-src/python python3 eval.py \
    --n 125 --offset $((i*125)) --port $((30000+i)) --concurrency 125 \
    --label dp8_r$i --out /tmp/sgl_dp8_r$i.json &
done
wait   # aggregate throughput = sum(out_tokens) / max(wall_s) across the 8 JSONs

# Or a single N-way data-parallel server (SGLang router) with identical knobs:
DP=8 BUFFER=128 ./launch_server.sh rkv 256
```

`DP=N ./launch_server.sh rkv 256` adds `--dp-size N --tp-size 1` (one server, a
router fanning out across N replicas); the per-GPU independent-server loop above
is the equivalent that pins each replica to a dedicated GPU for a clean
per-replica load.
