# R-KV under Data Parallelism (DP) — Scaling

How R-KV behaves under vLLM **data parallelism**: N independent replicas, each
with its own KV pool and its own R-KV compressor, over a disjoint slice of the
request stream. Companion to [`RESULTS.md`](./RESULTS.md) /
[`RESULTS_tp.md`](./RESULTS_tp.md).

> **Plain DP works trivially.** Requests never cross replicas, so each replica
> runs its own R-KV over a disjoint request set — there is no cross-rank
> eviction-agreement problem (that only arises under tensor parallelism, which
> needs a score all-reduce; see [`RESULTS_tp.md`](./RESULTS_tp.md)).

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (bf16), single node, **8× NVIDIA H100 80GB**.
- **R-KV**: `budget=256, window=8, buffer_size=128`; default PIECEWISE cudagraph,
  `gpu_memory_utilization=0.85`, `block_size=16`.
- **Harness**: [`eval.py`](./eval.py) (few-shot GSM8K, `data/gsm8k_fewshot.jsonl`),
  `max_tokens=512`, `temperature=0`, **offline batched**. To keep each replica
  equally loaded (so this measures *scaling*, not amortization), every replica
  processes a **fixed 125 questions** on its own GPU (via `eval.py --offset`), so
  the total stream grows with DP (125 → 1000). Aggregate throughput = Σ decode
  tokens ÷ wall (replicas run concurrently).
- **Parallelism**: N replicas × `--tp 1`, `N ∈ {1, 2, 4, 8}`.

## Scaling

| DP | GPUs | Total Q | Accuracy | Throughput | vs dp=1 | Compactions/rank |
| ---: | ---: | ---: | :---: | ---: | :---: | ---: |
| 1 | 1 | 125 | 0.896 (112/125) | 3180 tok/s | 1.00× | 112 |
| 2 | 2 | 250 | 0.908 (227/250) | 6463 tok/s | 2.03× | 115 |
| 4 | 4 | 500 | 0.904 (452/500) | 11551 tok/s | 3.63× | 108 |
| 8 | 8 | 1000 | 0.905 (905/1000) | **24110 tok/s** | **7.58×** | 108 |

## Findings

- **Throughput scales near-linearly** — 3180 → 24110 tok/s, **7.58× on 8 GPUs**.
  Each replica is an independent R-KV instance over its own request slice, so the
  compression work is fully parallelized with no cross-replica coordination.
- **Accuracy is flat** across DP degrees (0.896–0.908): DP does not perturb
  correctness — each replica compresses its own requests exactly as the
  single-GPU path does.
- **Per-replica compaction count is constant** (~108–115 at every DP degree,
  since each replica handles the same 125-question load): R-KV work is set by the
  *request stream per replica*, and DP simply adds more independent replicas.
- Because plain DP is embarrassingly parallel for R-KV (no score all-reduce,
  unlike TP), it is the scaling path that most directly multiplies R-KV's
  constant-KV-footprint benefit across GPUs.

## Scope

- **Plain data parallel** (independent replicas, `--tp 1`) — validated here.
- **Tensor parallel** (`--tp N`) — supported and validated separately; see
  [`RESULTS_tp.md`](./RESULTS_tp.md).
- **DP attention** (padded/all-gathered attention layout) — **unsupported**:
  its metadata layout is unverified against R-KV's hooks and is rejected at
  startup. See the fail-closed matrix in [`../docs/OPTIMIZATIONS.md`](../docs/OPTIMIZATIONS.md).

## Reproduce

```bash
cd vLLM/benchmark   # uses the .venv-rkv interpreter that has vllm installed

# 8 independent R-KV replicas (budget 256, buffer 128), 125 questions each,
# one GPU per replica (offset shards the stream):
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
    python eval.py --n 125 --offset $((i*125)) --label dp8_r$i --out /tmp/dp8_r$i.json &
done
wait   # aggregate throughput = sum(out_tokens) / max(wall_s) across the 8 JSONs

# Or a single N-way data-parallel server with the identical knobs:
DP=8 BUFFER=128 ./launch_server.sh rkv 256
```
