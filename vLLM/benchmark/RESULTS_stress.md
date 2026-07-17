# R-KV under Memory Pressure — Single-GPU Concurrency Scaling

The [main sweep](./RESULTS.md) runs at `mem_frac 0.85` where the KV pool holds
every request easily, so it only exposes R-KV's *overhead*. This report shows the
**benefit** side: on a **single GPU with a shrinking KV pool**, R-KV's bounded
footprint (`budget + buffer` tokens/request) lets **more requests run
concurrently** than Full-KV, so throughput **rises above** Full-KV exactly when
memory is the bottleneck. Companion to [`RESULTS.md`](./RESULTS.md),
[`RESULTS_tp.md`](./RESULTS_tp.md), [`RESULTS_dp.md`](./RESULTS_dp.md).

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (bf16), single **NVIDIA H100 80GB**.
- **Workload**: [`eval.py`](./eval.py) with `--ignore-eos --max-tokens 1024`, so
  every request generates a **fixed 1024 tokens** on top of the ~700-token
  few-shot prompt (~1724 tokens/request) — a constant, KV-heavy load. **N=256**
  requests, all submitted at once (offline batched), `max_model_len=2048`,
  `block_size=16`.
- **Knob**: `gpu_memory_utilization ∈ {0.50, 0.40, 0.30, 0.25}` shrinks the KV
  pool from 430K down to 60K tokens.
- **R-KV**: `budget=256, buffer=128, window=8`, **`FREE_BLOCKS=1`** (returns
  evicted blocks to the allocator — *this is what makes the footprint shrink*)
  and `ASYNC=1` (best-throughput scheduling, matching Full-KV's async default).
- **Full-KV**: constrained (prefix caching off), the fair A/B baseline.
- **Metric**: **peak concurrency** = the maximum `Running: N reqs` the engine
  sustained (achieved in-flight requests), plus offline decode throughput.

## Result — R-KV holds concurrency as the pool shrinks

| `gpu_mem` | KV pool (tokens) | **Full-KV** peak conc. | Full-KV tok/s | **R-KV** peak conc. | R-KV tok/s | R-KV vs Full-KV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 430K | 256 | 13902 | 256 | 12177 | **−12 %** |
| 0.40 | 282K | 231 | 10503 | 256 | 12282 | **+17 %** |
| 0.30 | 134K | **100** | 7838 | **256** | 10371 | **+32 %** |
| 0.25 | 60K | **60** | 4958 | **124** | 6566 | **+32 %** |

*(R-KV performs ~1790 physical compactions per run at every setting — the same
256×1024-token workload — so the pool stays bounded at `budget+buffer` regardless
of pressure.)*

## Findings

1. **The crossover is the whole point.** With an **ample** pool (`0.50`, both fit
   all 256 in flight) R-KV is **−12 %** — pure compaction overhead, no benefit,
   exactly as [`RESULTS.md`](./RESULTS.md) shows at `0.85`. As the pool shrinks,
   Full-KV's concurrency **collapses** (256 → 231 → **100** → **60**) because each
   request pins its full ~1724-token KV, while R-KV **holds 256** (then 124)
   because each request is capped at `budget+buffer = 384` tokens after its first
   compaction. R-KV **flips to +17 % … +32 %** throughput.
2. **Concurrency is the mechanism.** At `0.30`, Full-KV can keep only **100**
   requests resident (the rest queue), whereas R-KV runs **all 256** — a **2.6×**
   concurrency advantage that directly buys the +32 % throughput. At `0.25` both
   are memory-bound but R-KV still fits **2.1×** more (124 vs 60).
3. **Same VRAM, more work in flight.** R-KV and Full-KV get the *same* KV pool at
   each `gpu_mem`; R-KV simply spends it on ~4.5× more (shorter) sequences instead
   of a few full-length ones. This is the memory-bound serving win the DP report's
   note alludes to, isolated on a single GPU. **Notably, the vLLM port *realizes*
   this win where the reference SGLang R-KV does not** — vLLM R-KV's pool is only
   ~7 % smaller than Full-KV's (124K vs 134K at `0.30`), whereas SGLang R-KV
   carries a ~7 GB static buffer that shrinks its pool 2–4× and even fails under
   tight memory, so it delivers the concurrency but not the throughput. See
   [`SGLang/benchmark/RESULTS_stress.md`](../../SGLang/benchmark/RESULTS_stress.md).
4. **When R-KV helps.** Below the crossover (here ~`gpu_mem 0.4`, i.e. when the
   working set no longer fits) R-KV is a net throughput win; above it, prefer
   Full-KV (or a larger `budget`). The tighter the memory, the larger R-KV's edge.

## Reproduce

```bash
# Prereq: build + install the patched vLLM once (see ../README.md):
#   scripts/apply_rkv.sh && pip install -e vllm-src
# then, in that Python env (set RKV_MODEL to a local path to skip the HF download):
cd vLLM/benchmark

# Full-KV under a tight KV pool, 256 fixed-length (1024-token) requests:
python eval.py --n 256 --no-prefix --ignore-eos --max-tokens 1024 --stats \
  --mem-frac 0.30 --max-model-len 2048 --label stress_fullkv

# R-KV (budget 256, buffer 128) — FREE_BLOCKS returns evicted blocks to the pool:
VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
VLLM_V1_R_KV_FREE_BLOCKS=1 VLLM_V1_R_KV_ASYNC=1 \
  python eval.py --n 256 --ignore-eos --max-tokens 1024 --stats \
    --mem-frac 0.30 --max-model-len 2048 --label stress_rkv
```

`--stats` prints the engine's periodic `Running: N reqs` (peak = achieved
concurrency); the startup log prints `GPU KV cache size: N tokens`. `FREE_BLOCKS`
is opt-in / experimental (see [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md)
§6.5) but is what bounds R-KV's allocator footprint — required for this benefit.
