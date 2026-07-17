# R-KV under Memory Pressure — Single-GPU Concurrency (SGLang)

Companion to the vLLM port's
[`vLLM/benchmark/RESULTS_stress.md`](../../vLLM/benchmark/RESULTS_stress.md).
Same question: on a **single GPU with a tight KV pool**, does R-KV's bounded
footprint (`budget + buffer` tokens/request) let **more requests run
concurrently**, and does that raise throughput? The answer here is **nuanced**:
the concurrency benefit is real, but this reference SGLang R-KV does **not** turn
it into a throughput win (the vLLM port does — see the contrast at the bottom).

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (bf16), single **NVIDIA H100 80GB**.
- **Workload**: [`eval.py`](./eval.py) with `--ignore-eos --max-tokens 1024`, so
  every request generates a **fixed 1024 tokens** on the ~700-token few-shot
  prompt (~1724 tokens/request). Requests sent at high concurrency.
- **Knob**: `--mem-fraction-static` shrinks the KV pool.
- **R-KV**: `budget=256, buffer=128, window=8` (radix/overlap off, page_size 1 —
  R-KV's required flags). **Full-KV** = the same *constrained* flags, no
  compression (fair A/B).
- **Metric**: **peak `#running-req`** the server sustained (achieved concurrency)
  from the server log, plus end-to-end throughput from `eval.py`, plus the KV
  pool (`max_total_num_tokens`) the server allocated.

## The dominant effect — SGLang R-KV's static memory overhead

At the **same** `--mem-fraction-static`, the R-KV server allocates a KV pool that
is a **fixed ~126K tokens (~7 GB) smaller** than Full-KV's — its per-layer
observation-query rings and redundancy-score workspace are static allocations:

| `mem-frac` | Full-KV pool | Full-KV peak conc. | R-KV pool | R-KV peak conc. |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 466K | 256 | 340K | 256 |
| 0.40 | 319K | 185 | 193K | **232** |
| 0.30 | 172K | 100 | 45K | **54** |
| 0.25 | 99K | 57 | *fails to start* | — |

(N=256, concurrency 256.) Because the ~7 GB overhead is **fixed**, R-KV's pool
shrinks *faster* than Full-KV's as memory tightens: at `0.40` R-KV still fits
**more** (232 vs 185), but by `0.30` its pool has collapsed to 45K and it fits
**fewer** (54 vs 100), and at `0.25` it can't allocate a single running slot
(`AssertionError: max_running_request is zero`). But most of this overhead is the
rolling-query buffer sized for 4096 requests — the next section shrinks it to the
pool.

## Shrinking the overhead — size the rolling buffer to the pool (auto-cap)

Most of that "~7 GB fixed" cost is **not inherently fixed**. It is the per-layer
**rolling observation-query buffer** (`rolling_q`), sized

    rolling_q = layers × window × (max_running_requests + 1) × q_heads × head_dim

i.e. **one row per possibly-concurrent request**. SGLang's generic estimate sets
`max_running_requests = 4096`, so for Qwen2.5-Math-7B (28 layers, window 8, 28
heads, dim 128, bf16) `rolling_q` = **6.13 GiB** — yet at `mem-frac 0.50` the pool
only holds ~411 concurrent requests, so **~90 % of those rows can never be used**.

R-KV holds each request's KV at a bounded ceiling of `budget + buffer` tokens, so
the pool serves at most `ceil(pool / (budget + buffer))` concurrent requests.
Capping `max_running_requests` to that ceiling ties `rolling_q` to the KV cache.
This is now the **default** when neither `--rkv-max-active-requests` nor
`--max-running-requests` is set (and is still tunable):

| `mem-frac 0.50` config | `max_running` | `rolling_q` | R-KV KV pool | vs Full-KV (466K) |
| --- | ---: | ---: | ---: | ---: |
| default (before) | 4096 | 6.13 GiB | 340K | −27 % |
| **auto-cap (now default)** | 1096 | 1.82 GiB | **421K** | **−9.8 %** |
| `--rkv-max-active-requests 512` | 512 | 0.77 GiB | **440K** | **−5.5 %** |

The auto-cap (1096) stays **well above** the ~500 requests the pool can actually
admit, so no achievable concurrency is lost — the reclaimed ~80–100K tokens go
straight back to the KV pool, nearly closing the gap to Full-KV. Lower
`--rkv-max-active-requests` reclaims even more when you are memory-bound. (Numbers
are from the server startup log's `R-KV decode: reserving …` line.)

## Isolating the concurrency benefit — healthy pool, heavier load

To test the *per-request footprint* effect without the static-overhead confound,
give both a healthy pool (`mem-frac 0.50`) and raise the offered load. R-KV's
concurrency advantage holds, and its throughput **improves from a loss to a wash**
as the load saturates:

| Load (N = concurrency) | Full-KV conc. / tok/s | R-KV conc. / tok/s | R-KV vs Full-KV |
| --- | ---: | ---: | ---: |
| 512  | 270 / 12550 | 410 / 11654 | −7 % |
| 1319 | 271 / 12442 | **411** / 12597 | **+1 %** (break-even) |

At the higher load R-KV sustains **+52 % concurrency** (411 vs 271) and **matches
Full-KV throughput** (the −7 % at N=512 was partly the shorter run's ramp-down;
saturated, it evens out). So on SGLang, R-KV's bounded footprint buys **more
concurrency at no throughput cost** — but still not the throughput *gain* the
vLLM port shows.

**Why it doesn't go further (a scheduler limit, not a client one).** R-KV's peak
plateaus at **~411** at *both* N=512 and N=1319, even though its 340K pool could
hold ~885 compacted requests (384 tokens each). SGLang admits each request
reserving its *pre-compaction* footprint (~828 tokens: prompt + buffer), fills the
pool at 411, and then **does not re-admit new prefills into the headroom that R-KV
frees on compaction** — so 411×384 = 158K of the 340K pool sits used while ~180K
is idle. Raising client concurrency to 2048 would only deepen the queue, not the
running batch (`max_running_requests` was already 4096, not the binding limit).

## Findings

1. **The concurrency benefit is real** — R-KV's `budget+buffer` cap lets the
   server keep 411 vs 271 requests resident on a smaller pool (and 232 vs 185 at
   `mem-frac 0.40`). The bounded-footprint mechanism holds on SGLang too.
2. **…but at best it breaks even on throughput — no win.** Two SGLang-specific
   costs eat the concurrency advantage: (a) the **~7 GB static R-KV buffers**
   shrink the KV pool (so the benefit only shows where the pool is still healthy,
   and it *fails* under tight memory), and (b) each compaction forces the
   surrounding decode steps **out of the CUDA graph into eager mode**, a
   per-compaction cost that grows with the ~1800 compactions this workload
   triggers. Under saturating load R-KV *ties* Full-KV (+1 %) while running 52 %
   more requests, but never pulls ahead.
3. **The scheduler leaves R-KV's headroom on the table.** Because SGLang reserves
   the pre-compaction footprint at admission and does not re-admit into the space
   R-KV frees, R-KV plateaus at ~411 concurrent instead of the ~885 its pool could
   hold — so the biggest lever (running far more short-KV requests) never engages.
   bottleneck Full-KV, R-KV's static overhead has already crippled its own pool.

## Contrast with the vLLM port

The vLLM port **does** convert the concurrency benefit into throughput — at
`gpu_mem 0.30` it sustains **256 vs 100** concurrent and **+32 %** throughput
(see [`vLLM/benchmark/RESULTS_stress.md`](../../vLLM/benchmark/RESULTS_stress.md)).
Two implementation differences explain the gap, and motivate the port:

- **Lean footprint.** vLLM R-KV uses a small fixed-address query ring and
  memory-guarded *tiled* scoring (no pre-materialized `L×L` matrix), so its pool
  is only **~7 % smaller** than Full-KV's (124K vs 134K at `gpu_mem 0.30`), vs
  SGLang's ~7 GB fixed overhead. It never fails under tight memory.
- **No extra eager cost.** vLLM auto-selects **PIECEWISE cudagraph**, which keeps
  attention eager on *every* step anyway, so compaction adds no CUDA-graph
  break — SGLang graphs attention and pays an eager window per compaction.

So the memory-bound throughput win R-KV promises is realized by the vLLM port;
the reference SGLang R-KV delivers the concurrency but not the throughput in this
regime.

## Reproduce

```bash
# Prereq: scripts/apply_rkv.sh, then activate the sglang env (see ../README.md).
cd SGLang/benchmark

# Full-KV constrained server + R-KV server, on two GPUs:
CUDA_VISIBLE_DEVICES=0 MEM_FRAC=0.5 PORT=30010 ./launch_server.sh constrained 256 &
CUDA_VISIBLE_DEVICES=1 MEM_FRAC=0.5 BUFFER=128 PORT=30011 ./launch_server.sh rkv 256 &
# wait for both /health, then drive a saturating load of fixed-length (1024-token)
# requests (N is capped by the 1319-prompt dataset -> use it as the concurrency):
python eval.py --port 30010 --n 1319 --concurrency 1319 --max-tokens 1024 --ignore-eos &
python eval.py --port 30011 --n 1319 --concurrency 1319 --max-tokens 1024 --ignore-eos &
wait
```

Peak concurrency is the max `#running-req: N` in each server log; `eval.py` prints
end-to-end throughput; the startup log prints `max_total_num_tokens` (the pool).
