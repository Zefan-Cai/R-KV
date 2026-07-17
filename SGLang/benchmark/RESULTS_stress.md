# R-KV under Memory Pressure — Single-GPU Concurrency (SGLang)

Companion to the vLLM port's
[`vLLM/benchmark/RESULTS_stress.md`](../../vLLM/benchmark/RESULTS_stress.md).
Same question: on a **single GPU with a tight KV pool**, does R-KV's bounded
footprint (`budget + buffer` tokens/request) let **more requests run
concurrently**, and does that raise throughput? **Yes on both** — once the
rolling observation-query buffer is sized to the pool (the **auto-cap**, now the
default; see below). Under memory pressure R-KV sustains **+38–78 % concurrency**
and **+19–33 % throughput** over constrained Full-KV, and it now starts at memory
fractions where it previously could not allocate a single slot.

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

## R-KV under memory pressure — the concurrency-into-throughput win

As `--mem-fraction-static` shrinks the KV pool, Full-KV's ~1724-token/request
footprint forces its running batch down fast, while R-KV's **bounded** footprint
(prompt + `buffer`, compacted to `budget + buffer` = 384 tokens) keeps far more
requests resident. At `N=256` (concurrency 256):

| `mem-frac` | Full-KV pool / conc / tok/s | R-KV pool / conc / tok/s | R-KV vs Full-KV |
| ---: | ---: | ---: | ---: |
| 0.50 | 466K / 256 / 13005 | 421K / 256 / 12335 | −5 % (both cap at offered 256) |
| 0.40 | 319K / 185 / 9298 | 284K / **256** / 12286 | **+32 %** |
| 0.30 | 172K / 100 / 7487 | 148K / **178** / 8906 | **+19 %** |
| 0.25 | 99K / 57 / 5346 | 80K / **96** / 7095 | **+33 %** |

At `mem-frac 0.50` both fit the offered 256 requests, so only R-KV's residual
overhead shows (−5 %). As memory tightens, Full-KV's concurrency collapses
(185 → 100 → 57) while R-KV's holds up (256 → 178 → 96), and that concurrency gap
converts **directly** into throughput: **+19 % to +33 %**. R-KV's peak is
pool-limited at `pool / (prompt + buffer)` ≈ `pool / 828` (178 ≈ 148K/828,
96 ≈ 80K/828); Full-KV's at `pool / ~1724`.

This is the regime KV compression is *for*, and it is exactly where the reference
implementation used to **lose** — until the rolling buffer was sized to the pool.

## Sizing the rolling buffer to the pool (the auto-cap)

The win above is **not** what this reference implementation did by default. Most
of R-KV's static overhead is the per-layer **rolling observation-query buffer**
(`rolling_q`), sized

    rolling_q = layers × window × (max_running_requests + 1) × q_heads × head_dim

i.e. **one row per possibly-concurrent request**. SGLang's generic estimate sets
`max_running_requests = 4096`, so for Qwen2.5-Math-7B (28 layers, window 8, 28
heads, dim 128, bf16) `rolling_q` = **6.13 GiB** — yet at `mem-frac 0.50` the pool
only serves ~411 concurrent requests, so **~90 % of those rows can never be used**.
That fixed 6.13 GiB is carved out of the *same* GPU budget as the KV pool, so it
shrank the pool hardest exactly when memory was tight: R-KV *lost* at `mem-frac
0.30` (45K pool, 54 vs 100 concurrent) and could not allocate a single slot at
`0.25` (`AssertionError: max_running_request is zero`).

R-KV holds each request's KV at a bounded ceiling of `budget + buffer` tokens, so
the pool serves at most `ceil(pool / (budget + buffer))` concurrent requests.
Capping `max_running_requests` to that ceiling ties `rolling_q` to the KV cache.
This is now the **default** when neither `--rkv-max-active-requests` nor
`--max-running-requests` is set (and is still tunable):

| `mem-frac 0.50` config | `max_running` | `rolling_q` | R-KV KV pool | vs Full-KV (466K) |
| --- | ---: | ---: | ---: | ---: |
| pre-auto-cap default | 4096 | 6.13 GiB | 340K | −27 % |
| **auto-cap (now default)** | 1096 | 1.82 GiB | **421K** | **−9.8 %** |
| `--rkv-max-active-requests 512` | 512 | 0.77 GiB | **440K** | **−5.5 %** |

The auto-cap (1096) stays **well above** the ~500 requests the `mem-frac 0.50` pool
can actually admit, so no achievable concurrency is lost — the reclaimed ~80K
tokens go straight back to the KV pool. The same mechanism scales the buffer down
as memory tightens (1.82 → 1.25 → 0.67 → 0.39 GiB across the four fractions above),
which is what lets R-KV **start and win** at `0.30` / `0.25` where it used to
collapse. Lower `--rkv-max-active-requests` reclaims even more when you are
memory-bound. (Numbers are from the server startup log's `R-KV decode:
reserving …` line.)

## At an ample pool — the benefit needs saturating load

With memory *not* tight (`mem-frac 0.50`, so both fit the offered load until it
saturates), R-KV's residual overhead has to be amortized before its concurrency
edge pays off. Raising the offered load at this fraction:

| Load (N = concurrency) | Full-KV conc. / tok/s | R-KV conc. / tok/s | R-KV vs Full-KV |
| --- | ---: | ---: | ---: |
| 512  | 270 / 12533 | 507 / 11387 | −9 % |
| 1319 | 271 / 12431 | **508** / 13107 | **+5 %** |

At `N=512` the pool is not yet the binding constraint, so R-KV runs 507 vs 270 but
its per-step compaction cost leaves it −9 %. At the saturating `N=1319` the
concurrency edge (**+87 %**, 508 vs 271) converts to **+5 %** throughput. So even
where memory is ample, R-KV pulls ahead once the load fully saturates the pool —
and the auto-cap raised that plateau from 411 (pre-auto-cap) to **508** by growing
the pool 340K → 421K.

**The plateau is a scheduler choice, not a client limit.** R-KV's peak is
`pool / (prompt + buffer)` ≈ `421K / 828` = 508, even though the pool could hold
~1096 *compacted* requests (384 tokens each). SGLang admits each request reserving
its *pre-compaction* footprint (prompt + buffer ≈ 828 tokens) and **does not
re-admit new prefills into the headroom R-KV frees on compaction**, so 508×384 =
195K of the 421K pool is used while ~226K sits idle. Closing that gap (re-admitting
into freed space) is the remaining lever on SGLang.

## Findings

1. **The concurrency benefit is real and large.** R-KV's `budget + buffer` cap
   keeps far more requests resident on a smaller pool — **+38–78 %** under memory
   pressure (256 vs 185 at `0.40`, 178 vs 100 at `0.30`, 96 vs 57 at `0.25`) and
   **+87 %** (508 vs 271) at a saturated ample pool.
2. **Under memory pressure it converts to a throughput win.** Where the KV pool is
   the binding constraint — the regime compression targets — R-KV delivers
   **+19 % to +33 %** end-to-end throughput. This is new: it depends entirely on
   the auto-cap. With the old fixed 6.13 GiB rolling buffer, R-KV *lost* at
   `mem-frac 0.30` and failed to start at `0.25`.
3. **At an ample pool the win is smaller and load-dependent.** At `mem-frac 0.50`
   R-KV needs the load to fully saturate the pool (−9 % at N=512, **+5 %** at
   N=1319); with light load and slack memory its per-compaction cost (each
   compaction forces the surrounding decode steps out of the CUDA graph into eager
   mode) shows as a small overhead.
4. **A scheduler limit still caps the upside.** SGLang reserves each request's
   pre-compaction footprint (prompt + buffer ≈ 828 tokens) at admission and does
   not re-admit into the space R-KV frees on compaction, so R-KV plateaus at
   `pool / 828` (508 at `mem-frac 0.50`) instead of the `pool / 384` ≈ 1096 its
   compacted footprint could support. Re-admitting into freed headroom is the
   remaining lever.

## Contrast with the vLLM port

Both ports now realize the memory-bound throughput win — SGLang at **+19–33 %**
here, the vLLM port at **+17–32 %** (256 vs 100 concurrent at `gpu_mem 0.30`; see
[`vLLM/benchmark/RESULTS_stress.md`](../../vLLM/benchmark/RESULTS_stress.md)). Two
implementation differences still separate them:

- **Footprint.** vLLM R-KV uses a small fixed-address query ring and
  memory-guarded *tiled* scoring (no pre-materialized `L×L` matrix), so its pool is
  only **~7 % smaller** than Full-KV's at `gpu_mem 0.30`. SGLang's auto-cap closes
  most of its gap (−9.8 % at `mem-frac 0.50`) but the per-layer buffers still cost
  more than vLLM's ring.
- **Cudagraph.** vLLM auto-selects **PIECEWISE cudagraph**, which keeps attention
  eager on *every* step anyway, so compaction adds no CUDA-graph break; SGLang
  graphs attention and pays a short eager window per compaction, which is what
  keeps its ample-pool win modest (+5 % at N=1319) relative to vLLM.

The reference SGLang R-KV, once its rolling buffer is sized to the pool, delivers
the same qualitative win as the port — larger under memory pressure, smaller when
memory is slack.

## Reproduce

```bash
# Prereq: scripts/apply_rkv.sh, then activate the sglang env (see ../README.md).
cd SGLang/benchmark

# Memory-pressure sweep: for each fraction, a constrained Full-KV server and an
# R-KV server (auto-cap is the default — no extra flag needed), on two GPUs.
for MEM in 0.50 0.40 0.30 0.25; do
  CUDA_VISIBLE_DEVICES=0 MEM_FRAC=$MEM PORT=30010 ./launch_server.sh constrained 256 &
  CUDA_VISIBLE_DEVICES=1 MEM_FRAC=$MEM BUFFER=128 PORT=30011 ./launch_server.sh rkv 256 &
  # wait for both /health, then drive N=256 at concurrency 256:
  python eval.py --port 30010 --n 256 --concurrency 256 --max-tokens 1024 --ignore-eos &  c0=$!
  python eval.py --port 30011 --n 256 --concurrency 256 --max-tokens 1024 --ignore-eos &  c1=$!
  wait $c0 $c1          # wait ONLY on the clients, not the never-exiting servers
  pkill -f sglang.launch_server
done
```

Peak concurrency is the max `#running-req: N` in each server log; `eval.py` prints
end-to-end throughput; the startup log prints `max_total_num_tokens` (the pool) and,
for R-KV, the `auto-capping max_running_requests …` line. Override the auto-cap with
`--rkv-max-active-requests N` or `--max-running-requests N`.
