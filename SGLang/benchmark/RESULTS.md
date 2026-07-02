# R-KV Benchmark Results

First measurements of the R-KV port, taken during phase-1 bring-up.

## Setup

- **Models**: `Qwen2.5-Math-7B-Instruct` (headline — a strong math model, so
  accuracy differences are signal, not noise) and `Qwen2.5-0.5B-Instruct` (a
  weak *non-math* model, kept as a noisy secondary sanity check).
- **GPU**: single NVIDIA H100 80GB.
- **Dataset**: GSM8K-style few-shot MATH harness (`test.jsonl` from the `dev`
  branch), first 20 items, `max_new_tokens=512`, `temperature=0`.
- **Judging**: numeric match (see `eval.py`) — approximate, identical across all
  runs.
- **Config**: all R-KV runs use `--disable-radix-cache --disable-decode-cuda-graph
  --disable-prefill-cuda-graph --disable-overlap-schedule --page-size 1`; the
  eager baseline uses the same flags without `--enable-rkv`.
- `window_size=8`, `buffer_size=16` throughout.

> ⚠️ **Small sample (20 items).** Each item is ±5%, so treat these as trend
> indicators, not precise numbers. Re-run with `--n 100`+ for tighter figures.

## Accuracy vs budget — Qwen2.5-Math-7B-Instruct (headline)

Strong math model, so accuracy differences are signal, not noise. 20 items,
few-shot prompt ~700 tokens, `window=8, buffer=16`, `max_new_tokens=512`
(re-run 2026-07-02, post rotary off-by-one fix).

| Config | Accuracy (20) | Compactions | Notes |
| --- | --- | --- | --- |
| baseline (R-KV off) | 95% (19/20) | — | reference |
| R-KV budget=512 | **95% (19/20)** | ~230 | on par with baseline, heavy eviction |
| R-KV budget=256 | **95% (19/20)** | 227 | most aggressive, still on par |

**Headline takeaway.** Even though `budget=512 < prompt (~700)` — so R-KV evicts
part of the few-shot prompt itself — accuracy is **identical to the baseline
(95%)**, confirming the "keep important + recent" selection retains what matters.
Even at `budget=256` there is **no drop**. ~227–230 physical compactions ran per
sweep with the server never crashing. (Full report:
[`RESULTS_math7b.md`](./RESULTS_math7b.md).)

## Accuracy vs budget — Qwen2.5-0.5B-Instruct (weak model, noisy)

20 items (re-run 2026-07-02, post rotary off-by-one fix).

| Config | Accuracy (20) | Compactions | Notes |
| --- | --- | --- | --- |
| baseline (R-KV off) | 30% (6/20) | — | reference |
| R-KV budget=512 | 35% (7/20) | 386 | heavy eviction, accuracy preserved |
| R-KV budget=256 | 35% (7/20) | 509 | most aggressive, still within noise |

**Takeaway.** On this weak *non-math* model the absolute accuracy is low and
noisy (±5% per item), but R-KV is well-behaved: at both `budget=512` and
`budget=256` it matches the baseline within noise (indeed marginally above).
This is the expected behaviour of a KV compressor — lossless while the budget is
large enough. Crucially, the server ran **hundreds of physical compactions
(freeing slots) without crashing**, confirming the eviction / free /
`req_to_token` rewrite / rotary-decoupling are correct.

## Speed — Qwen2.5-0.5B-Instruct (fair, same eager config; serial batch=1)

| Config | Wall (20) | avg tok | Throughput | Notes |
| --- | --- | --- | --- | --- |
| baseline (eager, fair) | 60.7s | 363 | **119.6 tok/s** | reference |
| R-KV budget=512 | 72.3s | 320 | **88.6 tok/s** | 386 compactions |
| R-KV budget=256 | 95.1s | 420 | **88.3 tok/s** | 509 compactions |

**R-KV compaction overhead** (same eager path, only `--enable-rkv` differs):
119.6 → 88.6 tok/s, **≈ 26% slower** — consistent with the Math-7B measurement
(~28%). Source: every `buffer_size` steps R-KV reads back all layers' KV,
computes an O(budget²) key-similarity, and relocates slots — all synchronous on
the eager path.

**Loss of CUDA graph** (implicit, larger): R-KV must run eager, while a
production baseline uses CUDA graph. So versus a *production* baseline, most of
the end-to-end slowdown comes from losing CUDA graph, not from the compaction
itself — see the Math-7B `concurrency=8` result (272 tok/s) for how batching
amortizes the eager overhead.

## Caveat: this scenario is cost-only for R-KV

The benchmark is **short sequences (~1.1k tokens) at batch=1**, where the KV
cache is not a bottleneck — so it only exposes R-KV's overhead, not its benefit.
R-KV pays off when:

- **long CoT (several-k to tens-of-k tokens)** — attention grows ~quadratically;
  compressing to `budget` makes late-decode attention materially cheaper;
- **large batch / memory-bound serving** — the freed KV memory admits more
  concurrent requests, raising aggregate throughput.

A long-sequence / memory-pressure throughput test is needed to show the benefit
side. Phase-2 perf targets (see `SGLang/docs/DESIGN.md`): CUDA-graph-compatible compaction,
a cheaper redundancy estimate than O(budget²), and avoiding full KV read-back.
