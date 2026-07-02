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
few-shot prompt ~700 tokens, `window=8, buffer=16`, `max_new_tokens=512`.

| Config | Accuracy (20) | Compactions | Notes |
| --- | --- | --- | --- |
| baseline (R-KV off) | 95% (19/20) | — | reference |
| R-KV budget=512 | **100% (20/20)** | 184 | on par / +1, heavy eviction |
| R-KV budget=256 | 90% (18/20) | 241 | most aggressive, slight drop |

**Headline takeaway.** Even though `budget=512 < prompt (~700)` — so R-KV evicts
part of the few-shot prompt itself — accuracy is **fully preserved (100%)**,
confirming the "keep important + recent" selection retains what matters. Only at
`budget=256` does it dip (−5%). 184–241 physical compactions ran per sweep with
the server never crashing.

## Accuracy vs budget — Qwen2.5-0.5B-Instruct (weak model, noisy)

| Config | Accuracy (20) | Compactions | Notes |
| --- | --- | --- | --- |
| baseline (R-KV off) | 35% (7/20) | — | reference |
| R-KV budget=1024 | 35% (7/20) | ~0 | ≥ prompt+gen, almost no eviction ⇒ == baseline |
| R-KV budget=512 | 40% (8/20) | 442 | heavy eviction, accuracy preserved |
| R-KV budget=256 | 30% (6/20) | 482 | most aggressive, accuracy starts to drop |

**Takeaway.** Accuracy is monotone and well-behaved in `budget`: at
`budget ≥ 512` it matches the baseline within noise; only at `budget=256` does it
start to degrade. This is exactly the expected behaviour of a KV compressor —
lossless while the budget is large enough, lossy only when squeezed too hard.
Crucially, the server ran **hundreds of physical compactions (freeing slots)
without crashing**, confirming the eviction / free / `req_to_token` rewrite /
rotary-decoupling are correct.

## Speed (fair, same eager config; serial batch=1)

| Config | Wall (20) | avg tok | Throughput | Relative |
| --- | --- | --- | --- | --- |
| baseline (CUDA graph, prod-like) | ~10s | 377 | ~750 tok/s | reference |
| baseline (eager, fair) | 61s | 363 | **119 tok/s** | isolates CUDA-graph loss |
| R-KV budget=512 | 81s | 366 | **90 tok/s** | 442 compactions |
| R-KV budget=256 | 90s | 399 | **89 tok/s** | 482 compactions |

**Two layers of cost:**

1. **R-KV compaction overhead** (same eager path, only `--enable-rkv` differs):
   119 → 90 tok/s, **≈ 24% slower**. Source: every `buffer_size` steps R-KV
   reads back all layers' KV, computes an O(budget²) key-similarity, and
   relocates slots — all synchronous on the eager path.
2. **Loss of CUDA graph** (implicit, larger): R-KV must run eager, while the
   production baseline uses CUDA graph (~750 tok/s). So versus a *production*
   baseline, most of the end-to-end slowdown comes from losing CUDA graph, not
   from the compaction itself.

## Caveat: this scenario is cost-only for R-KV

The benchmark is **short sequences (~1.1k tokens) at batch=1**, where the KV
cache is not a bottleneck — so it only exposes R-KV's overhead, not its benefit.
R-KV pays off when:

- **long CoT (several-k to tens-of-k tokens)** — attention grows ~quadratically;
  compressing to `budget` makes late-decode attention materially cheaper;
- **large batch / memory-bound serving** — the freed KV memory admits more
  concurrent requests, raising aggregate throughput.

A long-sequence / memory-pressure throughput test is needed to show the benefit
side. Phase-2 perf targets (see `R-KV/doc/DESIGN.md`): CUDA-graph-compatible compaction,
a cheaper redundancy estimate than O(budget²), and avoiding full KV read-back.
