# R-KV Benchmark Results — Qwen2.5-Math-7B-Instruct

Dedicated report for the R-KV port on a **strong math model**, where accuracy
differences are signal rather than noise. Companion to [`RESULTS.md`](./RESULTS.md)
(which also covers the weak Qwen2.5-0.5B sanity check).

## Setup

- **Model**: `Qwen/Qwen2.5-Math-7B-Instruct` (dense, GQA, rotary; `bf16`).
  Downloaded to `/data/model/Qwen2.5-Math-7B-Instruct` via direct CDN `curl -L`
  (HF Xet transfer hangs on large files).
- **GPU**: single NVIDIA H100 80GB, `--mem-fraction-static 0.85`
  (weights ≈14.3 GB, KV pool ≈52 GB → ~981k tokens).
- **Dataset**: GSM8K-style few-shot MATH harness (`test.jsonl` from the `dev`
  branch), first **20 items**, `max_new_tokens=512`, `temperature=0`.
  Few-shot prompt is ~700 tokens.
- **Judging**: numeric match (`eval.py`), identical across all runs.
- **Config**: all R-KV runs use the benchmark's `rkv` mode
  (`--disable-radix-cache --disable-decode-cuda-graph --disable-prefill-cuda-graph
  --disable-overlap-schedule --page-size 1`); the eager baseline uses the same
  flags without `--enable-rkv`. `window_size=8`, `buffer_size=16` throughout.

> ⚠️ **Small sample (20 items).** Each item is ±5%, so treat these as trend
> indicators. Re-run with `--n 100`+ for tighter figures. The trend is clean.
>
> **Re-run 2026-07-02, post rotary off-by-one fix.** Earlier numbers (budget=512
> 100%, budget=256 90%) were n=20 noise; with the fix accuracy is a flat 95% at
> every budget — i.e. *identical* to the baseline. For a 5× larger sample on a
> different GPU see [`RESULTS_a100_n100.md`](./RESULTS_a100_n100.md) (n=100,
> A100: parity within noise, 90 vs 91).

## Accuracy vs budget

| Config | Accuracy (20) | Compactions | avg tok | Notes |
| --- | --- | --- | --- | --- |
| baseline (R-KV off) | 95% (19/20) | — | 181 | reference |
| **R-KV budget=512** | **95% (19/20)** | ~230 | 197 | on par with baseline, heavy eviction |
| R-KV budget=256 | 95% (19/20) | 227 | 190 | most aggressive, still on par |

**Takeaway.** Accuracy is **lossless at every budget tested** — R-KV matches the
baseline exactly (95%, 19/20):

- At `budget=512` accuracy is identical to baseline — **even though
  `budget=512 < prompt (~700)`**, so R-KV is evicting part of the few-shot prompt
  itself. The "keep important + recent" selection retains what matters.
- Even at the aggressive `budget=256` there is **no drop** (still 95%); the one
  miss is the same item the baseline also misses.
- The server ran **227–230 physical compactions per sweep with zero crashes**,
  confirming eviction / `free` / `req_to_token` rewrite / rotary decoupling are
  correct under sustained load on a 7B model.

## Speed (fair, same eager config)

| Config | Wall (20) | avg tok | Throughput | Notes |
| --- | --- | --- | --- | --- |
| baseline (eager, fair) | 35.6s | 181 | 101.8 tok/s | serial batch=1, reference |
| R-KV budget=512 | 53.4s | 197 | 73.7 tok/s | serial, ~230 compactions |
| R-KV budget=256 | 49.1s | 190 | 77.3 tok/s | serial, 227 compactions |
| R-KV budget=512, concurrency=8 | 14.3s | 194 | **272.2 tok/s** | 8 in-flight (batch path) |

**R-KV compaction overhead** (same eager path, only `--enable-rkv` differs):
≈102 → 74 tok/s at batch=1, **~28% slower** — consistent with the 0.5B
measurement (~26%). The overhead comes from the per-`buffer_size` compaction
(read back all layers' KV, O(budget²) key-similarity, slot relocation), all
synchronous on the eager path.

**Batching recovers it and then some.** At `concurrency=8`, R-KV budget=512
sustains **272 tok/s** — ~2.7× the eager serial baseline and ~3.7× serial R-KV —
because the batch>=2 per-request path amortizes the compaction work across
in-flight requests. As with 0.5B, the larger *implicit* cost is that R-KV must
run eager and so gives up CUDA graph relative to a production baseline.

## Caveat: benefit side not measured here

This is **short sequences (~1k tokens), batch=1** — the KV cache is not the
bottleneck, so only R-KV's overhead shows, not its benefit (memory saved →
longer context / larger batch under fixed VRAM, and cheaper late-decode
attention on long CoT). A long-sequence / memory-pressure throughput test is
still needed to show the upside.

## Reproduce

```bash
cd SGLang/benchmark
./prepare_data.sh   # or use an existing test.jsonl via --data

# baseline (eager)
MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 ./launch_server.sh baseline
python3 eval.py --n 20 --label base7b

# R-KV budget=512
MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 ./launch_server.sh rkv 512
python3 eval.py --n 20 --label rkv7b_b512

# R-KV budget=512, batched (concurrency=8)
python3 eval.py --n 20 --concurrency 8 --label rkv7b_b512_c8
```
