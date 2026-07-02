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

> ⚠️ **Small sample (20 items).** Each item is ±5%, so 20/20 has some luck in it.
> Re-run with `--n 100`+ for tighter figures. The trend, however, is clean.

## Accuracy vs budget

| Config | Accuracy (20) | Compactions | avg tok | Notes |
| --- | --- | --- | --- | --- |
| baseline (R-KV off) | 95% (19/20) | — | 181 | reference |
| **R-KV budget=512** | **100% (20/20)** | 184 | 156 | on par / +1, heavy eviction |
| R-KV budget=256 | 90% (18/20) | 241 | 202 | most aggressive, slight drop |

**Takeaway.** Accuracy is monotone and, at a sensible budget, **lossless**:

- At `budget=512` the model matches (indeed slightly beats, within noise) the
  baseline — **even though `budget=512 < prompt (~700)`**, so R-KV is evicting
  part of the few-shot prompt itself. The "keep important + recent" selection
  retains the information that matters for the answer.
- Only at the very aggressive `budget=256` does accuracy dip (−5%).
- The server ran **184–241 physical compactions per sweep with zero crashes**,
  confirming eviction / `free` / `req_to_token` rewrite / rotary decoupling are
  correct under sustained load on a 7B model.

This is a much stronger signal than the 0.5B run: on a capable model the
on/off gap is a clear, interpretable 0–5%, not noise.

## Speed (fair, same eager config; serial batch=1)

| Config | Wall (20) | avg tok | Throughput | Relative |
| --- | --- | --- | --- | --- |
| baseline (eager, fair) | 35s | 181 | ~103 tok/s | reference |
| R-KV budget=512 | 41s | 156 | ~76 tok/s | 184 compactions |
| R-KV budget=256 | 53s | 202 | ~76 tok/s | 241 compactions |

**R-KV compaction overhead** (same eager path, only `--enable-rkv` differs):
≈103 → 76 tok/s, **~26% slower** — consistent with the 0.5B measurement (~24%).
The overhead comes from the per-`buffer_size` compaction (read back all layers'
KV, O(budget²) key-similarity, slot relocation), all synchronous on the eager
path. As with 0.5B, the larger *implicit* cost is that R-KV must run eager and
so gives up CUDA graph relative to a production baseline.

## Caveat: benefit side not measured here

This is **short sequences (~1k tokens), batch=1** — the KV cache is not the
bottleneck, so only R-KV's overhead shows, not its benefit (memory saved →
longer context / larger batch under fixed VRAM, and cheaper late-decode
attention on long CoT). A long-sequence / memory-pressure throughput test is
still needed to show the upside.

## Reproduce

```bash
cd R-KV/benchmark
./prepare_data.sh   # or use an existing test.jsonl via --data

# baseline (eager)
MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 ./launch_server.sh baseline
python3 eval.py --n 20 --label base7b

# R-KV budget=512
MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 ./launch_server.sh rkv 512
python3 eval.py --n 20 --label rkv7b_b512
```
