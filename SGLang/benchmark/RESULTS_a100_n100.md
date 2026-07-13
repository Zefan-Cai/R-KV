# Independent validation — Qwen2.5-Math-7B-Instruct, n=100, 1×A100-80G

Independent rerun of the H100 n=20 benchmark
([`RESULTS.md`](./RESULTS.md)) with a 5× larger sample, on a
different GPU, from a clean environment — run 2026-07-01 as part of verifying
this port against the repo's reference implementation before publishing.

## Setup

- **Code**: this directory's port (source: `wanke1997/sglang-compress` @
  `9ed5f084`, batch>=1 per-request triggering). "fixed" rows additionally
  carry the two correctness fixes now shipped here (rotary position
  off-by-one in `rkv/integration.py` + `--enable-rkv` startup validation in
  the wiring patch).
- **GPU**: 1× NVIDIA A100-SXM4-80GB (`tp=1`), driver CUDA 13.0.
- **Env**: PyPI `sglang==0.5.14` dependency stack (torch 2.11.0+cu130,
  flashinfer 0.6.12, sglang-kernel 0.4.4, transformers 5.8.1), port source via
  `PYTHONPATH`.
- **Dataset**: `benchmark/data/gsm8k_fewshot.jsonl` (the standard 1319-item
  GSM8K test set with MATH-style few-shot prompts, prompt ≈ 700 tokens),
  first **100 items**, `max_new_tokens=512`, `temperature=0`.
- **Config**: benchmark defaults — `window_size=8`, `buffer_size=16`,
  `MEM_FRAC=0.85`, eager decode, radix cache off, overlap off, `page_size=1`.
  Baseline = same flags without `--enable-rkv`.

## Results

| Config | Accuracy (100) | avg tok | Wall | Throughput | Compactions |
| --- | --- | --- | --- | --- | --- |
| baseline (eager, R-KV off) | **91/100 = 91.0%** | 169 | 344s | 49.2 tok/s | 0 |
| R-KV budget=512, as published (`9ed5f084`) | 89/100 = 89.0% | 168 | 388s | 43.3 tok/s | 998 |
| R-KV budget=512, **fixed** | **90/100 = 90.0%** | 171 | 388s | 44.2 tok/s | 1012 |
| R-KV budget=256, **fixed** | 89/100 = 89.0% | 191 | 450s | 42.4 tok/s | 1138 |
| R-KV budget=512, **fixed**, `--concurrency 8` | **90/100 = 90.0%** | 170 | 94s | **181.8 tok/s** | 1007 |

## Takeaways

1. **Accuracy is preserved under aggressive eviction.** With `budget=512 <
   prompt (~700)` — i.e. R-KV is evicting part of the few-shot prompt itself —
   accuracy is within 1–2 points of baseline (noise band at n=100 is ±~3
   points). Even `budget=256` (≈ one quarter of the peak KV) loses only 2
   points. ~1000 physical compactions per sweep, zero crashes, no
   leak-checker aborts at idle.
2. **The H100 n=20 numbers reproduce in trend, not in letter.** The original
   pre-fix n=20 run showed an optimistic "100% vs 95%" (serial) that was
   small-sample luck; the post-fix n=20 re-run was a flat 95%, and at n=100
   here the honest statement is *parity within noise* (90 vs 91).
3. **The rotary off-by-one fix does not change GSM8K accuracy** (89 → 90 is
   one item, well within noise) — consistent with a uniform +1 position shift
   of all decode tokens being nearly invisible to relative-position attention.
   The fix matters for correctness discipline (`--enable-rkv` should be
   output-equivalent to baseline until compression fires), not for this
   benchmark's score.
4. **`batch >= 2` works and scales.** With 8 concurrent requests the
   per-request trigger path compacted 1007 times with identical accuracy
   (90/100) and 4.1× end-to-end throughput (181.8 vs 44.2 tok/s serial).
5. **Compression overhead on the eager path is ~10% at these lengths**
   (49.2 → 44.2 tok/s serial; shorter generations than the H100 run, where it
   was ~25%). The O(budget²) similarity + full KV read-back per trigger were the
   phase-2 optimization targets, since addressed by the fused Triton redundancy
   kernel and batched cross-layer scoring (see
   [`../docs/OPTIMIZATIONS.md`](../docs/OPTIMIZATIONS.md)).
