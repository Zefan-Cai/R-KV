# FlashInfer backend — H100 validation results

- **Date**: 2026-07-08/09 (62-run sweep, `results/validation-2026-07-08-h100/flashinfer_sweep/`)
- **GPU**: 1× pod, 2× NVIDIA H100 80GB HBM3 (Pluto `rkv-fi-bench-0`)
- **Stack**: exactly `requirements-rkv.txt` — torch 2.10.0 cu128, flashinfer-python 0.6.12, transformers 5.8.1, Python 3.12
- **Sampling**: temperature 0.6, top_p 0.95, seed 42; R-KV `buffer=128`, `mix_lambda=0.1` unless noted
- Raw artifacts: per-run logs, per-sample jsonl records, runlist, and `harvest.json` under
  `results/validation-2026-07-08-h100/flashinfer_sweep/`.

## Accuracy

GSM8K / MATH-500 `--max-samples 100`; AIME24 full n=30 (small-n: ±1 answer = ±3.3).

### DeepSeek-R1-Distill-Qwen-1.5B

| dataset | FullKV | rkv 512 | rkv 1024 | rkv 2048 | SnapKV-style 1024 (λ=1.0) |
|---|---|---|---|---|---|
| gsm8k | 71.0 | 71.0 | 71.0 | 71.0 | 71.0 |
| math | 77.0 | 64.0 | 73.0 | 75.0 | 71.0 |
| aime24 | 20.0 | 0.0 | 6.7 | 13.3 | 6.7 |

### DeepSeek-R1-Distill-Qwen-7B

| dataset | FullKV | rkv 512 | rkv 1024 | rkv 2048 |
|---|---|---|---|---|
| gsm8k | 82.0 | 82.0 | 82.0 | 82.0 |
| math | 88.0 | 76.0 | 79.0 | 85.0 |
| aime24 | 53.3 | 6.7 | 16.7 | 40.0 |

Reading: GSM8K is budget-insensitive (exact parity at every budget, both models).
MATH degrades gracefully and budget 2048 sits within 2-3 points of FullKV.
AIME24's very long CoT needs budget ≥ 2048 (7B: 40.0 vs 53.3); 512 collapses it.
The λ=1.0 ablation (attention-only scoring = SnapKV-style) trails joint R-KV
scoring by 2 points on MATH at the same budget, and matches elsewhere.
AIME24 FullKV 20.0 (1.5B) exactly reproduces the 2026-07-02 A100 reference
(`results/validation-2026-07-02-a100/aime24_n30/`); rkv1024 6.7 vs 10.0 there
is a 2-vs-3-answers gap, inside small-n noise.

## Throughput / memory (synthetic bench, `bench_rkv.py`, pre-optimization code)

DeepSeek-R1-Distill-Qwen-1.5B / 7B, prompt 512, decode tok/s (mean of trials),
peak GiB from `torch.cuda.max_memory_allocated`:

| config | FullKV tok/s | R-KV 1024 tok/s | FullKV peak GiB | R-KV peak GiB |
|---|---|---|---|---|
| 1.5B b=1 g=2048 | 70.7 | 63.3 | 3.60 | 3.57 |
| 1.5B b=8 g=2048 | 633.3 | 524.9 | 4.34 | 4.05 |
| 1.5B b=8 g=8192 | 619.2 | 516.4 | 5.66 | 4.05 |
| 1.5B b=32 g=2048 | 2493.7 | 1787.0 | 6.91 | 5.73 |
| 7B b=1 g=2048 | 68.3 | 60.2 | 14.62 | 14.57 |
| 7B b=8 g=2048 | 602.4 | 513.2 | 16.16 | 15.57 |
| 7B b=8 g=8192 | 599.1 | 491.9 | 18.78 | 15.57 |
| 7B b=32 g=2048 | 2386.8 | 1780.0 | 21.42 | 19.06 |

R-KV's KV footprint is flat in generation length (b=8: identical peak at g=2048
and g=8192) while FullKV grows with it; at these gen lengths the compression
work costs 10-28% decode throughput on this eager engine. Long-prompt
compression (p=4096, budget 512) runs at 270.6 / 257.0 tok/s (1.5B / 7B, b=4).

## Numerics diagnostics

- `probe_logits.py` (logit-level FlashInfer-vs-HF comparison): **PROBE_OK on
  all four architectures** — DeepSeek-R1-Distill-Qwen-7B, Qwen3-0.6B,
  Qwen3-8B, Llama-3.2-1B-Instruct.
- `compare_hf` free-running text comparison: 7B passes verbatim; Qwen3-0.6B /
  Llama-1B diverge after a short shared prefix — expected fp16 trajectory
  divergence under sampling, not a numerical defect (see probe results).
- GPU smokes: all pass except sporadic small-model flakes (Qwen3-0.6B under a
  deliberately tiny smoke budget degenerates into repetition and trips the
  health heuristic; one Llama-1B run answered too briefly to trigger
  compaction). Same flake pattern on pre- and post-optimization code.

## Post-sweep optimization pass (A/B on the same node)

After this sweep was recorded, the backend received a parity-safe optimization
pass (fused qkv / gate_up GEMMs, batched compaction with a first-call bitwise
gate — **gate passed on H100, batched path adopted** — plus decode host-path
and scoring-transient reductions; `torch.equal` algo parity verified by CPU
tests). See `docs/DESIGN.md` §5.2-5.3 and the A/B numbers below; the accuracy
tables above were produced by the pre-optimization code at commit `437579ba`.

| config (1.5B, b=16, g=2048, 2 trials) | pre-opt | post-opt | Δ |
|---|---|---|---|
| fullkv decode tok/s | 1234.1 | 1394.4 | **+13.0%** |
| rkv 1024 decode tok/s | 996.1 | 1357.8 | **+36.3%** |
| rkv 1024 compaction s/trial | 2.927 | 0.268 | −90.8% |
| rkv 1024 peak GiB | 4.61 | 5.45 | +0.84 |

The R-KV-vs-FullKV throughput gap shrinks from −19.3% to −2.6%. The peak-GiB
increase is the batched compaction's transient working set (engine-side
stacked K/V copies + scoring buffers, the latter chunk-bounded to ~512MB via
`compressor._BATCH_CHUNK_BYTES`); R-KV still holds its flat-in-gen-length
memory advantage. Post-opt GPU smokes pass on DeepSeek-1.5B (rkv on/off) and
Llama-3.2-1B; Qwen3-0.6B reproduces the same pre-existing tiny-model
repetition flake as the pre-opt code.
