# R-KV Benchmark Results — Qwen2.5-Math-7B-Instruct (Mini-SGLang)

Accuracy + throughput for the Mini-SGLang R-KV port on a **strong math model**,
comparing **full KV** against R-KV at `budget=256` / `budget=512`. Mirrors the
sibling [`SGLang/`](../../SGLang) port's harness; the difference is that
Mini-SGLang has no R-KV server flags, so this runs the **offline** `LLM` engine
with batched generation (the concurrency proxy).

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (bf16), single **NVIDIA H100 80GB**.
- **Dataset**: GSM8K-style few-shot MATH harness (`gsm8k_fewshot.jsonl`, the same
  set the SGLang port uses), first **40 items**, `max_new_tokens=512`,
  `temperature=0`. Few-shot prompt ≈ 700 tokens — larger than either budget, so
  R-KV compresses immediately.
- **Judging**: numeric match (`eval_math.py`), identical across runs. Completions
  are truncated at the next `\nProblem` (emulating the SGLang harness's stop).
- **Config**: `attention_backend="fi"` (FlashInfer), `page_size=1`, CUDA graph
  off. R-KV runs force **overlap scheduling off** (required — see
  [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) §4); `window_size=8`,
  `buffer=64`.

> ⚠️ **Small sample (40 items).** Treat these as trend indicators; re-run with a
> larger `--n` for tighter figures. Date: 2026-07-02.

## Accuracy vs budget

| Config | Accuracy (40) | Compactions | avg tok | Notes |
| --- | --- | --- | --- | --- |
| full KV (R-KV off) | 95.0% (38/40) | — | 492 | reference |
| **R-KV budget=512** | **95.0% (38/40)** | 224 | 486 | on par, heavy eviction |
| R-KV budget=256 | **95.0% (38/40)** | 224 | 462 | most aggressive, still on par |

**Takeaway.** Accuracy is **lossless at both budgets** — R-KV matches full KV
exactly (95.0%), even though `budget=256` / `512` are both **below the ~700-token
prompt**, so R-KV is evicting part of the few-shot prompt itself. The "keep
important + recent" selection retains what matters. ~224 physical compactions ran
per sweep with zero crashes and no degeneration (validated under normal async
execution after the overlap-scheduling fix).

## Speed (offline, batch=40)

| Config | Wall (40) | avg tok | Throughput | Notes |
| --- | --- | --- | --- | --- |
| full KV (overlap on, prod-like) | 4.0s | 492 | **4885 tok/s** | reference |
| full KV (overlap off, fair eager) | 4.5s | 491 | **4403 tok/s** | isolates overlap loss |
| R-KV budget=512 | 15.8s | 486 | **1231 tok/s** | 224 compactions |
| R-KV budget=256 | 16.7s | 462 | **1109 tok/s** | 224 compactions |

**Two costs, and the compaction dominates here:**

1. **Loss of overlap scheduling** (R-KV requires it off): 4885 → 4403 tok/s,
   **only ~10%**.
2. **R-KV compaction overhead** (fair eager baseline → R-KV): 4403 → ~1.1–1.2k
   tok/s, **≈ 3.6× slower**. Unlike the SGLang port (~28%), Mini-SGLang compacts
   **per layer, inside the forward** — every `buffer` steps each of the model's
   layers runs an `index_select` + O(budget²) similarity + scatter, all
   synchronous — so at these short sequences the compaction is the dominant cost.
   A cheaper redundancy estimate / batched post-forward compaction is the
   phase-2 optimization target.

## Caveat: this scenario is cost-only for R-KV

This is **short sequences (~1.2k tokens) at batch=40**, where the KV cache is not
the bottleneck — so it only exposes R-KV's overhead, not its benefit (memory
saved → longer context / larger batch under fixed VRAM, and cheaper late-decode
attention on long CoT). The **accuracy** result (lossless at ¼–½ the KV) is the
headline; a long-sequence / memory-pressure test is needed to show the upside.

## Reproduce

```bash
cd Mini-SGLang
scripts/apply_rkv.sh                    # build the patched, pinned tree
cd mini-sglang-src && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .

../benchmark/prepare_data.sh            # fetch gsm8k_fewshot.jsonl into benchmark/data/

# full KV, then R-KV at budget 512 / 256 (0 = full KV):
python3 ../benchmark/eval_math.py --budget 0   --n 40
python3 ../benchmark/eval_math.py --budget 512 --n 40
python3 ../benchmark/eval_math.py --budget 256 --n 40
```
