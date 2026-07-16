# R-KV on vLLM — GSM8K budget × buffer sweep (H100)

Accuracy and throughput of the R-KV decode-time KV-cache compressor across the
full `budget × buffer` grid, measured on the SGLang-parity differential harness
after the async-scheduling fix (see
[`../docs/RESIDUAL_GAP_INVESTIGATION.md`](../docs/RESIDUAL_GAP_INVESTIGATION.md)).

## Setup

- **Model:** Qwen2.5-Math-7B-Instruct (bf16), `--enforce-eager`.
- **Engine:** vLLM v0.25.1 + R-KV patch (this repo), V1 GPU runner,
  FlashAttention backend, `block_size=16`, `gpu_memory_utilization=0.85`,
  `max_model_len=4096`. Prefix caching and async scheduling are auto-disabled
  when R-KV is on.
- **Workload:** SGLang's few-shot GSM8K harness
  (`SGLang/benchmark/data/gsm8k_fewshot.jsonl`), prompt ≈ 700 tokens (> every
  budget), **200 questions**, greedy (`temperature=0`), `max_tokens=512`,
  stop `"\nProblem"`. Decode-only R-KV (`window=8`), same extraction as the
  `对拍` harness.
- **Hardware / execution:** 8× H100 80GB. The 13 configs were run **one per GPU,
  8 in parallel** (two waves), each an offline `LLM.generate` over all 200
  prompts (natural high-concurrency batching). Total wall for the whole sweep:
  **~2m40s**.
- **Throughput** = decode (output) tokens ÷ generation wall time. Because the
  sweep runs 8 processes concurrently on one host, absolute tok/s carries ~8%
  host contention (Full-KV alone = 5330 tok/s vs 4910 tok/s inside the parallel
  wave); the **relative** buffer/budget trends are unaffected.

## Accuracy (% correct, 200 questions)

Full-KV ceiling on this harness: **91.0%** (182/200).

| budget \ buffer | 16 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- |
| **128** | 64.0 | 75.5 | 82.0 | 90.5 |
| **256** | 83.0 | 87.5 | 90.5 | 91.5 |
| **512** | 89.5 | 90.0 | 91.0 | 91.5 |

- **Buffer is the accuracy knob at tight budgets.** At `budget=128`, raising the
  buffer 16 → 256 recovers **+26.5 points** (64.0 → 90.5). At `budget=512` the
  model is already near-lossless at every buffer (89.5 – 91.5).
- **`budget=256 buffer=64` = 87.5%** matches the SGLang reference (88.5%) within
  n=200 noise, and `buffer ≥ 128` is **lossless** (90.5 – 91.5% ≥ the 91.0%
  ceiling — the ≥100% cases are within noise).
- Tighter configs also **generate longer** (avg gen-len 258 tok at
  `b128/buf16` vs 171 at Full-KV): once critical context is evicted the model
  rambles, which both lowers accuracy and adds work.

## Decode throughput (output tok/s, 8-way parallel)

Full-KV: 4910 tok/s (async, in-wave) / 5330 tok/s (alone).

| budget \ buffer | 16 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- |
| **128** | 890 | 1637 | 1827 | 1882 |
| **256** | 836 | 1556 | 1835 | 2005 |
| **512** | 809 | 1562 | 1841 | 2070 |

- **Buffer is also the throughput knob**: it sets how often compaction fires
  (every `buffer` decode steps), and compaction — the O((budget+buffer)²)
  scoring pass run eager — dominates R-KV's cost. Raising the buffer 16 → 256 is
  **~2.5×** faster (≈ 830 → ≈ 2000 tok/s) at every budget.
- **Budget barely moves throughput** at a fixed buffer (the scoring cost grows
  only mildly with budget here), so a larger budget is nearly free for accuracy.
- **Full-KV is faster than any R-KV config** on this workload: at 200 GSM8K
  prompts the 7B model's KV pool fits comfortably in 80 GB, so the run is **not
  memory-bound** and R-KV is pure overhead (compaction + forced synchronous
  scheduling). R-KV's constant-footprint advantage — more concurrency / longer
  context in the same memory — only pays off in the memory-bound regime (long
  outputs, high concurrency); see [`../docs/OPTIMIZATIONS.md`](../docs/OPTIMIZATIONS.md).

## Full per-config results

| Config | Accuracy | Decode tok/s | Total tok/s | Wall (s) | Avg gen len |
| --- | --- | --- | --- | --- | --- |
| Full-KV (async) | 91.0% (182/200) | 4910 | 24908 | 7.0 | 171 |
| Full-KV (sync) | 90.0% (180/200) | 5330 | 27407 | 6.3 | 168 |
| budget=128 buffer=16 | 64.0% (128/200) | 890 | 3292 | 58.1 | 258 |
| budget=128 buffer=64 | 75.5% (151/200) | 1637 | 7178 | 25.2 | 206 |
| budget=128 buffer=128 | 82.0% (164/200) | 1827 | 9033 | 19.3 | 177 |
| budget=128 buffer=256 | 90.5% (181/200) | 1882 | 9624 | 18.0 | 169 |
| budget=256 buffer=16 | 83.0% (166/200) | 836 | 3545 | 51.5 | 215 |
| budget=256 buffer=64 | 87.5% (175/200) | 1556 | 7040 | 25.4 | 198 |
| budget=256 buffer=128 | 90.5% (181/200) | 1835 | 8542 | 20.8 | 191 |
| budget=256 buffer=256 | 91.5% (183/200) | 2005 | 10271 | 16.9 | 169 |
| budget=512 buffer=16 | 89.5% (179/200) | 809 | 4109 | 42.2 | 171 |
| budget=512 buffer=64 | 90.0% (180/200) | 1562 | 7846 | 22.2 | 173 |
| budget=512 buffer=128 | 91.0% (182/200) | 1841 | 9496 | 18.2 | 168 |
| budget=512 buffer=256 | 91.5% (183/200) | 2069 | 10481 | 16.6 | 172 |

## Takeaways & recommendations

1. **`buffer ≈ budget` is the sweet spot** — it maximises both accuracy (lossless
   at `budget ≥ 256`) and throughput (fewest compactions). `budget=256
   buffer=256` = **91.5% @ 2005 tok/s**.
2. **Don't run tiny buffers.** `buffer=16` is the worst on both axes (aggressive,
   frequent compaction): lowest accuracy *and* ~2.5× slower.
3. **`budget=512` is robustly near-lossless** (89.5 – 91.5%) at any buffer — use
   it when accuracy matters more than memory savings.
4. **`budget=128` only works with a large buffer** (`buffer=256` → 90.5%); at
   small buffers it collapses (64 – 82%).
5. Throughput here reflects R-KV **overhead** (non-memory-bound); the memory /
   concurrency **benefit** requires a memory-bound workload.

## Reproduce

```bash
# per-config runner (accuracy + throughput), one GPU:
CUDA_VISIBLE_DEVICES=0 VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=64 \
  RKV_N=200 RKV_MAXTOK=512 RKV_OUT=/tmp/r.json python bench_sweep.py

# full sweep across 8 GPUs (two waves):
bash run_sweep.sh
```

`bench_sweep.py` and `run_sweep.sh` are included in this folder. `budget=0
buffer=0` selects the Full-KV baseline; `RKV_NOASYNC=1` forces synchronous
scheduling for the Full-KV baseline (R-KV forces it automatically).
