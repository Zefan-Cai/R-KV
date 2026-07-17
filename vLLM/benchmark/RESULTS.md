# R-KV Benchmark Results — Qwen2.5-Math-7B-Instruct (GSM8K)

Decode-time R-KV on a strong math model, measured with the **few-shot GSM8K
harness** in [`eval.py`](./eval.py) (the same `data/gsm8k_fewshot.jsonl` prompts
as the SGLang port, for a like-for-like comparison). Companion reports:
[`RESULTS_tp.md`](./RESULTS_tp.md) (tensor-parallel scaling),
[`RESULTS_dp.md`](./RESULTS_dp.md) (data-parallel scaling).

> **Methodology note — offline, not served.** vLLM's R-KV path is a drop-in
> serving change gated by env vars, so this harness runs the model **offline**
> (`LLM.generate` over all `--n` prompts at once) rather than over HTTP. The
> reported throughput is therefore an **offline batched** decode rate (every
> prompt in flight at once, on one GPU), not a served-request rate at a fixed
> concurrency. [`launch_server.sh`](./launch_server.sh) starts a real
> OpenAI-compatible server with the identical R-KV knobs if you want to drive a
> serving benchmark instead. Compaction is confirmed active via the compactor's
> own counter (read back from every worker), **not** inferred from accuracy.

## Setup

- **Model**: `Qwen2.5-Math-7B-Instruct` (bf16, GQA: 28 Q / 4 KV heads), single
  **NVIDIA H100 80GB**.
- **Harness**: [`eval.py`](./eval.py), `data/gsm8k_fewshot.jsonl` (4-shot,
  prompt ≈ 700 tokens > budget), first **200 questions**, `max_tokens=512`,
  `temperature=0`, output ≈ 220 tokens/req.
- **R-KV**: `window_size=8`, default PIECEWISE cudagraph (attention stays eager
  so the in-forward hooks fire), `gpu_memory_utilization=0.85`, `block_size=16`.
- **Two Full-KV baselines** (both matter — see the note):
  - **production** — prefix caching + full cudagraph, upstream defaults (fastest
    Full-KV; `eval.py` with no R-KV env).
  - **constrained** — the *exact* constraint R-KV requires (prefix caching OFF),
    no compression (`eval.py --no-prefix`). This is the **fair A/B baseline**.

> **Why two baselines?** R-KV structurally cannot use the prefix (radix) cache —
> it frees KV slots the cache would still reference, so it force-disables it. With
> a 4-shot prompt every request shares a large prefix, so the prefix cache *alone*
> makes production Full-KV much faster, an advantage unrelated to compression. The
> **constrained** baseline removes it from both sides, isolating R-KV's true cost.

## Baselines

| Full-KV | Accuracy | Throughput (offline batched) |
| --- | :---: | ---: |
| production (prefix cache + full cudagraph) | 0.910 (182/200) | **8117 tok/s** |
| constrained (R-KV's flags, no compression) | 0.910 (182/200) | **4785 tok/s** |

Disabling the prefix cache alone costs Full-KV **~41 %** (8117 → 4785 tok/s) — the
shared-prefix prefill-dedup advantage R-KV cannot use, **not** a compression cost.
All R-KV rows below are compared to the **constrained** baseline.

## R-KV — `budget` × `buffer_size`

200 questions, offline batched. **Compactions** = total physical KV evictions
during the run (from the compactor's counter).

| `budget` | `buffer` | Accuracy | Throughput | vs constrained | Compactions |
| ---: | ---: | :---: | ---: | ---: | ---: |
| *constrained Full-KV* | — | 0.910 (182/200) | 4785 | — | 0 |
| 512 | 256 | 0.920 (184/200) | 3883 | −19 % | 32 |
| 512 | 128 | 0.920 (184/200) | 3664 | −23 % | 154 |
| 512 | 64 | 0.920 (184/200) | 3443 | −28 % | 425 |
| 512 | 16 | 0.910 (182/200) | 2698 | −44 % | 2064 |
| 256 | 256 | 0.900 (180/200) | 3982 | −17 % | 33 |
| 256 | 128 | 0.910 (182/200) | 4056 | −15 % | 184 |
| 256 | 64 | 0.870 (174/200) | 4427 | −7 % | 565 |
| 256 | 16 | 0.880 (176/200) | 3648 | −24 % | 2602 |
| 128 | 256 | 0.880 (176/200) | 3912 | −18 % | 33 |
| 128 | 128 | 0.835 (167/200) | 3897 | −19 % | 181 |
| 128 | 64 | 0.750 (150/200) | 4327 | −10 % | 547 |
| 128 | 16 | 0.660 (132/200) | 4441 | −7 % | 3052 |

## Findings

1. **`budget` sets the accuracy wall.** `budget=512` is **lossless** (0.910–0.920
   vs constrained/production 0.910 — at or above Full-KV, within n=200 noise);
   `budget=256` is **near-lossless at `buffer ≥ 128`** (0.900–0.910) and dips to
   0.870 at `buffer=64`; `budget=128` holds only at large buffers and **collapses
   to 0.660** at `buffer=16` — there it evicts most of the ~700-token prompt+CoT
   every 16 steps. Recommended: **`budget = 256–512`, `buffer ≥ 128`**.
2. **`buffer_size` sets the compaction frequency**, `~1/buffer`: `buffer=256`→~32
   compactions, `128`→~180, `64`→~500, `16`→~2000–3000 over 200 requests. The
   observation window is warm at large buffers and hammered at `buffer=16`.
3. **R-KV stays correct *while* compacting.** e.g. `budget=256, buffer=128` runs
   **184 physical compactions at 0.910 accuracy** (= Full-KV), and `budget=512,
   buffer=64` sustains **425 compactions still lossless** (0.920).
4. **This bench is not memory-bound, so it shows R-KV's *overhead*, not its
   benefit.** At `mem_frac 0.85` the KV pool holds all 200 short-CoT requests
   easily, so R-KV only *adds* work here (compaction compute + the eager
   attention window around each compaction) — hence the −7 % to −44 % throughput
   vs constrained Full-KV, tracking compaction frequency and budget (a smaller
   budget shrinks decode attention, partly offsetting more-frequent compaction).
   The **benefit** side — a constant, prompt-independent KV footprint that lets
   more requests run concurrently under fixed VRAM — appears only under memory
   pressure; the [DP report](./RESULTS_dp.md) fans the work across replicas.

**Sweet spot:** `budget = 256–512`, `buffer = 128` — lossless-to-near-lossless
accuracy with ~180 compactions per 200 requests.

## Reproduce

```bash
# Prereq: build + install the patched vLLM once (see ../README.md):
#   scripts/apply_rkv.sh && pip install -e vllm-src
# then, in that Python env (set RKV_MODEL to a local path to skip the HF download):
cd vLLM/benchmark

# Full-KV production baseline:
python eval.py --n 200 --label fullkv_production
# Full-KV constrained (fair) baseline:
python eval.py --n 200 --no-prefix --label fullkv_constrained
# R-KV, budget 256, buffer 128:
VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
  python eval.py --n 200 --label rkv_b256_buf128
```

`eval.py` prints accuracy, offline throughput, and the physical **compaction
count** (read from every worker's compactor via `collective_rpc`). Set
`VLLM_V1_R_KV_TRACE=1` to log each `[RKV-COMPACT]` / `[RKV-SKIP]` event — always
confirm compaction is active this way, since a broken build silently runs Full-KV
at a nearly identical accuracy.
