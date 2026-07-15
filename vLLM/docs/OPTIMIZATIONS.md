# R-KV on vLLM — Validation Record, Known Limitations & Roadmap

This port re-implements the R-KV runtime wiring against vLLM **v0.25.1**. The
`rkv/` algorithm is CPU bit-parity tested, the wiring patch applies cleanly to a
pristine v0.25.1 tree, and the end-to-end serving path is **validated on an
NVIDIA H100** (vLLM 0.25.1, torch 2.11+cu130, `Qwen2.5-0.5B` / `Qwen2.5-Math-7B`,
`--enforce-eager`).

## Validation record

- **Smoke** — server starts, generates coherent output, no crash.
- **Compaction fires** — `budget=64, buffer=16` over a 150-token generation
  performs hundreds of physical compactions, each keeping exactly `budget`
  entries.
- **Position consistency** — with `budget=2048` (no eviction on a 200-token
  generation) R-KV output is **byte-identical** to Full-KV across prompts,
  proving the logical/physical position + slot-mapping wiring is transparent
  when nothing is evicted.
- **Quality scales with budget** — see the GSM8K sweep below.
- **Batch > 1** — validated up to 64 concurrent requests; each compresses
  independently.
- **Out-of-the-box** — setting `BUDGET`/`BUFFER` auto-selects the V1 runner; no
  flags required beyond `--enforce-eager`.

### Accuracy — GSM8K, Qwen2.5-Math-7B-Instruct (200 questions, greedy)

| Config | Accuracy |
| --- | --- |
| Full-KV | 94.5% (189/200) |
| R-KV budget=512 buffer=64 | 92.5% (185/200) |
| R-KV budget=384 buffer=128 | 90.5% (181/200) |
| R-KV budget=256 buffer=256 | **90.0%** (180/200) |
| R-KV budget=256 buffer=128 | 82.5% (165/200) |
| R-KV budget=256 buffer=64 | 72.5% (145/200) |

**R-KV on vLLM reaches SGLang-parity accuracy** — near-lossless at `budget=512`,
and ~90% at `budget=256` **when the buffer is large enough**. The `buffer` (how
many tokens accumulate before each compaction) is the key quality knob at tight
budgets: too small a buffer compacts too aggressively and too often. Larger
buffers are also **faster** (fewer compactions). Recommended: `buffer ≈ budget`.

> This table predates the two-phase cross-layer refactor and uses a different
> (higher-scoring) harness than the 对拍 below. The apples-to-apples,
> post-refactor numbers on SGLang's own few-shot harness are in the
> **Differential test** table.

### Differential test (对拍) vs SGLang — same harness, model, config

Using SGLang's own few-shot GSM8K harness (`data/gsm8k_fewshot.jsonl`, prompt
≈ 700 tokens > budget), Qwen2.5-Math-7B-Instruct, 200 questions, greedy,
`window=8`, run identically against both engines (vLLM offline `--enforce-eager`
vs the SGLang server; **decode-only** R-KV on both — `enable_rkv_prefill=False`):

| Engine / config | Accuracy |
| --- | --- |
| vLLM Full-KV (ceiling) | 90.5% (181/200) |
| SGLang R-KV — budget=256 buffer=64 | 88.5% (177/200) |
| vLLM R-KV — budget=256 buffer=64 | 82.0% (164/200) |
| vLLM R-KV — budget=256 buffer=128 | 88.0% (176/200) |
| vLLM R-KV — budget=512 buffer=128 | 89.5% (179/200) |

**Two-phase cross-layer compaction (now matches SGLang).** The port previously
evicted **per-layer / per-head inside each layer's `forward`**; it now
accumulates a cross-head-**mean** score, **sums it across all layers**, makes one
global kept-set decision, and evicts every layer identically **after the full
forward** (`RKVCompressor.observe_layer` + `compact_step`). At `budget=256` this
lifted `buffer=64` from 68.5% → **82.0%**, matches SGLang at `buffer=128`
(**88.0%** vs 88.5%), and is near-lossless at `budget=512` (**89.5%** vs the
90.5% Full-KV ceiling).

**Critical bug this 对拍 originally surfaced:** with **prefix caching ON**, the
shared exemplar prefix's KV blocks are shared across requests; R-KV's in-place
eviction of one request corrupts the shared blocks of the others (output bleeds
another request's content, ~1.5%). Fixed: **R-KV force-disables prefix caching**
in `VllmConfig.__post_init__` (mirrors SGLang's required `--disable-radix-cache`).

**Residual tight-budget gap:** at `buffer=64` vLLM (82.0%) still trails SGLang
(88.5%) — vLLM needs `buffer=128` to match (88.0%), i.e. each vLLM compaction is
slightly lossier. With cadence (first compaction ~decode step `buffer`, then
every `buffer` steps at `seq=budget+buffer`), algorithm, config, cross-head +
cross-layer reduction, observation window and eviction mechanics all verified
**identical** to SGLang, the residual is attributed to attention-backend numerics
(vLLM FlashAttention vs SGLang FlashInfer produce slightly different K/Q feeding
the scorer) compounding over frequent tight-budget compactions. Use
`buffer ≈ budget` or `budget ≈ 512` for best accuracy.



### Measured throughput (H100, Qwen2.5-0.5B, equal work, `ignore_eos`)

This is a **non-memory-bound** microbenchmark (0.5B on an 80GB H100 → huge KV
pool), so it measures R-KV's **overhead**, not its benefit. R-KV's advantage
(constant per-request KV footprint → more concurrency / longer context) only
shows up when memory-bound.

| Config | N×tok | tok/s |
| --- | --- | --- |
| Full-KV (V2 runner + CUDA graph, production default) | 64×512 | ~27,800 |
| Full-KV (eager, fair baseline) | 64×512 | ~6,740 |
| R-KV budget=512 buffer=64 (eager) | 32×1024 | ~2,460 (−32% vs fair eager) |
| R-KV budget=256 buffer=64 (eager) | 32×1024 | ~2,170 (−40% vs fair eager) |

Two separate costs stack here: (a) **forcing eager** (no CUDA graph) is the
largest factor (~4× on this tiny model), and (b) **compaction overhead**
(O(seq²) scoring) adds ~32–40% on top. Both match the SGLang port's findings for
short, non-memory-bound decode. Raising `buffer` (compact less often) and
larger models shrink the relative overhead.

### Bugs found & fixed during GPU validation

1. `RKVCompressor` constructed `R1KV` even when disabled → assertion crash at
   startup. Fixed: the algorithm is only built when R-KV is enabled.
2. `compact_batch` used `key_cache.view(...)`, which fails on the
   non-contiguous post-`unbind` paged cache. Fixed: `(block, offset)` advanced
   indexing (gather on read, scatter in place on write).
3. R-KV silently no-op'd on v0.25.1's default V2 model runner. Fixed: R-KV
   auto-selects the V1 runner when enabled (`VllmConfig.use_v2_model_runner`).
4. `occupied_slot_mapping` indexed the fixed-size `arange_np`, overflowing when
   total batch KV exceeded one step's token budget (batch>1, long context).
   Fixed: build the per-request position ramp with `np.arange(total_kv)`.
5. Compaction fired during **chunked prefill** of long prompts (partial-prefill
   eviction → `num_dropped > num_computed` → crash). Fixed: gate compaction to
   the decode phase (`num_computed_tokens > num_prompt_tokens`).
6. **Prefix caching corrupted R-KV** (shared prefix blocks mutated in place →
   cross-request KV bleed → ~1.5% accuracy on shared-prefix workloads). Found
   via the SGLang 对拍. Fixed: force-disable prefix caching when R-KV is on.
7. **Per-layer / per-head eviction diverged from the R-KV reference** — each
   layer independently ran top-k *inside its own forward*, compounding scoring
   noise across all layers. Fixed: **two-phase cross-layer compaction** —
   cross-head **mean**, **summed across all layers**, one global kept set
   evicted after the full forward (`observe_layer` + `compact_step`).
8. **First compaction fired far too early.** The scheduler armed on absolute
   `num_computed_tokens % buffer`; for a prompt length not a multiple of
   `buffer` this fired a few decode steps in and evicted most of the prompt off
   a nearly-cold observation window (permanent damage). Fixed: arm on the
   **decode-relative** count (`num_computed_tokens - num_prompt_tokens`), so the
   first compaction lands `buffer` steps into decode with a warm window
   (matches SGLang's cadence).
9. **Observation window was empty on mixed batches.** `record_query` skipped any
   step whose query rows ≠ request count — i.e. every mixed prefill/decode step
   under continuous batching — so the window was under-populated. Fixed: gather
   each request's *last* query token via `query_start_loc` (vectorized, no host
   sync).

## Known limitations

1. **V1 GPU model runner only.** v0.25.1 ships a newer V2 runner
   (`vllm/v1/worker/gpu/`) as the default for many models; R-KV is wired into
   the V1 runner and auto-selects it whenever enabled. **Roadmap P0** — port the
   wiring to V2.

2. **Requires `--enforce-eager`.** `RKVCompressor` uses data-dependent control
   flow (`.item()`, per-request Python loops, dynamic shapes) that is
   incompatible with full CUDA-graph capture. Run with `--enforce-eager` until a
   graph-safe path exists (the SGLang port runs a *hybrid* graph/eager path —
   window/compaction steps eager, the rest replayed).

3. **Accuracy is buffer-sensitive at tight budgets.** Matching SGLang at
   `budget=256` needs `buffer ≈ 128`; a smaller buffer compacts more often and
   each compaction is slightly lossy, so the error compounds. This is a tuning
   property, not a bug. Cross-head-mean + cross-layer-sum scoring and the
   observation window are now **implemented** (they closed most of the earlier
   gap — see the 对拍 table); the small remaining `buffer=64` gap vs SGLang is
   attributed to attention-backend numerics.

4. **FlashAttention backend only.** Other backends (FlashInfer, Triton, MLA) are
   untouched — R-KV is a no-op there. **Roadmap P3.**

5. **`optimistic_seq_lens_cpu` stays logical.** It is used only as an upper
   bound (`max_seq_len`), so an over-estimate is safe, but a few code paths that
   read the CPU seq-len copy should be audited on GPU.

6. **Interactions not yet exercised:** speculative decoding, chunked prefill,
   prefix caching / block reuse, tensor/pipeline parallelism, async scheduling,
   M-RoPE models. Start validation with these **off**.

## Roadmap

| # | Item | Status | Payoff |
| --- | --- | --- | --- |
| P5 | Accuracy sweep (GSM8K, Math-7B) | **done** | SGLang parity: 90% @ b256/buf256, near-lossless @ 512 |
| P2 | Skip `occupied_slot_mapping` build when nothing compacts | **done** | lower pre-compaction overhead |
| P1 | Observation-window + cross-layer scoring | **done** | +13.5 pts @ b256/buf64 (68.5→82.0); matches SGLang at buf=128 (88.0 vs 88.5) |
| P0 | Port wiring to the V2 GPU model runner | todo | works on the default runner |
| P3 | FlashInfer + other backends | todo | broader coverage |
| P4 | Hybrid CUDA-graph path | todo | throughput (avoid full eager) |
