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
| **vLLM R-KV — budget=256 buffer=64** | **87.5% (175/200)** |
| vLLM R-KV — budget=256 buffer=128 | 90.5% (181/200) |

**vLLM R-KV now matches SGLang at every budget/buffer** (buffer=64: 87.5% vs
88.5% — a 1-question difference, within n=200 noise; buffer=128: 90.5% =
Full-KV, lossless). Both engines degrade only ~2–3 points from their identical
90.5% Full-KV ceiling.

Two fixes closed the original 68.5% → 87.5% gap at `budget=256 buffer=64`:

**1. Two-phase cross-layer compaction (68.5% → 82.0%).** The port previously
evicted **per-layer / per-head inside each layer's `forward`**; it now
accumulates a cross-head-**mean** score, **sums it across all layers**, makes one
global kept-set decision, and evicts every layer identically **after the full
forward** (`RKVCompressor.observe_layer` + `compact_step`).

**2. Async-scheduling cache corruption (82.0% → 87.5%).** R-KV compaction runs
inside the forward and reports each request's evicted-token count as a model
output (`num_dropped_tokens`) that the scheduler must apply **before** the next
step's *physical* KV positions are computed. vLLM V1's **async scheduling**
prepares step N+1 before step N's output is processed, so for one step after a
compaction the dropped-token count is stale and the next decode token overwrites
a surviving KV slot. The corruption is **batch-dependent** — it only appears
under the memory pressure of many concurrent requests (a prompt correct in
isolation and at N≤40 would run to the token cap producing garbage at N=100),
and it compounds with the number of compactions, so it hit long generations and
tight buffers hardest (explaining the earlier apparent "buffer sensitivity").
Fixed: **R-KV force-disables async scheduling** in `VllmConfig.__post_init__`
(alongside prefix caching), so the eviction accounting is always current. This
alone lifted `buffer=64` from 82.0% → **87.5%** and restored `buffer=128` to the
lossless **90.5%**.

**Prefix-caching bug (earlier):** with **prefix caching ON**, the shared
exemplar prefix's KV blocks are shared across requests; R-KV's in-place eviction
of one request corrupts the shared blocks of the others (output bleeds another
request's content). Fixed: **R-KV force-disables prefix caching** in
`VllmConfig.__post_init__` (mirrors SGLang's required `--disable-radix-cache`).

With these fixes the port is byte-consistent per-prompt regardless of batch
composition, and matches SGLang to within noise. The final ~1-point spread at
`buffer=64` is attention-backend numerics (vLLM FlashAttention vs SGLang
FlashInfer produce slightly different K/Q feeding the scorer) amplified by
R-KV's discrete top-k selection near the score cutoff — vLLM's post-RoPE keys
match the HuggingFace reference bit-closely. Use `buffer ≈ budget` for best
accuracy.



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

### Batched cross-layer scoring (ported from SGLang)

The compaction score is O(seq²) per layer (the redundancy cosine matrix) and
dominates R-KV's overhead. The default `score_mode="batched"` (env
`VLLM_V1_R_KV_SCORE_MODE`) defers scoring out of the per-layer `forward` and runs
it as a **single GEMM over all layers** at compaction time (`num_layers` as the
batch dim, chunked so the transient cosine matrix stays under
`VLLM_V1_R_KV_SCORE_CHUNK_MB`), instead of 28 separate bsz=1 scoring calls. The
batch GEMM computes independent per-layer results, so the kept set is **identical**
to the per-layer `score_mode="reference"` path — verified: both score 87.5%
(175/200) at `budget=256 buffer=64`, with the **same 508 compactions** — while
cutting kernel launches.

Measured (b256/buf64, Qwen2.5-Math-7B, 200q, H100): **1987 vs 1622 decode tok/s
(+22.5%)**, wall 24.4s → 19.9s, matching the SGLang port's ~+23% batched-compaction
win. The restructure stays entirely inside `rkv/integration.py` (the per-layer
window queries ride along in the already-shared `rkv_layer_caches` list), so the
wiring patch is untouched.

### Throughput profile — where the time goes (next optimization)

Measured decomposition under PIECEWISE cudagraph (b256, Qwen2.5-Math-7B, 200q,
H100 decode tok/s):

| config | tok/s | isolates |
| --- | --- | --- |
| Full-KV FULL cudagraph | 8317 | ceiling |
| Full-KV PIECEWISE (attention eager) | 8054 | **graphing attention = +3% only** |
| R-KV buf512 (rare compaction) | 2784 | **R-KV per-step hooks = −65%** |
| ↳ `record_query` skipped | **4267** | **`record_query` alone = +53%** |
| R-KV buf16 (frequent compaction) | 2103 | compaction cost = −24% |

> `record_query` skipped = the +53% ceiling that motivated **P4a below**, now
> largely recovered by the ring-buffer rewrite (buf64 PIECEWISE 2623 → **3880**).

Two conclusions redirect the roadmap:

1. **`record_query` is the dominant cost, not compaction scoring.** It runs
   28×/step (once per attention layer) with a **per-request Python loop that does
   a tiny GPU `copy_` per request** — thousands of micro-kernel-launches per step
   for a large batch. Skipping it recovers +53% at `buffer=512`.
2. **Graphing attention buys only ~3%** here, so a full SGLang-style hybrid that
   FULL-captures attention (roadmap P4b) is *low* value on this workload.

**Implemented (roadmap P4a): vectorized `record_query`.** Mirrors the SGLang
port — the per-request dict + per-request GPU copies are replaced with a single
**`index_copy_` scatter** into a **fixed-address** ring buffer
`(window, max_slots, q_heads, head_dim)` (one ring per layer), indexed by a
persistent per-request column (`_slot`, freed + reused on finish) and a
per-request step counter (cursor = `count % window`). One scatter per layer
per step replaces the thousands of micro-launches; the score means over the
window so the ring needs no un-rotation. The ring grows (realloc + copy) only on
a new concurrency peak, so it is amortized after the first prefill wave.

Measured (b256, 200q, H100 decode tok/s; accuracy **unchanged** — eager buf64 =
89.0% = 178/200, exactly matching pre-change):

| config | before | after | Δ |
| --- | --- | --- | --- |
| buf64 eager | 2122 | **3030** | **+43%** |
| buf64 PIECEWISE | 2623 | **3880** | **+48%** |
| buf16 PIECEWISE | 2103 | 2807 | +33% |
| buf128 PIECEWISE | 2705 | 3747 | +39% |
| buf256 PIECEWISE | 2686 | 3611 | +34% |

Note: SGLang's query recording is *not* a Triton kernel — it is exactly this
vectorized `index_copy_` (`_rolling_q_flat[layer].index_copy_(0, index, q)`); its
fused **Triton kernel is for the redundancy cosine-similarity in the *scoring***
(`cal_similarity`), a different hot path already handled here by batched scoring.

**Remaining P4a headroom** (not yet done): the per-layer version repeats the
Python slot loop + the small H2D of the flat index **28×/step** (once per layer,
identical each time). Centralizing it — compute the ring index **once per step**
in the runner and pass it via `attn_metadata` (matching SGLang's single
`rolling_q_flat`) — removes that redundancy and makes recording graph-capturable,
which unlocks FULL cudagraph. A cheap complementary win is to **gate recording to
the observation window** — only the last `window` (8) steps before each
compaction need queries, so record `window / buffer` of steps (8× fewer at buf64,
64× fewer at buf512).

### Decode CUDA graph (PIECEWISE) — implemented

R-KV runs under vLLM's PIECEWISE cudagraph by default (auto-selected in
`VllmConfig.__post_init__`; no `--enforce-eager`). Attention stays eager so the
R-KV hooks fire; the rest of the decode layer is graphed. Same accuracy as eager
(within n=200 noise) at **+30–40% decode throughput** on R-KV configs and **+62%**
on Full-KV. See Known limitations #2.

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
10. **Async scheduling corrupted the post-compaction KV (batch-dependent).**
    Compaction reports its evicted-token count as a model output
    (`num_dropped_tokens`) that the scheduler must apply before the next step's
    *physical* KV positions are computed; vLLM V1 async scheduling prepares step
    N+1 before step N's output is processed, so the count was stale for one step
    after every compaction and the next decode token overwrote a surviving KV
    slot. Only surfaced under many concurrent requests (a prompt correct in
    isolation ran to the token cap producing garbage in a 100-request batch) and
    compounded with the number of compactions, so it hit long generations /
    tight buffers hardest. Fixed: **force-disable async scheduling when R-KV is
    on** (`VllmConfig.__post_init__`). Lifted `budget=256 buffer=64` from
    82.0% → **87.5%** (SGLang: 88.5%) and restored `buffer=128` to lossless
    90.5%.
11. **Preemption left the dropped-token counter stale.** `_preempt_request`
    resets `num_computed_tokens = 0` (a preempted request recomputes from
    scratch) but left `num_dropped_tokens` untouched, so on resume the physical
    position (`logical − dropped`) would go **negative** and corrupt the paged
    KV. Fixed: reset `num_dropped_tokens = 0` alongside `num_computed_tokens`.
    Defensive — preemption does not fire on the GSM8K sweep (no accuracy change),
    but the reset is required for correctness under high-concurrency / memory
    pressure. Found while investigating the `buffer=16` gap (see below).
12. **`mix_lambda` default mismatched the reference (the tight-buffer gap).** The
    joint score is `mix_lambda·importance − (1−mix_lambda)·redundancy`. vLLM
    defaulted to **0.07** (the R-KV *algorithm class* default), but SGLang's
    runtime `RKVConfig` and the reference HF eval scripts use **0.1**. The
    importance/redundancy imbalance shifts the kept set, and the error
    **compounds with compaction frequency** — so it hit `buffer=16` (≈12
    compactions/request) far harder than `buffer=64` (≈3). This was the main
    driver of the vLLM-vs-SGLang *buffer sensitivity*. Fixed: default
    `mix_lambda = 0.1` (matches SGLang). Effect at `budget=256`: buf16
    83.0→**85.5**, buf64 87.5→**89.0** (now ≥ SGLang's 88.5), buf128
    90.5→**91.5** (> SGLang's 89.0). (My earlier "0.07 is better" reading was
    confounded by the pre-fix async corruption, bug #10.)

## Tight-buffer (`buffer=16`) gap — root-caused to `mix_lambda`, residual numerics

At `budget=256 buffer=16` vLLM originally scored **83.0%** vs SGLang's **88.0%**,
and unlike SGLang was *buffer-sensitive* (buf64 87.5% → buf16 83.0%, vs SGLang's
flat 88.5% → 88.0%). A full step-by-step differential (as for bug #10) traced
most of that asymmetry to the **`mix_lambda` config mismatch** (bug #12):

- Cadence, single-prompt correctness, physical positions, preemption, slot
  collisions and the algorithm were all verified **identical/correct**; two
  identical batch runs are byte-for-byte deterministic (not a race).
- The remaining lever was the R-KV **algorithm parameters**. Comparing them
  against SGLang's runtime config exposed `mix_lambda` (0.07 vs 0.1). A clean
  sweep confirmed it: at `mix_lambda=0.1`, buf64 matches SGLang and buf16 gains
  +2.5.

After the fix, `budget=256` `buffer≥64` **meets or beats SGLang**; only `buf16`
still trails by ~2.5 pts (85.5% vs 88.0%). That residual is a genuine
borderline-top-k sensitivity to the FlashAttention-vs-FlashInfer K/Q numerics,
amplified ~4× by buf16's compaction frequency (a `mix_lambda` of 0.12 reaches
89% at buf16 but that overfits and lowers buf64 — so 0.1, the SGLang/HF value, is
the principled default). **Use `buffer ≥ 64`** (`buffer=128` is lossless).

## Known limitations

1. **V1 GPU model runner only.** v0.25.1 ships a newer V2 runner
   (`vllm/v1/worker/gpu/`) as the default for many models; R-KV is wired into
   the V1 runner and auto-selects it whenever enabled. **Roadmap P0** — port the
   wiring to V2.

2. **Decode CUDA graphs: PIECEWISE, auto-selected (no `--enforce-eager` needed).**
   R-KV runs its scoring/eviction hooks *inside* the attention forward (eager)
   plus a post-forward compaction. vLLM's default FULL / FULL_AND_PIECEWISE
   cudagraph **captures** the decode attention, so those Python hooks run only
   at capture → R-KV silently no-ops (runs as Full-KV). vLLM's **PIECEWISE**
   cudagraph keeps attention eager (a graph *splitting* op), so the hooks fire
   every decode step while the rest of the layer is graphed. R-KV therefore
   **force-selects PIECEWISE** in `VllmConfig.__post_init__` when enabled and
   cudagraphs are on (a no-op under `--enforce-eager`). Measured: **same accuracy
   as eager** (within n=200 noise) at **+30–40% decode throughput** (Full-KV:
   +62%). A full SGLang-style hybrid that *also* graphs attention on
   non-compaction steps (in-graph fixed-address query buffer + eager dispatch on
   compaction steps) would recover the remaining attention-eager cost —
   **Roadmap P4b**.

3. **Tight-buffer accuracy dip.** At `budget=256`, `buffer=64` = 89.0% and
   `buffer=128` = 91.5% (≥ SGLang); only `buffer=16` (85.5%) trails SGLang
   (88.0%) by ~2.5 pts — residual FlashAttention-vs-FlashInfer K/Q numerics near
   the top-k cutoff, amplified by buf16's ~12 compactions/request (bug #12 fixed
   the larger part via `mix_lambda`). Recommended: `buffer ≥ 64`.

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
| P6 | Batched cross-layer scoring (SGLang parity) | **done** | +22.5% decode tok/s @ b256/buf64 (1622→1987), identical accuracy |
| P0 | Port wiring to the V2 GPU model runner | todo | works on the default runner |
| P3 | FlashInfer + other backends | todo | broader coverage |
| P4 | Decode CUDA graph (PIECEWISE, auto-selected) | **done** | +30–40% decode tok/s (Full-KV +62%), same accuracy; no `--enforce-eager` |
| **P4a** | **Vectorize `record_query`** (fixed-address ring + single `index_copy_` per layer) | **done** | **+43–48% decode tok/s @ b256/buf64 (2623→3880 PIECEWISE), accuracy unchanged** |
| P4a′ | Centralize `record_query` (compute ring index once/step in the runner; gate to observation window) | todo | removes the 28×/step redundant Python+H2D; unlocks FULL cudagraph |
| P4b | Full hybrid graph (also graph attention, eager only on compaction) | todo (low value here: attention-graph = +3%) | recover the attention-eager cost |
| P7 | Memory-bound benchmark (long context / high concurrency) | todo | demonstrate R-KV's *benefit* (constant KV footprint), not just its overhead |
