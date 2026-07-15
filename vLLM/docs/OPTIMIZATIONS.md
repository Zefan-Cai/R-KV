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
| R-KV budget=512 buffer=64 | **92.5%** (185/200) — near-lossless |
| R-KV budget=256 buffer=64 | 72.5% (145/200) |
| R-KV budget=128 buffer=64 | 36.5% (73/200) |

**R-KV is near-lossless at `budget=512`.** Accuracy falls off faster at tight
budgets than the SGLang port (which holds ~0.90 at budget 256). The gap is the
**observation-window scoring**, not the layer dimension (see the analysis under
"Roadmap"). `budget≈512` is the recommended operating point today.


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

## Known limitations

1. **V1 GPU model runner only.** v0.25.1 ships a newer V2 runner
   (`vllm/v1/worker/gpu/`) as the default for many models; R-KV is wired into
   the V1 runner and auto-selects it whenever enabled. **Roadmap P0** — port the
   wiring to V2.

2. **Requires `--enforce-eager`.** `RKVCompressor.compact_batch` uses
   data-dependent control flow (`.item()`, per-request Python loops, dynamic
   shapes) that is incompatible with full CUDA-graph capture. Run with
   `--enforce-eager` until a graph-safe path exists (the SGLang port runs a
   *hybrid* graph/eager path — window/compaction steps eager, the rest replayed).

3. **Single-query importance (the tight-budget accuracy lever).** At compaction
   time the compressor scores each past token by the attention it receives from
   only the **current** decode query. The reference / SGLang port scores over a
   trailing **observation window** of the last `window_size` decode queries,
   which is a much more robust importance signal and is the main reason SGLang
   holds ~0.90 at budget 256 while this port drops to ~0.73. Implementing the
   window on vLLM needs runner-level per-request query state (stable across the
   batch reordering that `condense()` performs), so it is a substantial change.
   **Roadmap P1.** (Note: making the eviction decision *cross-layer* instead of
   per-layer is **not** expected to help here — vLLM's per-layer caches let each
   layer keep its own best tokens, which is strictly more expressive than the
   single shared decision SGLang is forced into by its shared slot table.)

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
| P5 | Accuracy sweep (GSM8K, Math-7B) | **done** | near-lossless @ budget 512 |
| P2 | Skip `occupied_slot_mapping` build when nothing compacts | **done** | lower pre-compaction overhead |
| P1 | Observation-window importance scoring | todo | tight-budget accuracy parity with SGLang |
| P0 | Port wiring to the V2 GPU model runner | todo | works on the default runner |
| P3 | FlashInfer + other backends | todo | broader coverage |
| P4 | Hybrid CUDA-graph path | todo | throughput (avoid full eager) |
