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
- **Quality scales with budget** — `budget=256` matches Full-KV's reasoning;
  `budget=64` stays coherent with minor artifacts (expected at an aggressive
  budget on a 0.5B model).
- **Batch > 1** — multiple concurrent requests compress independently.
- **Out-of-the-box** — setting `BUDGET`/`BUFFER` auto-selects the V1 runner; no
  flags required beyond `--enforce-eager`.

### Bugs found & fixed during GPU validation

1. `RKVCompressor` constructed `R1KV` even when disabled → assertion crash at
   startup. Fixed: the algorithm is only built when R-KV is enabled.
2. `compact_batch` used `key_cache.view(...)`, which fails on the
   non-contiguous post-`unbind` paged cache. Fixed: `(block, offset)` advanced
   indexing (gather on read, scatter in place on write).
3. R-KV silently no-op'd on v0.25.1's default V2 model runner. Fixed: R-KV
   auto-selects the V1 runner when enabled (`VllmConfig.use_v2_model_runner`).

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

3. **Per-layer independent compression.** Each attention layer evicts its own
   tokens (keeping the same *count* but possibly different *tokens* per layer).
   This matches the original PoC but is a fidelity compromise; the SGLang port's
   single cross-layer decision (cross-layer score sum, cross-head mean) is the
   principled fix. **Roadmap P1.**

4. **`occupied_slot_mapping` is rebuilt (numpy, CPU) every compaction step** and
   can be large for big batches × long contexts. Fine for correctness; a GPU
   kernel or an incremental scheme would cut overhead. **Roadmap P2.**

5. **FlashAttention backend only.** Other backends (FlashInfer, Triton, MLA) are
   untouched — R-KV is a no-op there. **Roadmap P3.**

6. **`optimistic_seq_lens_cpu` stays logical.** It is used only as an upper
   bound (`max_seq_len`), so an over-estimate is safe, but a few code paths that
   read the CPU seq-len copy should be audited on GPU.

7. **Interactions not yet exercised:** speculative decoding, chunked prefill,
   prefix caching / block reuse, tensor/pipeline parallelism, async scheduling,
   M-RoPE models. Start validation with these **off**.

## Roadmap

| # | Item | Payoff |
| --- | --- | --- |
| P0 | Port wiring to the V2 GPU model runner | works on the default runner |
| P1 | Single cross-layer eviction decision | correctness/fidelity parity with SGLang |
| P2 | GPU / incremental `occupied_slot_mapping` | lower per-compaction overhead |
| P3 | FlashInfer + other backends | broader coverage |
| P4 | Hybrid CUDA-graph path | throughput (avoid full eager) |
| P5 | Accuracy sweep (GSM8K / MATH, 7B) | quantify the budget/quality curve |
