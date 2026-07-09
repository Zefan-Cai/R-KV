# R-KV on FlashInfer — Design

This backend is a **complete rewrite** of the private `RKV-HS` prototype into a clean,
self-contained FlashInfer serving harness for R-KV decode-time KV-cache compression.
It is *not* a port of RKV-HS code: RKV-HS used flash-attn's `flash_attn_with_kvcache`
over a contiguous static cache and used FlashInfer only for element-wise kernels; it also
carried a silent scoring bug (advanced-indexing `.zero_()` no-op in the redundancy
"retain most-recent duplicate" exemption) that made bit-parity with the reference
impossible. Here the algorithm is ported faithfully from the repo-root reference
(`rkv/compression/r1_kv.py` + `rkv/utils.py`, same port shape as `SGLang/rkv/algo.py`),
and the attention/KV path is built on FlashInfer's paged-KV wrappers.

## 1. Scope

- Single GPU, single process. Plain data-parallel sharding across GPUs is done by the
  benchmark scripts (one process per GPU, disjoint data shards). TP is **not supported**.
- Eager execution only (no CUDA graphs, no torch.compile) — same constraint set as the
  SGLang / Nano-vLLM / Mini-SGLang backends. Graph capture is an explicit phase-2 item.
- Static request batch per `generate()` call; finished rows are dropped from planning
  (no wasted decode compute — fixes RKV-HS's finished-rows-keep-spinning behavior).
- Decode-time compression **plus** HF-reference-style prefill compression (see §5.4),
  so long prompts fit the fixed-size R-KV region and the query window is seeded exactly
  like the HuggingFace accuracy-reference implementation.

## 2. Package layout (target: `R-KV/FlashInfer/`)

```
FlashInfer/
├── README.md                   # house template (tagline, layout tree, quick start,
│                               # pinned env table, config reference, ✅/❌ table)
├── requirements-rkv.txt        # pinned verified stack (see §9)
├── rkv/
│   ├── __init__.py             # exports RKVConfig, R1KV, FlashInferEngine, GenOutput
│   ├── config.py               # RKVConfig frozen dataclass + validation
│   ├── algo.py                 # faithful port of reference scoring/selection (parity-tested)
│   ├── compressor.py           # R1KV: per-batch window-query state + compaction driver
│   ├── models.py               # Llama / Qwen2 / Qwen3 forward on FlashInfer kernels
│   ├── engine.py               # FlashInferEngine: paged pool, prefill/decode, generate()
│   └── loader.py               # safetensors weight loading (no torch full-model 2x copy)
├── examples/
│   ├── example.py              # FullKV baseline decode
│   └── example_rkv.py          # same prompt with RKVConfig enabled
├── benchmark/
│   ├── README.md
│   ├── bench_rkv.py            # decode throughput + peak memory, R-KV on/off matrix
│   ├── eval_math.py            # GSM8K / AIME24 / MATH-500 accuracy harness
│   └── RESULTS_*.md            # dated, hardware-tagged (filled from GPU validation)
├── docs/
│   ├── DESIGN.md               # this file
│   └── IMPLEMENTATION.md       # code map + bring-up log (filled during GPU validation)
└── tests/
    ├── test_rkv_algo.py            # CPU-only unit tests, __main__ runner
    └── test_cross_repo_parity.py   # bit-parity vs ../../rkv/compression/r1_kv.py
```

Companion (repo conventions): `R-KV/tests/smoke/flashinfer_smoke.py` + row in
`tests/smoke/README.md`; CPU tests wired into `.github/workflows/cpu-tests.yml`;
raw artifacts under `results/validation-<date>-<gpu>/flashinfer_*/`.

## 3. Configuration — `rkv/config.py`

```python
@dataclass(frozen=True)
class RKVConfig:
    budget: int = 1024          # tokens kept per layer/head after compaction
    buffer: int = 128           # extra slots; compaction fires when len == budget + buffer
    window_size: int = 8        # trailing observation-window queries used for scoring
    kernel_size: int = 7        # 1-D max-pool over attention scores
    mix_lambda: float = 0.1     # score = λ·attention + (1-λ)·(-redundancy)  [CLI convention;
                                # the reference class default is 0.07 — documented, same as SGLang]
    retain_ratio: float = 0.1   # only used by *_percent modes of the reference; kept for parity
    retain_direction: str = "last"
    compress_prefill: bool = True   # HF-reference prefill compression (§5.4)
```

Validation (raise `ValueError` in `__post_init__`): `budget > window_size`,
`buffer >= window_size` (first decode compaction must have a full real query window
when `compress_prefill` seeds it; and non-negative), `kernel_size` odd,
`0 < mix_lambda <= 1` (`1.0` = attention-only scoring, i.e. SnapKV-style — replaces
RKV-HS's separate 550-line `SnapKV` class), `retain_direction in {"last", "first"}`.
Similarity threshold stays hardcoded at 0.5 inside the algorithm (house convention).

## 4. Algorithm — `rkv/algo.py`

Faithful port of the repo-root reference; same function names and semantics as
`SGLang/rkv/algo.py` so `tests/test_cross_repo_parity.py` (copied nearly verbatim from
`SGLang/tests/test_cross_repo_parity.py`) passes with `torch.equal`:

- `compute_attention_scores(window_q, k, ...)`: GQA group **max** pooling, divide by
  sqrt(head_dim), softmax in fp32 → cast back, mean over window, `max_pool1d(kernel, pad=k//2, stride=1)`.
- `cal_similarity(k, threshold=0.5)`: cosine similarity over normalized keys, diagonal
  masked, **`scatter_`-based** exemption of each token's most-recent near-duplicate
  (the semantics RKV-HS silently lost), mean over rows, softmax.
- `select_indices(scores, budget, window_size, sort=True)`: top-(budget−window) over past
  tokens, ascending sort, then append trailing window indices.
- `update_kv(...)`: gather kept K/V — kept in the exact reference order so `torch.equal`
  parity holds.

No `flashinfer` import in this module (CPU CI runs it).

## 5. Engine — `rkv/engine.py`

### 5.1 Paged KV pool

- `page_size = 1` (slot == token). Layout `NHD`:
  `kv_pool: [num_layers, max_slots, 2, 1, num_kv_heads, head_dim]`, bf16 default —
  indexed per layer when calling wrapper `.run(q, kv_pool[layer])`.
- Slot allocation is **static per request region**:
  - FullKV mode: region size = `max_seq_len` per request.
  - R-KV mode: region size = `budget + buffer` per request. This is where the memory
    saving physically comes from.
  - Request r's slots are `[r*region, r*region + phys_len_r)`; `kv_indices` is just
    `arange` over the region prefix — rebuilt each step (cheap, eager).
- Logical vs physical length split (all backends converge on this):
  - `logical_len[r]`: total tokens ever seen — drives RoPE positions; never shrinks.
  - `phys_len[r]`: slots in use — drives `kv_indptr`/`last_page_len`; collapses to
    `budget` at each compaction. Keys are rotated **once** at their logical position and
    never re-rotated after compaction.

### 5.2 Per-step decode flow

One `plan()` per step (not per layer) on `BatchDecodeWithPagedKVCacheWrapper`
(`float workspace 128MB`, `"NHD"`), shared by all layers — valid because every layer keeps
the same count. `pos_encoding_mode="NONE"` (we pre-rotate q/k). Plan inputs are built
cheaply: `kv_indptr` / `last_page_len` stay **CPU** tensors (flashinfer 0.6.x `plan()`
reads them host-side via `.to("cpu")`, so device tensors would force a per-step D2H
sync), `kv_indices` are slices of one precomputed device arange, and the per-step
scalars (slots / rows / input ids / positions) ride a single staged H2D transfer.

Per layer: rmsnorm → **fused** qkv proj (one GEMM; split + contiguous copies, since the
flashinfer rope/attention kernels are only known-safe on packed layouts) →
`flashinfer.apply_rope_with_cos_sin_cache_inplace(positions=logical_pos, query, key,
head_size, cos_sin_cache, is_neox=True)` → write k/v into slot `phys_len` of each active
request's region (direct indexing; page_size=1) → push post-RoPE q into the rolling
window-query cache (R-KV mode; circular slot write, no shift) →
`decode_wrapper.run(q, kv_pool[layer])` → o proj → residual/MLP (`fused_add_rmsnorm`,
fused gate_up GEMM feeding `silu_and_mul` directly — no concat).

After the step: `phys_len += 1`, `logical_len += 1`; sample next token
(`flashinfer.sampling.top_p_sampling_from_probs`, greedy if `temperature == 0`);
mark EOS/max-len rows finished; **compaction check** (§5.3); re-plan next step over
active rows only.

### 5.3 Decode compaction (R-KV mode)

Trigger, per request independently (SGLang "method A"): after append,
`phys_len[r] == budget + buffer` → for every layer: score with that request's window
queries (post-RoPE, rolling buffer `[num_layers, bsz, num_q_heads, window, head_dim]`,
stored circularly and rotated back to temporal order at compaction — a pure
permutation, so scoring is bit-identical to a shifted window),
`select_indices`, gather kept K/V **per kv-head** (each head keeps its own token set —
reference-faithful; slot i then holds different logical tokens per layer/head, which is
fine because only counts must agree across layers) to the front of the region;
`phys_len[r] = budget`. Not captured in any graph; runs eagerly on the main stream —
correctness first, the overlap engineering of RKV-HS (dual threads, busy-wait
`layer_flags`, graph-captured eviction) is deliberately dropped (phase 2).

All (layer, row) pairs of a trigger step batch into few `update_kv` calls
(`compressor.compact_batch`, chunked to keep the `[pairs, kv_heads, S, S]` scoring
transients under ~512MB, ≤32 pairs — peak memory is a headline metric, so launch
amortization must not eat the pool savings) instead of `num_layers × rows`
sequential bsz=1 calls. Batched GEMMs
are not provably bit-identical to bsz=1 calls, so the first compaction A/B-checks
batched vs per-pair on the real pool data (`torch.equal`) and permanently falls back to
per-pair if they differ; the per-(layer, request) `compact()` surface stays for tests.

### 5.4 Prefill

- Attention: `BatchPrefillWithRaggedKVCacheWrapper` (causal) over ragged q/k/v — the
  prompt KV does not need to be paged during prefill. RoPE applied in-place at logical
  positions 0..L-1 first.
- After attention, per request: if `compress_prefill` and `prompt_len > budget`:
  run reference `update_kv` on the prompt K/V using the **last `window_size` prompt
  queries** (exactly `HuggingFace/rkv/modeling.py` behavior), write the kept `budget`
  tokens into the region, `phys_len = budget`. Else write all prompt tokens.
- Window-query cache seeded with the last `window_size` prompt queries (HF behavior)
  in R-KV mode, regardless of whether prefill compression fired.
- `logical_len = prompt_len`. No padding tokens anywhere (ragged prefill) — fixes
  RKV-HS's zero-padded static batches.

### 5.5 Public API

```python
engine = FlashInferEngine(
    model_path,                  # HF repo dir (config.json + safetensors)
    max_batch_size=8,
    max_seq_len=32768,           # logical cap (fullkv region size; rkv logical stop)
    rkv=None | RKVConfig(...),   # None → FullKV baseline
    dtype=torch.bfloat16, device="cuda",
)
outs = engine.generate(
    prompts,                     # list[list[int]] token ids (caller tokenizes)
    max_new_tokens, temperature=0.6, top_p=0.95, stop_token_ids=(eos,),
    seed=42,
)  # -> list[GenOutput]

@dataclass
class GenOutput:
    token_ids: list[int]
    num_prefill_tokens: int
    num_output_tokens: int
    num_compactions: int
    finish_reason: str           # "stop" | "length"
```

`engine.stats` additionally exposes wall-clock prefill/decode seconds and
`torch.cuda.max_memory_allocated()` snapshots for the benchmark scripts.
`compaction_seconds` is attributed with CUDA event pairs read after the final
sync (GPU time of the compaction kernels), so compaction rounds no longer
bracket the decode pipeline with two host syncs each.

## 6. Models — `rkv/models.py` + `rkv/loader.py`

- Families: **Llama** (`LlamaForCausalLM`), **Qwen2** (`Qwen2ForCausalLM`, qkv bias),
  **Qwen3** (`Qwen3ForCausalLM`, per-head q/k rmsnorm, `head_dim` from config). Family
  detected from `config.architectures[0]`; `AutoConfig`/`AutoTokenizer` are the only
  transformers surfaces used.
- `head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)` —
  RKV-HS's hardcoded `hidden // n_heads` breaks Qwen3; do not repeat it.
- RoPE cos/sin cache built fp32 from `rope_theta` (+ scaling config passthrough for
  Llama-3-style `rope_scaling` when present), laid out as FlashInfer's
  `cos_sin_cache` `[max_pos, head_dim]` (cos ‖ sin halves), `is_neox=True`.
- Kernels: `flashinfer.norm.rmsnorm` / `fused_add_rmsnorm`, `flashinfer.activation.silu_and_mul`;
  projections are plain `F.linear`. `tie_word_embeddings` honored.
- q/k/v and gate/up are **fused GEMMs** (`qkv_proj`, `gate_up_proj`, Nano-vLLM-style);
  `CausalLM.packed_map` declares the checkpoint-name → (fused param, row offset)
  mapping the loader uses. Fusing changes GEMM shapes, so logits may differ from the
  split-projection build at floating-point rounding level (algo scoring parity is
  unaffected — it never consumes attention outputs).
- `loader.py` streams safetensors shards straight into pre-allocated parameters
  (nanovllm-style), no intermediate full-model materialization; `packed_map` rows are
  scattered into their fused parameter slices, and a fused parameter counts as loaded
  only when all of its sources landed.

## 7. Tests

- `tests/test_rkv_algo.py` (CPU): config validation; score shapes/dtypes; scatter
  exemption actually modifies the similarity matrix (regression for the RKV-HS bug);
  selection keeps trailing window; below-budget no-op; GQA layouts (Qwen2.5-style
  n_q=28/n_kv=4 etc.); bf16 + fp32.
- `tests/test_cross_repo_parity.py` (CPU): near-verbatim copy of the SGLang one,
  pointed at `../../rkv/compression/r1_kv.py`, `torch.equal` bit-parity on
  `compute_attention_scores` / `cal_similarity` / `select_indices` / `update_kv`,
  seeds `1234+i`, class-default hyperparams (`mix_lambda=0.07`, `retain_ratio=0.1`).
- Both: `importlib` by file path, **no flashinfer import**, `__main__` runner printing
  `all tests passed`; wired into `.github/workflows/cpu-tests.yml` (compileall excludes
  `engine.py`/`models.py`? No — compileall is syntax-only, include everything; only the
  *executed* tests must avoid importing flashinfer).
- `R-KV/tests/smoke/flashinfer_smoke.py` (GPU): `RKV_ON=1/0`, `RKV_SMOKE_MODEL`
  override (default Qwen3-0.6B), a few hundred sampled tokens at temp 0.6,
  `health_check.report()`, and **fail if compression never fired** when `RKV_ON=1`
  (Mini-SGLang convention).

## 8. GPU validation plan (H100 node)

1. CPU tests on-node (sanity).
2. FullKV correctness vs HF transformers: same model/prompt, greedy; assert prefill
   logits `max|Δ|` small and long common token prefix.
3. R-KV behavior: compaction count matches trigger schedule; phys_len bounded;
   health-checked outputs.
4. Accuracy matrix (`eval_math.py`): DeepSeek-R1-Distill-Qwen-1.5B + 7B ×
   {fullkv, rkv-512, rkv-1024, rkv-2048} × {GSM8K n≥100, MATH-500 n≥100, AIME24 n=30}.
5. Perf matrix (`bench_rkv.py`): same models × batch {1, 8, 32} × gen-len {2k, 8k} ×
   {fullkv, rkv-1024}: decode tok/s, peak allocated GiB, compaction overhead %.
6. Qwen3 + Llama family smoke (Qwen3-0.6B/8B, Llama-3.2-1B/3.1-8B as available).

## 9. Pinned environment (validated on H100 — see IMPLEMENTATION.md §8)

Python 3.12, `torch==2.10.0` (cu128; `flashinfer-python==0.6.12` requires
torch<2.11), `flashinfer-python==0.6.12`, `flashinfer-cubin==0.6.12`,
`transformers==5.8.1` (config/tokenizer only), `safetensors`, `datasets`
(eval data comes from committed jsonl — no hub dependency).

## 10. Explicitly dropped from RKV-HS (and why)

| RKV-HS behavior | Status here |
|---|---|
| flash-attn `flash_attn_with_kvcache` attention | replaced by FlashInfer paged wrappers |
| decode CUDA graph + graph-captured eviction + dual threads/`layer_flags` busy-wait | dropped (eager; phase-2) |
| advanced-indexing `.zero_()` retain exemption (silent no-op) | fixed via reference `scatter_` |
| `cache_seqlens` undefined attribute crash in prefill-compress branch | prefill compression reimplemented (§5.4) |
| zero-padded static batches, `len(prompts) % bsz == 0` assert | ragged prefill, active-row planning |
| finished rows keep replaying the decode graph | finished rows dropped from plan |
| hand-rolled DeepSeek/hybrid chat templates | `tokenizer.apply_chat_template` in harnesses |
| `mix_alpha` name | `mix_lambda` (house vocabulary) |
| SnapKV as a separate 550-line class copy | `mix_lambda=1.0`≈attention-only; not a separate class |
