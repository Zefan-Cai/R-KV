# R-KV on FlashInfer

**Decoding-time, redundancy-aware KV-cache compression on a self-contained
FlashInfer paged-attention engine.**

While a model generates a long output, R-KV periodically evicts the
**unimportant** and **redundant** past tokens, keeping only a fixed `budget` of
KV entries per layer/head per request — freeing GPU memory while preserving
generation quality. Unlike the SGLang / vLLM backends in this repo, this
directory is not a patch on an upstream serving framework: it is a **standalone
engine** (`rkv/engine.py`) built directly on FlashInfer's paged-KV wrappers,
small enough to read end to end.

## What this backend is

A **complete rewrite** of the private `RKV-HS` prototype (design reference
only; no code reused). The attention/KV path moves from flash-attn's
contiguous static cache to FlashInfer's paged wrappers
(`BatchPrefillWithRaggedKVCacheWrapper` for prefill,
`BatchDecodeWithPagedKVCacheWrapper` for decode, `page_size=1`), and the
algorithm is re-ported faithfully from this repo's reference
(`rkv/compression/r1_kv.py` + `rkv/utils.py`, same port shape as
`SGLang/rkv/algo.py`, bit-parity tested).

Concrete RKV-HS defects fixed in the rewrite (full table:
[`docs/DESIGN.md`](docs/DESIGN.md) §10):

- **Retain-exemption silent no-op** — the redundancy scoring's "retain the
  most-recent near-duplicate" exemption used advanced-indexing `.zero_()`,
  which zeroes a copy and changes nothing; the reference `scatter_` semantics
  are restored (and regression-tested).
- **Undefined `cache_seqlens` crash** — the prefill-compression branch read an
  attribute that was never set; prefill compression is reimplemented from the
  HuggingFace reference behavior ([`docs/DESIGN.md`](docs/DESIGN.md) §5.4).
- **Qwen3 `head_dim` breakage** — `hidden_size // num_attention_heads` was
  hardcoded; `config.head_dim` is honored when present.
- Also dropped: zero-padded static batches (now ragged prefill), finished rows
  replaying the decode graph (now dropped from planning), and the dual-thread /
  busy-wait / graph-captured eviction machinery (eager correctness first;
  overlap is phase 2).

```
FlashInfer/
├── README.md                   # you are here
├── requirements-rkv.txt        # pinned, verified dependency stack
├── rkv/                        # the engine package
│   ├── config.py               # RKVConfig frozen dataclass + validation
│   ├── algo.py                 # faithful port of the reference scoring/selection
│   ├── compressor.py           # R1KV: window-query state + compaction driver
│   ├── models.py               # Llama / Qwen2 / Qwen3 on FlashInfer kernels
│   ├── engine.py               # FlashInferEngine: paged pool, prefill/decode, generate()
│   └── loader.py               # safetensors streaming weight loader
├── examples/
│   ├── example.py              # FullKV baseline decode
│   └── example_rkv.py          # same prompt with R-KV enabled
├── benchmark/                  # bench_rkv.py, eval_math.py, RESULTS_*.md
├── docs/                       # DESIGN.md, IMPLEMENTATION.md (deep-dive)
└── tests/                      # GPU-free CPU unit + cross-repo parity tests
```

## Quick start

### Step 0 — environment

```bash
conda create -n rkv-flashinfer python=3.12 -y
conda activate rkv-flashinfer
pip install -r requirements-rkv.txt
```

### Step 1 — CPU tests (no GPU needed)

```bash
python tests/test_rkv_algo.py            # config validation + algorithm unit tests
python tests/test_cross_repo_parity.py   # bit-parity vs this repo's rkv/ reference
python tests/test_fa3_engine.py           # CPU contracts for the optional FA3 adapter
```

### Step 2 — examples

```bash
python examples/example.py       # FullKV baseline
python examples/example_rkv.py   # same prompt, R-KV on
```

Or drive the engine directly:

```python
import torch
from rkv import FlashInferEngine, RKVConfig

engine = FlashInferEngine(
    "/path/to/Qwen3-0.6B",           # HF repo dir (config.json + safetensors)
    max_batch_size=8,
    max_seq_len=32768,
    rkv=RKVConfig(budget=1024, buffer=128),   # None -> FullKV baseline
    dtype=torch.bfloat16,
)
outs = engine.generate(prompt_token_ids, max_new_tokens=2048,
                       temperature=0.6, top_p=0.95,
                       stop_token_ids=(eos_id,), seed=42)
print(outs[0].num_output_tokens, outs[0].num_compactions, outs[0].finish_reason)
```

### Step 3 — benchmarks

```bash
python benchmark/bench_rkv.py    # decode throughput + peak memory, R-KV on/off matrix
python benchmark/eval_math.py    # GSM8K / AIME24 / MATH-500 accuracy harness
```

There is also a repo-level GPU smoke
(`RKV_ON=1 python ../tests/smoke/flashinfer_smoke.py` from this directory),
which fails if compression never fires.

## Pinned environment

Pins carried over from the SGLang backend's validated stack; this backend's
own GPU validation is pending (see `docs/IMPLEMENTATION.md`).

| Component | Pin |
| --- | --- |
| Hardware (validated) | NVIDIA H100 80GB (SM90) |
| CUDA | 12.8 (driver) |
| Python | 3.12 |
| torch | `2.10.0` (cu128; `flashinfer-python==0.6.12` requires torch<2.11) |
| flashinfer | `flashinfer-python==0.6.12`, `flashinfer-cubin==0.6.12` |
| transformers | `5.8.1` (config/tokenizer only) |

## Configuration reference

Pass `rkv=RKVConfig(...)` to `FlashInferEngine` (`rkv=None` is the FullKV
baseline). Invalid combinations raise `ValueError` at construction.

| Field | Default | Meaning |
| --- | --- | --- |
| `budget` | `1024` | KV entries kept per layer/head after compaction |
| `buffer` | `128` | extra slots; compaction fires when length hits `budget + buffer` |
| `window_size` | `8` | trailing observation-window queries used for scoring, always retained |
| `kernel_size` | `7` | 1-D max-pool over attention scores (must be odd) |
| `mix_lambda` | `0.1` | score = λ·attention + (1−λ)·(−redundancy); `1.0` = attention-only (SnapKV-style). CLI convention; the reference class default is `0.07`, same note as the SGLang backend |
| `retain_ratio` | `0.1` | only used by the `*_percent` modes of the reference; kept for parity |
| `retain_direction` | `"last"` | which near-duplicate to exempt (`"last"` / `"first"`) |
| `compress_prefill` | `True` | HF-reference prefill compression for prompts longer than `budget` |

The redundancy similarity threshold stays hardcoded at 0.5 inside the
algorithm (house convention across backends).

## Constraints

- **Eager execution only** — no CUDA graphs, no torch.compile; dynamic
  mid-generation eviction cannot live inside a captured graph (phase-2 item).
- **`page_size=1`** — slot == token, so compaction gathers survivors without
  page-granularity bookkeeping.
- **No prefix/radix cache** — R-KV evicts KV that prefix reuse assumes
  immutable; the engine simply does not implement one.
- **Static batch per `generate()` call** — finished rows are dropped from
  planning, but new requests are not admitted mid-call.
- **Caller tokenizes** — `generate()` takes token ids; harnesses use
  `tokenizer.apply_chat_template`.

## Supported / not supported

| Config | Status |
| --- | --- |
| `batch = 1`, single GPU | ✅ |
| `batch > 1`, single GPU (per-request compaction triggering) | ✅ |
| FlashAttention-3 drop-in attention line (`--attention fa3` / `RKV_ATTN=fa3`; shared interleaved KV pool, needs the hopper `flash_attn_interface` build, SM90) | ✅ |
| Data parallel — plain (one process per GPU, disjoint shards, via benchmark scripts) | ✅ |
| Tensor parallel | ❌ **not supported — silently incorrect**: per-rank head shards would each keep their own token set with no cross-rank score reduction. Do not shard this engine across GPUs. |
| Prefix / radix cache | ❌ not implemented (incompatible with destructive eviction) |
| CUDA-graph decode / torch.compile | ❌ eager only (phase 2) |
| `page_size > 1` | ❌ engine is built around `page_size=1` |

## Results

See [`benchmark/RESULTS_H100.md`](benchmark/RESULTS_H100.md) — full H100
accuracy matrix (GSM8K / MATH-500 / AIME24 on DeepSeek-R1-Distill 1.5B/7B,
budgets 512/1024/2048 + SnapKV-style ablation), throughput/memory bench,
logit-level HF parity probes, and the post-sweep optimization A/B. Raw
artifacts live under `results/validation-2026-07-08-h100/flashinfer_sweep/`.

## Learn more

- [`docs/DESIGN.md`](docs/DESIGN.md) — scope, paged-pool slot accounting,
  logical-vs-physical lengths, prefill compression, and what was deliberately
  dropped from RKV-HS.
- [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) — code map and the
  bring-up log (filled during GPU validation).
- [`../rkv/compression/r1_kv.py`](../rkv/compression/r1_kv.py) — the algorithm
  source of truth this port is parity-tested against.

## Acknowledgements

Built on [FlashInfer](https://github.com/flashinfer-ai/flashinfer)'s paged
attention, sampling, norm, and activation kernels. The R-KV algorithm and the
HuggingFace / SGLang / vLLM / Nano-vLLM / Mini-SGLang implementations live in
the parent [R-KV repository](https://github.com/Zefan-Cai/R-KV). This backend
supersedes the private RKV-HS prototype, which served as design reference only.
