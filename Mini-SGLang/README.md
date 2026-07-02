# R-KV on Mini-SGLang

**Decoding-time, redundancy-aware KV-cache compression for [Mini-SGLang](https://github.com/sgl-project/mini-sglang).**

While a model generates a long output, R-KV periodically evicts the
**unimportant** and **redundant** past tokens, keeping only a fixed `budget` of
KV entries per request — freeing GPU memory while preserving generation quality.
This directory ports R-KV onto a **pinned** Mini-SGLang baseline, packaged the
same way as the sibling [`SGLang/`](../SGLang) port: a standalone R-KV package
plus a small wiring patch, applied onto a known-good upstream commit.

- **Algorithm** — joint scoring of *importance* (attention over a recent
  observation window) and *redundancy* (key cosine-similarity); keep the top
  `budget` tokens per request. Pure PyTorch, bit-for-bit equivalent to the parent
  repo's Nano-vLLM / HuggingFace references.
- **Integration** — true physical eviction in Mini-SGLang's paged KV pool:
  compact surviving slots to the front of the page allocation, return dropped
  tail slots to the cache manager, and keep RoPE positions consistent after the
  sequence physically shrinks (`req.kv_len = device_len - num_dropped_tokens`).

## Layout

```
Mini-SGLang/
├── README.md                          # you are here
├── requirements-rkv.txt               # dependency notes (R-KV adds none)
├── rkv/                               # R-KV package (algo.py, integration.py)
├── patch/rkv-mini-sglang-9a91cfa.patch # wiring diff (9 upstream files)
├── scripts/apply_rkv.sh               # clone pinned Mini-SGLang + drop in rkv/ + apply patch
├── benchmark/                         # bench_rkv.py (offline), README.md
├── docs/                              # DESIGN.md, IMPLEMENTATION.md (deep-dive)
└── tests/                             # GPU-free CPU unit test
```

Nothing upstream is vendored here — `scripts/apply_rkv.sh` clones
`sgl-project/mini-sglang` at the pinned commit into a git-ignored
`mini-sglang-src/`, drops in `rkv/`, and applies the wiring patch, so you always
see *exactly* what R-KV changes.

## Quick start

```bash
cd Mini-SGLang
scripts/apply_rkv.sh                    # -> ./mini-sglang-src (patched, pinned)
cd mini-sglang-src && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
python3 ../benchmark/bench_rkv.py       # offline R-KV benchmark
```

CPU unit test (no GPU, no install — loads `rkv/` by path):

```bash
python3 tests/test_rkv_algorithm.py
```

## Pinned upstream

| Component | Pin |
| --- | --- |
| **Mini-SGLang** | `sgl-project/mini-sglang` @ `9a91cfafe754aa85daee49998176275667eb58f2` |

R-KV is pure PyTorch and adds no dependencies; the runtime stack is upstream
Mini-SGLang's own at that commit (see `requirements-rkv.txt`).

## Supported / not supported

| Config | Status |
| --- | --- |
| `page_size = 1`, single GPU, eager decode | ✅ ported, CPU-tested |
| Batch > 1 (per-`(uid, layer_id)` triggering) | ✅ ported |
| CUDA graph / `torch.compile` | ❌ eager only (engine force-disables capture when `rkv_enabled=True`) |
| `page_size > 1` | ❌ slot resolution assumes `page_size = 1` |
| Prefix-cache reuse of compressed pages | ❌ compressed pages are not the verbatim prefix |
| Tensor parallel (`tp > 1`) | ❌ prototype compresses rank-0's shard only |
| End-to-end GPU serving | ⏳ not yet validated (open follow-up) |

See [`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)
for the design and the concrete wiring.
