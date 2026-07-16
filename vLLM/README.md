# R-KV on vLLM (v0.25.1)

**Decoding-time, redundancy-aware KV-cache compression for vLLM v1.**

While a model generates a long output, R-KV periodically evicts the
**unimportant** and **redundant** past tokens, keeping only a fixed `budget` of
KV entries per request — freeing GPU memory while preserving generation quality.
This directory ports R-KV onto a **pinned** vLLM **v0.25.1** baseline, using the
same *patch-not-fork* layout as the [SGLang port](../SGLang/README.md).

- **Algorithm** — joint scoring of *importance* (attention over a recent
  observation window) and *redundancy* (key cosine-similarity); keep the top
  `budget` tokens per request.
- **Integration** — true physical eviction in vLLM's paged KV cache: the
  attention backend overwrites each request's surviving KV in place, and the
  model runner keeps rotary positions *logical* while writing new KV to the
  *physical* (shrunken) slots.

> **Status: re-ported against v0.25.1 and validated on an NVIDIA H100.** The
> previous proof-of-concept lived on a much older vLLM V1. This port
> re-implements the runtime wiring against v0.25.1's rewritten v1 stack. The
> `rkv/` algorithm is CPU-tested for bit-parity with the reference; the wiring
> patch applies cleanly to a pristine v0.25.1 tree; and the end-to-end serving
> path is validated on GPU (Qwen2.5-0.5B/Math-7B): compaction fires, output is
> coherent, and with a budget large enough that no eviction occurs the output is
> **byte-identical** to Full-KV (proving the position/slot wiring is
> transparent). See [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) for the
> validation record, known limitations, and roadmap.
>
> **R-KV runs on vLLM's V1 GPU model runner.** v0.25.1 defaults to a newer V2
> runner; the port **auto-selects V1 whenever R-KV is enabled** so compression
> is never silently skipped (V2 support is on the roadmap).

## Why a patch, not a fork?

R-KV touches vLLM in a **small, purely additive** way: one self-contained
package (`rkv/`) plus ~632 lines of wiring across **13** existing files. Instead
of vendoring the entire vLLM tree, this directory ships:

- `rkv/` — the R-KV code (browsable, the source of truth);
- `patch/rkv-vllm-0.25.1.patch` — the 13-file wiring diff;
- `scripts/apply_rkv.sh` — clones the **exact pinned** vLLM commit, drops in
  `rkv/`, and applies the patch.

So you always see *exactly* what R-KV changes, and you build against a
known-good upstream commit.

```
vLLM/
├── README.md                      # you are here
├── requirements-rkv.txt           # dependency notes (R-KV adds none)
├── rkv/                           # R-KV package (algo, integration)
├── patch/rkv-vllm-0.25.1.patch    # wiring diff (13 upstream files)
├── scripts/apply_rkv.sh           # clone pinned vLLM + drop in rkv/ + apply patch
├── benchmark/                     # benchmarking notes
├── docs/                          # DESIGN, IMPLEMENTATION, OPTIMIZATIONS, REPRODUCE
└── tests/                         # GPU-free CPU unit tests
```

## Pinned upstream

| Component | Pin |
| --- | --- |
| **vLLM** | release tag `v0.25.1` (commit `752a3a504485790a2e8491cacbb35c137339ad34`) |
| Runtime path | vLLM **v1** engine, FlashAttention backend |
| Torch / CUDA / kernels | resolved by vLLM's own build for your platform |

R-KV itself adds **no** new runtime dependencies — the algorithm uses only
`torch`, which vLLM already requires.

## Step-by-step

```bash
# 1. Build the patched vLLM tree (clones v0.25.1, copies rkv/, applies patch).
scripts/apply_rkv.sh

# 2. Install it (source build — needs CUDA + a GPU).
pip install -e vllm-src

# 3. Serve a model with R-KV at **best throughput**. Compression activates
#    automatically once BUDGET and BUFFER are both > 0. Do NOT pass
#    --enforce-eager: R-KV auto-selects PIECEWISE cudagraph (attention stays
#    eager so its hooks fire; the rest of the layer is graphed). ASYNC=1 turns
#    on async scheduling (+16.7% decode tok/s). See benchmark/launch_server.sh.
VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=64 VLLM_V1_R_KV_ASYNC=1 \
  vllm serve Qwen/Qwen2.5-Math-7B-Instruct

#    ...or use the wrapper (best-throughput flags baked in):
#    benchmark/launch_server.sh rkv 256
```

### Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_V1_R_KV_BUDGET` | `0` (off) | KV entries kept per request after compression |
| `VLLM_V1_R_KV_BUFFER` | `0` (off) | compress every N generated tokens |
| `VLLM_V1_R_KV_WINDOW` | `8` | trailing observation window (always retained) |
| `VLLM_V1_R_KV_KERNEL` | `7` | max-pool kernel size for the importance term |
| `VLLM_V1_R_KV_ASYNC` | `0` (off) | allow async scheduling for best throughput (+16.7%); the runner applies its evicted-token count itself so async stays correct |
| `VLLM_V1_R_KV_FREE_BLOCKS` | `1` (on) | free the KV blocks R-KV evicts so the footprint is bounded at `budget+buffer`; set `0` for the pre-fix behavior |

When `BUDGET` or `BUFFER` is `0`, **every** R-KV code path is inert and vLLM
behaves exactly as upstream.

### Multi-GPU (tensor & data parallelism)

R-KV works with **tensor parallelism** and **data parallelism** — add
`--tensor-parallel-size` / `--data-parallel-size` as usual (or pass them through
`EXTRA` to `launch_server.sh`). Under TP each rank holds only a shard of the KV
heads, so R-KV all-reduces its per-token eviction scores across the TP group,
guaranteeing every rank evicts the identical set (TP=2 few-shot GSM8K accuracy
bit-matches single-GPU). DP replicas are independent, and DP+TP reduces within
each replica's TP sub-group. Pipeline parallelism is not yet supported.

## Tests

```bash
python tests/test_rkv_algo.py     # GPU-free CPU parity tests for the algorithm
```

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — the algorithm and the core design tension.
- [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) — the per-step data flow and
  the exact wiring points in v0.25.1.
- [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) — known limitations, the
  GPU validation checklist, and the roadmap.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — how the patch was generated and how
  to regenerate it.
