# R-KV on SGLang (v0.5.14)

**Decoding-time, redundancy-aware KV-cache compression for SGLang.**

While a model generates a long output, R-KV periodically evicts the
**unimportant** and **redundant** past tokens, keeping only a fixed `budget` of
KV entries per request — freeing GPU memory while preserving generation quality.
This directory ports R-KV onto a **pinned** SGLang v0.5.14 baseline.

- **Algorithm** — joint scoring of *importance* (attention over a recent
  observation window) and *redundancy* (key cosine-similarity); keep the top
  `budget` tokens per request.
- **Integration** — true physical eviction in SGLang's paged KV pool: relocate
  surviving slots, `free()` the rest, rewrite `req_to_token`, and keep rotary
  positions consistent after the sequence physically shrinks. Runs on the
  FlashInfer decode path.

## Headline result — Qwen2.5-Math-7B-Instruct (single NVIDIA H100)

GSM8K-style math harness, first 20 items, **8 concurrent requests** (server-side
`batch` up to 8):

| Config | Accuracy | KV compactions |
| --- | --- | --- |
| baseline (R-KV off) | 95% | — |
| **R-KV, budget=512** | **95% (19/20)** | **~235** |

R-KV kept full accuracy while running **~235 physical KV compactions with zero
crashes**, each shrinking a request from ~700+ tokens back to the 512-token
budget. See [`benchmark/RESULTS_math7b.md`](benchmark/RESULTS_math7b.md).

**Data parallel** (`DP=N ./benchmark/launch_server.sh rkv 512`, plain DP with
`tp=1`) is validated: each replica runs its own R-KV over a disjoint request set,
and throughput scales up to **5.2× on 8× H100** with unchanged accuracy — see
[`benchmark/RESULTS_dp.md`](benchmark/RESULTS_dp.md).

---

## Independent verification (2026-07-01)

The port was independently verified against this repo's reference
implementation before being published here:

- **Cross-repo bit-level parity** —
  [`tests/test_cross_repo_parity.py`](tests/test_cross_repo_parity.py) feeds
  identical random tensors through the port's `R1KV` (`rkv/algo.py`) and the
  repo-root reference (`rkv/compression/r1_kv.py`): `update_kv` outputs, the
  attention/similarity primitives, and `select_indices` selections are
  **bit-for-bit identical** across MHA, Qwen2.5-7B/0.5B GQA shapes, batch>1,
  fp32/bf16, and below-budget no-op cases.
- **GPU rerun at n=100** — GSM8K few-shot, first 100 items,
  Qwen2.5-Math-7B-Instruct, 1×A100-80G, `temperature=0`:

  | Config | Accuracy (100) | Throughput | Compactions |
  | --- | --- | --- | --- |
  | baseline (eager, R-KV off) | **91.0%** | 49.2 tok/s | 0 |
  | R-KV budget=512 | **90.0%** | 44.2 tok/s | 1012 |
  | R-KV budget=256 | 89.0% | 42.4 tok/s | 1138 |
  | R-KV budget=512, 8 concurrent | **90.0%** | **181.8 tok/s** | 1007 |

  Accuracy holds within noise (±~3 pts at n=100) even with the budget below
  the few-shot prompt length, and the batch path keeps identical accuracy at
  4.1× throughput. Details:
  [`benchmark/RESULTS_a100_n100.md`](benchmark/RESULTS_a100_n100.md).

This distribution tracks the hardened R-KV source (upstream fork
`wanke1997/sglang-compress`); two correctness fixes found during the original
verification (below) are now baked into the source, along with substantial later
work — prefill-phase R-KV, a fused Triton redundancy kernel, **decode CUDA-graph
support**, two-phase compaction, and full KV-pool memory accounting (see
[`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md)).

1. **Rotary position off-by-one** (`rkv/integration.py`,
   `override_decode_positions`): the override used
   `len(origin_input_ids) + len(output_ids)`, but at forward time the
   just-sampled token is already in `output_ids`, so the current token's
   0-based position is that count **minus one** (baseline:
   `clamp_position(seq_lens) = seq_lens - 1`). Without the fix every
   R-KV-managed decode token was rotated at `position + 1` from the first
   decode step — a uniform shift that measurably did not hurt GSM8K accuracy
   (89 vs 90 at n=100), but made `--enable-rkv` non-equivalent to baseline
   even before any compression fires.
2. **Startup validation** (`server_args.py`, `_handle_rkv_validation`):
   `--enable-rkv` rejects configurations the port's memory safety depends on but
   previously did not enforce (radix cache on, overlap schedule on,
   `page_size > 1`, `tp_size > 1`) instead of silently corrupting the KV pool;
   plus `RKVConfig` guard checks (`buffer_size >= window_size`,
   `min_seq_len >= budget`). (Decode CUDA graph was **later made compatible** and
   is no longer rejected.)

---

## Why a patch, not a fork?

R-KV touches SGLang in a **small, purely additive** way: one self-contained
package (`rkv/`) plus ~780 lines of wiring across **9** existing files. Instead
of vendoring the entire (~6700-file) SGLang tree, this directory ships:

- `rkv/` — the R-KV code (browsable, the source of truth);
- `patch/` — the 9-file wiring diff;
- `scripts/apply_rkv.sh` — clones the **exact pinned** SGLang commit, drops in
  `rkv/`, and applies the patch.

So you always see *exactly* what R-KV changes, and you build against a known-good
upstream commit.

```
SGLang/
├── README.md                      # you are here
├── requirements-rkv.txt           # pinned, verified dependency stack
├── rkv/                           # R-KV package (algo, integration, prefill, redundancy_fused)
├── patch/rkv-sglang-0.5.14.patch  # wiring diff (9 upstream files)
├── scripts/apply_rkv.sh           # clone pinned SGLang + drop in rkv/ + apply patch
├── benchmark/                     # eval.py, launch_server.sh, prepare_data.sh, data/, RESULTS*.md
├── docs/                          # DESIGN, IMPLEMENTATION, OPTIMIZATIONS, REPRODUCE
└── tests/                         # GPU-free CPU unit tests (+ fused-kernel GPU test)
```

---

## Verified environment (pinned)

Everything below was validated end-to-end on this exact configuration:

| Component | Pin |
| --- | --- |
| Hardware | NVIDIA H100 80GB |
| CUDA | 12.9 |
| Python | 3.12 |
| **SGLang** | upstream commit `49e384ce9d304648e9959666ecb8ce8cd98d0deb` (release/v0.5.14) |
| torch | `2.11.0+cu129` (torchvision `0.26.0+cu129`, torchaudio `2.11.0+cu129`) |
| sglang-kernel | `0.4.4+cu129` |
| flashinfer | `flashinfer-python==0.6.12`, `flashinfer-cubin==0.6.12` |
| transformers | `5.8.1` |
| triton | `3.6.0` |

> For a different CUDA version, use the matching SGLang wheel index
> (`https://docs.sglang.ai/whl/<cuXXX>/`) and torch/flashinfer/kernel builds for
> that CUDA. The versions must agree with each other.

---

## Step-by-step

### Step 0 — Create a clean environment

```bash
conda create -n rkv-sglang python=3.12 -y
conda activate rkv-sglang
```

### Step 1 — Apply R-KV to a pinned SGLang checkout

```bash
cd SGLang            # this directory
bash scripts/apply_rkv.sh
```

This clones SGLang at the pinned commit into `./sglang-src/` (git-ignored),
copies `rkv/` into `sglang-src/python/sglang/srt/mem_cache/rkv/`, and applies
`patch/rkv-sglang-0.5.14.patch`. If the patch does not apply cleanly the script
stops with an error (it won't leave a half-patched tree).

### Step 2 — Install the pinned dependency stack

```bash
# SGLang framework + its dependency closure (editable, from the patched source):
pip install -e "sglang-src/python" \
  --extra-index-url https://docs.sglang.ai/whl/cu129/

# Pin the exact versions R-KV was validated with:
pip install -r requirements-rkv.txt \
  --extra-index-url https://docs.sglang.ai/whl/cu129/
```

### Step 3 — Smoke test (no GPU needed)

The algorithm and integration logic have GPU-free CPU unit tests (the fused-kernel
test needs a GPU + Triton and self-skips otherwise):

```bash
python3 tests/test_rkv_algo.py                 #  7 tests — algorithm parity vs reference
python3 tests/test_rkv_integration.py          # 24 tests — compaction, lifecycle, memory, batch>=2
python3 tests/test_rkv_prefill.py              #  9 tests — prefill algorithm (oneshot/buffered)
python3 tests/test_rkv_prefill_integration.py  # 11 tests — prefill compaction & admission
python3 tests/test_rkv_redundancy_fused.py     # 11 tests — fused Triton redundancy kernel (GPU)
python3 tests/test_cross_repo_parity.py        # bit-level parity vs this repo's rkv/ reference
```

All should print `OK` / `ALL PARITY CHECKS PASSED`.

### Step 4 — Launch a server with R-KV on

```bash
# check the bundled eval dataset is present
bash benchmark/prepare_data.sh

# start the server (R-KV on, budget=512). Point MODEL at your local weights.
MODEL=/path/to/Qwen2.5-Math-7B-Instruct \
  bash benchmark/launch_server.sh rkv 512
```

`launch_server.sh` sets `PYTHONPATH` to the patched `sglang-src/` and enables the
flags R-KV **requires** (see [Constraints](#constraints-required-flags)). Wait
for `The server is fired up and ready to roll!`.

For a fair baseline (R-KV off, same eager config):

```bash
MODEL=/path/to/Qwen2.5-Math-7B-Instruct bash benchmark/launch_server.sh baseline
```

### Step 5 — Run the eval

In another shell (same conda env):

```bash
# batch>1: send 8 requests in parallel so the server batches them
python3 benchmark/eval.py --n 20 --concurrency 8 --label rkv_b512
```

Expected (Math-7B, budget=512): `accuracy : 19/20 = 0.950`, and the server log
shows ~180+ `R-KV compacted req_pool_idx=... phys N -> 512` lines with decode
`#running-req` up to 8.

---

## Configuration reference

Enable with `--enable-rkv`. Every hyper-parameter has a flat CLI flag (defaults
mirror the R-KV reference); a `--rkv-config` JSON string **overrides** the flags.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--rkv-budget` | `1024` | KV entries kept per request after compression |
| `--rkv-window-size` | `8` | trailing observation window, always retained |
| `--rkv-kernel-size` | `7` | pooling kernel for importance smoothing |
| `--rkv-mix-lambda` | `0.1` | importance vs redundancy mix (1 = importance only) |
| `--rkv-retain-ratio` | `0.1` | fraction of recent similar neighbours exempted |
| `--rkv-retain-direction` | `last` | `last` / `first` / `last_percent` / `first_percent` |
| `--rkv-buffer-size` | `128` | compress every N newly generated tokens per request |
| `--rkv-min-seq-len` | `budget` | min KV length before compression is considered |
| `--rkv-config` | — | JSON that overrides any of the above, e.g. `'{"budget":512,"buffer_size":16}'` |
| `--rkv-max-active-requests` | — | cap concurrent R-KV requests → shrinks the `rolling_q` observation buffer (trades peak concurrency for memory) |
| `--rkv-fused-validation` | `startup` | when to validate the fused redundancy kernel: `startup` / `first-request` / `off` |

Prompt-phase compression is a separate mode: `--enable-rkv-prefill` (+
`--rkv-prefill-config` JSON, `mode` = `oneshot` or `buffered`). It cannot combine
with `--enable-rkv` and additionally requires `--disable-prefill-cuda-graph`.

Example (decode R-KV, CUDA-graph decode left **on** — now supported):

```bash
python3 -m sglang.launch_server --model-path <model> \
  --attention-backend flashinfer \
  --disable-prefill-cuda-graph \
  --disable-overlap-schedule --disable-radix-cache --page-size 1 \
  --enable-rkv --rkv-budget 512 --rkv-buffer-size 16
```

<a name="constraints-required-flags"></a>
## Constraints (required flags)

R-KV performs **destructive, mid-generation** eviction, so it needs a specific
server configuration (all set for you by `launch_server.sh`):

- `--attention-backend flashinfer` — the only backend wired for R-KV in phase 1.
- `--disable-radix-cache` — R-KV frees KV slots the radix/prefix cache would
  still reference; leaving it on double-counts the pool and crashes the leak
  checker. Prefix reuse assumes KV is immutable; R-KV evicts it.
- **Decode CUDA graph is supported** and left on (the observation-window queries
  are collected inside the captured graph, with a hybrid eager path only for the
  compaction steps). `--enable-rkv-prefill` additionally requires
  `--disable-prefill-cuda-graph` (prompt-phase scoring is dynamic-shape).
- `--disable-overlap-schedule` — simpler, deterministic timing, and it avoids an
  allocator free-list race during compaction (see [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) §2.2/§3).
- `--page-size 1` — clean per-slot `free()` (the paged allocator frees at page
  granularity otherwise).

## Supported / not supported

| Config | Status |
| --- | --- |
| `batch = 1`, `tp = 1`, `dp = 1` | ✅ validated |
| `batch > 1` (`tp = 1`, `dp = 1`) | ✅ validated (per-request triggering) |
| Tensor parallel (`tp ≥ 2`) | ❌ **not supported — silently incorrect** without a cross-rank score all-reduce (see [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) §11.2). Do not combine `--enable-rkv` with `--tp > 1`. |
| Data parallel — plain (`dp ≥ 2`, `tp = 1`) | ✅ validated — each replica runs its own R-KV over a disjoint request set; throughput scales up to 5.2× on 8× H100 (see [`benchmark/RESULTS_dp.md`](benchmark/RESULTS_dp.md)) |
| DP attention (`--enable-dp-attention`) | ❌ untested — implies `tp > 1` (currently blocked at startup); padded `forward_batch` layout unverified |
| CUDA-graph decode | ✅ supported (in-graph observation + hybrid eager compaction steps) |

---

## Learn more

- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, the paged-pool tension, the
  rotary/position scheme, and why the `sparsity/` framework was not reused.
- [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) — the practical code map:
  the 9 wiring points, bring-up bugs, and the parallelism/batching support matrix.
- [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) — performance optimizations and
  production-hardening (CUDA graph, fused kernel, two-phase compaction, admission).
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — exact, validated reproduction & usage.
- [`benchmark/RESULTS.md`](benchmark/RESULTS.md) — Qwen2.5-0.5B sanity numbers.
- [`benchmark/RESULTS_math7b.md`](benchmark/RESULTS_math7b.md) — Math-7B results.

## Manual apply (without the script)

```bash
git clone https://github.com/sgl-project/sglang.git sglang-src
cd sglang-src && git checkout 49e384ce9d304648e9959666ecb8ce8cd98d0deb
cp -r ../rkv python/sglang/srt/mem_cache/rkv
git apply ../patch/rkv-sglang-0.5.14.patch
```
