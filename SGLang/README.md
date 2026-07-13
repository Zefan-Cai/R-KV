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

## Highlights

**Accuracy *and* performance — both strong.** R-KV keeps generation quality
**lossless at `budget=512`** (0.90–0.915 vs Full-KV's 0.910 on GSM8K) while
compressing the KV cache to a fixed budget — and the port is engineered to make
that compression cheap:

- **Fused Triton redundancy kernel** — computes the O(n²) key-similarity term as
  a single row-blocked kernel (no full n×n matrix), bit-parity-validated with a
  permanent reference fallback.
- **CUDA-graph decode** — observation queries are gathered *inside* the captured
  decode graph; only the compaction steps run eager (hybrid graph/eager path).
- **Batched cross-layer scoring** — one GEMM pass instead of `num_layers`
  (up to 8× faster scoring on short prompts).
- **Two-phase compaction** — relocate-in-forward + free-in-scheduler decouples KV
  eviction from the allocator, so it stays race-free at scale.
- **Compression-aware admission** — the scheduler reserves each request's
  *constant* compressed footprint, admitting many more concurrent requests under
  a fixed KV pool.
- **Tensor & data parallel** (validated on 8× H100) — DP scales to **5.1×**; TP is
  supported via a cross-rank eviction-score all-reduce, so every rank evicts
  identical tokens.

Full list with measured effects: [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md).

## Headline result — Qwen2.5-Math-7B-Instruct (NVIDIA H100)

SGLang's own GSM8K harness (`bench_sglang.py`, 5-shot, first 200 questions,
`--parallel 32`), comparing R-KV to a **Full-KV baseline under the same required
flags** (radix/overlap off, `page_size 1`) so the delta is purely compression:

| Config | Accuracy | Throughput | KV compactions |
| --- | --- | --- | --- |
| Full-KV (same flags, no compression) | 0.910 | 1792 tok/s | — |
| **R-KV, budget=256, buffer=128** | **0.900** | **1679 tok/s** | **64** |
| **R-KV, budget=512, buffer=64** | **0.910** | 1549 tok/s | 245 |

R-KV holds accuracy **lossless at budget=512** while running dozens–hundreds of
physical KV compactions, at **within ~3–14 % of the fair Full-KV throughput**. See
[`benchmark/RESULTS.md`](benchmark/RESULTS.md) for the full `budget × buffer_size`
sweep and the production-vs-constrained baseline discussion.

**Data parallel** (`DP=N ./benchmark/launch_server.sh rkv 256`, plain DP with
`tp=1`) scales throughput up to **5.1× on 8× H100** with unchanged accuracy
([`benchmark/RESULTS_dp.md`](benchmark/RESULTS_dp.md)). **Tensor parallel** is
supported too — the per-token eviction score is all-reduced across the attention-TP
group so every rank evicts identical tokens — scaling to **1.56× at tp=4**
([`benchmark/RESULTS_tp.md`](benchmark/RESULTS_tp.md)).

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
| Tensor parallel (`tp ≥ 2`) | ✅ supported — the per-token eviction score is all-reduced across the attention-TP group before top-k, so every rank evicts the identical tokens (see [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) §11.2); validated on 8× H100 |
| Data parallel — plain (`dp ≥ 2`, `tp = 1`) | ✅ validated — each replica runs its own R-KV over a disjoint request set; throughput scales up to 5.1× on 8× H100 (see [`benchmark/RESULTS_dp.md`](benchmark/RESULTS_dp.md)) |
| DP attention (`--enable-dp-attention`) | ❌ unsupported — padded `forward_batch` layout unverified against the R-KV hooks |
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
- [`benchmark/RESULTS.md`](benchmark/RESULTS.md) — Math-7B GSM8K `budget × buffer_size` sweep (two baselines).
- [`benchmark/RESULTS_dp.md`](benchmark/RESULTS_dp.md) — data-parallel scaling (up to 5.1× on 8× H100).
- [`benchmark/RESULTS_tp.md`](benchmark/RESULTS_tp.md) — tensor-parallel scaling & cross-rank correctness.
- [`benchmark/RESULTS_a100_n100.md`](benchmark/RESULTS_a100_n100.md) — independent A100 n=100 rerun.

## Manual apply (without the script)

```bash
git clone https://github.com/sgl-project/sglang.git sglang-src
cd sglang-src && git checkout 49e384ce9d304648e9959666ecb8ce8cd98d0deb
cp -r ../rkv python/sglang/srt/mem_cache/rkv
git apply ../patch/rkv-sglang-0.5.14.patch
```
