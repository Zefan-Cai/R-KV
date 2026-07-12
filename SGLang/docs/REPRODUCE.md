# Reproduction & Usage Guide

Exact, copy-pasteable steps to reproduce the R-KV port on SGLang v0.5.14,
validated on **8× NVIDIA H100 (80 GB), CUDA 12.9, Python 3.12**. Every command
and every number below was run on that box.

- Design → [`DESIGN.md`](./DESIGN.md)
- Code map → [`IMPLEMENTATION.md`](./IMPLEMENTATION.md)
- Optimizations & hardening → [`OPTIMIZATIONS.md`](./OPTIMIZATIONS.md)

---

## 0. Pinned stack

| Component | Pin |
| --- | --- |
| Hardware | NVIDIA H100 80 GB |
| CUDA | 12.9 |
| Python | 3.12 |
| **SGLang** | upstream commit `49e384ce9d304648e9959666ecb8ce8cd98d0deb` (release/v0.5.14) |
| torch | `2.11.0+cu129` |
| sglang-kernel | `0.4.4+cu129` |
| flashinfer | `flashinfer-python==0.6.12`, `flashinfer-cubin==0.6.12` |
| transformers | `5.8.1` |
| triton | `3.6.0` |

The exact versions are in [`../requirements-rkv.txt`](../requirements-rkv.txt).

---

## 1. Build the patched tree

```bash
cd SGLang                       # this directory
bash scripts/apply_rkv.sh       # clone pinned SGLang -> sglang-src/, drop in rkv/, apply patch
```

This clones upstream SGLang at the pinned commit into `sglang-src/` (git-ignored),
copies the 6-file `rkv/` package into
`sglang-src/python/sglang/srt/mem_cache/rkv/`, and applies the **9-file** wiring
patch. It fails loudly (`git apply --check`) if the patch would not apply cleanly.

> To build from a local SGLang clone instead of GitHub, set `SGLANG_REPO`:
> `SGLANG_REPO=/path/to/local/sglang bash scripts/apply_rkv.sh --force`.

Install the dependency stack (skip if your env already matches §0):

```bash
pip install -e "sglang-src/python" --extra-index-url https://docs.sglang.ai/whl/cu129/
pip install -r requirements-rkv.txt --extra-index-url https://docs.sglang.ai/whl/cu129/
```

---

## 2. Unit tests (GPU-free algorithm + integration; fused test needs a GPU)

All modules load by file path, so no installed `sglang` is required.

```bash
python3 tests/test_rkv_algo.py                 #  7 tests — algorithm parity vs reference
python3 tests/test_rkv_integration.py          # 24 tests — compaction / lifecycle / memory / batch>=2
python3 tests/test_rkv_prefill.py              #  9 tests — prefill algorithm (tiled/oneshot/buffered)
python3 tests/test_rkv_prefill_integration.py  # 11 tests — prefill compaction & admission
python3 tests/test_rkv_redundancy_fused.py     # 11 tests — fused Triton redundancy kernel (needs CUDA+Triton)
python3 tests/test_cross_repo_parity.py        # bit-level parity vs this repo's rkv/ reference
```

**Validated result:** `62/62` unit tests pass; `test_cross_repo_parity.py` prints
`ALL PARITY CHECKS PASSED (bit-for-bit)` across MHA / GQA / batch>1 / fp32 / bf16.

---

## 3. Equivalence check (optional but recommended)

The tree produced by `apply_rkv.sh` is byte-identical to the development tree.
To confirm the patch reconstructs exactly the intended source:

```bash
# after apply_rkv.sh has produced sglang-src/
diff -rq --no-dereference \
  <(cd sglang-src/python/sglang/srt && find . -name '*.py' | sort) \
  <(...your reference tree...)   # empty diff == identical
```

During porting this was verified against the development fork: with
`--no-dereference` the `python/sglang/srt` diff was **empty**, and all 15 changed
files (6 `rkv/` + 9 wiring) matched by `sha256`.

---

## 4. Serve + evaluate (Qwen2.5-Math-7B, single H100)

```bash
bash benchmark/prepare_data.sh                                   # bundled GSM8K few-shot data
MODEL=/data/model/Qwen2.5-Math-7B-Instruct bash benchmark/launch_server.sh rkv 512
# in another shell (same env):
python3 benchmark/eval.py --n 20 --concurrency 8 --label rkv_b512
```

`launch_server.sh` sets `PYTHONPATH=sglang-src/python` and the required flags.

### Validated numbers

**Decode R-KV, budget 512, `n=20`, concurrency 8** — three back-to-back runs
(eager decode path):

| Run | Accuracy | avg_tokens | Throughput | KV compactions |
| --- | --- | --- | --- | --- |
| 1 | 19/20 = 0.950 | 195 | 423.6 tok/s | 232 |
| 2 | 19/20 = 0.950 | 195 | 427.1 tok/s | 232 |
| 3 | 19/20 = 0.950 | 195 | 424.4 tok/s | 232 |

Stable at **95 % (== eager baseline)** with **~232 physical KV compactions per
run, zero crashes / leaks / double-frees**.

**Decode R-KV + decode CUDA graph ON** (the hardening headline — the old port was
eager-only). Launch without `--disable-decode-cuda-graph`:

```bash
export PYTHONPATH="$PWD/sglang-src/python"
python3 -m sglang.launch_server \
  --model-path /data/model/Qwen2.5-Math-7B-Instruct \
  --attention-backend flashinfer \
  --disable-radix-cache --disable-overlap-schedule --disable-prefill-cuda-graph \
  --page-size 1 --mem-fraction-static 0.6 --host 127.0.0.1 --port 30011 \
  --enable-rkv --rkv-config '{"budget":512,"window_size":8,"buffer_size":16}'
python3 benchmark/eval.py --n 20 --concurrency 8 --port 30011 --label rkv_graphON
```

Result: decode graph captures all 36 batch sizes and the decode loop runs under
it (`cuda graph: True`); **18/20 = 0.900** (a single-item flip vs eager, within
n=20 noise), **579.6 tok/s (+37 % over eager)**, 228 compactions, **zero errors**.
The server log shows `R-KV decode fused-redundancy gate: OK -> fused adopted`
(the `--rkv-fused-validation startup` gate).

---

## 5. Configuration reference

Enable decode R-KV with `--enable-rkv`; every hyper-parameter has a flat flag and
a `--rkv-config` JSON override.

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
| `--rkv-config` | — | JSON overriding any of the above, e.g. `'{"budget":512,"buffer_size":16}'` |
| `--rkv-max-active-requests` | — | cap concurrent R-KV requests → shrinks the `rolling_q` buffer (trades concurrency for memory) |
| `--rkv-fused-validation` | `startup` | when to validate the fused kernel: `startup` / `first-request` / `off` |

Prompt-phase compression uses `--enable-rkv-prefill` (+ `--rkv-prefill-config`
JSON; `mode` is `oneshot` or `buffered`). It additionally requires
`--disable-prefill-cuda-graph`.

### Required flags

Decode R-KV (`--enable-rkv`): `--disable-radix-cache --disable-overlap-schedule
--page-size 1`. **Decode CUDA graph is supported** (enabled by default;
pass `--disable-decode-cuda-graph` only for a fully-eager, bit-reproducible run).
**Tensor parallel (`--tp-size N`) and plain data parallel (`--dp-size N`) are
supported**; only DP attention (`--enable-dp-attention`) is not. Prefill R-KV
(`--enable-rkv-prefill`) additionally needs `--disable-prefill-cuda-graph` and
cannot combine with `--enable-rkv`. Unsupported runtimes (non-FlashInfer backend,
MLA, hybrid-SWA, speculative decoding) are rejected at startup.

---

## 6. Gotchas

- Launch servers detached (`setsid ... </dev/null & disown`) so they survive an
  agent's terminal cleanup.
- Temperature-0 eval is deterministic, so accuracy repeats exactly across runs of
  the same config; run multiple times to confirm **stability under compaction**
  (compaction count / no crashes), not to average accuracy.
- The one-item accuracy delta between eager and CUDA-graph decode is expected
  kernel-selection noise at `n=20`, not a correctness bug — both are ≈95 %.
- `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` defaults on: any idle KV leak
  crashes the scheduler, so a clean run is itself a no-leak proof.
