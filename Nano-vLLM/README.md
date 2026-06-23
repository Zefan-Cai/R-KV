# Nano-vLLM with R-KV

A lightweight vLLM implementation, vendored from
[Zefan-Cai/Nano-vLLM](https://github.com/Zefan-Cai/Nano-vLLM) (forked from
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)) with
**R-KV decode-time KV cache compression** integrated into the attention
layer. This is the smallest of the R-KV reference implementations and is
meant to be the easiest way to read and modify the R-KV decode loop.

## What R-KV adds

R-KV compresses the KV cache during decoding by combining attention
scores with a redundancy-aware similarity penalty, so long
chain-of-thought / self-reflection traces fit in a bounded budget instead
of growing linearly with the generation length. The Nano-vLLM port keeps
the compression on the paged KV cache so prefix-cache and block-table
paths still work.

The integration touches a small set of files:

- `nanovllm/layers/rkv.py` — `R1KV` class with the attention + similarity
  score combination, top-k selection, and the per-step `update_kv()` call.
- `nanovllm/layers/attention.py` — extends `Attention` with
  `configure_rkv(...)`, caches the last `window_size` queries per
  sequence in `_rkv_query_cache`, and applies compression on the paged
  cache in the decode branch via `_maybe_compress_kvcache`.
- `nanovllm/utils/context.py` — adds `seq_ids`, `rkv_source_lens`,
  `rkv_target_lens` to `Context` so the runner can pass per-step
  per-sequence state down to the attention call.
- `nanovllm/engine/model_runner.py` — propagates seq IDs into the
  attention context, applies `configure_rkv(...)` to every attention
  module at load, and reads back `rkv_target_lens` to update
  `seq.rkv_cache_len`.
- `nanovllm/engine/sequence.py` — `rkv_cache_len` per-sequence field so
  later steps know the current compressed length.
- `nanovllm/config.py` — `rkv_*` config flags.

## Installation

The original Nano-vLLM install command still works because the package
name is unchanged:

```bash
pip install git+https://github.com/Zefan-Cai/R-KV.git#subdirectory=Nano-vLLM
```

Or from a local checkout of this repo:

```bash
cd R-KV/Nano-vLLM
pip install -e .
```

## Model Download

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

The API mirrors vLLM's `LLM` interface; pass the R-KV flags through the
`LLM` constructor (they forward to `Config`):

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/YOUR/MODEL/PATH",
    enforce_eager=True,          # R-KV requires eager mode
    tensor_parallel_size=1,
    rkv_enabled=True,
    rkv_budget=1024,
    rkv_buffer=128,
    rkv_window_size=8,
    rkv_kernel_size=7,
    rkv_mix_lambda=0.07,
    rkv_retain_ratio=0.1,
    rkv_retain_direction="last",
)

sampling_params = SamplingParams(temperature=0.6, max_tokens=4096)
outputs = llm.generate(["Solve: 12 * 13."], sampling_params)
```

The compression budget (`rkv_budget`) and trigger buffer (`rkv_buffer`)
match the same knobs in the HuggingFace and SGLang R-KV implementations
in the parent repo.

## Files copied from upstream

This directory is a vendored copy of Nano-vLLM with the R-KV prototype
already applied. See `nanovllm/layers/rkv.py` and the integration in
`nanovllm/layers/attention.py` for the algorithm. For evaluation and
the HuggingFace reference R-KV implementation, see `../HuggingFace/`.

## Limitations / known gaps

- The decode-time compression is implemented in Python over the paged KV
  cache; for the largest budgets and very long generations a fused Triton
  rewrite would help.
- `enforce_eager=True` is required: CUDA graphs and torch.compile are
  disabled when R-KV is enabled because cache lengths change per step.
- This is a prototype port; the full benchmark numbers from the R-KV
  paper are produced with the HuggingFace path in `../HuggingFace/`.

## Acknowledgements

Built on top of GeeeekExplorer/nano-vllm. R-KV algorithm and
HuggingFace / SGLang / vLLM implementations live in the parent
[R-KV repository](https://github.com/Zefan-Cai/R-KV).
