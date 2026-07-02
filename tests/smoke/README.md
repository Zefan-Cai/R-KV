# R-KV serving-integration smoke tests

One GPU smoke per integration. Each script generates a few hundred tokens
with R-KV enabled, prints the text, and applies a lexical health check
(distinct-word count, repeated-chunk detection, printable ratio) that
catches the two classic KV-corruption signatures: token salad (e.g.
`GraphUnits`-style vocabulary soup) and tight repetition loops. Exit code
0 = pass. Set `RKV_SMOKE_MODEL` to override the model.

Each integration needs its own environment — the four stacks pin
conflicting torch/transformers versions (see the per-tree requirements).
All four were validated on A100-40GB (sm80), CUDA 12.4-12.6 wheels,
Python 3.10, at repo `main`:

| Script | Env | Model (default) | R-KV switch |
|---|---|---|---|
| `vllm_smoke.py` | `pip install vllm==0.8.5 transformers==4.51.3`, then overlay the whole checked-in `vLLM/vllm/` package onto site-packages | Qwen/Qwen3-0.6B | `VLLM_USE_V1=1 VLLM_V1_R_KV_BUDGET=64 VLLM_V1_R_KV_BUFFER=8` (`BUFFER=0` disables; backend must be FLASH_ATTN) |
| `nano_smoke.py` | `pip install -e Nano-vLLM` + flash-attn>=2.5 (model path must be a **local directory**) | Qwen/Qwen3-0.6B | `RKV_ON=1` -> `rkv_enabled=True, rkv_budget=256, rkv_buffer=32` |
| `minisgl_smoke.py` | `pip install -e Mini-SGLang` | Qwen/Qwen3-0.6B | `RKV_ON=1` -> `rkv_enabled=True, rkv_budget=512, rkv_buffer=64`; instruments `RKVCompressor.maybe_compress` and fails if compression never fires |
| `sglang_smoke.py` | `pip install -r SGLang/requirements-rkv.txt && pip install -e SGLang/python --no-build-isolation` | Qwen/Qwen2.5-0.5B-Instruct | `RKV_ON=1` -> `compress_algorithm="RKV"`, triton backend, cuda graph off; `BATCH=2` for the batch variant |
| `hf_smoke.py` | `pip install -e HuggingFace` + flash-attn (transformers must be `>=4.48.1,<4.56`) | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | `MODE=rkv` (monkeypatch) vs `MODE=fullkv` (stock forward) |

Usage examples:

```bash
# vLLM
VLLM_USE_V1=1 VLLM_V1_R_KV_BUDGET=64 VLLM_V1_R_KV_BUFFER=8 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN python tests/smoke/vllm_smoke.py

# Mini-SGLang (R-KV on, then baseline)
RKV_ON=1 python tests/smoke/minisgl_smoke.py
RKV_ON=0 python tests/smoke/minisgl_smoke.py

# Nano-vLLM needs a local model directory
RKV_SMOKE_MODEL=~/huggingface/Qwen3-0.6B RKV_ON=1 python tests/smoke/nano_smoke.py

# SGLang single request + batch
RKV_ON=1 BATCH=1 python tests/smoke/sglang_smoke.py
RKV_ON=1 BATCH=2 python tests/smoke/sglang_smoke.py

# HuggingFace monkeypatch vs FullKV
MODE=rkv python tests/smoke/hf_smoke.py
MODE=fullkv python tests/smoke/hf_smoke.py
```

Notes:

- Baseline runs use sampling (temperature 0.6-0.8) on purpose: greedy
  decoding makes sub-1B models fall into repetition loops with or without
  R-KV, which would look like a compression failure.
- The Mini-SGLang engine force-disables CUDA graph capture when
  `rkv_enabled=True`; graph replay would silently skip the compression
  hooks (`cuda_graph_max_bs=None` means auto-enable, not disable).
- Mini-SGLang and Nano-vLLM both bind hardcoded localhost rendezvous
  ports (2333); do not run their smokes concurrently on one host.
