# R-KV on FlashInfer — Implementation Notes

This document is the practical map of the code: which file owns what, how a
step flows through the engine, and (once GPU validation runs) the bring-up log
of everything that broke on real hardware. For the architecture and the
rationale — scope, slot accounting, prefill compression, what was dropped from
RKV-HS — read [`DESIGN.md`](./DESIGN.md) first.

## 1. Code map

| File | Responsibility |
| --- | --- |
| [`rkv/config.py`](../rkv/config.py) | `RKVConfig` frozen dataclass; `__post_init__` validation (`budget > window_size`, `buffer >= window_size`, odd `kernel_size`, `0 < mix_lambda <= 1`, `retain_direction` in `{"last", "first"}`) |
| [`rkv/algo.py`](../rkv/algo.py) | Faithful port of the reference scoring/selection: `compute_attention_scores`, `cal_similarity` (`scatter_` exemption), `select_indices`, `update_kv`. No flashinfer import — CPU CI runs it. |
| [`rkv/compressor.py`](../rkv/compressor.py) | `R1KV`: per-batch rolling window-query cache and the compaction driver that scores, selects, and gathers survivors per layer/kv-head. |
| [`rkv/models.py`](../rkv/models.py) | Llama / Qwen2 / Qwen3 forward passes on FlashInfer kernels (`rmsnorm`, `fused_add_rmsnorm`, `silu_and_mul`, rope cache); family detected from `config.architectures[0]`; `head_dim` from config when present. |
| [`rkv/engine.py`](../rkv/engine.py) | `FlashInferEngine`: paged KV pool (`page_size=1`, NHD), ragged prefill + optional prefill compression, per-step decode planning over active rows, sampling, compaction trigger, `generate()` / `GenOutput` / `stats`. Attention plan/run are overridable hooks (`_plan_prefill` / `_run_prefill_attention` / `_plan_decode` / `_run_decode_attention`). |
| [`rkv/engine_fa3.py`](../rkv/engine_fa3.py) | `FA3Engine`: same engine with the two attention calls swapped to FlashAttention-3 (`flash_attn_varlen_func` prefill, `flash_attn_with_kvcache` decode over the per-request contiguous regions). Select with `--attention fa3` (benchmarks) or `RKV_ATTN=fa3` (smoke). Requires `flash_attn_interface` (hopper build, SM90). |
| [`rkv/loader.py`](../rkv/loader.py) | Streams safetensors shards into pre-allocated parameters (no full-model 2x copy). |
| [`examples/`](../examples/) | `example.py` (FullKV baseline) and `example_rkv.py` (same prompt, R-KV on). |
| [`benchmark/`](../benchmark/) | `bench_rkv.py` (throughput + peak memory matrix), `eval_math.py` (GSM8K / AIME24 / MATH-500), `RESULTS_*.md`. |
| [`tests/`](../tests/) | `test_rkv_algo.py` (CPU unit tests), `test_cross_repo_parity.py` (bit-parity vs `../../rkv/compression/r1_kv.py`), and `test_fa3_engine.py` (CPU contracts for the optional FA3 adapter); importlib-by-path, no flashinfer import, wired into `.github/workflows/cpu-tests.yml`. |
| `../../tests/smoke/flashinfer_smoke.py` | Repo-level GPU smoke: `RKV_ON=0/1`, health check, fails if `num_compactions == 0` with R-KV on. |
| `../../tests/smoke/fa3_parity.py` | H100 teacher-forced FA3-vs-FlashInfer logits probe covering ragged prefill, non-contiguous active rows, and post-compaction decode. |

## 2. Per-step data flow

Summary; authoritative description in [`DESIGN.md`](./DESIGN.md) §5.2–5.3.

1. One `plan()` per decode step (shared by all layers) on
   `BatchDecodeWithPagedKVCacheWrapper`, active rows only. `kv_indptr` /
   `last_page_len` are CPU tensors (0.6.x `plan()` reads them host-side;
   device tensors would D2H-sync every step); `kv_indices` are slices of a
   precomputed device arange; per-step scalars share one staged H2D copy.
2. Per layer: rmsnorm → fused qkv GEMM (split + contiguous) → RoPE in-place at
   `logical_len` → write k/v to slot `phys_len` of each region → push post-RoPE
   q into the circular window cache → `decode_wrapper.run` → o proj → MLP
   (fused gate_up GEMM → `silu_and_mul`).
3. After the step: lengths advance, sampling, EOS/max-len bookkeeping, then the
   per-request compaction check (`phys_len == budget + buffer`).

## 3. Slot accounting

`logical_len` (RoPE positions, never shrinks) vs `phys_len` (slots in use,
collapses to `budget` at compaction); static per-request regions sized
`max_seq_len` (FullKV) or `budget + buffer` (R-KV). See
[`DESIGN.md`](./DESIGN.md) §5.1.

## 4. Compaction path

All (layer, row) pairs of a trigger step batch into few `update_kv` calls via
`compressor.compact_batch` (chunked to a ~512MB scoring-transient budget, ≤32
pairs; first-call bit-parity gate vs the per-pair path with permanent fallback
if batched GEMMs differ); kept token sets gather per kv-head to the front of
each region; eager, main stream, not graph-captured. See
[`DESIGN.md`](./DESIGN.md) §5.3.

## 5. Prefill and prefill compression

Ragged causal prefill (no padding); HF-reference prefill compression with the
last `window_size` prompt queries when `prompt_len > budget`; window-query
cache seeded from the prompt either way. See [`DESIGN.md`](./DESIGN.md) §5.4.

## 6. Tests

- CPU: `python tests/test_rkv_algo.py`, `python tests/test_cross_repo_parity.py`,
  and `python tests/test_fa3_engine.py` — all print `all tests passed`;
  run on every push via `cpu-tests.yml`.
- GPU: `RKV_ON=1 python tests/smoke/flashinfer_smoke.py` and
  `python tests/smoke/fa3_parity.py --model /path/to/model` from the repo root,
  then the validation plan in [`DESIGN.md`](./DESIGN.md) §8.

## 7. Bring-up log

GPU validation ran on a 2× H100-80GB node (2026-07-08/09). Bugs found on the
way, in discovery order, symptom → root cause → fix:

1. **`generate()` crashed on chat-templated prompts** (`ValueError: too many
   dimensions 'str'`). transformers 5.x changed `tokenizer.apply_chat_template`
   defaults: it now returns a string, and with `tokenize=True` returns a
   `BatchEncoding` dict. Harness fix only (examples + smoke): pass
   `tokenize=True, return_dict=False` to get plain `list[int]`.
2. **Qwen3 / Llama-3 numerically wrong from prefill step 0** while
   Qwen2-family models matched HF exactly. Teacher-forced logits probe showed
   max|Δ| 6–9 vs HF's own prefill/decode noise floor of 0.3–0.6. Root cause:
   transformers 5.x deletes the top-level `config.rope_theta` attribute
   (standardized into `config.rope_parameters`), so
   `getattr(config, "rope_theta", 10000.0)` silently built every cos/sin cache
   with θ=10000 — coincidentally correct for Qwen2.5/R1-Distill (true θ is
   10000), wrong for Qwen3 (θ=1e6) and Llama-3.2 (θ=5e5 + llama3 scaling).
   Fix in `models.py`: `_rope_parameters()` normalizes 4.x
   (`rope_theta`/`rope_scaling`, legacy `type` key) and 5.x
   (`rope_parameters`) into one dict, plus a loud `NotImplementedError` for
   `partial_rotary_factor != 1.0`. After the fix all four probed models
   (Qwen3-0.6B/8B, Llama-3.2-1B, R1-Distill-1.5B) sit at or below HF's own
   kernel-path noise floor with 62–64/64 greedy argmax agreement over 64
   teacher-forced steps.
3. **Smoke test failed on short-answer instruct models** (Llama-3.2-1B answers
   in ~150 tokens; compaction with `budget=256, buffer=32` never fired and the
   RKV_NEVER_TRIGGERED guard tripped). Smoke now defaults to
   `budget=64, buffer=16` (override via `RKV_SMOKE_BUDGET`/`RKV_SMOKE_BUFFER`).
4. **`torch==2.11.0+cu129` is not installable next to
   `flashinfer-python==0.6.12`** — the flashinfer wheel requires `torch<2.11`
   and pip/uv resolve the pair by downgrading to `torch==2.10.0` (cu128).
   `requirements-rkv.txt` now pins the stack this backend was actually
   validated with.

FullKV correctness vs HF transformers (greedy, bf16): R1-Distill-Qwen-1.5B and
-7B produce a 200/200-token identical greedy continuation; the teacher-forced
logits probe (three-way: engine vs HF decode path vs HF prefill path) is the
per-family acceptance gate described above.

## 8. Validated environment

NVIDIA H100 80GB (SM90), driver CUDA 12.8 — Pluto `p5.48xlarge`, 2 GPUs used
as independent single-GPU workers. Python 3.12.11, `torch==2.10.0` (cu128),
`flashinfer-python==0.6.12` + `flashinfer-cubin==0.6.12`,
`transformers==5.8.1`, `safetensors==0.6.2`. Models validated: Qwen2 family
(DeepSeek-R1-Distill-Qwen-1.5B/7B), Qwen3 (0.6B, 8B), Llama (3.2-1B-Instruct).
