# R-KV Benchmark (Mini-SGLang)

Offline throughput benchmark for the R-KV port. Unlike the sibling
[`SGLang/`](../../SGLang) port (which drives an HTTP server over `/generate`),
Mini-SGLang's benchmark is **offline** — it constructs an `LLM` engine directly
and flips R-KV on via the engine config.

| File | Purpose |
| --- | --- |
| `eval_math.py` | GSM8K accuracy + throughput vs full KV / `--budget` (numeric judging; the Math-7B report driver) |
| `bench_rkv.py` | Offline generation throughput benchmark with R-KV enabled (mirrors upstream `benchmark/offline/bench.py`) |
| `prepare_data.sh` | Fetch the GSM8K few-shot eval set into `data/` |
| `RESULTS_math7b.md` | Measured full KV vs budget 256/512 on Qwen2.5-Math-7B (H100) |

## Run

```bash
cd Mini-SGLang
scripts/apply_rkv.sh                    # build the patched, pinned tree
cd mini-sglang-src && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .

../benchmark/prepare_data.sh            # fetch the GSM8K eval set into benchmark/data/

# GSM8K accuracy + throughput (0 = full KV; see RESULTS_math7b.md):
python3 ../benchmark/eval_math.py --budget 0   --n 40
python3 ../benchmark/eval_math.py --budget 512 --n 40
python3 ../benchmark/eval_math.py --budget 256 --n 40

python3 ../benchmark/bench_rkv.py       # offline R-KV throughput micro-bench
```

## Config

R-KV knobs are forwarded through `SchedulerConfig(EngineConfig)` →
`EngineConfig.rkv_*` (`rkv_enabled`, `rkv_budget`, `rkv_window_size`,
`rkv_buffer`, ...), which build an `RKVConfig` (`from minisgl.rkv import
RKVConfig`).

> **CUDA graph must stay off.** R-KV needs a variable per-step `cache_seqlens`.
> `cuda_graph_max_bs=None` means *auto-enable* (not disable) — only
> `cuda_graph_max_bs=0` disables capture. The engine also force-disables graph
> capture whenever `rkv_enabled=True`. See [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) §4.

## Status

GPU-validated on Qwen2.5-Math-7B (FlashInfer, budget 256/512, batched):
**accuracy is lossless (95.0% == full KV at both budgets)** with ~224 physical
compactions per sweep and no crashes. See [`RESULTS_math7b.md`](./RESULTS_math7b.md).
(Validation surfaced and fixed an overlap-scheduling race — see
[`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) §4/§6.)
