# R-KV Benchmark (Mini-SGLang)

Offline throughput benchmark for the R-KV port. Unlike the sibling
[`SGLang/`](../../SGLang) port (which drives an HTTP server over `/generate`),
Mini-SGLang's benchmark is **offline** — it constructs an `LLM` engine directly
and flips R-KV on via the engine config.

| File | Purpose |
| --- | --- |
| `bench_rkv.py` | Offline generation benchmark with R-KV enabled (mirrors upstream `benchmark/offline/bench.py`, forwarding `rkv_*` config) |

## Run

```bash
cd Mini-SGLang
scripts/apply_rkv.sh                    # build the patched, pinned tree
cd mini-sglang-src && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .

python3 ../benchmark/bench_rkv.py       # offline R-KV throughput
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

The algorithm and wiring are CPU-tested and the patch applies cleanly to the
pinned upstream, but an **end-to-end GPU serve run is not yet validated** — that
is the open follow-up (see [`../docs/IMPLEMENTATION.md`](../docs/IMPLEMENTATION.md) §6).
