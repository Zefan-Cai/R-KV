# R-KV in Mini-SGLang

This document describes the R-KV decode-time KV cache compression port
into Mini-SGLang. The compression algorithm and its driver live in
[`python/minisgl/compress/`](../python/minisgl/compress/); the attention
layer needs a small wire-up to call them. This port mirrors the
Nano-vLLM reference at `Zefan-Cai/R-KV/Nano-vLLM/` and the original
HuggingFace and full-vLLM/SGLang implementations in
[`Zefan-Cai/R-KV`](https://github.com/Zefan-Cai/R-KV).

## Pieces in place

- `python/minisgl/compress/rkv.py` — pure-Python `R1KV` algorithm:
  attention + redundancy-aware similarity scoring, top-k selection,
  `update_kv()` that returns the compacted `(key, value)` tensors.
- `python/minisgl/compress/integration.py` — `RKVCompressor`, a backend-
  agnostic driver that:
  - keeps a sliding window of the last `window_size` queries per
    request,
  - reads a request's KV slots out of the paged cache pool,
  - runs `R1KV.update_kv(...)`,
  - writes the compacted KV back to the first `new_len` slots of the
    same page allocation,
  - returns the new per-request cache lengths for the scheduler to
    propagate.
- `python/minisgl/engine/config.py` — `EngineConfig.rkv_*` fields and
  the `rkv_config` cached property that builds an `RKVConfig` instance.

## Wiring (now in main)

The `RKVCompressor` lives on the global `Context` as `ctx.rkv` and is
instantiated by `Engine.__init__` when `EngineConfig.rkv_enabled=True`.
`AttentionLayer.forward` invokes it directly, so no per-layer
constructor change is needed:

```python
# python/minisgl/layers/attention.py (now in main)
def forward(self, qkv: torch.Tensor) -> torch.Tensor:
    ctx = get_global_ctx()
    q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
    ...
    q = q.view(-1, self.num_qo_heads, self.head_dim)
    if ctx.rkv is not None and ctx.rkv.is_enabled:
        ctx.rkv.update_query_buffer(q, ctx.batch, self.layer_id)
    o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
    if ctx.rkv is not None and ctx.rkv.is_enabled and ctx.batch.is_decode:
        new_lens = ctx.rkv.maybe_compress(
            self.layer_id, ctx.kv_cache,
            ctx.batch.attn_metadata.page_table, ctx.batch,
        )
        if new_lens is not None and self.layer_id == ctx.rkv.num_layers - 1:
            for req, new_len in zip(ctx.batch.padded_reqs, new_lens):
                if new_len < req.device_len:
                    req.device_len = new_len
                    req.cached_len = max(0, new_len - 1)
    return o.view(-1, self.qo_attn_dim)
```

Compression schedule per step:

1. *Before* `attn_backend.forward` — record the trailing window of
   per-layer queries via `update_query_buffer(q, batch, layer_id)`. The
   per-`(uid, layer_id)` buffering matters because each layer projects
   its own `q`, and R-KV scores layer-specific K/V against the
   matching layer's queries.
2. *Inside* `attn_backend.forward` — the backend stores this step's
   K/V into the cache and attends over the full (uncompressed) cache.
   The current step's outputs are unaffected by R-KV.
3. *After* `attn_backend.forward` — `maybe_compress(layer_id, ...)`
   compacts this layer's K/V cache in place: each request's kept K/V
   move to the first `budget` slots of its page allocation, so the
   shared `page_table` stays valid.
4. On the **last** layer only, the request's `device_len` /
   `cached_len` are updated to the new shorter length. The scheduler
   sees the shorter cache on the next step and rebuilds
   `metadata.cache_seqlens` from it.

When a request finishes, the scheduler should call
`ctx.rkv.drop_request(uid)` so the cached query window for that uid
across all layers is freed. (Hook this into the scheduler's
finished-request path; not yet wired.)

## Constraints

- **Eager mode required.** R-KV mutates `cache_seqlens` per step. CUDA
  graphs and `torch.compile` must be disabled. Set
  `cuda_graph_bs=None` and `cuda_graph_max_bs=None` in `EngineConfig`
  when enabling R-KV.
- **Page size 1.** The current integration assumes `page_size=1`, which
  matches Mini-SGLang's default. For `page_size > 1`, slot resolution
  in `maybe_compress` needs to multiply page indices by `page_size`
  before reading from the flat cache.
- **No prefix-cache reuse of compressed pages.** Once a sequence's KV
  cache has been compressed, the page allocation no longer represents
  the verbatim prefix. Make sure `RadixCache` does not match incoming
  prefixes against compressed branches.
- **Single-GPU first.** TP is not yet supported by this prototype.
  Compression runs on rank 0's slice of the KV cache; for TP > 1, each
  rank should compress its own KV shard and the per-request `new_len`
  list must agree across ranks.

## Validation status

The algorithm module, integration helper, and `AttentionLayer` wiring
are all in `main`. The only thing not yet GPU-validated is the
end-to-end run on a real Mini-SGLang serve job; pure-Python parsing of
all touched files succeeds. Open items:

- [x] Algorithm port (`python/minisgl/compress/rkv.py`).
- [x] Per-(uid, layer_id) compressor (`python/minisgl/compress/integration.py`).
- [x] `EngineConfig.rkv_*` knobs and `rkv_config` cached property.
- [x] `ctx.rkv` field on the global `Context`, instantiated by the
  Engine when `rkv_enabled=True`.
- [x] `AttentionLayer.forward` calls `update_query_buffer` /
  `maybe_compress` with the last-layer guard on `device_len`.
- [ ] Smoke-test on an H100 with `Qwen3-0.6B` and a long-CoT prompt.
- [ ] Hook `ctx.rkv.drop_request(uid)` into the scheduler's
  finished-request path so the query buffer can't leak.
- [ ] Confirm the offline `LLM` constructor surfaces `rkv_enabled`
  through `**kwargs` (it should — `SchedulerConfig(EngineConfig)`
  inherits the new fields).
- [ ] Compare token throughput and answer fidelity against the
  HuggingFace R-KV reference at matching budgets.

Track progress in the parent R-KV repo. The pure-Python algorithm has
been kept bit-for-bit equivalent to the Nano-vLLM reference, so any
algorithm-level bugs should reproduce there too.
