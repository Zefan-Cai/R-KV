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

## Wiring contract for the attention layer

The attention layer should construct one `RKVCompressor` per attention
backend (it carries per-request query history, so it is stateful) and
invoke it in the decode path. A reference wiring inside
`python/minisgl/layers/attention.py` looks like this:

```python
from minisgl.compress import RKVCompressor

class AttentionLayer(StateLessOP):
    def __init__(self, layer_id, ..., rkv: RKVCompressor | None = None):
        ...
        self.rkv = rkv

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        ...
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        if self.rkv is not None:
            # Stash the trailing query window before scattering into the
            # attention kernel; the compressor uses this on the *next*
            # step to score what to keep.
            self.rkv.update_query_buffer(q, ctx.batch)
            new_lens = self.rkv.maybe_compress(
                self.layer_id,
                ctx.kv_cache,
                ctx.batch.attn_metadata.page_table,
                ctx.batch,
            )
            if new_lens is not None:
                # Patch in the new cache lengths so the kernel ignores
                # the freed tail. For FlashAttentionBackend this means
                # rewriting ``metadata.cache_seqlens``; other backends
                # have an equivalent field.
                ctx.batch.attn_metadata.cache_seqlens = torch.tensor(
                    new_lens,
                    device=ctx.batch.attn_metadata.cache_seqlens.device,
                    dtype=ctx.batch.attn_metadata.cache_seqlens.dtype,
                )
                for req, new_len in zip(ctx.batch.padded_reqs, new_lens):
                    req.device_len = new_len
                    req.cached_len = new_len - 1
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
```

Construction in the engine path:

```python
# python/minisgl/engine/engine.py (sketch)
from minisgl.compress import RKVCompressor

rkv = RKVCompressor(
    config=engine_config.rkv_config,
    num_layers=engine_config.model_config.num_hidden_layers,
)
# Pass `rkv` to each AttentionLayer during model construction so the
# layers share one compressor (it indexes requests by uid, not layer).
```

When a request finishes, the scheduler should call
`rkv.drop_request(uid)` so the cached query window is freed.

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

The algorithm module and integration helper are ported but have not
been GPU-validated against Mini-SGLang's runtime. Open items:

- [ ] Hook `RKVCompressor` into `AttentionLayer` per the snippet above.
- [ ] Add an `--rkv` flag (or equivalent) to the offline `LLM`
  constructor and benchmark scripts.
- [ ] Smoke-test on an H100 with `Qwen3-0.6B` and a long-CoT prompt.
- [ ] Compare token throughput and answer fidelity against the
  HuggingFace R-KV reference at matching budgets.

Track progress in the parent R-KV repo. The pure-Python algorithm has
been kept bit-for-bit equivalent to the Nano-vLLM reference, so any
algorithm-level bugs should reproduce there too.
