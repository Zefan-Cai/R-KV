# R-KV on Mini-SGLang — Design

**Decoding-time, redundancy-aware KV-cache compression for Mini-SGLang.**

While a model generates a long output, R-KV periodically evicts the
**unimportant** and **redundant** past tokens, keeping only a fixed `budget` of
KV entries per request — freeing GPU memory while preserving generation quality.
This directory ports R-KV onto a **pinned** Mini-SGLang baseline
(`sgl-project/mini-sglang@9a91cfa`) as a standalone package plus a small wiring
patch, mirroring the layout of the sibling [`SGLang/`](../../SGLang) port.

## 1. Algorithm (`rkv/algo.py`)

Per compression trigger, for each layer's K/V:

1. **Importance** — attention of the last `window_size` observation queries over
   the cached keys.
2. **Redundancy** — key cosine-similarity (a token that duplicates another adds
   little).
3. **Joint score** — `importance * mix_lambda - redundancy * (1 - mix_lambda)`.
4. **Selection** — keep the top `budget - window_size` past tokens **plus** the
   trailing `window_size` observation tokens (always retained), then
   `update_kv()` returns the compacted `(key, value)`.

`R1KV` is pure PyTorch (zero `minisgl` deps) and is **bit-for-bit equivalent** to
the Nano-vLLM / HuggingFace references in the parent repo.

## 2. Integration (`rkv/integration.py`)

`RKVCompressor` is a backend-agnostic driver, keyed per `(uid, layer_id)`:

- keeps a sliding window of the last `window_size` queries per request/layer
  (`update_query_buffer`, called *before* attention);
- after a decode layer attends, `maybe_compress(layer_id, ...)` reads that
  request's KV slots out of the paged pool, runs `R1KV.update_kv`, writes the
  compacted KV back to the **first `new_len` slots of the same page allocation**
  (so the shared `page_table` stays valid), and returns the new per-request
  lengths;
- `drop_request(uid)` frees a finished request's query window across all layers;
- dropped tail slots are enqueued on `_pending_free_slots` and drained by the
  scheduler back to the cache manager.

## 3. Key design decision — logical vs physical length

Mini-SGLang's `req.device_len` / `cached_len` are used for **both** RoPE
positions / token-pool indexing / the `max_tokens` stop condition **and** KV page
allocation. R-KV breaks that identity: after eviction the physical KV shrinks,
but future tokens must keep their **original absolute positions**.

Resolution (mirrors the SGLang port's "scheme A"):

- `device_len` / `cached_len` stay **logical** (total token count) — shrinking
  them would reset RoPE and make requests generate forever.
- A new `req.num_dropped_tokens` tracks evictions; the **physical** KV length is
  the derived `req.kv_len = device_len - num_dropped_tokens`, which the scheduler
  and attention backends use for page allocation, out-slot lookup, and
  `cache_seqlens`.
- `num_dropped_tokens` is bumped only on the **last** layer, once all layers have
  compacted consistently.

## 4. Constraints / support matrix

| Config | Status |
| --- | --- |
| `page_size = 1`, single GPU, eager decode | ✅ ported, CPU-tested |
| Batch > 1 (per-`(uid, layer_id)` triggering) | ✅ ported |
| CUDA graph / `torch.compile` | ❌ eager only — the engine force-disables graph capture when `rkv_enabled=True` (replay would skip the Python compression hooks) |
| `page_size > 1` | ❌ `maybe_compress` slot resolution assumes `page_size=1` |
| Prefix-cache reuse of compressed pages | ❌ compressed pages no longer represent the verbatim prefix; `RadixCache` must not match against compressed branches |
| Tensor parallel (`tp > 1`) | ❌ prototype compresses rank-0's shard; per-rank `new_len` must agree across ranks |
| End-to-end GPU serving | ⏳ not yet validated (open follow-up) |

See [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) for the concrete wiring and the
build/apply flow.
