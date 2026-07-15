# R-KV Design — vLLM port

For the *why* behind R-KV as a method, see the SGLang port's
[`DESIGN.md`](../../SGLang/docs/DESIGN.md); this document focuses on the design
decisions specific to bringing R-KV onto **vLLM v1 (v0.25.1)**.

## 1. The algorithm

R-KV is a **decoding-time** KV-cache compressor. While a model generates a long
output, R-KV periodically evicts the *unimportant* and *redundant* past tokens,
keeping only a fixed `budget` of KV entries per request.

Each past token gets a joint score

$$\text{score} = \lambda \cdot \text{importance} - (1-\lambda)\cdot\text{redundancy}$$

- **importance** — max-pooled attention mass from a trailing observation window
  of `window_size` queries (`compute_attention_scores` + `F.max_pool1d`).
- **redundancy** — a key cosine-similarity term (`cal_similarity`): a token that
  is highly similar to others is redundant and cheaper to drop.

The top `budget - window_size` past tokens by score are kept, together with the
trailing `window_size` observation tokens. This is implemented device-agnostic
in [`rkv/algo.py`](../rkv/algo.py) (`R1KV.update_kv`) and is CPU bit-parity
tested against the reference.

## 2. The core tension (per-head vs. per-token)

The algorithm is **per-head / per-layer**: different heads may keep different
tokens. But vLLM's paged KV cache is addressed by a shared `slot_mapping` /
block table — one logical position maps to one physical slot **per layer**.

This vLLM port resolves the tension the same way the original proof-of-concept
did: compression runs **per attention layer** inside `forward`, keeping exactly
`budget` entries and writing the survivors into that layer's leading physical
slots. Because every layer keeps the same *count* (`budget`), the block table
stays consistent across layers even though the *identity* of the kept tokens can
differ per layer. (Making a single cross-layer decision, as the SGLang port
does, is a fidelity improvement tracked in
[`OPTIMIZATIONS.md`](./OPTIMIZATIONS.md).)

## 3. Logical vs. physical positions

After eviction a request's KV cache physically **shrinks** to `budget`, but its
*logical* sequence length keeps growing. The port keeps the two separate:

| Quantity | Value | Used for |
| --- | --- | --- |
| **logical position** | `num_computed_tokens + offset` | RoPE (`self.positions`) |
| **physical position** | `logical − num_dropped_tokens` | slot writes, `seq_lens` |

Rotary embeddings stay relative/consistent because RoPE sees logical positions,
while new KV is written to the compacted physical slots and attention runs over
the physical KV length. This is computed in
`GPUModelRunner._rkv_prepare_physical`.

## 4. Lifecycle of `num_dropped_tokens`

`num_dropped_tokens` is the per-request count of evicted KV entries. It flows:

```
scheduler arms request (should_compress, every BUFFER tokens)
  → CachedRequestData carries should_compress + num_dropped_tokens
    → model runner builds occupied_slot_mapping + physical positions
      → attention backend compacts KV, writes num_dropped_tokens_list
        → ModelRunnerOutput.num_dropped_tokens_list
          → scheduler accumulates request.num_dropped_tokens (grows the offset)
```

See [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) for the exact code sites.
