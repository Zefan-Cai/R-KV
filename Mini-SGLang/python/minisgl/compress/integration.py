"""Mini-SGLang integration helpers for the R-KV compressor.

The :class:`RKVCompressor` glues the pure-algorithm :class:`R1KV` against
Mini-SGLang's paged KV cache pool. It is intentionally backend-agnostic:
the attention layer calls :meth:`update_query_buffer` before forwarding
into FA/FlashInfer/TRT-LLM, and :meth:`maybe_compress` after each layer
finishes attention. Compression is per ``(uid, layer_id)`` so each
layer's K/V is scored against its own recent queries. See
``docs/RKV.md`` for the full wiring contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .rkv import R1KV

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCachePool


@dataclass(frozen=True)
class RKVConfig:
    """User-facing knobs that mirror the HuggingFace R-KV reference."""

    enabled: bool = False
    budget: int = 1024
    buffer: int = 128
    window_size: int = 8
    kernel_size: int = 7
    mix_lambda: float = 0.07
    retain_ratio: float = 0.1
    retain_direction: str = "last"

    def make_r1kv(self) -> R1KV | None:
        if not self.enabled:
            return None
        return R1KV(
            budget=self.budget,
            window_size=self.window_size,
            kernel_size=self.kernel_size,
            mix_lambda=self.mix_lambda,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )


class RKVCompressor:
    """Per-(uid, layer_id) R-KV driver against a paged KV cache pool.

    The driver maintains one sliding query window per ``(uid, layer_id)``
    pair because the projection that produces ``q`` is layer-specific.
    Compression triggers when ``req.device_len > budget + buffer`` and
    always compacts the kept K/V into the first ``budget`` slots of the
    request's page allocation, so every layer ends a step at the same
    ``device_len`` and the shared page table stays in sync.

    Compression schedule per step:

    1. ``AttentionLayer`` calls :meth:`update_query_buffer` *before*
       invoking the attention backend, so the buffer reflects the
       queries used to attend over this step's K/V.
    2. ``AttentionLayer`` calls :meth:`maybe_compress` *after* the
       backend has stored this step's K/V into the cache. The cache for
       this layer is compacted in place.
    3. On the *last* layer only, ``AttentionLayer`` updates each
       request's ``device_len`` / ``cached_len`` from the returned
       ``new_lens``, so the scheduler rebuilds metadata with the
       shorter cache lengths on the next step.
    """

    def __init__(self, config: RKVConfig, num_layers: int) -> None:
        self.config = config
        self.num_layers = num_layers
        self.r1kv = config.make_r1kv()
        # (uid, layer_id) -> (num_qo_heads, window, head_dim) cached
        # query slice. Stored on the query tensor's original device.
        self._query_buffer: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def is_enabled(self) -> bool:
        return self.r1kv is not None

    # ------------------------------------------------------------------
    # query buffer maintenance
    # ------------------------------------------------------------------
    def update_query_buffer(
        self, q: torch.Tensor, batch: "Batch", layer_id: int
    ) -> None:
        """Stash the last ``window_size`` queries from this forward step.

        ``q`` is the per-token query tensor *after* the QK projection and
        rotary embedding, immediately before the attention kernel call.
        Shape is ``(total_tokens, num_qo_heads, head_dim)`` in
        Mini-SGLang's convention.
        """
        if self.r1kv is None:
            return
        win = self.config.window_size
        reqs = list(batch.padded_reqs)
        if batch.is_prefill:
            offset = 0
            for req in reqs:
                ext = req.extend_len
                if ext <= 0:
                    continue
                end = offset + ext
                tail = q[max(offset, end - win) : end].detach()
                # (window, num_qo_heads, head_dim) -> (num_qo_heads, window, head_dim)
                self._query_buffer[(req.uid, layer_id)] = (
                    tail.permute(1, 0, 2).contiguous()
                )
                offset = end
            return
        # decode: one token per request
        for i, req in enumerate(reqs):
            cur = q[i : i + 1].detach().permute(1, 0, 2).contiguous()
            key = (req.uid, layer_id)
            prev = self._query_buffer.get(key)
            if prev is None:
                self._query_buffer[key] = cur
            else:
                merged = torch.cat([prev.to(cur.device), cur], dim=1)
                self._query_buffer[key] = merged[:, -win:, :].contiguous()

    def drop_request(self, uid: int) -> None:
        for key in list(self._query_buffer):
            if key[0] == uid:
                self._query_buffer.pop(key, None)

    # ------------------------------------------------------------------
    # compression
    # ------------------------------------------------------------------
    def maybe_compress(
        self,
        layer_id: int,
        kvcache: "BaseKVCachePool",
        page_table: torch.Tensor,
        batch: "Batch",
    ) -> list[int] | None:
        """Compress each request's KV cache for ``layer_id`` if oversized.

        Returns a list of new per-request cache lengths (same order as
        ``batch.padded_reqs``) when at least one request was compressed,
        or ``None`` when R-KV is disabled / the batch is in prefill
        mode / no request crossed the trigger threshold.

        The cache is compacted in place: the kept K/V are written into
        the first ``new_len`` slots of each request's page allocation,
        so the page table itself does not move. The caller is
        responsible for propagating ``new_lens`` into the request's
        ``device_len`` exactly once per step (typically on the last
        layer) so the scheduler frees the dropped slots.
        """
        if self.r1kv is None or batch.is_prefill:
            return None

        trigger_len = self.r1kv.budget + self.config.buffer
        k_cache = kvcache.k_cache(layer_id)
        v_cache = kvcache.v_cache(layer_id)
        num_kv_heads = k_cache.shape[-2]
        head_dim = k_cache.shape[-1]
        flat_k = k_cache.reshape(-1, num_kv_heads, head_dim)
        flat_v = v_cache.reshape(-1, num_kv_heads, head_dim)

        any_compressed = False
        new_lens: list[int] = []
        reqs = list(batch.padded_reqs)
        for i, req in enumerate(reqs):
            source_len = req.device_len
            if source_len <= trigger_len:
                new_lens.append(source_len)
                continue

            slots = page_table[i, :source_len].long()
            keys = (
                flat_k.index_select(0, slots)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )
            values = (
                flat_v.index_select(0, slots)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
            )

            queries = self._query_buffer.get((req.uid, layer_id))
            if queries is None:
                # Without a recorded window we have nothing to score
                # against; skip this request rather than guessing.
                new_lens.append(source_len)
                continue
            queries = queries.unsqueeze(0).to(device=keys.device, dtype=keys.dtype)

            new_keys, new_values = self.r1kv.update_kv(keys, queries, values)
            new_len = new_keys.size(-2)
            dst = slots[:new_len]
            flat_k[dst] = new_keys.squeeze(0).permute(1, 0, 2)
            flat_v[dst] = new_values.squeeze(0).permute(1, 0, 2)
            new_lens.append(new_len)
            if new_len < source_len:
                any_compressed = True
        return new_lens if any_compressed else None
