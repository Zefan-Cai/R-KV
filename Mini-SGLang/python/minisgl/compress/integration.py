"""Mini-SGLang integration helpers for the R-KV compressor.

The :class:`RKVCompressor` glues the pure-algorithm :class:`R1KV` against
Mini-SGLang's paged KV cache pool. It is intentionally backend-agnostic:
it reads/writes through ``kvcache.k_cache(layer_id)`` /
``kvcache.v_cache(layer_id)`` slots, and the attention backend only
needs to invoke ``maybe_compress`` in the decode path before forwarding
into FA/FlashInfer/TRT-LLM. See ``docs/RKV.md`` for the wiring contract.
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
    """Per-layer R-KV driver that keeps a sliding query buffer per request.

    Mini-SGLang's attention path stores K/V into a paged cache via
    ``kvcache.store_kv(...)``. To compress during decode we need to:

    1. Maintain the last ``window_size`` queries for each live request
       (so the score can be computed without re-running prefill).
    2. Read the request's KV slots out of the paged cache into a dense
       ``(1, kv_heads, kv_len, head_dim)`` tensor.
    3. Run :meth:`R1KV.update_kv`.
    4. Write the compressed K/V back into the *first* ``new_len`` slots
       of the same page allocation and update the request's logical
       cache length.

    The Mini-SGLang scheduler must then reflect the new shorter cache
    length on the next step (e.g. through ``cache_seqlens``) so the
    attention kernel skips the freed tail.
    """

    def __init__(self, config: RKVConfig, num_layers: int) -> None:
        self.config = config
        self.num_layers = num_layers
        self.r1kv = config.make_r1kv()
        # uid -> (window_size, num_kv_heads, head_dim) cached query slice
        self._query_buffer: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # query buffer maintenance
    # ------------------------------------------------------------------
    def update_query_buffer(self, q: torch.Tensor, batch: "Batch") -> None:
        """Stash the last ``window_size`` queries from this forward step.

        ``q`` is the per-token query tensor before scattering into the
        attention kernel. It has shape ``(total_tokens, num_qo_heads,
        head_dim)`` in Mini-SGLang's convention.
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
                self._query_buffer[req.uid] = tail.permute(1, 0, 2).contiguous()
                offset = end
            return
        # decode: one token per request
        for i, req in enumerate(reqs):
            cur = q[i : i + 1].detach().permute(1, 0, 2).contiguous()
            prev = self._query_buffer.get(req.uid)
            if prev is None:
                self._query_buffer[req.uid] = cur
            else:
                merged = torch.cat([prev.to(cur.device), cur], dim=1)
                self._query_buffer[req.uid] = merged[:, -win:, :].contiguous()

    def drop_request(self, uid: int) -> None:
        self._query_buffer.pop(uid, None)

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
        ``batch.padded_reqs``) when compression ran, or ``None`` when
        R-KV is disabled / the batch is in prefill mode. The caller is
        responsible for propagating the new lengths into the attention
        metadata (``cache_seqlens``) and back into ``req.cached_len`` so
        the scheduler frees the dropped slots.
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

            queries = self._query_buffer.get(req.uid)
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
        return new_lens
