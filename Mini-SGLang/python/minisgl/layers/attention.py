from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        if ctx.rkv is not None and ctx.rkv.is_enabled:
            ctx.rkv.update_query_buffer(q, ctx.batch, self.layer_id)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        if ctx.rkv is not None and ctx.rkv.is_enabled and ctx.batch.is_decode:
            # Compress this layer's cache in place after attention has
            # already attended over the full cache. The shorter
            # device_len takes effect on the next scheduler step.
            # Use the global page table (indexed by req.table_idx): backend
            # metadata does not uniformly expose per-batch slot rows (the
            # FlashInfer FIMetadata has no page_table attribute).
            new_lens = ctx.rkv.maybe_compress(
                self.layer_id,
                ctx.kv_cache,
                ctx.page_table,
                ctx.batch,
            )
            if new_lens is not None and self.layer_id == ctx.rkv.num_layers - 1:
                # Only the last layer mutates request state, so all
                # layers see the same source_len during this step.
                for req, new_len in zip(ctx.batch.padded_reqs, new_lens):
                    if new_len < req.device_len:
                        req.device_len = new_len
                        req.cached_len = max(0, new_len - 1)
        return o.view(-1, self.qo_attn_dim)
