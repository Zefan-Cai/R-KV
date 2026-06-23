import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.rkv import R1KV
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.rkv_enabled = False
        self.rkv_buffer = 128
        self.rkv_window_size = 8
        self.rkv: R1KV | None = None
        self._rkv_query_cache: dict[int, torch.Tensor] = {}

    def configure_rkv(
        self,
        enabled: bool = False,
        budget: int = 1024,
        buffer: int = 128,
        window_size: int = 8,
        kernel_size: int = 7,
        mix_lambda: float = 0.07,
        retain_ratio: float = 0.1,
        retain_direction: str = "last",
    ):
        self.rkv_enabled = enabled
        self.rkv_buffer = buffer
        self.rkv_window_size = window_size
        self._rkv_query_cache.clear()
        self.rkv = R1KV(
            budget=budget,
            window_size=window_size,
            kernel_size=kernel_size,
            mix_lambda=mix_lambda,
            retain_ratio=retain_ratio,
            retain_direction=retain_direction,
        ) if enabled else None

    def _logical_slots(self, block_table: torch.Tensor, length: int, block_size: int) -> torch.Tensor:
        positions = torch.arange(length, device=block_table.device, dtype=torch.long)
        block_ids = block_table.index_select(0, positions // block_size).long()
        return block_ids * block_size + positions % block_size

    def _update_rkv_query_cache(self, q: torch.Tensor):
        context = get_context()
        if not self.rkv_enabled or context.seq_ids is None:
            return

        seq_ids = context.seq_ids.detach().cpu().tolist()
        if context.is_prefill:
            if context.cu_seqlens_q is None:
                return
            cu_seqlens_q = context.cu_seqlens_q.detach().cpu().tolist()
            for i, seq_id in enumerate(seq_ids):
                start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
                if end <= start:
                    continue
                query_window = q[start:end].detach().permute(1, 0, 2).contiguous()
                self._rkv_query_cache[int(seq_id)] = query_window[:, -self.rkv_window_size:, :].clone()
            return

        for i, seq_id in enumerate(seq_ids):
            cur_query = q[i].detach().unsqueeze(1)
            prev_query = self._rkv_query_cache.get(int(seq_id))
            if prev_query is None:
                query_window = cur_query
            else:
                query_window = torch.cat([prev_query.to(cur_query.device), cur_query], dim=1)
            self._rkv_query_cache[int(seq_id)] = query_window[:, -self.rkv_window_size:, :].contiguous()

    def _maybe_compress_kvcache(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if (
            not self.rkv_enabled
            or self.rkv is None
            or context.is_prefill
            or context.seq_ids is None
            or context.context_lens is None
            or context.block_tables is None
        ):
            return context.context_lens

        if context.rkv_source_lens is None:
            context.rkv_source_lens = context.context_lens.clone()
        if context.rkv_target_lens is None:
            context.rkv_target_lens = context.context_lens.clone()

        source_lens = context.rkv_source_lens
        target_lens = context.rkv_target_lens
        seq_ids = context.seq_ids.detach().cpu().tolist()
        block_size = k_cache.size(1)
        flat_k = k_cache.view(-1, self.num_kv_heads, self.head_dim)
        flat_v = v_cache.view(-1, self.num_kv_heads, self.head_dim)
        trigger_len = self.rkv.budget + self.rkv_buffer

        for i, seq_id in enumerate(seq_ids):
            source_len = int(source_lens[i].item())
            if source_len <= trigger_len:
                target_lens[i] = source_len
                continue

            slots = self._logical_slots(context.block_tables[i], source_len, block_size)
            key_states = flat_k.index_select(0, slots).permute(1, 0, 2).unsqueeze(0).contiguous()
            value_states = flat_v.index_select(0, slots).permute(1, 0, 2).unsqueeze(0).contiguous()
            query_states = self._rkv_query_cache.get(int(seq_id))
            if query_states is None:
                query_states = q[i].detach().unsqueeze(1)
            query_states = query_states.unsqueeze(0).to(device=key_states.device, dtype=key_states.dtype)

            key_compress, value_compress = self.rkv.update_kv(key_states, query_states, value_states)
            new_len = key_compress.size(-2)
            dst_slots = slots[:new_len]
            flat_k[dst_slots] = key_compress.squeeze(0).permute(1, 0, 2)
            flat_v[dst_slots] = value_compress.squeeze(0).permute(1, 0, 2)
            target_lens[i] = new_len

        return target_lens

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        self._update_rkv_query_cache(q)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            cache_lens = self._maybe_compress_kvcache(q, k_cache, v_cache)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=cache_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
