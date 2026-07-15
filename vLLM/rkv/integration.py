"""R-KV integration layer for vLLM v1 (FlashAttention backend).

This module bridges the pure :class:`~rkv.algo.R1KV` algorithm to vLLM's paged
KV cache. It encapsulates the per-request scoring and **physical** KV eviction
that would otherwise live inline in the attention backend's ``forward`` — so the
runtime wiring patch stays small and additive.

Design (mirrors the SGLang R-KV port):

* The algorithm is **per-head / per-layer**, but vLLM's paged layout shares one
  ``slot_mapping`` slot across all layers. Compression therefore runs per
  attention layer's ``forward``, physically overwriting that layer's KV cache
  in place and reporting how many tokens were dropped so the scheduler /
  model-runner can shrink the logical→physical position mapping consistently.
* Config is read from environment variables (see :meth:`RKVConfig.from_env`) so
  the wiring patch does not have to touch vLLM's argument parser.

Expected ``attn_metadata`` fields (added by the wiring patch):

* ``num_reqs``: int
* ``seq_lens``: ``(num_reqs,)`` physical KV length per request
* ``query_start_loc``: ``(num_reqs + 1,)`` cumulative query offsets
* ``occupied_slot_mapping``: ``(total_num_kv_cache_tokens,)`` physical slot ids
  of every currently-occupied KV entry, laid out per request
* ``should_compress_list``: ``tuple[bool, ...]`` per-request compression gate
* ``num_dropped_tokens_list``: ``list[int]`` written back in place
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.rkv.algo import R1KV

__all__ = ["RKVConfig", "RKVCompressor"]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class RKVConfig:
    """Decode-time R-KV configuration.

    ``budget`` and ``buffer_size`` mirror the two ``VLLM_V1_R_KV_*`` environment
    variables of the original proof-of-concept; the remaining fields expose the
    algorithm knobs with the reference defaults.
    """

    budget: int = 64
    buffer_size: int = 64
    window_size: int = 8
    kernel_size: int = 7
    mix_lambda: float = 0.07
    retain_ratio: float = 0.1
    retain_direction: str = "last"

    @classmethod
    def from_env(cls) -> "RKVConfig":
        return cls(
            budget=_env_int("VLLM_V1_R_KV_BUDGET", 64),
            buffer_size=_env_int("VLLM_V1_R_KV_BUFFER", 64),
            window_size=_env_int("VLLM_V1_R_KV_WINDOW", 8),
            kernel_size=_env_int("VLLM_V1_R_KV_KERNEL", 7),
        )

    @property
    def enabled(self) -> bool:
        return self.budget > 0 and self.buffer_size > 0


class RKVCompressor:
    """Coordinates per-request physical KV eviction for one attention layer."""

    def __init__(self, config: RKVConfig | None = None):
        self.config = config or RKVConfig.from_env()
        # Only build the algorithm when R-KV is actually enabled; when disabled
        # (budget/buffer == 0) this stays a no-op so the attention backend can
        # construct it unconditionally.
        self.algo = (
            R1KV(
                budget=self.config.budget,
                window_size=self.config.window_size,
                kernel_size=self.config.kernel_size,
                mix_lambda=self.config.mix_lambda,
                retain_ratio=self.config.retain_ratio,
                retain_direction=self.config.retain_direction,
            )
            if self.config.enabled
            else None
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def compact_batch(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata,
    ) -> None:
        """Physically evict redundant KV for every armed request in the batch.

        ``key_cache`` / ``value_cache`` are the paged caches for the current
        layer, shape ``[num_blocks, block_size, num_kv_heads, head_dim]``. The
        surviving KV entries are written back into the request's leading
        physical slots (``occupied_slot_mapping[kv_start : kv_start + kept]``)
        and ``attn_metadata.num_dropped_tokens_list`` is updated in place.
        """
        should_compress = attn_metadata.should_compress_list
        num_reqs = attn_metadata.num_reqs
        if not should_compress:
            return
        # ``num_reqs`` may be padded (CUDA-graph); the R-KV lists cover only the
        # real requests, so clamp the loop bound and require the occupied map.
        num_reqs = min(num_reqs, len(should_compress))
        slot_map = attn_metadata.occupied_slot_mapping
        if slot_map is None or not any(should_compress[:num_reqs]):
            return

        budget = self.config.budget
        threshold = budget + self.config.buffer_size

        # Paged KV cache is [num_blocks, block_size, num_kv_heads, head_dim].
        # After ``kv_cache.unbind`` the per-layer cache is non-contiguous, so we
        # address it with (block, offset) advanced indexing (which gathers on
        # read and scatters in place on write) rather than a flat ``.view``.
        block_size = key_cache.size(1)

        seq_lens = attn_metadata.seq_lens
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        query_start_loc = attn_metadata.query_start_loc

        for i in range(num_reqs):
            if not should_compress[i]:
                continue
            if seq_lens[i].item() < threshold:
                continue

            kv_start = seq_starts[i].item()
            kv_end = seq_ends[i].item()
            slots = slot_map[kv_start:kv_end]
            blk = slots // block_size
            off = slots % block_size

            current_key_cache = key_cache[blk, off]
            current_value_cache = value_cache[blk, off]

            # [num_heads, num_tokens, head_dim]
            q_start = query_start_loc[i].item()
            q_end = query_start_loc[i + 1].item()
            current_query = query[q_start:q_end].transpose(0, 1)
            current_key_cache = current_key_cache.transpose(0, 1)
            current_value_cache = current_value_cache.transpose(0, 1)

            # [batch_size, num_heads, num_tokens, head_dim]
            current_query = current_query.unsqueeze(0)
            current_key_cache = current_key_cache.unsqueeze(0)
            current_value_cache = current_value_cache.unsqueeze(0)

            current_kv_len = current_key_cache.size(2)
            compressed_key_cache, compressed_value_cache = self.algo.update_kv(
                current_key_cache,
                current_query,
                current_value_cache,
            )
            compressed_key_cache = compressed_key_cache.squeeze(0)
            compressed_value_cache = compressed_value_cache.squeeze(0)

            compressed_kv_len = compressed_key_cache.size(1)
            dst = slot_map[kv_start : kv_start + compressed_kv_len]
            dst_blk = dst // block_size
            dst_off = dst % block_size
            key_cache[dst_blk, dst_off] = compressed_key_cache.transpose(0, 1)
            value_cache[dst_blk, dst_off] = compressed_value_cache.transpose(0, 1)

            num_dropped_tokens_i = current_kv_len - compressed_kv_len
            if num_dropped_tokens_i != attn_metadata.num_dropped_tokens_list[i]:
                assert attn_metadata.num_dropped_tokens_list[i] == 0
                attn_metadata.num_dropped_tokens_list[i] = num_dropped_tokens_i
