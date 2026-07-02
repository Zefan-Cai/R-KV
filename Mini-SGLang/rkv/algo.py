"""Core R-KV algorithm — attention + redundancy-aware KV cache scoring.

Ported from the Nano-vLLM R-KV reference at
``Zefan-Cai/R-KV/Nano-vLLM/nanovllm/layers/rkv.py`` (commit 957482e) so
the algorithm matches across implementations bit-for-bit. The HuggingFace
reference lives at ``Zefan-Cai/R-KV/HuggingFace/rkv/compression/r1_kv.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_attention_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    pooling: str = "max",
) -> torch.Tensor:
    """Compute scaled-dot-product attention scores with GQA pooling.

    Args:
        query_states: shape ``(batch, q_heads, q_len, head_dim)``.
        key_states: shape ``(batch, kv_heads, k_len, head_dim)``.
        pooling: how to reduce across query heads in a GQA group
            (``"max"`` matches the R-KV paper).

    Returns:
        Attention logits with shape ``(batch, kv_heads, q_len, k_len)``.
    """
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    if query_group_size == 1:
        return torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    query_states = query_states.view(
        batch_size, kv_heads, query_group_size, q_len, head_dim
    )
    key_states = key_states.unsqueeze(2)
    attn_weights = torch.matmul(query_states, key_states.transpose(3, 4)) / math.sqrt(
        head_dim
    )
    if pooling == "mean":
        return attn_weights.mean(dim=2)
    if pooling == "max":
        return attn_weights.max(dim=2).values
    raise ValueError("Pooling method not supported")


def cal_similarity(
    key_states: torch.Tensor,
    threshold: float = 0.5,
    retain_ratio: float = 0.2,
    retain_direction: str = "last",
) -> torch.Tensor:
    """Compute the redundancy score used as the R-KV penalty term.

    The score is the row-mean of a masked cosine-similarity matrix over
    keys, softmaxed across keys so it can be linearly mixed with the
    attention score in :class:`R1KV.update_kv`.
    """
    _, _, seq_len, _ = key_states.shape

    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_states.device)
    similarity_cos.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)

    similarity_mask = similarity_cos > threshold
    k = max(1, int(seq_len * retain_ratio))
    indices = torch.where(
        similarity_mask,
        torch.arange(seq_len, device=similarity_mask.device).view(1, 1, 1, seq_len),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]
    else:
        raise ValueError("retain_direction not supported")

    similarity_cos.scatter_(-1, similarity_retain.unsqueeze(-1), 0)
    return similarity_cos.mean(dim=-2).softmax(dim=-1)


class R1KV:
    """R-KV decode-time KV cache compressor for a single sequence.

    Calling :meth:`update_kv` with the current per-head key/value cache
    and the last few queries returns a compacted ``(key, value)`` pair
    where only the highest-scoring ``budget`` tokens are retained.
    """

    def __init__(
        self,
        budget: int = 128,
        window_size: int = 8,
        kernel_size: int = 7,
        mix_lambda: float = 0.07,
        retain_ratio: float = 0.1,
        retain_direction: str = "last",
    ) -> None:
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len <= self.budget:
            return key_states, value_states

        local_window = min(self.window_size, kv_cache_len - 1)
        if local_window <= 0:
            return key_states, value_states

        attn_weights = compute_attention_scores(query_states, key_states)
        attn_weights_sum = (
            F.softmax(
                attn_weights[:, :, -local_window:, :-local_window],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)
            .to(query_states.dtype)
        )
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        similarity_cos = cal_similarity(
            key_states,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )[:, :, :-local_window]

        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
        keep_past = self.budget - local_window
        indices = final_score.topk(keep_past, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_states[:, :, :-local_window, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-local_window, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, -local_window:, :]
        v_cur = value_states[:, :, -local_window:, :]
        return (
            torch.cat([k_past_compress, k_cur], dim=2),
            torch.cat([v_past_compress, v_cur], dim=2),
        )
