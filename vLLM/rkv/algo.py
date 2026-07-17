"""Pure R-KV compression algorithm (device-agnostic).

This is a faithful port of the reference R-KV implementation (``R1KV`` from the
R-KV repository) and of the algorithm that was validated in the original vLLM
proof-of-concept (``vllm/r1_kv/modeling.py``). It intentionally has **no** vLLM
dependencies so it can be unit-tested on CPU and compared bit-for-bit against
the original, and it runs on whatever device the input tensors live on (GPU in
production).

Tensor conventions (matching the reference):

* ``query_states``: ``(bsz, q_heads, q_len, head_dim)``
* ``key_states`` / ``value_states``: ``(bsz, kv_heads, kv_len, head_dim)``

R-KV supports grouped-query attention: ``q_heads`` may be a multiple of
``kv_heads``, in which case importance scores are pooled across each query
group.

The joint score is

    score = mix_lambda * importance - (1 - mix_lambda) * redundancy

where *importance* is a max-pooled attention mass over a trailing observation
window and *redundancy* is a key cosine-similarity term. The top
``budget - window_size`` past tokens by score are kept, together with the
trailing ``window_size`` observation tokens.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["R1KV", "cal_similarity", "compute_attention_scores"]

logger = logging.getLogger(__name__)


def compute_attention_scores(query_states, key_states, pooling="max"):
    """Importance signal: scaled dot-product attention logits ``q @ k^T``.

    Grouped-query attention is handled by reshaping queries into their kv
    groups and pooling the per-group logits (``max`` or ``mean``) so the result
    is indexed by kv head.
    """
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"q_heads ({q_heads}) must be a multiple of kv_heads ({kv_heads}) "
            "for grouped-query attention pooling."
        )
    query_group_size = q_heads // kv_heads

    if query_group_size == 1:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
    else:
        query_states = query_states.view(
            batch_size, kv_heads, query_group_size, q_len, head_dim
        )
        key_states = key_states.unsqueeze(2)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 4)
        ) / math.sqrt(head_dim)
        if pooling == "mean":
            attn_weights = attn_weights.mean(dim=2)
        elif pooling == "max":
            attn_weights = attn_weights.max(dim=2).values
        else:
            raise ValueError("Pooling method not supported")

    return attn_weights


def cal_similarity(
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    """Redundancy signal: per-token key cosine-similarity mass.

    Builds the ``(kv_len, kv_len)`` key cosine-similarity matrix, zeroes the
    diagonal, keeps only entries above ``threshold``, discounts a retained
    representative per redundant group (per ``retain_direction``), then reduces
    to a per-token redundancy score via a softmax over the column mean. A high
    score marks a token that is highly similar to (redundant with) others.
    """
    _, _, seq_len, _ = key_states.shape

    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_states.device)
    similarity_cos.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)

    similarity_mask = similarity_cos > threshold
    k = max(1, int(seq_len * retain_ratio))
    positions = torch.arange(
        seq_len, device=similarity_mask.device
    ).view(1, 1, 1, seq_len)

    if retain_direction in ("last", "last_percent"):
        # Non-matching entries -> 0 so the largest-index reductions ignore them.
        indices = torch.where(
            similarity_mask,
            positions,
            torch.zeros_like(similarity_mask, dtype=torch.long),
        )
        if retain_direction == "last":
            similarity_retain = torch.max(indices, dim=-1)[0]
        else:
            similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]
    elif retain_direction in ("first", "first_percent"):
        # Non-matching entries -> ``seq_len`` sentinel so the smallest-index
        # reductions skip them. (Using 0 like the reference makes every row pick
        # position 0, because masking the diagonal leaves a 0 in almost every
        # row -- the bug this fixes.)
        sentinel = torch.full_like(similarity_mask, seq_len, dtype=torch.long)
        indices = torch.where(similarity_mask, positions, sentinel)
        if retain_direction == "first":
            similarity_retain = torch.min(indices, dim=-1)[0]
        else:
            similarity_retain = torch.topk(
                indices, k=k, dim=-1, largest=False
            )[0][:, :, -1]
        # Rows with no match hold the out-of-range sentinel; clamp so the
        # scatter below targets a valid (harmless) slot, not out of bounds.
        similarity_retain = similarity_retain.clamp(max=seq_len - 1)
    else:
        raise ValueError("retain_direction not supported")

    similarity_cos.scatter_(-1, similarity_retain.unsqueeze(-1), 0)
    return similarity_cos.mean(dim=-2).softmax(dim=-1)


class R1KV:
    """Redundancy-aware KV cache compressor (decode-time).

    ``update_kv`` scores the current per-request KV cache and returns the
    compressed ``(key_states, value_states)`` down to ``budget`` entries
    (``budget - window_size`` selected past tokens plus the trailing
    ``window_size`` observation tokens). If the cache is shorter than
    ``budget`` it is returned unchanged.
    """

    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction

    def _scores(self, key_states, query_states):
        """Per-past-token joint R-KV score.

        Returns ``final_score`` of shape
        ``(bsz, kv_heads, kv_cache_len - window_size)``.

        Split out of :meth:`update_kv` so the serving integration can compute a
        per-layer score, reduce it across KV heads, and accumulate it across
        layers into a single global eviction decision (mirroring the SGLang
        R-KV port's cross-head-mean + cross-layer-sum reduction).
        """
        attn_weights = compute_attention_scores(query_states, key_states)
        attn_weights_sum = (
            nn.functional.softmax(
                attn_weights[:, :, -self.window_size :, : -self.window_size],
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
        )[:, :, : -self.window_size]

        return attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            return key_states, value_states

        final_score = self._scores(key_states, query_states)
        indices = final_score.topk(self.budget - self.window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states
