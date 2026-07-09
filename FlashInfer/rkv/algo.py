"""Pure R-KV scoring/selection algorithm (device-agnostic, no flashinfer).

Faithful port of the repo-root reference (``rkv/utils.py`` +
``rkv/compression/r1_kv.py``), same port shape as ``SGLang/rkv/algo.py`` but
as module functions: ``tests/test_cross_repo_parity.py`` checks ``torch.equal``
bit-parity against the reference. Default hyper-parameters match the reference
class defaults (``mix_lambda=0.07``, ``retain_ratio=0.1``).

Tensor conventions (matching the reference):

* ``query_states``: ``(bsz, q_heads, q_len, head_dim)``
* ``key_states`` / ``value_states``: ``(bsz, kv_heads, kv_len, head_dim)``

``q_heads`` may be a multiple of ``kv_heads`` (grouped-query attention), in
which case scores are pooled across each query group.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "R1KV",
    "cal_similarity",
    "compute_attention_scores",
    "compute_scores",
    "select_indices",
    "update_kv",
]


def compute_attention_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    pooling: str = "max",
) -> torch.Tensor:
    """Importance signal: scaled dot-product attention logits ``q @ k^T``.

    Grouped-query attention is handled by reshaping queries into their kv
    groups and pooling the per-group logits (``max`` or ``mean``) so the
    result is indexed by kv head. Shape: ``(bsz, kv_heads, q_len, kv_len)``.
    """
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
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
    key_states: torch.Tensor,
    threshold: float = 0.5,
    retain_ratio: float = 0.2,
    retain_direction: str = "last",
) -> torch.Tensor:
    """Redundancy signal derived from pairwise key cosine similarity.

    For each key, near-duplicate neighbours (cosine similarity above
    ``threshold``) are found; the most-recent such neighbour (per
    ``retain_direction``) is exempted via ``scatter_`` (the semantics the
    RKV-HS prototype silently lost), and the remaining similarity mass is
    aggregated into a per-key redundancy distribution.
    Shape: ``(bsz, kv_heads, seq_len)``.
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


def compute_scores(
    key_states: torch.Tensor,
    query_states: torch.Tensor,
    window_size: int = 8,
    kernel_size: int = 7,
    mix_lambda: float = 0.07,
    retain_ratio: float = 0.1,
    retain_direction: str = "last",
) -> torch.Tensor:
    """Joint R-KV score per past token: ``λ·attention − (1−λ)·redundancy``.

    ``query_states`` holds (at least) the trailing ``window_size`` observation
    queries. Softmax runs in fp32 and is cast back to the query dtype.
    Shape: ``(bsz, kv_heads, kv_len - window_size)``.
    """
    attn_weights = compute_attention_scores(query_states, key_states)

    attn_weights_sum = (
        nn.functional.softmax(
            attn_weights[:, :, -window_size:, :-window_size],
            dim=-1,
            dtype=torch.float32,
        )
        .mean(dim=-2)
        .to(query_states.dtype)
    )

    attn_cache = F.max_pool1d(
        attn_weights_sum,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    )

    similarity_cos = cal_similarity(
        key_states,
        retain_ratio=retain_ratio,
        retain_direction=retain_direction,
    )[:, :, :-window_size]

    return attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)


def select_indices(
    scores: torch.Tensor,
    budget: int,
    window_size: int,
    sort: bool = True,
) -> torch.Tensor:
    """Kept-token indices per (batch, kv head) from precomputed joint scores.

    ``scores`` covers past tokens ``[0, kv_len - window_size)``; the result is
    the ``budget - window_size`` top-scoring past tokens plus the trailing
    ``window_size`` window tokens, shape ``(bsz, kv_heads, budget)`` with
    indices into ``[0, kv_len)``. ``sort=True`` returns ascending (temporal)
    order, which is what paged-cache compaction wants; order is otherwise
    semantically irrelevant because rotary positions are baked into the keys.
    """
    past_indices = scores.topk(budget - window_size, dim=-1).indices
    kv_len = scores.shape[-1] + window_size

    bsz, kv_heads = past_indices.shape[0], past_indices.shape[1]
    window_indices = (
        torch.arange(kv_len - window_size, kv_len, device=scores.device)
        .view(1, 1, -1)
        .expand(bsz, kv_heads, -1)
    )

    kept = torch.cat([past_indices, window_indices], dim=-1)
    if sort:
        kept, _ = torch.sort(kept, dim=-1)
    return kept


def update_kv(
    key_states: torch.Tensor,
    query_states: torch.Tensor,
    value_states: torch.Tensor,
    budget: int = 128,
    window_size: int = 8,
    kernel_size: int = 7,
    mix_lambda: float = 0.07,
    retain_ratio: float = 0.1,
    retain_direction: str = "last",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-faithful compression: returns compressed ``(keys, values)``.

    Below-budget caches are returned unchanged. Token order matches the
    reference exactly (top-scoring past tokens in score order, then the
    observation window), so ``torch.equal`` parity holds.
    """
    head_dim = query_states.shape[-1]
    kv_cache_len = key_states.shape[-2]

    if kv_cache_len < budget:
        return key_states, value_states

    scores = compute_scores(
        key_states,
        query_states,
        window_size=window_size,
        kernel_size=kernel_size,
        mix_lambda=mix_lambda,
        retain_ratio=retain_ratio,
        retain_direction=retain_direction,
    )

    # shape: (bsz, kv_heads, budget - window_size)
    indices = scores.topk(budget - window_size, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

    k_past_compress = key_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    v_past_compress = value_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    k_cur = key_states[:, :, -window_size:, :]
    v_cur = value_states[:, :, -window_size:, :]
    key_states = torch.cat([k_past_compress, k_cur], dim=2)
    value_states = torch.cat([v_past_compress, v_cur], dim=2)
    return key_states, value_states


class R1KV:
    """R-KV compressor with the SGLang-shaped class API (DESIGN §4).

    Thin stateless wrapper over the module functions above — a single scoring
    implementation keeps ``torch.equal`` bit-parity trivially intact. This is
    the surface ``tests/test_cross_repo_parity.py`` (near-verbatim from the
    SGLang backend) exercises; the engine-side compaction *driver* of the same
    name lives in ``rkv.compressor``.
    """

    def __init__(
        self,
        budget: int = 128,
        window_size: int = 8,
        kernel_size: int = 7,
        mix_lambda: float = 0.07,
        retain_ratio: float = 0.1,
        retain_direction: str = "last",
        **kwargs: object,
    ) -> None:
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction

    def _scores(
        self, key_states: torch.Tensor, query_states: torch.Tensor
    ) -> torch.Tensor:
        """Joint per-past-token score, shape ``(bsz, kv_heads, kv_len - window)``."""
        return compute_scores(
            key_states,
            query_states,
            window_size=self.window_size,
            kernel_size=self.kernel_size,
            mix_lambda=self.mix_lambda,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )

    def select_indices(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        sort: bool = True,
    ) -> torch.Tensor | None:
        """Kept-token indices ``(bsz, kv_heads, budget)``; ``None`` below budget."""
        if key_states.shape[-2] < self.budget:
            return None
        scores = self._scores(key_states, query_states)
        return select_indices(scores, self.budget, self.window_size, sort=sort)

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference-faithful compression; below-budget caches pass through."""
        return update_kv(
            key_states,
            query_states,
            value_states,
            budget=self.budget,
            window_size=self.window_size,
            kernel_size=self.kernel_size,
            mix_lambda=self.mix_lambda,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )
