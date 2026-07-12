"""Pure R-KV compression algorithm (device-agnostic).

This is a faithful port of the reference implementation in the R-KV repository
(``rkv/utils.py`` and ``rkv/compression/r1_kv.py``). It intentionally has **no**
SGLang dependencies so it can be unit-tested on CPU and compared bit-for-bit
against the original, and it runs on whatever device the input tensors live on
(GPU in production).

Tensor conventions (matching the reference):

* ``query_states``: ``(bsz, q_heads, q_len, head_dim)``
* ``key_states`` / ``value_states``: ``(bsz, kv_heads, kv_len, head_dim)``

R-KV supports grouped-query attention: ``q_heads`` may be a multiple of
``kv_heads``, in which case importance scores are pooled across each query group.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["R1KV", "cal_similarity", "compute_attention_scores"]

logger = logging.getLogger(__name__)

# Lazily-loaded fused Triton redundancy kernel (CUDA-only). ``None`` = untried,
# ``False`` = unavailable (no CUDA / no Triton), else the callable. Adopted per
# R1KV instance via a one-time smoke check against the full-matrix reference.
_FUSED_REDUNDANCY = None


def _get_fused_redundancy():
    global _FUSED_REDUNDANCY
    if _FUSED_REDUNDANCY is None:
        try:
            if torch.cuda.is_available():
                from sglang.srt.mem_cache.rkv.redundancy_fused import (
                    cal_similarity_fused,
                )

                _FUSED_REDUNDANCY = cal_similarity_fused
            else:
                _FUSED_REDUNDANCY = False
        except Exception:  # pragma: no cover - triton/build issue -> fallback
            _FUSED_REDUNDANCY = False
    return _FUSED_REDUNDANCY


def compute_attention_scores(query_states, key_states, pooling="max"):
    """Importance signal: scaled dot-product attention logits q @ k^T.

    Grouped-query attention is handled by reshaping queries into their kv
    groups and pooling the per-group logits (``max`` or ``mean``) so the result
    is indexed by kv head.

    Returns a tensor of shape ``(bsz, kv_heads, q_len, kv_len)``.
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
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    """Redundancy signal derived from pairwise key cosine similarity.

    For each key, near-duplicate neighbours (cosine similarity above
    ``threshold``) are found; the most-recent such neighbour (per
    ``retain_direction``) is exempted, and the remaining similarity mass is
    aggregated into a per-key redundancy distribution.

    Returns a tensor of shape ``(bsz, kv_heads, seq_len)``.
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
    """R-KV decoding-time KV cache compressor.

    Combines an attention-based *importance* score with a key-similarity
    *redundancy* score into a single objective::

        score = importance * mix_lambda - redundancy * (1 - mix_lambda)

    and keeps the ``budget - window_size`` highest-scoring past tokens plus the
    most recent ``window_size`` tokens (the observation window, always kept).
    """

    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        fused_validation="first-request",
        **kwargs,
    ):
        if budget - window_size <= 0:
            raise ValueError("R-KV budget must be greater than window_size")
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        # Fused redundancy backend: None=untried, False=reference fallback, else fn.
        self._fused_redundancy = None
        # Fused-kernel adoption policy: "off" (never use the fused kernel),
        # "startup" (validate once with a synthetic tensor via
        # warmup_fused_kernel, so the first real compaction pays no gate cost),
        # or "first-request" (lazy — validate on the first real compaction).
        self._fused_validation = fused_validation
        if fused_validation == "off":
            self._fused_redundancy = False

    def warmup_fused_kernel(self, kv_heads, head_dim, device, dtype, seq_len=None):
        """Startup validation of the fused redundancy kernel.

        Runs the fused-vs-reference A/B gate once on a synthetic key tensor so
        the first real compaction does not pay the gate cost, and a broken /
        unavailable kernel is caught at startup instead of on a user's first
        (possibly long) request. No-op unless ``fused_validation == "startup"``,
        ``retain_direction == "last"``, and the device is CUDA; the adoption
        decision it latches uses the real model's ``kv_heads`` / ``head_dim`` /
        ``dtype``, which is what the kernel's correctness depends on.
        """
        if self._fused_validation != "startup":
            return
        dev = torch.device(device)
        if self.retain_direction != "last" or dev.type != "cuda":
            return
        if self._fused_redundancy is not None:  # already decided (e.g. "off")
            return
        n = int(seq_len or max(self.budget, self.window_size + 1))
        keys = torch.randn((1, kv_heads, n, head_dim), device=dev, dtype=dtype)
        self._redundancy(keys)  # triggers the lazy gate, latches the decision

    def _scores(self, key_states, query_states):
        """Compute the per-past-token joint R-KV score.

        Returns ``final_score`` of shape
        ``(bsz, kv_heads, kv_cache_len - window_size)``.
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

        similarity_cos = self._redundancy(key_states)[:, :, : -self.window_size]

        final_score = attn_cache * self.mix_lambda - similarity_cos * (
            1 - self.mix_lambda
        )
        return final_score

    def _reference_redundancy(self, key_states):
        # Reference redundancy for the fused-kernel smoke gate. On CUDA use the
        # O(n)-memory tiled implementation: the full n x n matrix built by
        # ``cal_similarity`` OOMs on long sequences (e.g. decode-mode compaction
        # of a long prompt, where n = prompt length). ``cal_similarity_tiled`` is
        # bit-parity with ``cal_similarity`` for ``retain_direction="last"``. On
        # CPU (small test tensors, and no serving package on the import path) the
        # full-matrix reference is fine.
        if self.retain_direction == "last" and key_states.is_cuda:
            from sglang.srt.mem_cache.rkv.prefill import cal_similarity_tiled

            return cal_similarity_tiled(
                key_states, threshold=0.5, retain_direction="last"
            )
        return cal_similarity(
            key_states,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )

    def _redundancy(self, key_states):
        """Key-similarity redundancy per past token. On CUDA with
        ``retain_direction='last'`` this uses the fused Triton kernel, adopted
        once via a smoke gate against the full-matrix reference (a gross
        mismatch -> permanent reference fallback, logged). On CPU, without
        Triton, or for other retain directions it uses ``cal_similarity``.
        """
        if self.retain_direction != "last" or not key_states.is_cuda:
            return self._reference_redundancy(key_states)
        if self._fused_redundancy is False:
            return self._reference_redundancy(key_states)
        if self._fused_redundancy is not None:
            # Guard the adopted kernel against SYNCHRONOUS failures (Triton
            # compile / shape / launch errors, wrapper bugs): degrade to the
            # reference permanently instead of crashing. Note the failure model:
            # an ASYNCHRONOUS CUDA fault (illegal access / device-side assert)
            # from the kernel does NOT surface here — it poisons the CUDA context
            # and raises at a later sync point, which is (correctly) worker-fatal;
            # we do not try to recover a poisoned context.
            try:
                return self._fused_redundancy(key_states, threshold=0.5)
            except Exception as e:  # pragma: no cover - hardware/shape dependent
                logger.warning(
                    "R-KV decode fused-redundancy kernel failed at runtime (%s); "
                    "falling back to the reference permanently.",
                    e,
                )
                self._fused_redundancy = False
                return self._reference_redundancy(key_states)
        fn = _get_fused_redundancy()
        if fn is False:
            self._fused_redundancy = False
            return self._reference_redundancy(key_states)
        ref = self._reference_redundancy(key_states)
        try:
            got = fn(key_states, threshold=0.5)
        except Exception as e:  # pragma: no cover - hardware/shape dependent
            logger.warning(
                "R-KV decode fused-redundancy kernel unavailable at runtime (%s); "
                "using the reference permanently.",
                e,
            )
            self._fused_redundancy = False
            return ref
        ok = ref.shape == got.shape and torch.allclose(
            ref.float(), got.float(), atol=1e-3, rtol=1e-2
        )
        self._fused_redundancy = fn if ok else False
        logger.info(
            "R-KV decode fused-redundancy gate: %s",
            "OK -> fused adopted" if ok else "DIVERGED -> reference fallback",
        )
        return got if ok else ref

    def select_indices(self, key_states, query_states, sort=True):
        """Return the token indices to keep, per (batch, kv head).

        This is the primitive the SGLang integration layer needs: it answers
        "which tokens survive compression" without materialising new K/V
        tensors. The result has shape ``(bsz, kv_heads, budget)`` with indices
        into ``[0, kv_cache_len)``.

        The kept set is the ``budget - window_size`` top-scoring past tokens
        plus the trailing ``window_size`` window tokens. When ``sort`` is True
        the indices are returned in ascending (temporal) order, which is what
        the paged-cache compaction wants; ordering is otherwise semantically
        irrelevant because rotary position information is already baked into the
        stored keys.

        Returns ``None`` when ``kv_cache_len < budget`` (no compression).
        """
        kv_cache_len = key_states.shape[-2]
        if kv_cache_len < self.budget:
            return None

        final_score = self._scores(key_states, query_states)
        past_indices = final_score.topk(self.budget - self.window_size, dim=-1).indices

        bsz, kv_heads = past_indices.shape[0], past_indices.shape[1]
        window_indices = (
            torch.arange(
                kv_cache_len - self.window_size,
                kv_cache_len,
                device=past_indices.device,
            )
            .view(1, 1, -1)
            .expand(bsz, kv_heads, -1)
        )

        kept = torch.cat([past_indices, window_indices], dim=-1)
        if sort:
            kept, _ = torch.sort(kept, dim=-1)
        return kept

    def update_kv(self, key_states, query_states, value_states):
        """Faithful reference-compatible compression.

        Returns the compressed ``(key_states, value_states)``. When the cache is
        below budget the inputs are returned unchanged. Token order matches the
        reference implementation (top-scoring past tokens in score order,
        followed by the observation window).
        """
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            return key_states, value_states

        final_score = self._scores(key_states, query_states)

        # shape: (bsz, num_kv_heads, budget - window_size)
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
