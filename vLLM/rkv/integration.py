"""R-KV integration layer for vLLM v1 (FlashAttention backend).

This module bridges the pure :class:`~rkv.algo.R1KV` algorithm to vLLM's paged
KV cache. It encapsulates the per-request scoring and **physical** KV eviction
that would otherwise live inline in the attention backend's ``forward`` — so the
runtime wiring patch stays small and additive.

Design (mirrors the SGLang R-KV port):

* The algorithm is **per-head / per-layer**, but a single global eviction
  decision is applied across all of them (the R-KV reference reduces the
  per-head / per-layer scores to one kept set). Compaction is therefore
  **two-phase**:

  - :meth:`RKVCompressor.observe_layer` runs inside every attention layer's
    ``forward`` (after attention has read the full KV). It computes that
    layer's per-past-token score, reduces it across KV heads (**mean**), and
    accumulates it into a shared per-request buffer (**sum** across layers).
  - :meth:`RKVCompressor.compact_step` runs once after the full forward pass
    (from the model runner). It turns the summed score into one kept set and
    physically evicts that same set from **every** layer's paged KV, reporting
    how many tokens were dropped so the scheduler / model-runner can shrink the
    logical->physical position mapping consistently.

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
* ``rkv_score_acc``: ``list`` per-request cross-layer score accumulator
* ``rkv_layer_caches``: ``list`` of ``(key_cache, value_cache)`` registered per
  layer during the forward, evicted together after it
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
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
            mix_lambda=_env_float("VLLM_V1_R_KV_MIX_LAMBDA", 0.07),
            retain_ratio=_env_float("VLLM_V1_R_KV_RETAIN_RATIO", 0.1),
        )

    @property
    def enabled(self) -> bool:
        return self.budget > 0 and self.buffer_size > 0


class RKVCompressor:
    """Coordinates two-phase, cross-layer R-KV eviction for a decode step."""

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
        # Rolling observation-window queries, keyed by request id so they
        # survive vLLM's batch reordering (condense on finish). Each entry is
        # this layer's last ``window_size`` decode queries; the R-KV importance
        # score means over them (order-invariant), matching the SGLang port's
        # observation window. Populated every decode step by ``record_query``.
        self._qwin: dict[str, torch.Tensor] = {}
        self._qcount: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def record_query(self, query: torch.Tensor, attn_metadata) -> None:
        """Append each request's most recent query to its rolling window.

        Runs inside every attention layer's ``forward`` on every step (before
        the score is ever needed) so the compaction step has the last
        ``window_size`` queries to probe past-token importance -- the single
        largest fidelity gap vs. the SGLang port when only the current step's
        query was used. Keyed by request id, so a request keeps its own window
        across vLLM's batch reordering.

        Works for mixed prefill/decode batches: it gathers each request's *last*
        query token (via ``query_start_loc``) rather than assuming one row per
        request. Recording a prefilling request's chunk-tail query is harmless
        -- by the time a request compacts (``buffer_size`` decode steps in) its
        window holds only decode queries.
        """
        if self.algo is None:
            return
        req_ids = getattr(attn_metadata, "rkv_req_ids", None)
        if req_ids is None:
            return
        num_reqs = len(req_ids)
        query_start_loc = attn_metadata.query_start_loc
        if query_start_loc is None or query_start_loc.shape[0] <= num_reqs:
            return

        # Each request's most recent query token (vectorized, no host sync).
        last_idx = (query_start_loc[1 : num_reqs + 1] - 1).long()
        last_q = query.index_select(0, last_idx)

        window = self.config.window_size
        present = set(req_ids)
        # Drop finished requests so the buffers do not grow unbounded.
        if len(self._qwin) > num_reqs:
            for rid in [r for r in self._qwin if r not in present]:
                self._qwin.pop(rid, None)
                self._qcount.pop(rid, None)

        for i, rid in enumerate(req_ids):
            buf = self._qwin.get(rid)
            if buf is None:
                buf = last_q.new_zeros((window, last_q.shape[1], last_q.shape[2]))
                self._qwin[rid] = buf
                self._qcount[rid] = 0
            cnt = self._qcount[rid]
            # No host sync: a plain GPU slice copy into the ring slot.
            buf[cnt % window].copy_(last_q[i])
            self._qcount[rid] = cnt + 1

    def observe_layer(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata,
    ) -> None:
        """Accumulate this layer's R-KV score for every armed request.

        **Phase 1** of the two-phase, cross-layer compaction (mirrors the SGLang
        R-KV port). Runs inside each attention layer's ``forward`` after
        attention has read this step's full KV. For each armed request it
        computes the per-past-token joint score for THIS layer, reduces it
        across KV heads (**mean**), and adds it into the shared per-request
        accumulator ``attn_metadata.rkv_score_acc`` (**sum** across layers). It
        also registers this layer's ``(key_cache, value_cache)`` so the
        post-forward :meth:`compact_step` can evict every layer with the single
        global decision. No eviction happens here -- the decision needs every
        layer's contribution first.

        ``key_cache`` / ``value_cache`` are this layer's paged caches, shape
        ``[num_blocks, block_size, num_kv_heads, head_dim]``. After
        ``kv_cache.unbind`` they are non-contiguous, so R-KV addresses them with
        ``(block, offset)`` advanced indexing (gather on read, scatter on write)
        rather than a flat ``.view``.
        """
        if self.algo is None:
            return
        should_compress = attn_metadata.should_compress_list
        if not should_compress:
            return
        # ``num_reqs`` may be padded (CUDA-graph); the R-KV lists cover only the
        # real requests, so clamp the loop bound and require the occupied map.
        num_reqs = min(attn_metadata.num_reqs, len(should_compress))
        slot_map = attn_metadata.occupied_slot_mapping
        if slot_map is None or not any(should_compress[:num_reqs]):
            return

        threshold = self.config.budget + self.config.buffer_size
        block_size = key_cache.size(1)

        seq_lens = attn_metadata.seq_lens
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        query_start_loc = attn_metadata.query_start_loc

        score_acc = attn_metadata.rkv_score_acc
        req_ids = getattr(attn_metadata, "rkv_req_ids", None)
        # Register this layer's caches once (all armed requests evict the same
        # global kept set from every layer in ``compact_step``).
        attn_metadata.rkv_layer_caches.append((key_cache, value_cache))

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

            # [1, num_heads, num_tokens, head_dim]
            keys = key_cache[blk, off].transpose(0, 1).unsqueeze(0)
            # Observation queries: this request's rolling window (last
            # ``window_size`` decode queries) collected by ``record_query``.
            # The importance score means over these queries (order-invariant),
            # so the ring buffer needs no un-rotation. Falls back to the current
            # step's query if the window has not been recorded (e.g. first step).
            qbuf = self._qwin.get(req_ids[i]) if req_ids is not None else None
            if qbuf is not None:
                queries = qbuf.transpose(0, 1).unsqueeze(0)
            else:
                q_start = query_start_loc[i].item()
                q_end = query_start_loc[i + 1].item()
                queries = query[q_start:q_end].transpose(0, 1).unsqueeze(0)

            # [1, kv_heads, kv_len - window] -> cross-head mean -> [kv_len - window]
            layer_score = self.algo._scores(keys, queries).mean(dim=1).squeeze(0)
            if score_acc[i] is None:
                score_acc[i] = layer_score
            else:
                score_acc[i] = score_acc[i] + layer_score

    def compact_step(
        self,
        num_reqs: int,
        seq_lens: torch.Tensor,
        occupied_slot_mapping,
        should_compress,
        score_acc,
        layer_caches,
        num_dropped_tokens_list,
    ) -> None:
        """Evict every armed request with one global cross-layer decision.

        **Phase 2** of the two-phase compaction, run once after the full forward
        pass (all layers have contributed to ``score_acc`` and registered their
        caches in :meth:`observe_layer`). For each armed request it selects the
        ``budget - window_size`` highest-scoring past tokens (summed across all
        layers) plus the trailing ``window_size`` observation tokens, sorts them
        to preserve temporal order, then physically relocates that ONE kept set
        to the leading ``budget`` slots in **every** layer's paged KV and
        records the per-request evicted count.
        """
        if (
            self.algo is None
            or not should_compress
            or occupied_slot_mapping is None
            or not layer_caches
        ):
            return
        num_reqs = min(num_reqs, len(should_compress))
        if not any(should_compress[:num_reqs]):
            return

        budget = self.config.budget
        window = self.config.window_size
        num_past = budget - window
        threshold = budget + self.config.buffer_size

        block_size = layer_caches[0][0].size(1)
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens

        for i in range(num_reqs):
            if not should_compress[i]:
                continue
            if seq_lens[i].item() < threshold or score_acc[i] is None:
                continue

            kv_start = seq_starts[i].item()
            kv_end = seq_ends[i].item()
            seq_len = kv_end - kv_start

            # One global kept set: top past tokens + trailing observation window,
            # sorted ascending to keep the survivors in temporal order.
            past_idx = score_acc[i].topk(num_past).indices
            window_idx = torch.arange(
                seq_len - window, seq_len, device=past_idx.device
            )
            kept = torch.sort(torch.cat([past_idx, window_idx])).values

            slots = occupied_slot_mapping[kv_start:kv_end]
            src = slots[kept]
            src_blk = src // block_size
            src_off = src % block_size
            dst = occupied_slot_mapping[kv_start : kv_start + budget]
            dst_blk = dst // block_size
            dst_off = dst % block_size

            # Advanced-index gather returns a fresh tensor before the scatter, so
            # the overlapping src/dst front-slot ranges do not alias-corrupt.
            for key_cache, value_cache in layer_caches:
                kept_k = key_cache[src_blk, src_off]
                kept_v = value_cache[src_blk, src_off]
                key_cache[dst_blk, dst_off] = kept_k
                value_cache[dst_blk, dst_off] = kept_v

            num_dropped_tokens_list[i] = seq_len - budget
