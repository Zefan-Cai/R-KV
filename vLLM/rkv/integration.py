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
* ``rkv_layer_caches``: ``list`` of ``(key_cache, value_cache, window_queries)``
  registered per layer during the forward, scored/evicted together after it
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
    # Scoring path: "batched" (default; one cross-layer GEMM at compaction time,
    # ported from the SGLang R-KV port) or "reference" (per-layer scoring inside
    # each layer's forward). Both produce the same kept set; batched issues far
    # fewer kernel launches. ``score_chunk_bytes`` bounds the transient
    # cosine-similarity matrix so batching cannot blow up memory at large
    # ``budget + buffer``.
    score_mode: str = "batched"
    score_chunk_bytes: int = 512 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "RKVConfig":
        return cls(
            budget=_env_int("VLLM_V1_R_KV_BUDGET", 64),
            buffer_size=_env_int("VLLM_V1_R_KV_BUFFER", 64),
            window_size=_env_int("VLLM_V1_R_KV_WINDOW", 8),
            kernel_size=_env_int("VLLM_V1_R_KV_KERNEL", 7),
            mix_lambda=_env_float("VLLM_V1_R_KV_MIX_LAMBDA", 0.07),
            retain_ratio=_env_float("VLLM_V1_R_KV_RETAIN_RATIO", 0.1),
            score_mode=os.getenv("VLLM_V1_R_KV_SCORE_MODE", "batched"),
            score_chunk_bytes=_env_int("VLLM_V1_R_KV_SCORE_CHUNK_MB", 512)
            * 1024
            * 1024,
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
        # Running count of physical compactions performed (one per request
        # eviction). Cheap observability; the per-event log is env-gated.
        self._n_compactions = 0

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
        batched = self.config.score_mode == "batched"

        # Per-layer observation-window queries for the requests armed this step,
        # keyed by batch index. In batched mode ``compact_step`` reads these to
        # score every layer in one GEMM; in reference mode they go unused (the
        # per-layer score is summed into ``score_acc`` below).
        layer_wq: dict[int, torch.Tensor] = {}

        for i in range(num_reqs):
            if not should_compress[i]:
                continue
            if seq_lens[i].item() < threshold:
                continue

            # Observation queries: this request's rolling window (last
            # ``window_size`` decode queries) collected by ``record_query``.
            # The importance score means over them (order-invariant), so the
            # ring buffer needs no un-rotation. Falls back to the current step's
            # query if the window has not been recorded yet (e.g. first step).
            qbuf = self._qwin.get(req_ids[i]) if req_ids is not None else None
            if qbuf is None:
                q_start = query_start_loc[i].item()
                q_end = query_start_loc[i + 1].item()
                qbuf = query[q_start:q_end]

            if batched:
                # Defer scoring to ``compact_step`` (one batched cross-layer
                # GEMM). Only the window queries need stashing; the keys are
                # gathered there from the registered layer caches.
                layer_wq[i] = qbuf
                continue

            # Reference path: score THIS layer now and sum across layers.
            kv_start = seq_starts[i].item()
            kv_end = seq_ends[i].item()
            slots = slot_map[kv_start:kv_end]
            blk = slots // block_size
            off = slots % block_size
            keys = key_cache[blk, off].transpose(0, 1).unsqueeze(0)
            queries = qbuf.transpose(0, 1).unsqueeze(0)
            layer_score = self.algo._scores(keys, queries).mean(dim=1).squeeze(0)
            if score_acc[i] is None:
                score_acc[i] = layer_score
            else:
                score_acc[i] = score_acc[i] + layer_score

        # Register this layer's caches (+ window queries for batched scoring) so
        # ``compact_step`` can score/evict every layer with one global decision.
        attn_metadata.rkv_layer_caches.append((key_cache, value_cache, layer_wq))

    def _batched_scores(self, layer_caches, i, slots, block_size):
        """Cross-layer R-KV score for request ``i`` in one batched GEMM.

        Ports the SGLang R-KV optimization: instead of ``num_layers`` bsz=1
        scoring calls interleaved in the forward, gather every registered
        layer's K for this request, stack them with ``num_layers`` as the batch
        dim, and run a single :meth:`R1KV._scores` call (cross-head **mean**,
        then cross-layer **sum** in layer order -- identical to the per-layer
        reference, since the batch GEMM computes independent per-layer results).
        Chunked over layers so the transient cosine-similarity matrix stays
        under ``score_chunk_bytes``. Returns ``(seq_len - window_size,)`` or
        ``None`` if the observation window was not recorded for this request.
        """
        wq = [lc[2].get(i) for lc in layer_caches]
        if any(q is None for q in wq):
            return None
        blk = slots // block_size
        off = slots % block_size
        num_layers = len(layer_caches)
        kv_heads = layer_caches[0][0].shape[2]
        seq_len = slots.shape[0]
        elt = layer_caches[0][0].element_size()
        # cosine matrix (key dtype, x2 for softmax r/w) + bool mask + int32 idx,
        # per (kv_heads, seq, seq) element -- bound the transient matrix memory.
        per_layer = max(1, (2 * elt + 1 + 4) * kv_heads * seq_len * seq_len)
        chunk = max(1, min(num_layers, self.config.score_chunk_bytes // per_layer))

        acc = None
        for c in range(0, num_layers, chunk):
            hi = min(c + chunk, num_layers)
            # (chunk, seq_len, kv_heads, hd) -> (chunk, kv_heads, seq_len, hd)
            keys = (
                torch.stack([layer_caches[l][0][blk, off] for l in range(c, hi)])
                .transpose(1, 2)
                .contiguous()
            )
            # (chunk, window, q_heads, hd) -> (chunk, q_heads, window, hd)
            queries = (
                torch.stack([wq[l] for l in range(c, hi)])
                .transpose(1, 2)
                .contiguous()
            )
            # (chunk, kv_heads, seq_len - window) -> cross-head mean
            layer_scores = self.algo._scores(keys, queries).mean(dim=1)
            for li in range(layer_scores.shape[0]):
                s = layer_scores[li]
                acc = s if acc is None else acc + s
        return acc

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
            if seq_lens[i].item() < threshold:
                continue

            kv_start = seq_starts[i].item()
            kv_end = seq_ends[i].item()
            seq_len = kv_end - kv_start
            slots = occupied_slot_mapping[kv_start:kv_end]

            # Cross-layer score: accumulated per layer during the forward
            # (reference mode) or computed now in one batched GEMM (batched
            # mode). Batched matches the reference sum (independent per-layer
            # GEMMs) with far fewer kernel launches.
            score = score_acc[i]
            if score is None:
                score = self._batched_scores(layer_caches, i, slots, block_size)
            if score is None:
                continue

            # One global kept set: top past tokens + trailing observation window,
            # sorted ascending to keep the survivors in temporal order.
            past_idx = score.topk(num_past).indices
            window_idx = torch.arange(
                seq_len - window, seq_len, device=past_idx.device
            )
            kept = torch.sort(torch.cat([past_idx, window_idx])).values

            src = slots[kept]
            src_blk = src // block_size
            src_off = src % block_size
            dst = occupied_slot_mapping[kv_start : kv_start + budget]
            dst_blk = dst // block_size
            dst_off = dst % block_size

            # Advanced-index gather returns a fresh tensor before the scatter, so
            # the overlapping src/dst front-slot ranges do not alias-corrupt.
            for key_cache, value_cache, _wq in layer_caches:
                kept_k = key_cache[src_blk, src_off]
                kept_v = value_cache[src_blk, src_off]
                key_cache[dst_blk, dst_off] = kept_k
                value_cache[dst_blk, dst_off] = kept_v

            num_dropped_tokens_list[i] = seq_len - budget

            self._n_compactions += 1
            if os.getenv("VLLM_V1_R_KV_TRACE") == "1":
                print(
                    f"[RKV-COMPACT] #{self._n_compactions} "
                    f"layers={len(layer_caches)} seq_len={seq_len} "
                    f"kept={int(kept.numel())} dropped={seq_len - budget}",
                    flush=True,
                )
