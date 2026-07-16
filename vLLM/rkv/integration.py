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
    # ``mix_lambda`` weights importance vs. redundancy in the joint score
    # (``mix_lambda*importance - (1-mix_lambda)*redundancy``). 0.1 matches the
    # SGLang R-KV port's runtime config and the reference HF eval scripts; the
    # R-KV *algorithm* class default is 0.07, but the eval-validated value is
    # 0.1. Using 0.07 measurably lowers accuracy and the loss compounds at tight
    # buffers (more compactions): b256 buf64 87.5->89.0, buf16 83.0->85.5.
    mix_lambda: float = 0.1
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
            mix_lambda=_env_float("VLLM_V1_R_KV_MIX_LAMBDA", 0.1),
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
        # Rolling observation-window queries in a fixed-address ring buffer
        # ``(window, max_slots, q_heads, head_dim)``, written with ONE vectorized
        # ``index_copy_`` per layer per step (the old per-request GPU copy was
        # ~4.5k micro-launches/step and R-KV's #1 decode cost). The slot
        # bookkeeping + flat write index are identical across layers, so the
        # model runner computes them ONCE per step on its own compactor via
        # :meth:`plan_qwrite` (below) and shares the result through
        # ``attn_metadata.rkv_qplan``; each layer only allocates its ring and
        # scatters its own queries. ``_slot`` maps a request id to a persistent
        # ring column (freed + reused on finish) so the window survives vLLM's
        # batch reordering; ``_qcount`` is the per-request step count (ring
        # cursor = ``count % window``); ``_ring_width`` is the shared,
        # monotonically growing column count. The score means over the window
        # (order-invariant), so no un-rotation is needed. The ``_slot`` /
        # ``_qcount`` / ``_ring_width`` state is used only on the runner's
        # compactor (the instance that drives :meth:`plan_qwrite`).
        self._qring: torch.Tensor | None = None
        self._slot: dict[str, int] = {}
        self._free: list[int] = []
        self._next_slot: int = 0
        self._qcount: dict[str, int] = {}
        self._ring_width: int = 0
        # Running count of physical compactions performed (one per request
        # eviction). Cheap observability; the per-event log is env-gated.
        self._n_compactions = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def plan_qwrite(self, req_ids, device) -> tuple | None:
        """Compute this step's shared ring write-plan (runner-owned, once/step).

        Called ONCE per decode step by the model runner on its compactor, then
        shared with every attention layer through ``attn_metadata.rkv_qplan``.
        The slot->column assignment, the per-request ring cursor and the flat
        scatter index are the same for all 28 layers (only the query *data*
        differs), so doing this here removes the 28x-per-step Python slot loop
        and host->device index copy that :meth:`record_query` used to repeat.

        Returns ``(flat_index, slots, max_slots)`` -- or ``None`` when R-KV is
        disabled:

        * ``flat_index`` -- GPU ``long`` tensor, the ring row each request
          writes to (``cursor * max_slots + column``); used by ``record_query``.
        * ``slots`` -- per-request persistent ring column (Python ints); used
          by ``observe_layer`` to read a request's window.
        * ``max_slots`` -- current ring width; every layer grows its ring to it.
        """
        if self.algo is None:
            return None
        num_reqs = len(req_ids)
        window = self.config.window_size
        # Free the ring columns of finished requests for reuse.
        present = set(req_ids)
        if len(self._slot) > num_reqs:
            for rid in [r for r in self._slot if r not in present]:
                self._free.append(self._slot.pop(rid))
                self._qcount.pop(rid, None)

        # Assign each request a persistent ring column + its current cursor
        # (``step_count % window``); advance the step count once per step.
        slots = [0] * num_reqs
        curs = [0] * num_reqs
        for i, rid in enumerate(req_ids):
            s = self._slot.get(rid)
            if s is None:
                s = self._free.pop() if self._free else self._next_slot
                if s == self._next_slot:
                    self._next_slot += 1
                self._slot[rid] = s
                self._qcount[rid] = 0
            cnt = self._qcount[rid]
            slots[i] = s
            curs[i] = cnt % window
            self._qcount[rid] = cnt + 1

        # Ring width grows monotonically (doubling) with the concurrency peak,
        # so it is stable after the first prefill wave and identical for every
        # layer's ring, keeping the flat index valid across all of them.
        if self._next_slot > self._ring_width:
            new_w = max(self._next_slot, 16)
            if self._ring_width:
                new_w = max(new_w, self._ring_width * 2)
            self._ring_width = new_w
        max_slots = self._ring_width
        flat = (
            torch.as_tensor(curs, device=device, dtype=torch.long) * max_slots
            + torch.as_tensor(slots, device=device, dtype=torch.long)
        )
        return flat, slots, max_slots

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
        plan = getattr(attn_metadata, "rkv_qplan", None)
        if plan is None:
            return
        flat, slots, max_slots = plan
        num_reqs = len(slots)
        query_start_loc = attn_metadata.query_start_loc
        if query_start_loc is None or query_start_loc.shape[0] <= num_reqs:
            return

        # Each request's most recent query token (vectorized, no host sync).
        window = self.config.window_size
        last_idx = (query_start_loc[1 : num_reqs + 1] - 1).long()
        last_q = query.index_select(0, last_idx)

        # Grow this layer's ring to the shared width, then scatter every
        # request's query in one ``index_copy_`` with the runner-precomputed
        # flat index -- no per-layer Python loop or host->device copy.
        self._ensure_ring(last_q, window, max_slots)
        self._qring.view(
            window * max_slots, last_q.shape[1], last_q.shape[2]
        ).index_copy_(0, flat, last_q)

    def _ensure_ring(
        self, last_q: torch.Tensor, window: int, max_slots: int
    ) -> None:
        """Grow this layer's fixed-address query ring to ``max_slots`` columns.

        Growth (a realloc + copy) only happens when the runner's shared width
        advances (a new concurrency peak), so it is rare after the first prefill
        wave. Reading ``_qring[:, slot]`` (2-D) stays correct across widths; only
        the flat ``index_copy_`` uses the current width.
        """
        if self._qring is None or self._qring.shape[1] < max_slots:
            new_ring = last_q.new_zeros(
                (window, max_slots, last_q.shape[1], last_q.shape[2])
            )
            if self._qring is not None:
                new_ring[:, : self._qring.shape[1]] = self._qring
            self._qring = new_ring

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
        # One host copy per layer instead of a GPU->CPU ``.item()`` sync per
        # request inside the armed loop (this runs once per attention layer).
        seq_lens_cpu = seq_lens.tolist()
        query_start_loc = attn_metadata.query_start_loc

        score_acc = attn_metadata.rkv_score_acc
        req_ids = getattr(attn_metadata, "rkv_req_ids", None)
        plan = getattr(attn_metadata, "rkv_qplan", None)
        plan_slots = plan[1] if plan is not None else None
        batched = self.config.score_mode == "batched"

        # Per-layer observation-window queries for the requests armed this step,
        # keyed by batch index. In batched mode ``compact_step`` reads these to
        # score every layer in one GEMM; in reference mode they go unused (the
        # per-layer score is summed into ``score_acc`` below).
        layer_wq: dict[int, torch.Tensor] = {}

        for i in range(num_reqs):
            if not should_compress[i]:
                continue
            if seq_lens_cpu[i] < threshold:
                continue

            # Observation queries: this request's rolling window (last
            # ``window_size`` decode queries) collected by ``record_query``.
            # The importance score means over them (order-invariant), so the
            # ring buffer needs no un-rotation. Falls back to the current step's
            # query if the window has not been recorded yet (e.g. first step).
            slot = plan_slots[i] if plan_slots is not None else None
            if slot is not None and self._qring is not None:
                qbuf = self._qring[:, slot]  # (window, q_heads, head_dim)
            else:
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

    def _score_group(self, layer_caches, req_indices, slots_list, block_size):
        """Cross-layer R-KV scores for a GROUP of requests sharing ``seq_len``.

        Generalizes the single-request batched score across requests: requests
        in a group have the same cache length, so their per-layer keys stack
        into one ``(num_layers*num_reqs, kv_heads, seq_len, hd)`` batch and the
        whole group's cosine-similarity + attention GEMMs run in one
        :meth:`R1KV._scores` call, with a single concatenated K gather per layer
        for all requests (instead of a gather + score per request). Chunked over
        layers to bound the transient cosine matrix. The cross-layer reduction
        stays a **sequential** per-layer sum, so the result is bit-identical to
        the per-request path. Returns ``(num_reqs, seq_len - window_size)`` or
        ``None`` if any request's observation window was not recorded.
        """
        num_reqs = len(req_indices)
        num_layers = len(layer_caches)
        kv_heads = layer_caches[0][0].shape[2]
        head_dim = layer_caches[0][0].shape[3]
        window = self.config.window_size
        seq_len = slots_list[0].shape[0]

        for lc in layer_caches:
            wqd = lc[2]
            for i in req_indices:
                if wqd.get(i) is None:
                    return None

        # One K gather per layer for the whole group (concatenated slots), in
        # request order so the flat rows reshape back to (num_reqs, seq_len).
        slots_cat = torch.cat(slots_list)
        blk = slots_cat // block_size
        off = slots_cat % block_size

        elt = layer_caches[0][0].element_size()
        # Transient cosine matrix per (layer, request) unit; bound total memory
        # by chunking over layers (each chunk covers every request in the group).
        per_unit = max(1, (2 * elt + 1 + 4) * kv_heads * seq_len * seq_len)
        chunk = max(
            1, min(num_layers, self.config.score_chunk_bytes // (per_unit * num_reqs))
        )

        acc = None  # (num_reqs, seq_len - window)
        for c in range(0, num_layers, chunk):
            hi = min(c + chunk, num_layers)
            cl = hi - c
            # (cl, num_reqs*seq_len, kv_heads, hd)
            #   -> (cl*num_reqs, kv_heads, seq_len, hd)
            keys = (
                torch.stack([layer_caches[l][0][blk, off] for l in range(c, hi)])
                .view(cl, num_reqs, seq_len, kv_heads, head_dim)
                .permute(0, 1, 3, 2, 4)
                .reshape(cl * num_reqs, kv_heads, seq_len, head_dim)
                .contiguous()
            )
            # (cl, num_reqs, window, q_heads, hd)
            #   -> (cl*num_reqs, q_heads, window, hd)
            queries = torch.stack(
                [
                    torch.stack([layer_caches[l][2][i] for i in req_indices])
                    for l in range(c, hi)
                ]
            )
            q_heads = queries.shape[3]
            queries = (
                queries.permute(0, 1, 3, 2, 4)
                .reshape(cl * num_reqs, q_heads, window, head_dim)
                .contiguous()
            )
            # (cl*num_reqs, kv_heads, seq_len - window) -> cross-head mean
            layer_scores = self.algo._scores(keys, queries).mean(dim=1)
            layer_scores = layer_scores.view(cl, num_reqs, seq_len - window)
            # Sequential per-layer sum (bit-identical to the per-request path).
            for li in range(cl):
                acc = layer_scores[li] if acc is None else acc + layer_scores[li]
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
        # One host copy up front instead of a GPU->CPU ``.item()`` sync per
        # request inside the loop.
        seq_lens_cpu = seq_lens.tolist()
        seq_starts_cpu = seq_starts.tolist()
        seq_ends_cpu = seq_ends.tolist()

        # Phase A0: collect the armed-and-over-threshold requests and their
        # occupied slots.
        armed: list[tuple[int, int, int, torch.Tensor]] = []
        for i in range(num_reqs):
            if not should_compress[i]:
                continue
            seq_len = seq_lens_cpu[i]
            if seq_len < threshold:
                continue
            kv_start = seq_starts_cpu[i]
            kv_end = seq_ends_cpu[i]
            armed.append((i, seq_len, kv_start, occupied_slot_mapping[kv_start:kv_end]))
        if not armed:
            return

        # Phase A1: cross-layer scoring. In batched mode, group requests by cache
        # length so every request in a group shares ONE cosine-similarity +
        # attention scoring call (and one K gather per layer) -- the redundancy
        # scoring is launch-bound, so amortizing it across the ~budget/buffer
        # requests that compact together is the dominant win at small buffers.
        # Reference mode already has each request's score in ``score_acc``.
        scores: dict[int, torch.Tensor] = {}
        if self.config.score_mode == "batched":
            groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
            for (i, seq_len, kv_start, slots) in armed:
                groups.setdefault(seq_len, []).append((i, slots))
            for members in groups.values():
                idxs = [m[0] for m in members]
                grp = self._score_group(
                    layer_caches, idxs, [m[1] for m in members], block_size
                )
                if grp is not None:
                    for r, i in enumerate(idxs):
                        scores[i] = grp[r]

        # Phase A2: pick each request's kept set and collect its source /
        # destination slot ids. The physical relocation is deferred to Phase B so
        # it can run as ONE gather+scatter per layer across ALL requests, instead
        # of per (request, layer) -- that per-request x per-layer loop was the
        # dominant eviction cost at small buffers (many tiny kernel launches).
        src_slots: list[torch.Tensor] = []
        dst_slots: list[torch.Tensor] = []
        for (i, seq_len, kv_start, slots) in armed:
            score = score_acc[i] if score_acc[i] is not None else scores.get(i)
            if score is None:
                continue

            # One global kept set: top past tokens + trailing observation window,
            # sorted ascending to keep the survivors in temporal order.
            past_idx = score.topk(num_past).indices
            window_idx = torch.arange(
                seq_len - window, seq_len, device=past_idx.device
            )
            kept = torch.sort(torch.cat([past_idx, window_idx])).values

            src_slots.append(slots[kept])
            dst_slots.append(occupied_slot_mapping[kv_start : kv_start + budget])
            num_dropped_tokens_list[i] = seq_len - budget

            self._n_compactions += 1
            if os.getenv("VLLM_V1_R_KV_TRACE") == "1":
                print(
                    f"[RKV-COMPACT] #{self._n_compactions} "
                    f"layers={len(layer_caches)} seq_len={seq_len} "
                    f"kept={int(kept.numel())} dropped={seq_len - budget}",
                    flush=True,
                )

        if not src_slots:
            return

        # Phase B: relocate the kept KV for every request with ONE gather +
        # scatter per layer, batched across all requests. Destination slots are
        # disjoint across requests, and the advanced-index gather returns a fresh
        # tensor before the scatter, so the overlapping front-slot ranges do not
        # alias-corrupt.
        src = torch.cat(src_slots)
        dst = torch.cat(dst_slots)
        src_blk = src // block_size
        src_off = src % block_size
        dst_blk = dst // block_size
        dst_off = dst % block_size
        for key_cache, value_cache, _wq in layer_caches:
            kept_k = key_cache[src_blk, src_off]
            kept_v = value_cache[src_blk, src_off]
            key_cache[dst_blk, dst_off] = kept_k
            value_cache[dst_blk, dst_off] = kept_v
