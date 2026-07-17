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

* **Tensor parallelism**: each rank holds only a shard of the KV heads, so its
  per-token score is partial. All ranks share one physical KV layout, so the
  eviction decision must be identical on every rank -- :meth:`compact_step`
  therefore sums the per-token scores across the tensor-parallel group before
  the top-k. The armed requests, cache lengths and grouping are already
  identical on every rank (replicated scheduler state), and the collective is
  skipped entirely when TP is off. **Data parallelism** needs no coordination:
  each replica owns an independent KV cache and compactor and evicts its own
  requests (the all-reduce above uses the TP sub-group, so DP+TP is correct).

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
    """Parse an integer env var. Unset -> ``default``; **invalid -> raise**.

    Silently falling back to a default on a malformed value hides typos and can
    silently change behaviour, so an env var that is *set* but unparseable is a
    hard error naming the variable and value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {name}={raw!r}: expected an integer.")


def _env_float(name: str, default: float) -> float:
    """Parse a float env var. Unset -> ``default``; **invalid -> raise**."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {name}={raw!r}: expected a number.")


@dataclass
class RKVConfig:
    """Decode-time R-KV configuration.

    ``budget`` and ``buffer_size`` mirror the two ``VLLM_V1_R_KV_*`` environment
    variables of the original proof-of-concept; the remaining fields expose the
    algorithm knobs with the reference defaults.
    """

    budget: int = 0
    buffer_size: int = 0
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

    def __post_init__(self) -> None:
        # Only validate an *enabled* config: with budget or buffer_size == 0
        # R-KV is a pure no-op, so the (otherwise out-of-range) zero values must
        # be tolerated rather than rejected.
        if not (self.budget > 0 and self.buffer_size > 0):
            return
        if self.window_size <= 0:
            raise ValueError(
                f"R-KV window_size ({self.window_size}) must be positive."
            )
        if self.budget <= self.window_size:
            raise ValueError(
                f"R-KV budget ({self.budget}) must be greater than window_size "
                f"({self.window_size})."
            )
        if self.buffer_size < self.window_size:
            raise ValueError(
                f"R-KV buffer_size ({self.buffer_size}) must be >= window_size "
                f"({self.window_size}); a smaller buffer compacts before the "
                "observation window is full, scoring against stale/zero queries."
            )
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"R-KV kernel_size ({self.kernel_size}) must be a positive odd "
                "integer (an even kernel makes the max-pool output one longer "
                "than the score, so scoring raises a shape-mismatch error)."
            )
        if not 0.0 <= self.mix_lambda <= 1.0:
            raise ValueError(
                f"R-KV mix_lambda ({self.mix_lambda}) must be in [0, 1]."
            )
        if not 0.0 < self.retain_ratio <= 1.0:
            raise ValueError(
                f"R-KV retain_ratio ({self.retain_ratio}) must be in (0, 1]."
            )
        if self.score_mode not in ("batched", "reference"):
            raise ValueError(
                f"R-KV score_mode ({self.score_mode!r}) must be 'batched' or "
                "'reference'."
            )
        if self.score_chunk_bytes <= 0:
            raise ValueError(
                "R-KV score_chunk_bytes (VLLM_V1_R_KV_SCORE_CHUNK_MB) must be "
                f"positive, got {self.score_chunk_bytes} bytes."
            )

    @classmethod
    def from_env(cls) -> "RKVConfig":
        return cls(
            budget=_env_int("VLLM_V1_R_KV_BUDGET", 0),
            buffer_size=_env_int("VLLM_V1_R_KV_BUFFER", 0),
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
        # Tensor-parallel group coordinator for the cross-rank score sum,
        # resolved lazily on first compaction (the distributed groups are set up
        # after this object is constructed) and cached. Stays ``None`` when TP is
        # off so single-GPU decode never touches a collective.
        self._tp_grp = None
        self._tp_grp_resolved = False
        # Expected tensor-parallel world size, set by the model runner via
        # ``set_parallel_context``. When > 1 the cross-rank score reduction is
        # mandatory: if the TP group is missing or its world size disagrees,
        # ``_tp_group`` raises instead of silently making a single-rank decision
        # (which would desync the sharded KV across ranks).
        self._expected_tp_world_size: int | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def set_parallel_context(self, tp_world_size: int) -> None:
        """Record the model's TP world size so the cross-rank score reduction
        can fail closed if the group is unavailable or inconsistent."""
        self._expected_tp_world_size = tp_world_size
        self._tp_grp_resolved = False  # re-resolve against the new expectation

    def _tp_group(self):
        """Tensor-parallel group for the cross-rank score sum, or ``None``.

        Fail-closed, keyed on the expected TP world size
        (:meth:`set_parallel_context`):

        * expected <= 1 (single GPU, or unset): no cross-rank reduction is
          needed; returns ``None`` and the collective is skipped.
        * expected > 1: the TP group **must** exist and its world size must
          match, else each rank would score only its KV-head shard and evict a
          different set -- silently divergent, unrecoverable KV. Any failure
          raises rather than degrading to a local decision.
        """
        if self._tp_grp_resolved:
            return self._tp_grp
        self._tp_grp_resolved = True
        expected = self._expected_tp_world_size
        if expected is None or expected <= 1:
            # Single-GPU / unset: reduction not required. Still avoid silently
            # running in a >1 world by checking the group if it is initialized.
            try:
                from vllm.distributed.parallel_state import get_tp_group

                grp = get_tp_group()
                self._tp_grp = grp if grp.world_size > 1 else None
            except (ImportError, AssertionError):
                self._tp_grp = None
            return self._tp_grp
        # expected > 1: the reduction is mandatory -- fail closed on any problem.
        from vllm.distributed.parallel_state import get_tp_group

        grp = get_tp_group()  # raises if the group is not initialized
        if grp.world_size != expected:
            raise RuntimeError(
                f"R-KV: tensor-parallel group world_size {grp.world_size} != "
                f"expected {expected}; refusing to make a per-rank-inconsistent "
                "eviction decision."
            )
        self._tp_grp = grp
        return self._tp_grp

    def _storage_id(self, t: torch.Tensor):
        """Identity of a tensor's underlying storage (data ptr + view geometry).

        Two layers that share a KV cache alias the same storage; a plain object
        or ``id()`` check would miss aliasing views, so identity also folds in
        the storage offset, shape and stride.
        """
        return (
            t.untyped_storage().data_ptr(),
            t.storage_offset(),
            tuple(t.shape),
            tuple(t.stride()),
        )

    def _dedup_relocation_caches(self, layer_caches):
        """Distinct (key, value) storages for physical relocation.

        Cross-layer KV sharing is rejected at startup, but even if a shared
        tensor slipped through it must be relocated **once** -- a second in-place
        gather/scatter would read already-relocated data and corrupt the kept
        set. Scoring still uses every layer entry (each contributes its own
        query window); only relocation dedups by storage identity.
        """
        seen = set()
        out = []
        for kc, vc, wq in layer_caches:
            sig = (self._storage_id(kc), self._storage_id(vc))
            if sig in seen:
                continue
            seen.add(sig)
            out.append((kc, vc, wq))
        return out

    def _tp_readiness_check(self, tp, groups, seq_lens_cpu, device) -> None:
        """One fixed collective so divergent ranks fail together, not deadlock.

        Each rank all-reduces a signature of its compaction plan (group count,
        total requests, seq-len signature). If any rank disagrees the sum will
        not equal ``local * world_size`` on every rank, so they all raise --
        rather than some ranks entering the per-group score all-reduce while
        others ``continue``/return and leave the collective hanging.
        """
        if tp is None:
            return
        import torch.distributed as dist

        total_reqs = sum(len(m) for m in groups.values())
        sig = (
            len(groups) * 1_000_003
            + total_reqs * 1009
            + sum(int(sl) * len(m) for sl, m in groups.items())
        )
        t = torch.tensor([sig], dtype=torch.int64, device=device)
        dist.all_reduce(t, group=tp.device_group)
        if int(t.item()) != sig * tp.world_size:
            raise RuntimeError(
                "R-KV: tensor-parallel ranks disagree on the compaction plan; "
                "refusing to make divergent eviction decisions."
            )

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

        elt = layer_caches[0][0].element_size()
        # Transient cosine matrix per (layer, request) unit (~seq_len^2). Bound
        # peak memory by tiling BOTH the layer and the request dimension so the
        # batch stays under the cap even when many requests compact together on
        # a long shared prefix. Per-request scores are independent and the
        # per-layer sum order is preserved, so the result is bit-identical to
        # scoring the whole group at once. compact_step guarantees each unit
        # itself fits the cap (it skips/raises otherwise), so ``units_cap >= 1``.
        per_unit = max(1, (2 * elt + 1 + 4) * kv_heads * seq_len * seq_len)
        units_cap = max(1, self.config.score_chunk_bytes // per_unit)
        req_chunk = max(1, min(num_reqs, units_cap))
        layer_chunk = max(1, min(num_layers, units_cap // req_chunk))

        acc_parts: list[torch.Tensor] = []  # per request-chunk, in idxs order
        for r0 in range(0, num_reqs, req_chunk):
            r_ids = req_indices[r0 : r0 + req_chunk]
            rc = len(r_ids)
            # One K gather per layer for this request-chunk (concatenated slots),
            # in request order so the flat rows reshape back to (rc, seq_len).
            slots_cat = torch.cat(slots_list[r0 : r0 + rc])
            blk = slots_cat // block_size
            off = slots_cat % block_size
            part = None  # (rc, seq_len - window)
            for c in range(0, num_layers, layer_chunk):
                hi = min(c + layer_chunk, num_layers)
                cl = hi - c
                # (cl, rc*seq_len, kv_heads, hd) -> (cl*rc, kv_heads, seq_len, hd)
                keys = (
                    torch.stack(
                        [layer_caches[l][0][blk, off] for l in range(c, hi)]
                    )
                    .view(cl, rc, seq_len, kv_heads, head_dim)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(cl * rc, kv_heads, seq_len, head_dim)
                    .contiguous()
                )
                # (cl, rc, window, q_heads, hd) -> (cl*rc, q_heads, window, hd)
                queries = torch.stack(
                    [
                        torch.stack([layer_caches[l][2][i] for i in r_ids])
                        for l in range(c, hi)
                    ]
                )
                q_heads = queries.shape[3]
                queries = (
                    queries.permute(0, 1, 3, 2, 4)
                    .reshape(cl * rc, q_heads, window, head_dim)
                    .contiguous()
                )
                # (cl*rc, kv_heads, seq_len - window) -> cross-head mean
                layer_scores = self.algo._scores(keys, queries).mean(dim=1)
                layer_scores = layer_scores.view(cl, rc, seq_len - window)
                # Sequential per-layer sum (bit-identical to the whole-group path).
                for li in range(cl):
                    part = (
                        layer_scores[li]
                        if part is None
                        else part + layer_scores[li]
                    )
            acc_parts.append(part)
        return torch.cat(acc_parts, dim=0)

    def compact_step(
        self,
        num_reqs: int,
        seq_lens: torch.Tensor,
        occupied_slot_mapping,
        should_compress,
        score_acc,
        layer_caches,
        num_dropped_tokens_list,
        expected_layer_count: int | None = None,
        prev_dropped=None,
    ) -> None:
        """Evict every required request with one global cross-layer decision.

        **Phase 2** of the two-phase compaction, run once after the full forward
        pass (all layers have contributed to ``score_acc`` and registered their
        caches in :meth:`observe_layer`). For each armed request it selects the
        ``budget - window_size`` highest-scoring past tokens (summed across all
        layers) plus the trailing ``window_size`` observation tokens, sorts them
        to preserve temporal order, then physically relocates that ONE kept set
        to the leading ``budget`` slots in **every** layer's paged KV and
        records the per-request evicted count.

        Compaction is **transactional**: once a request is at/over the
        threshold it MUST be handled this step, because a request that has
        already dropped KV cannot fall back to Full-KV (its old KV is gone and
        the block manager has capped its allocation). Missing state, incomplete
        layer registration, or a missing observation window therefore **raise**
        rather than silently skip. The only permitted skip is a request's
        *first* compaction whose scoring would exceed the memory cap: it is left
        Full-KV (``num_dropped`` stays 0, so the block manager never caps it).
        The evicted counts are published only after every relocation kernel is
        enqueued (``expected_layer_count`` / ``prev_dropped`` are supplied by
        the model runner; they default to permissive values for direct tests).
        """
        if self.algo is None or not should_compress:
            return
        num_reqs = min(num_reqs, len(should_compress))
        if not any(should_compress[:num_reqs]):
            return

        budget = self.config.budget
        window = self.config.window_size
        num_past = budget - window
        threshold = budget + self.config.buffer_size
        batched = self.config.score_mode == "batched"

        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        seq_lens_cpu = seq_lens.tolist()
        seq_starts_cpu = seq_starts.tolist()
        seq_ends_cpu = seq_ends.tolist()

        # Requests armed AND at/over the threshold: these MUST be handled now.
        required = [
            i
            for i in range(num_reqs)
            if should_compress[i] and seq_lens_cpu[i] >= threshold
        ]
        if not required:
            return

        # From here compaction is REQUIRED -- missing state or an incompletely
        # registered layer set is a hard error, not a silent skip.
        if occupied_slot_mapping is None or not layer_caches:
            raise RuntimeError(
                f"R-KV: compaction required for {len(required)} request(s) but "
                "occupied_slot_mapping/layer_caches are missing -- refusing to "
                "continue with inconsistent KV state."
            )
        if (
            expected_layer_count is not None
            and len(layer_caches) != expected_layer_count
        ):
            raise RuntimeError(
                f"R-KV: only {len(layer_caches)}/{expected_layer_count} attention "
                "layers registered for compaction; compacting a subset would "
                "desync the block table across layers."
            )

        block_size = layer_caches[0][0].size(1)
        elt = layer_caches[0][0].element_size()
        kv_heads = layer_caches[0][0].shape[2]
        cap = self.config.score_chunk_bytes
        prev = prev_dropped if prev_dropped is not None else [0] * num_reqs

        # Phase A: group required requests by cache length. A request whose
        # first-compaction scoring matrix (~seq_len^2) exceeds the cap is left
        # Full-KV (only ever safe before it has dropped anything); an already-
        # compacted request that no longer fits is a hard error.
        groups: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
        for i in required:
            seq_len = seq_lens_cpu[i]
            unit_bytes = (2 * elt + 5) * kv_heads * seq_len * seq_len
            if unit_bytes > cap:
                if int(prev[i]) > 0:
                    raise RuntimeError(
                        f"R-KV: request {i} has already compacted "
                        f"(dropped={int(prev[i])}) but its seq_len={seq_len} "
                        f"scoring needs ~{unit_bytes >> 20} MiB > {cap >> 20} MiB "
                        "cap and cannot fall back to Full-KV. Raise "
                        "VLLM_V1_R_KV_SCORE_CHUNK_MB or lower budget/buffer."
                    )
                if os.getenv("VLLM_V1_R_KV_TRACE") == "1":
                    print(
                        f"[RKV-SKIP] req {i} seq_len={seq_len} scoring "
                        f"~{unit_bytes >> 20} MiB > {cap >> 20} MiB cap; left "
                        "Full-KV (never compacted)",
                        flush=True,
                    )
                continue
            kv_start = seq_starts_cpu[i]
            groups.setdefault(seq_len, []).append(
                (i, kv_start, occupied_slot_mapping[kv_start : seq_ends_cpu[i]])
            )
        if not groups:
            return

        # Fail together (not deadlock) if TP ranks disagree on the plan.
        dev = occupied_slot_mapping.device
        tp = self._tp_group()
        self._tp_readiness_check(tp, groups, seq_lens_cpu, dev)

        src_slots: list[torch.Tensor] = []
        dst_slots: list[torch.Tensor] = []
        pending_drop: dict[int, int] = {}  # applied AFTER relocation (P1-2)
        for seq_len, members in groups.items():
            idxs = [m[0] for m in members]
            slots_list = [m[2] for m in members]
            if batched:
                grp_scores = self._score_group(
                    layer_caches, idxs, slots_list, block_size
                )
                if grp_scores is None:
                    raise RuntimeError(
                        "R-KV: observation window missing for a required "
                        f"compaction (requests {idxs}); refusing to skip -- the "
                        "block manager has already capped this request's KV."
                    )
            else:
                missing = [i for i in idxs if score_acc[i] is None]
                if missing:
                    raise RuntimeError(
                        f"R-KV: missing cross-layer score for required requests "
                        f"{missing}."
                    )
                grp_scores = torch.stack([score_acc[i] for i in idxs])

            # Tensor parallelism: this rank scored only its shard of the KV
            # heads, so ``grp_scores`` is partial. Summing across the TP group
            # makes every rank's top-k -- and thus the physically kept set --
            # identical. No-op when TP is off.
            if tp is not None:
                grp_scores = tp.all_reduce(grp_scores.contiguous())
            g = len(idxs)
            gdev = grp_scores.device

            # One global kept set per request: top past tokens + trailing
            # observation window, sorted ascending to keep temporal order.
            past_idx = grp_scores.topk(num_past, dim=-1).indices
            window_idx = torch.arange(
                seq_len - window, seq_len, device=gdev
            ).expand(g, window)
            kept = torch.sort(
                torch.cat([past_idx, window_idx], dim=-1), dim=-1
            ).values

            src_grp = torch.gather(torch.stack(slots_list), 1, kept)
            kv_starts = torch.tensor(
                [m[1] for m in members], device=gdev
            ).unsqueeze(1)
            dst_grp = occupied_slot_mapping[
                kv_starts + torch.arange(budget, device=gdev)
            ]
            src_slots.append(src_grp.reshape(-1))
            dst_slots.append(dst_grp.reshape(-1))
            for i in idxs:
                pending_drop[i] = seq_len - budget

        # Every grouped request must have a relocation planned (defensive).
        expected_planned = {m[0] for members in groups.values() for m in members}
        if set(pending_drop) != expected_planned:
            raise RuntimeError(
                "R-KV: internal invariant violated -- planned "
                f"{set(pending_drop)} but expected {expected_planned}."
            )

        # Phase B: relocate the kept KV once per DISTINCT storage (dedup guards
        # against any cross-layer KV sharing that slipped past the startup
        # check -- relocating a shared tensor twice would corrupt it). The
        # gather returns a fresh tensor before the scatter, so overlapping
        # src/dst ranges do not alias-corrupt.
        src = torch.cat(src_slots)
        dst = torch.cat(dst_slots)
        src_blk = src // block_size
        src_off = src % block_size
        dst_blk = dst // block_size
        dst_off = dst % block_size
        for key_cache, value_cache, _wq in self._dedup_relocation_caches(
            layer_caches
        ):
            kept_k = key_cache[src_blk, src_off]
            kept_v = value_cache[src_blk, src_off]
            key_cache[dst_blk, dst_off] = kept_k
            value_cache[dst_blk, dst_off] = kept_v

        # P1-2: publish the evicted counts ONLY after every relocation kernel is
        # enqueued, so a mid-relocation failure never leaves "compaction done"
        # metadata over half-moved KV.
        for i, dropped in pending_drop.items():
            num_dropped_tokens_list[i] = dropped
            self._n_compactions += 1
        if os.getenv("VLLM_V1_R_KV_TRACE") == "1":
            print(
                f"[RKV-COMPACT] #{self._n_compactions} groups={len(groups)} "
                f"reqs={len(pending_drop)} layers={len(layer_caches)}",
                flush=True,
            )
