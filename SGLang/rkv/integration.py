"""R-KV integration layer for SGLang v0.5.14.

Bridges the pure, device-agnostic R-KV algorithm (:mod:`.algo`) to SGLang's
paged KV cache and the FlashInfer decode path. Read ``R-KV/doc/DESIGN.md``
(especially
sections 5-9) for the architecture rationale behind the decisions encoded here.

Scope of this module (phase 1, correctness first):

* FlashInfer backend, ``batch >= 1`` with **per-request triggering**: each
  request independently decides when to compress based on its own KV length
  (see :meth:`RKVCompressor.begin_decode_step`).
* ``page_size == 1`` token pool, so evicted slots free cleanly one-by-one
  (the paged allocator frees at page granularity; see R-KV/doc/DESIGN.md
  section 8).
* Reduction of the algorithm's per-head / per-layer scores into a single global
  per-token eviction decision:

  - cross-head:  **mean** of the per-head joint scores.
  - cross-layer: **sum** over every layer's scores (aggregate all layers).

* Trigger cadence: compress once every ``buffer_size`` generated tokens, and
  only once the request's KV length has reached ``budget``.
* Physical compaction runs **after the full forward pass** (all layers have
  written their KV), not inside a single layer's ``forward_decode``.

The module is deliberately standalone: it imports only :mod:`.algo` and
``torch`` at module load time, so it can be unit-tested against mock pools
without pulling in the heavy serving stack.

Wiring into the runtime (``forward_decode`` hook, ``model_runner`` after-forward
call, ``scheduler`` request begin/end) is intentionally **not** done in this
file yet -- those touch core files and need on-GPU validation. See the TODOs at
the bottom and the roadmap in ``R-KV/doc/DESIGN.md`` section 9.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.rkv.algo import R1KV

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids heavy imports
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


# Cap on the batched-scoring transient (cosine matrix + mask + indices) per
# compaction. Because compactions run sequentially (one armed request at a time
# in ``maybe_compact``), this is also the per-step compaction-workspace peak the
# KV-pool sizing reserves so a compaction never OOMs on transient tensors even
# when the pool is full (ModelRunner._reserve_rkv_decode_aux_bytes).
RKV_SCORE_CHUNK_BYTES: int = 512 << 20


# Backends into which the R-KV observation / compaction hooks are actually
# wired. The hooks live only in the MHA FlashInfer attention backend and index
# every layer's KV buffer with full ``req_to_token`` slots.
_RKV_SUPPORTED_ATTENTION_BACKENDS = ("flashinfer",)

# KV-cache dtypes R-KV scoring can consume directly. The compressor reads the
# raw K/V buffer (``token_to_kv_pool.get_key_buffer``) and feeds it straight into
# RMSNorm / ``q @ k^T`` importance+redundancy scoring with NO dequantization
# path, so a quantized KV cache (fp8/fp4) would be scored in the wrong numeric
# representation. Restrict to unquantized float KV until scoring learns to apply
# the per-layer ``k_scale``/``v_scale`` dequant.
_RKV_SUPPORTED_KV_CACHE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def rkv_runtime_support_error(
    *,
    mode: str,
    prefill_backend: Optional[str],
    decode_backend: Optional[str],
    use_mla: bool,
    is_hybrid_swa: bool,
    spec_enabled: bool,
    page_size: Optional[int],
    kv_cache_dtype: Optional[torch.dtype] = None,
) -> Optional[str]:
    """Return a reason the *resolved* runtime cannot support R-KV, else ``None``.

    ``ServerArgs._handle_rkv*_validation`` runs before the attention backend,
    model architecture, page size, and speculative algorithm are resolved, so it
    cannot see what is actually wired. This pure check is called from
    ``ModelRunner`` right before the compressor is built, once everything is
    resolved, and hard-fails any configuration whose runtime the observation /
    compaction hooks do not cover:

    * the hooks exist ONLY in the MHA FlashInfer backend (prefill observe lives
      in ``forward_extend``; decode observe in ``forward_decode``);
    * they index every layer buffer with full ``req_to_token`` slots, so MLA
      (different KV layout) and hybrid-SWA (needs full->SWA slot translation)
      pools are unsupported;
    * speculative decoding (e.g. TARGET_VERIFY) drives extra forwards the hooks
      do not account for;
    * page_size must be 1 for per-slot free.
    * quantized KV-cache dtypes (fp8_e4m3 / fp8_e5m2 / fp4_e2m1) cannot be
      scored: the compressor reads the raw K/V buffer and feeds it into RMSNorm
      / ``q @ k^T`` with no ``k_scale``/``v_scale`` dequantization path.

    ``mode`` is ``"decode"`` or ``"prefill"`` (used only in the message).
    ``kv_cache_dtype`` is the *resolved* KV-cache ``torch.dtype`` (from
    ``ModelRunner.configure_kv_cache_dtype``); ``None`` skips the dtype check.
    """
    if (
        prefill_backend not in _RKV_SUPPORTED_ATTENTION_BACKENDS
        or decode_backend not in _RKV_SUPPORTED_ATTENTION_BACKENDS
    ):
        return (
            f"R-KV ({mode}) requires the FlashInfer attention backend for both "
            f"prefill and decode (resolved prefill={prefill_backend!r}, "
            f"decode={decode_backend!r}); the R-KV observation hooks are wired "
            "only into FlashInferAttnBackend."
        )
    if use_mla:
        return (
            f"R-KV ({mode}) does not support MLA models: the compressor indexes "
            "every layer's KV buffer with full req_to_token slots, which the MLA "
            "pool layout does not provide."
        )
    if is_hybrid_swa:
        return (
            f"R-KV ({mode}) does not support hybrid sliding-window (SWA) models: "
            "SWA layers require full-to-SWA slot translation that the compressor "
            "does not perform (it indexes every layer with full req_to_token "
            "slots)."
        )
    if spec_enabled:
        return (
            f"R-KV ({mode}) does not support speculative decoding: TARGET_VERIFY "
            "and draft forwards are not wired into the observation/compaction "
            "hooks."
        )
    if page_size not in (None, 1):
        return f"R-KV ({mode}) requires page_size == 1 (per-slot free)."
    if (
        kv_cache_dtype is not None
        and kv_cache_dtype not in _RKV_SUPPORTED_KV_CACHE_DTYPES
    ):
        return (
            f"R-KV ({mode}) does not support a quantized KV cache dtype "
            f"({kv_cache_dtype}): the compressor reads the raw K/V buffer and "
            "scores it with RMSNorm / q @ k^T with no dequantization path, so "
            "fp8_e4m3 / fp8_e5m2 / fp4_e2m1 would be ranked in the wrong "
            "representation. Use an unquantized KV cache (fp16/bf16/fp32) until "
            "R-KV scoring learns to dequantize (k_scale/v_scale)."
        )
    return None


class _CompactionCommit(msgspec.Struct):
    """One prepared compaction awaiting the scheduler-synced commit.

    Emitted by the forward-side *prepare* phase (``_prepare_compaction``), after
    the surviving K/V has already been relocated to the front ``budget`` slots.
    The scheduler drains these after the forward (``commit_compactions``) to free
    the tail slots and finalize the request bookkeeping, so the allocator is
    mutated only at a point where the forward stream is synced — not from inside
    the forward. ``freed_slots`` is a device tensor and ``req`` a duck-typed
    ``Req``; this struct is an in-process container only (never serialized).
    """

    req_pool_idx: int
    budget: int
    seq_len: int
    freed_slots: Any
    req: Any


@dataclass
class RKVConfig:
    """Hyper-parameters for R-KV compression.

    ``budget=1024`` is a serving-oriented default (the R-KV reference class and
    its HF eval scripts use ``budget=128``); ``buffer_size`` is the staging
    buffer ``B_buffer`` that controls how often compression fires. Note the
    reference HF eval scripts run ``mix_lambda=0.1, retain_ratio=0.2`` while the
    reference class defaults are ``mix_lambda=0.07, retain_ratio=0.1``; override
    via the ``--rkv-*`` flags / ``--rkv-config`` to match a specific
    configuration exactly.
    """

    budget: int = 1024
    window_size: int = 8
    kernel_size: int = 7
    mix_lambda: float = 0.1
    retain_ratio: float = 0.1
    retain_direction: str = "last"
    # Compress every ``buffer_size`` newly generated tokens per request.
    buffer_size: int = 128
    # Minimum KV length before compression is ever considered. Defaults to
    # ``budget`` (there is nothing to drop below budget).
    min_seq_len: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_seq_len is None:
            self.min_seq_len = self.budget
        # Config validation raises (not assert, which -O strips) so an invalid
        # --rkv-config can never silently start a corrupting server.
        if self.window_size <= 0:
            raise ValueError("R-KV window_size must be a positive integer")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(
                "R-KV kernel_size must be a positive ODD integer: max_pool1d "
                "with an even kernel emits n-window+1 importance values against "
                "n-window redundancy values, which fails deterministically in "
                "R1KV._scores"
            )
        if self.retain_direction not in (
            "last",
            "first",
            "last_percent",
            "first_percent",
        ):
            raise ValueError(
                "R-KV retain_direction must be one of 'last' / 'first' / "
                "'last_percent' / 'first_percent'"
            )
        if self.budget <= self.window_size:
            raise ValueError("R-KV budget must exceed window_size")
        if self.buffer_size < self.window_size:
            raise ValueError(
                "R-KV buffer_size must be >= window_size, otherwise the first "
                "compaction scores against zero-initialized observation queries"
            )
        if self.min_seq_len < self.budget:
            raise ValueError(
                "R-KV min_seq_len must be >= budget (select_indices keeps budget "
                "tokens)"
            )


class RKVRequestState:
    """Per-request bookkeeping the integration layer maintains.

    One instance per active request, keyed by ``req_pool_idx`` in
    :class:`RKVCompressor`. Holds the trigger counter, the per-layer observation
    query window (R-KV needs the last ``window_size`` queries, and queries are
    per-layer), and the running cross-layer score accumulator used during a
    compaction step.
    """

    def __init__(self, req_pool_idx: int) -> None:
        self.req_pool_idx = req_pool_idx

        # Generated steps since the last compaction (trigger cadence counter),
        # advanced once per decode step by ``RKVCompressor.begin_decode_step``
        # (which runs every step, unlike the in-graph query collection which is
        # captured once into the decode graph and replayed).
        self.steps_since_compact = 0

        # Logical absolute position of the request's next token (vestigial:
        # decode positions are derived from the request in
        # ``override_decode_positions``, not from this field).
        self.next_position = 0

        # Per-layer observation-window queries, shape
        # (num_layers, window_size, q_head_num, head_dim), set transiently in
        # ``maybe_compact`` from the in-graph ``rolling_q`` collection (read
        # un-rotated to temporal order) right before scoring this request.
        self.window_q: Optional[torch.Tensor] = None

        # Back-reference to the owning request, so compaction can update its
        # physical-length bookkeeping (kv_committed_len / kv_allocated_len).
        # Duck-typed: only needs kv_committed_len / kv_allocated_len /
        # origin_input_ids / output_ids. Set in ``on_request_begin``.
        self.req: Optional[Req] = None


class RKVCompressor:
    """Coordinates R-KV compression across a request's lifetime.

    Borrows the lifecycle-hook naming from ``mem_cache/sparsity`` but shares no
    code with it (that framework is non-destructive; R-KV truly evicts and frees
    slots -- see R-KV/doc/DESIGN.md section 8).
    """

    def __init__(
        self,
        config: RKVConfig,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        kv_allocator: BaseTokenToKVPoolAllocator,
        start_layer: int,
        end_layer: int,
        device: torch.device,
        q_head_num: int,
        head_dim: int,
        q_dtype: torch.dtype,
        fused_validation: str = "first-request",
        attn_tp_group=None,
    ) -> None:
        self.config = config
        # Serving is restricted to the memory-BOUNDED redundancy path. Only
        # retain_direction="last" uses the fused Triton kernel / O(n)-memory
        # tiled reference; every other direction falls back to cal_similarity,
        # which materializes a kv_heads x n x n cosine matrix (+ mask + int64
        # indices) that can reach many GiB and is NOT covered by the fixed
        # compaction-workspace reservation. Reject it here rather than OOM mid
        # serving. (The algorithm still supports other directions for offline
        # use; only the served RKVCompressor is gated.)
        if config.retain_direction != "last":
            raise ValueError(
                "R-KV serving supports only retain_direction='last' (the "
                f"memory-bounded path); got {config.retain_direction!r}. Other "
                "directions build an unbounded kv_heads x n x n similarity "
                "matrix that the compaction-workspace reservation cannot cap."
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.kv_allocator = kv_allocator
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_layers = end_layer - start_layer
        self.device = device

        # Attention-TP group. Under TP each rank holds only a SUBSET of the KV
        # heads, so its per-token score (a cross-head mean over LOCAL heads)
        # differs from other ranks -> different kept set -> the replicated
        # req_to_token diverges -> silent KV corruption. We sum the per-token
        # score across this group before top-k so every rank keeps the exact
        # same tokens (see _reduce_score_across_tp). ``None`` / world_size==1
        # means no TP: the reduce and check are no-ops (single-GPU unchanged).
        self.attn_tp_group = attn_tp_group
        self.attn_tp_size = (
            getattr(attn_tp_group, "world_size", 1) if attn_tp_group is not None else 1
        )
        # TP kept-set consistency check: self-validate on the first few
        # compactions (cheap), or every compaction when SGLANG_RKV_TP_CHECK=1.
        self._tp_check_always = os.environ.get("SGLANG_RKV_TP_CHECK", "0") == "1"
        self._tp_check_remaining = 8

        self.algo = R1KV(
            budget=config.budget,
            window_size=config.window_size,
            kernel_size=config.kernel_size,
            mix_lambda=config.mix_lambda,
            retain_ratio=config.retain_ratio,
            retain_direction=config.retain_direction,
            fused_validation=fused_validation,
        )
        # Startup fused-kernel validation (no-op unless fused_validation ==
        # "startup"): warm the fused-vs-reference gate now, using the real KV
        # head count / dtype, so the first real compaction pays no gate cost.
        k0 = self.token_to_kv_pool.get_key_buffer(self.start_layer)
        self.algo.warmup_fused_kernel(
            kv_heads=k0.shape[1],
            head_dim=head_dim,
            device=device,
            dtype=k0.dtype,
            seq_len=config.budget,
        )

        # Active per-request state, keyed by req_pool_idx.
        self.states: Dict[int, RKVRequestState] = {}
        # req_pool_idx values that armed a compaction this forward pass.
        self._armed: set[int] = set()
        # Two-phase compaction: the forward-side prepare phase relocates K/V and
        # appends a commit record here; the scheduler drains it after the forward
        # (``commit_compactions``) to free slots + finalize bookkeeping, keeping
        # allocator mutation out of the forward stream.
        self._pending_commits: List[_CompactionCommit] = []
        # req_pool_idx -> (new physical KV length, owning Req) after the latest
        # compaction. The scheduler drains this (``take_pending_length_updates``)
        # to update the batch-level seq_lens tensors, which it owns. The owning
        # Req is stored so the scheduler can VALIDATE identity before applying an
        # update: if the pool slot was released and reused between the compaction
        # and the drain, the update belongs to a dead request and must be
        # ignored (belt-and-braces with the clear-on-finish/retract below).
        self.pending_length_updates: Dict[int, Tuple[int, Req]] = {}
        # Batched-scoring A/B gate: None = not yet checked, True = batched
        # selects the same kept set as the per-layer reference (adopted), False
        # = they differed on the first compaction (per-layer fallback forever).
        self._batched_ok: Optional[bool] = None
        # Cap the batched-scoring transient (cosine matrix + mask + indices) so
        # peak memory stays bounded when budget (=> seq_len) is large. This is
        # also the per-request compaction-workspace bound the KV-pool sizing
        # reserves (ModelRunner._reserve_rkv_decode_aux_bytes).
        self._score_chunk_bytes: int = RKV_SCORE_CHUNK_BYTES

        # --- (1c) in-graph decode query collection ---
        # Rolling per-layer observation-window buffer, keyed by req_pool_idx (so
        # it survives batch reordering). It is written INSIDE the decode CUDA
        # graph via an index_copy_ scatter in forward_decode -- the same class of
        # op as set_kv_buffer writing token_to_kv_pool at out_cache_loc -- so the
        # observation-window steps no longer need to run eager. Circular over
        # window_size; the write slot is a global (batch-synchronous) counter.
        #
        # Row count == req_to_token rows == max_running_requests + 1, so there is
        # exactly one row per possible concurrent request (no oversized pool: the
        # decode concurrency ceiling IS the row count). Row 0 is the reserved
        # ReqToTokenPool padding slot (never assigned to a real request), which is
        # what makes the unconditional in-graph scatter safe under CUDA-graph
        # padding -- see collect_decode_query. This buffer is reserved in the KV
        # pool sizing (ModelRunner._reserve_rkv_decode_aux_bytes) so it cannot OOM
        # at startup.
        max_reqs = req_to_token_pool.req_to_token.shape[0]
        window = config.window_size
        self.rolling_q = torch.zeros(
            (self.num_layers, window, max_reqs, q_head_num, head_dim),
            device=device,
            dtype=q_dtype,
        )
        # Flattened (window * max_reqs) view for the in-graph scatter index.
        self._rolling_q_flat = self.rolling_q.view(
            self.num_layers, window * max_reqs, q_head_num, head_dim
        )
        self._rolling_max_reqs = max_reqs
        # Per-request decode-step counter, keyed by req_pool_idx (a fixed-address
        # GPU tensor). This step's circular write slot for request ``r`` is
        # ``step_count_of_req[r] % window_size``. begin_decode_step advances it
        # for every request in the batch each decode step (incl. graph-replay
        # steps), and collect_decode_query reads it in-graph. Per-request (not a
        # single global cursor) so the observation window stays correct even if a
        # request skips decode steps (future preemption/pipeline/overlap): each
        # request's slots only advance on the steps it actually participates in.
        # Padding safety does NOT depend on row 0's counter value: row 0 is the
        # reserved ReqToTokenPool padding slot (never a real request), so any
        # padding-row write lands on rolling_q row 0 harmlessly regardless of the
        # counter — the same reason the in-graph scatter is safe under padding.
        self.step_count_of_req = torch.zeros(
            (max_reqs,), device=device, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------
    def on_request_begin(self, req: Req) -> None:
        """Register a request and initialise its R-KV state."""
        if req.req_pool_idx is None:
            return
        state = RKVRequestState(req_pool_idx=req.req_pool_idx)
        state.next_position = len(req.origin_input_ids)
        state.req = req
        self.states[req.req_pool_idx] = state
        # Reset this slot's step counter so a reused req_pool_idx starts a fresh
        # observation window (does not inherit the previous request's cursor).
        self.step_count_of_req[req.req_pool_idx] = 0

    def _clear_request_state(self, idx: int) -> bool:
        """Drop ALL per-request R-KV bookkeeping for ``idx`` and return whether a
        state existed.

        Every finish/retract path routes through here so no stale compaction
        artifact (pending physical-length update or armed flag) can survive the
        release of ``idx`` and be applied to the next request that reuses the
        same pool slot. ``_pending_commits`` is intentionally NOT touched: the
        scheduler always drains it (``commit_compactions``) BEFORE the finish
        loop that calls ``on_request_end``, so a queued commit is never left
        behind; if one somehow is, the commit-path stale-plan identity guard in
        ``_commit_compaction`` must still fire rather than be silently dropped.
        """
        had = self.states.pop(idx, None) is not None
        self.pending_length_updates.pop(idx, None)
        self._armed.discard(idx)
        return had

    def on_request_end(self, req: Req) -> None:
        """Drop a request's R-KV state (and any pending compaction bookkeeping)
        when it finishes or aborts.

        Clearing the pending length update / armed flag / queued commit is
        REQUIRED, not just tidy: a request can compact and finish on the same
        step (or finish as the last request in the batch), after which the pool
        slot is released and reused. A leftover ``pending_length_updates`` entry
        would then be applied to whatever request next reuses this
        ``req_pool_idx`` in ``_apply_rkv_pre_decode``, corrupting its seq_lens.
        """
        if req.req_pool_idx is not None and self._clear_request_state(req.req_pool_idx):
            logger.debug(
                "R-KV on_request_end req_pool_idx=%d states_left=%d",
                req.req_pool_idx,
                len(self.states),
            )

    def on_request_retract(self, req: Req) -> None:
        """Discard a retracted request's R-KV state.

        Retraction frees the request's physical KV and sends it back to the
        waiting queue to be re-prefilled from scratch. Its old state (compaction
        counter, observation-window queries, physical-length bookkeeping) no
        longer matches the freed KV, so drop it here; a clean state is rebuilt by
        ``on_request_begin`` when the request re-enters decode. Must be called
        while ``req.req_pool_idx`` is still valid, i.e. BEFORE the pool frees it.
        """
        if req.req_pool_idx is not None:
            self._clear_request_state(req.req_pool_idx)

    # ------------------------------------------------------------------
    # Scheduler admission support
    # ------------------------------------------------------------------
    def admission_reserve(self, prompt_len: int, occupied: int) -> int:
        """Upper bound on the *future* physical KV a request can still consume.

        R-KV holds a request's physical KV cache at a constant ceiling no matter
        how many tokens it will still generate: it lets the length grow to at
        most ``max(prompt_len, min_seq_len) + buffer_size`` (the peak reached the
        step just before a compaction) and then frees back down to ``budget``.
        So the scheduler only needs to reserve ``ceiling - occupied`` for a
        request, NOT its full remaining ``max_new_tokens``. Reserving this (much
        smaller) bound is what lets many more R-KV requests run concurrently.

        The bound is deliberately the pre-compaction PEAK, so it is never an
        underestimate: a request's physical KV can never exceed it, hence
        admission can never over-commit the pool (memory-safe). ``occupied`` is
        the request's current physical KV length (its committed length, or its
        prompt length before prefill).
        """
        ceiling = max(prompt_len, self.config.min_seq_len) + self.config.buffer_size
        return max(0, ceiling - occupied)

    # ------------------------------------------------------------------
    # Decode-time observation (called per layer from forward_decode)
    # ------------------------------------------------------------------
    def begin_decode_step(self, forward_batch: ForwardBatch) -> bool:
        """Advance per-request decode counters and decide graph-vs-eager.

        Called once per decode step in ``ModelRunner.forward`` BEFORE the
        graph/eager decision, so it runs on EVERY step — including steps that
        replay a captured CUDA graph. (The per-request compaction counter must
        advance on graph-replay steps too, which is why it lives here rather
        than in a hook inside ``forward_decode``.)

        For each managed request it advances ``steps_since_compact`` (only once
        the request is long enough to compress, ``seq_len >= min_seq_len``) and
        arms the compaction on the final window step.

        Returns True only on a compaction step (``steps_since_compact >=
        buffer_size``): that step must run eager because ``maybe_compact`` is
        called on the eager path (the graph path returns before it). The
        observation-window queries are collected in-graph by
        ``collect_decode_query``, so the window steps no longer force eager --
        they replay the captured decode graph like any other step.
        """
        # Advance the per-request circular write cursor EVERY decode step (even
        # with no managed requests). Vectorized over the whole batch on the GPU
        # tensor the captured rolling_q scatter reads; begin_decode_step runs
        # BEFORE the graph/eager decision, so this advance reaches graph-replay
        # steps. Per-request (not one global cursor) so each request's window
        # only advances on the steps it actually participates in — correct even
        # if a request skips decode steps (future preemption/pipeline/overlap).
        # Reused req_pool_idx slots are additionally protected by
        # buffer_size >= window_size (asserted in RKVConfig), which forces
        # >= window_size fresh writes before the first compaction, and by the
        # on_request_begin counter reset.
        #
        # CUDA-graph ordering: this eager ``+= 1`` is issued on the current
        # stream, and the captured decode graph is replayed with a plain
        # ``cuda_graph.replay()`` on that same current stream (see the
        # full/tc-piecewise graph backends — no stream switch; overlap and
        # PDMux, which could introduce a second stream, are rejected for R-KV).
        # Same-stream ops execute in issue order, so the counter update
        # happens-before the graph read of ``step_count_of_req`` — the identical
        # guarantee ``set_kv_buffer`` (out_cache_loc written eagerly, read
        # in-graph) already relies on.
        self.step_count_of_req[forward_batch.req_pool_indices] += 1

        if not self.states:
            return False
        req_indices = forward_batch.req_pool_indices.tolist()
        seq_lens_src = forward_batch.seq_lens_cpu
        if seq_lens_src is None:
            seq_lens_src = forward_batch.seq_lens
        seq_lens = seq_lens_src.tolist()

        buffer = self.config.buffer_size
        need_eager = False
        for i, req_pool_idx in enumerate(req_indices):
            state = self.states.get(int(req_pool_idx))
            if state is None:
                continue
            state.next_position += 1
            # Only start the compaction clock once the request can actually be
            # compressed; below budget there is nothing to evict.
            if int(seq_lens[i]) < self.config.min_seq_len:
                continue
            state.steps_since_compact += 1
            if state.steps_since_compact >= buffer:
                # Compaction step: the ONLY step that must run eager, because
                # ``maybe_compact`` runs on the eager path (the graph path
                # returns before it). Every other decode step -- including the
                # ``window_size`` observation-window steps ending here -- replays
                # the captured decode graph; their queries are collected in-graph
                # by ``collect_decode_query``. Scoring is batched across all
                # layers in ``maybe_compact`` after the forward.
                need_eager = True
                self._armed.add(int(req_pool_idx))
        return need_eager

    def collect_decode_query(
        self, q: torch.Tensor, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> None:
        """In-graph rolling collection of this layer's decode query.

        Writes ``q`` ``(bs, q_head_num, head_dim)`` into the fixed ``rolling_q``
        buffer at each request's current circular write slot
        (``step_count_of_req[r] % window_size``), indexed by req_pool_idx.
        Pure-tensor scatter (no host sync / Python loop), so it is captured in
        the decode CUDA graph and runs on graph-replay steps too -- the same
        class of op as ``set_kv_buffer`` scattering into ``token_to_kv_pool`` at
        ``out_cache_loc``. This is what lets observation-window steps stop
        running eager (P3). Unconditional so CUDA-graph capture includes it.

        CUDA-graph padding safety: a padded replay batch fills the tail
        ``req_pool_indices`` with 0 (PaddingPolicy.ZERO), so padding rows scatter
        their (discarded) query into ``rolling_q`` row 0 -- the reserved
        ReqToTokenPool padding slot, never a real request -- exactly as those
        rows write ``token_to_kv_pool`` at ``out_cache_loc`` 0. No real request's
        observation window is corrupted.
        """
        layer_idx = layer.layer_id - self.start_layer
        # This step's circular write slot per request = step_count % window.
        # Read in-graph from the fixed-address per-request counter tensor that
        # begin_decode_step advanced this step. Flattened index into
        # (window * max_reqs): slot-major, req-minor.
        req_indices = forward_batch.req_pool_indices
        cur = self.step_count_of_req[req_indices] % self.config.window_size
        index = cur * self._rolling_max_reqs + req_indices
        self._rolling_q_flat[layer_idx].index_copy_(0, index, q)

    def _read_window_rolling(self, req_pool_idx: int) -> torch.Tensor:
        """Read a request's observation window from ``rolling_q``, un-rotated to
        temporal order (slot 0 oldest ... window-1 newest, matching
        ``RKVRequestState.window_q``). Called at compaction (eager), so a host
        ``roll`` is fine. Returns ``(num_layers, window_size, q_heads, head_dim)``.
        """
        w = self.rolling_q[:, :, req_pool_idx]  # (num_layers, window, q_heads, hd)
        window = self.config.window_size
        # This request's own newest write slot (per-request cursor).
        cur = int(self.step_count_of_req[req_pool_idx].item()) % window
        return torch.roll(w, shifts=-(cur + 1), dims=1)

    def _layer_score(
        self, state: RKVRequestState, layer_idx: int, seq_len: int
    ) -> torch.Tensor:
        """Per-token joint R-KV score for one layer, averaged across KV heads.

        Returns shape ``(seq_len - window_size,)``.
        """
        layer_id = self.start_layer + layer_idx
        r2t = self.req_to_token_pool.req_to_token
        slots = r2t[state.req_pool_idx, :seq_len].long()

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
        # keys: (1, kv_heads, seq_len, head_dim) -- NHD buffer is
        # (num_slots, kv_heads, head_dim); gather this request's slots.
        keys = k_buffer[slots].unsqueeze(0).transpose(1, 2).contiguous()

        # queries: (1, q_heads, window_size, head_dim). window_q holds the last
        # window_size decode queries in temporal order (slot 0 oldest ..
        # window_size-1 newest), read from the in-graph rolling collection
        # (_read_window_rolling) at compaction time.
        window_q = state.window_q[layer_idx]  # (window, q_heads, head_dim)
        queries = window_q.unsqueeze(0).transpose(1, 2).contiguous()

        # (1, kv_heads, seq_len - window) -> mean over heads -> (seq_len - window,)
        final_score = self.algo._scores(keys, queries)
        return final_score.mean(dim=1).squeeze(0)

    def _reference_scores(self, state: RKVRequestState, seq_len: int) -> torch.Tensor:
        """Per-layer sequential scoring (the original path; A/B reference).

        Sums the per-layer mean-over-heads scores in layer order, exactly as the
        old ``observe`` accumulation did. Returns ``(seq_len - window_size,)``.
        """
        acc: Optional[torch.Tensor] = None
        for layer_idx in range(self.num_layers):
            s = self._layer_score(state, layer_idx, seq_len)
            acc = s if acc is None else acc + s
        return acc

    def _batched_scores(self, state: RKVRequestState, seq_len: int) -> torch.Tensor:
        """Batched scoring over all layers in one GEMM (the optimization).

        Gathers every layer's K for this request, stacks them into a single
        ``(num_layers, kv_heads, seq_len, head_dim)`` batch, and runs one
        ``algo._scores`` call (num_layers as the batch dim) instead of
        ``num_layers`` bsz=1 calls. Cross-head mean, then cross-layer sum in
        layer order (matching ``_reference_scores``). Chunked over layers so the
        transient cosine-similarity matrix stays under ``_score_chunk_bytes``.
        Returns ``(seq_len - window_size,)``.
        """
        r2t = self.req_to_token_pool.req_to_token
        slots = r2t[state.req_pool_idx, :seq_len].long()
        # window_q: (num_layers, window, q_heads, hd) -> (num_layers, q_heads, window, hd)
        queries_all = state.window_q.transpose(1, 2).contiguous()

        k0 = self.token_to_kv_pool.get_key_buffer(self.start_layer)
        kv_heads = k0.shape[1]
        # cosine matrix (key dtype, x2 for softmax read/write) + bool mask + int32
        # indices, per element of (kv_heads, seq, seq) -- owner's per-pair budget.
        per_layer = (2 * k0.element_size() + 1 + 4) * kv_heads * seq_len * seq_len
        chunk = max(
            1, min(self.num_layers, self._score_chunk_bytes // max(1, per_layer))
        )

        acc: Optional[torch.Tensor] = None
        for c in range(0, self.num_layers, chunk):
            hi = min(c + chunk, self.num_layers)
            # (chunk, seq_len, kv_heads, hd) -> (chunk, kv_heads, seq_len, hd)
            keys = (
                torch.stack(
                    [
                        self.token_to_kv_pool.get_key_buffer(self.start_layer + l)[
                            slots
                        ]
                        for l in range(c, hi)
                    ]
                )
                .transpose(1, 2)
                .contiguous()
            )
            # (chunk, kv_heads, seq_len - window) -> mean over heads -> (chunk, seq_len - window)
            layer_scores = self.algo._scores(keys, queries_all[c:hi]).mean(dim=1)
            # Sum over layers in order (matches the sequential reference).
            for li in range(layer_scores.shape[0]):
                s = layer_scores[li]
                acc = s if acc is None else acc + s
        return acc

    # ------------------------------------------------------------------
    # Compaction (prepare in the forward, commit at a scheduler-synced point)
    # ------------------------------------------------------------------
    def maybe_compact(self, forward_batch: ForwardBatch) -> None:
        """Prepare physical compaction for any request armed this forward pass.

        This is the **prepare** phase, run at the end of the decode forward: it
        scores each armed request and relocates its surviving K/V to the front
        ``budget`` slots (forward-stream compute), then appends a commit record
        to ``_pending_commits``. The allocator free + request bookkeeping is
        deferred to ``commit_compactions`` (called by the scheduler after the
        forward), so allocator state is never mutated from inside the forward
        stream. Scoring is batched across all layers (see ``_batched_scores``);
        the first compaction runs an A/B gate against the per-layer reference.

        Fail-stop contract: the relocation is in-place, so a request that raises
        mid-relocation (OOM, a validation guard, a CUDA fault) is left partially
        moved. Any exception raised here is therefore UNRECOVERABLE — the caller
        (ModelRunner forward) must let it propagate and terminate the worker; it
        must NOT be caught-and-continued, or the partially-relocated request
        would serve corrupt KV. ``_armed`` is cleared up front (below) only so a
        *restarted* accumulator does not inherit stale arming, not to imply the
        current worker can recover.
        """
        if not self._armed:
            return

        # Clear the armed set up front so a mid-loop exception (Triton failure,
        # OOM, a validation guard) cannot carry stale armed requests forward.
        # This is fail-stop hygiene, not recovery (see the contract above).
        armed = self._armed
        self._armed = set()

        seq_len_by_req = self._seq_len_by_req(forward_batch)
        # sorted() (not list(set)) so every attention-TP rank iterates the armed
        # requests in the SAME order and therefore issues the per-request score
        # all-reduces in the same order — mismatched collective order across
        # ranks would pair the wrong tensors (silent corruption / hang).
        for req_pool_idx in sorted(armed):
            # An armed request was armed from THIS forward's req_pool_indices in
            # begin_decode_step, and no lifecycle hook runs between arming and
            # here (same forward). So a missing state or seq_len is a genuine
            # lifecycle desync, not an expected skip — fail fast rather than
            # silently drop the compaction and let the request's KV grow.
            state = self.states.get(req_pool_idx)
            if state is None:
                raise RuntimeError(
                    f"Armed R-KV request {req_pool_idx} has no compressor state "
                    "(lifecycle desync between arming and compaction)"
                )
            seq_len = seq_len_by_req.get(req_pool_idx)
            if seq_len is None:
                raise RuntimeError(
                    f"Armed R-KV request {req_pool_idx} is missing from the "
                    "ForwardBatch it was armed from"
                )

            # P2: the observation-window queries now come from the in-graph
            # rolling collection (un-rotated to temporal order), not the eager
            # window_q. Values are identical (P1 validated diff=0); this switch
            # is what lets P3 stop forcing the window steps to run eager.
            state.window_q = self._read_window_rolling(req_pool_idx)

            if self._batched_ok is None:
                ref_kept = self._assemble_kept(
                    self._reduce_score_across_tp(
                        self._reference_scores(state, seq_len)
                    ),
                    seq_len,
                )
                bat_kept = self._assemble_kept(
                    self._reduce_score_across_tp(self._batched_scores(state, seq_len)),
                    seq_len,
                )
                same_shape = ref_kept.shape == bat_kept.shape
                self._batched_ok = bool(same_shape and torch.equal(ref_kept, bat_kept))
                logger.info(
                    "R-KV batched-scoring gate: %s (kept diff=%s)",
                    (
                        "OK -> batched adopted"
                        if self._batched_ok
                        else "DIFFER -> per-layer fallback"
                    ),
                    int((ref_kept != bat_kept).sum()) if same_shape else "shape",
                )
                kept = bat_kept if self._batched_ok else ref_kept
            else:
                score = self._reduce_score_across_tp(
                    self._batched_scores(state, seq_len)
                    if self._batched_ok
                    else self._reference_scores(state, seq_len)
                )
                kept = self._assemble_kept(score, seq_len)

            # Under TP, verify every rank derived the identical kept set (turns a
            # silent cross-rank KV divergence into a loud failure).
            self._check_kept_consistent_across_tp(kept)
            self._pending_commits.append(self._prepare_compaction(state, seq_len, kept))

    def commit_compactions(self) -> None:
        """Commit phase: apply every prepared compaction.

        The scheduler calls this after the decode forward has completed (a
        stream-synced point), *before* it releases any finished request's KV, so
        the tail free + ``req_to_token`` clear land before ``release_kv_cache``
        (no double-free) and the allocator is mutated with the forward stream
        idle. No-op when nothing was prepared.
        """
        if not self._pending_commits:
            return
        # Drain up front so a mid-loop failure cannot re-commit an already-freed
        # plan on a later call.
        plans = self._pending_commits
        self._pending_commits = []
        for plan in plans:
            self._commit_compaction(plan)

    # ------------------------------------------------------------------
    # Tensor-parallel score agreement
    # ------------------------------------------------------------------
    def _reduce_score_across_tp(self, score: torch.Tensor) -> torch.Tensor:
        """Sum the per-token eviction score across the attention-TP group.

        R-KV's score is a cross-head MEAN, computed here over each rank's LOCAL
        kv heads only; the softmax/pool inside the score are per-head, so the
        cross-head reduction is LINEAR. Head sharding is uniform (either
        ``num_kv_heads % tp == 0`` for distinct heads, or ``tp % num_kv_heads ==
        0`` with uniform replication), so the all-reduced SUM equals the true
        global cross-head mean scaled by a positive constant. ``topk`` is
        invariant to positive scaling, so after this every rank selects the
        IDENTICAL kept set — which is what keeps the replicated ``req_to_token``
        consistent across ranks. No-op at tp==1 (single-GPU path unchanged).
        """
        if self.attn_tp_group is None or self.attn_tp_size <= 1:
            return score
        # fp32 all-reduce: identical across ranks (all-reduce semantics), and
        # bounded-magnitude so ``topk`` ties break identically on every rank.
        return self.attn_tp_group.all_reduce(score.float())

    def _check_kept_consistent_across_tp(self, kept: torch.Tensor) -> None:
        """Assert every attention-TP rank derived the identical kept set.

        Turns a silent cross-rank KV divergence (the failure mode that makes TP
        unsafe without the score all-reduce) into a loud error. Runs on the
        first few compactions (cheap startup self-validation) or on every
        compaction when ``SGLANG_RKV_TP_CHECK=1``. The fire schedule is a
        function of the (rank-identical) compaction count, so the extra
        collective is issued in lockstep across ranks.
        """
        if self.attn_tp_group is None or self.attn_tp_size <= 1:
            return
        if not self._tp_check_always:
            if self._tp_check_remaining <= 0:
                return
            self._tp_check_remaining -= 1
        # kept indices are < seq_len (<= a few 10k) and tp <= a handful, so the
        # summed values stay exactly representable in fp32.
        local = kept.to(torch.float32)
        summed = self.attn_tp_group.all_reduce(local.clone())
        if not torch.equal(summed, local * self.attn_tp_size):
            raise RuntimeError(
                "R-KV TP divergence: attention-TP ranks selected different kept "
                "sets. The per-token score all-reduce is not producing identical "
                "scores across ranks; continuing would corrupt the KV cache."
            )

    def _assemble_kept(self, score_accum: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Global kept-token indices: top past tokens + trailing window.

        ``score_accum`` covers the past tokens ``[0, seq_len - window)``.
        Returns ascending indices of length ``budget``.
        """
        num_past = self.config.budget - self.config.window_size
        past_idx = score_accum.topk(num_past).indices
        window_idx = torch.arange(
            seq_len - self.config.window_size,
            seq_len,
            device=score_accum.device,
        )
        kept = torch.cat([past_idx, window_idx])
        return torch.sort(kept).values

    def _prepare_compaction(
        self, state: RKVRequestState, seq_len: int, kept_local: torch.Tensor
    ) -> _CompactionCommit:
        """Prepare phase (runs in the forward): relocate one request's surviving
        K/V to the front ``budget`` slots for every layer (page_size == 1) and
        return a commit record. Does NOT free slots, clear ``req_to_token``, or
        touch request lengths — those are the scheduler-synced commit phase
        (``_commit_compaction``), so the allocator is mutated only with the
        forward stream idle. Kept indices must be ascending.
        """
        idx = state.req_pool_idx
        budget = self.config.budget
        r2t = self.req_to_token_pool.req_to_token

        slots = r2t[idx, :seq_len].long().clone()  # physical slots, temporal order

        # Validate the kept set and the physical slot table BEFORE mutating any
        # buffer, so a bad score/select or a corrupt req_to_token can never write
        # KV out of bounds, relocate from a duplicate slot, or double-free (every
        # check is on indices only — no KV writes have happened yet). These are
        # production safety barriers, so they RAISE rather than ``assert`` (which
        # ``python -O`` strips). Cheap: O(seq_len), once per compaction.
        if kept_local.numel() != budget:
            raise RuntimeError(
                f"R-KV kept set has {kept_local.numel()} entries, expected "
                f"budget {budget}"
            )
        if kept_local.numel() > 0:
            kept_min = int(kept_local.min())
            kept_max = int(kept_local.max())
            if kept_min < 0 or kept_max >= seq_len:
                raise RuntimeError(
                    f"R-KV kept indices out of range [0, {seq_len}): "
                    f"[{kept_min}, {kept_max}]"
                )
        if not bool(torch.all(kept_local[1:] > kept_local[:-1])):
            raise RuntimeError(
                "R-KV kept indices must be strictly ascending (unique, sorted)"
            )
        # The WHOLE physical slot table for this request must be a 1-to-1 map,
        # not just the survivors: a duplicate anywhere in slots[:seq_len] could
        # put the same slot in both the freed tail and the kept head, so commit
        # would free a slot req_to_token still references (use-after-free).
        # Checking the full table subsumes the survivors-unique check.
        if slots.unique().numel() != slots.numel():
            raise RuntimeError(
                "R-KV req_to_token contains duplicate physical KV slots "
                "(allocator corruption); refusing to compact"
            )

        src = slots[kept_local]  # surviving physical slots (budget,)
        dst = slots[:budget]  # target front slots (budget,)

        # Relocate K/V for every layer. Clone before write so overlapping
        # src/dst ranges don't corrupt each other. This is forward-stream compute
        # on the KV buffers; it does NOT touch the allocator.
        for layer_id in range(self.start_layer, self.end_layer):
            k_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
            v_buffer = self.token_to_kv_pool.get_value_buffer(layer_id)
            k_keep = k_buffer[src].clone()
            v_keep = v_buffer[src].clone()
            k_buffer[dst] = k_keep
            v_buffer[dst] = v_keep

        # Reset the trigger counter now (compressor-local state, not allocator).
        state.steps_since_compact = 0

        # The tail slots [budget, seq_len) are the survivors' old homes plus the
        # evicted tokens; they are freed in the commit phase. ``slots`` is a
        # clone, so this stays valid after req_to_token is cleared at commit.
        freed = slots[budget:seq_len].to(r2t.dtype)
        return _CompactionCommit(
            req_pool_idx=idx,
            budget=budget,
            seq_len=seq_len,
            freed_slots=freed,
            req=state.req,
        )

    def _commit_compaction(self, plan: _CompactionCommit) -> None:
        """Commit phase (scheduler-synced): free the tail slots, clear the
        ``req_to_token`` tail, and shrink the request's physical length.

        Runs after the forward at a stream-synced scheduler point, so the
        allocator free never races the forward. ``req_to_token[:budget]`` already
        holds the relocated kept KV (written in the prepare phase); here we only
        release the tail the survivors vacated.
        """
        idx = plan.req_pool_idx
        budget = plan.budget
        seq_len = plan.seq_len
        r2t = self.req_to_token_pool.req_to_token

        # Stale-plan guard: if the request slot was released or reused between
        # prepare (forward) and this commit, the tracked identity no longer
        # matches the plan. The prepare phase already relocated this request's
        # K/V, so a mismatch is an unrecoverable invariant break (the scheduler
        # order guarantees commit runs before any release), NOT something to skip
        # — fail the worker rather than free/rewrite the wrong request's slots.
        state = self.states.get(idx)
        if state is None or state.req is not plan.req:
            raise RuntimeError(
                f"Stale R-KV compaction plan for req_pool_idx={idx}: request "
                "slot was released or reused between prepare and commit"
            )

        # Free the tail slots (page_size == 1 => per-slot free).
        if plan.freed_slots.numel() > 0:
            self.kv_allocator.free(plan.freed_slots)

        # req_to_token[:budget] already equals the relocated kept KV in temporal
        # order. Clear the tail.
        r2t[idx, budget:seq_len] = 0

        # Physical-length bookkeeping. The scheduler normally treats seq_lens as
        # BOTH the physical KV length AND the rotary position source; R-KV breaks
        # that identity. We shrink the *physical* length to ``budget`` on the
        # owning request here (same process, shared pools), and expose the new
        # length via ``pending_length_updates`` so the scheduler can update its
        # batch-level seq_lens / seq_lens_cpu tensors. Rotary positions stay
        # *logical* and are supplied separately via ``logical_position``.
        # ``next_position`` is intentionally NOT rewound: future tokens keep
        # their absolute positions so their rotary stays consistent with the
        # retained keys.
        if plan.req is not None:
            plan.req.kv_committed_len = budget
            plan.req.kv_allocated_len = budget
        self.pending_length_updates[idx] = (budget, plan.req)

        logger.info(
            "R-KV compacted req_pool_idx=%d: phys %d -> %d slots (freed %d)",
            idx,
            seq_len,
            budget,
            plan.freed_slots.numel(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _seq_len_by_req(forward_batch: ForwardBatch) -> Dict[int, int]:
        """Map ``req_pool_idx -> physical seq_len`` for the current batch.

        Uses ``seq_lens_cpu`` when available to avoid a device sync, falling
        back to ``seq_lens``. Supports ``batch >= 1``.
        """
        req_indices = forward_batch.req_pool_indices.tolist()
        seq_lens_src = forward_batch.seq_lens_cpu
        if seq_lens_src is None:
            seq_lens_src = forward_batch.seq_lens
        seq_lens = seq_lens_src.tolist()
        return {int(r): int(s) for r, s in zip(req_indices, seq_lens)}

    @staticmethod
    def logical_position(req: Req) -> int:
        """Rotary position of a request's NEXT token.

        Equals the total number of tokens seen so far
        (``len(origin_input_ids) + len(output_ids)``), which is unaffected by
        KV eviction. Use this to override ``ForwardBatch.positions`` for
        R-KV-managed requests so rotary stays continuous after compaction, even
        though the physical seq_lens has shrunk.
        """
        return len(req.origin_input_ids) + len(req.output_ids)

    def take_pending_length_updates(self) -> Dict[int, Tuple[int, Req]]:
        """Return and clear the pending {req_pool_idx: (new_physical_len, req)} map.

        The scheduler calls this right after the forward pass to apply the new
        physical lengths to its batch-level seq_lens / seq_lens_cpu tensors
        (the request-level kv_committed_len / kv_allocated_len are already
        updated in-place during compaction). The owning Req is returned so the
        scheduler can drop an update whose slot was reused (identity mismatch).
        """
        updates = self.pending_length_updates
        self.pending_length_updates = {}
        return updates

    def override_decode_positions(self, forward_batch: ForwardBatch) -> None:
        """Replace decode positions with *logical* positions for R-KV requests.

        ``seq_lens`` now tracks the (possibly shorter) physical KV length, so
        ``clamp_position(seq_lens)`` would rewind rotary after compaction. For
        every R-KV-managed request in the batch we overwrite its position with
        ``logical_position(req)`` so future tokens keep absolute positions
        consistent with the retained keys. No-op for requests we don't manage.
        """
        if forward_batch.positions is None:
            return
        # One GPU->CPU sync for the whole batch (tolist) instead of a .item()
        # per request each decode step; collect the logical overrides and apply
        # them in a single batched scatter rather than an element write per req.
        req_indices = forward_batch.req_pool_indices.tolist()
        rows: List[int] = []
        values: List[int] = []
        for i, req_pool_idx in enumerate(req_indices):
            st = self.states.get(int(req_pool_idx))
            if st is not None and st.req is not None:
                # logical_position() counts all tokens seen so far INCLUDING the
                # token being decoded this step (it was appended to output_ids
                # when it was sampled), so the current token's 0-based rotary
                # position is that count minus one — for an un-compacted request
                # this equals the baseline clamp_position(seq_lens) = seq_lens-1.
                rows.append(i)
                values.append(self.logical_position(st.req) - 1)
        if rows:
            positions = forward_batch.positions
            idx = torch.tensor(rows, device=positions.device, dtype=torch.long)
            val = torch.tensor(values, device=positions.device, dtype=positions.dtype)
            positions[idx] = val


# ---------------------------------------------------------------------------
# Remaining wiring (NOT done here -- needs on-GPU / running-server validation;
# see R-KV/doc/DESIGN.md section 9 roadmap). Design = "scheme A": seq_lens tracks the
# PHYSICAL KV length, rotary positions stay LOGICAL.
#
#   1. FlashInferAttnBackend.forward_decode: after set_kv_buffer, call
#      compressor.collect_decode_query(q, layer, forward_batch) (in-graph).
#   2. model_runner (or the backend's end-of-forward hook): after the full
#      decode forward pass, call compressor.maybe_compact(forward_batch) (the
#      PREPARE phase — relocates surviving K/V, queues commit records). The
#      scheduler then, after the forward has completed, calls
#      compressor.commit_compactions() (the COMMIT phase — frees the evicted
#      tail slots off the forward stream) and applies take_pending_length_updates()
#      to batch.seq_lens / seq_lens_cpu (kv_committed_len / kv_allocated_len are
#      updated in the commit phase).
#   3. scheduler / schedule_batch: call on_request_begin / on_request_end around
#      a request's life; disable overlap scheduling for phase 1 (simpler timing).
#   4. ForwardBatch construction (forward_batch_info): for R-KV-managed requests
#      override positions with RKVCompressor.logical_position(req) - 1 instead of
#      clamp_position(seq_lens), so rotary stays continuous after eviction. There
#      is already an override slot on the decode path (~L802).
# ---------------------------------------------------------------------------
