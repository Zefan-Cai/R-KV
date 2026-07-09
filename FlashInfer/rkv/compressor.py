"""R-KV compaction driver for the FlashInfer engine (pure torch, CPU-testable).

Per-request observation-window and trigger bookkeeping follows the shape of
``SGLang/rkv/integration.py``; the scoring/selection math lives in ``.algo``.
Window-query seeding from the prompt follows ``HuggingFace/rkv/modeling.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from . import algo
from .config import RKVConfig


class R1KV:
    """Drives per-layer, per-kv-head R-KV compaction over a static batch.

    Holds the rolling post-RoPE window-query cache with shape
    ``[num_layers, max_batch_size, num_q_heads, window_size, head_dim]``
    as a circular buffer: each decode step overwrites one slot
    (``steps % window_size``, shared by all rows of the static batch) instead
    of shifting the whole window, and :meth:`compact` rotates the slots back
    into temporal order (newest last) before scoring. Also tracks per-request
    trigger state and compaction counters. The engine owns the paged KV pool;
    this class only transforms K/V tensors handed to :meth:`compact`.
    """

    def __init__(
        self,
        config: RKVConfig,
        *,
        num_layers: int,
        max_batch_size: int,
        num_q_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.window = torch.zeros(
            (num_layers, max_batch_size, num_q_heads, config.window_size, head_dim),
            device=device,
            dtype=dtype,
        )
        self.window_len: list[int] = [0] * max_batch_size
        self.num_compactions: list[int] = [0] * max_batch_size
        # Decode steps begun since reset(); rows of a static batch all push on
        # every step they are active, so one global write cursor is exact.
        self._steps = 0
        self._write_slot = 0
        # compact_batch gate: None = not yet A/B-checked on this device,
        # True = batched update_kv proved bit-identical to per-pair calls,
        # False = it differed, per-pair fallback forever. Device-level fact,
        # deliberately NOT cleared by reset().
        self._batched_ok: bool | None = None

    def reset(self) -> None:
        """Clear all per-request state for a new ``generate()`` call."""
        self.window.zero_()
        self.window_len = [0] * self.max_batch_size
        self.num_compactions = [0] * self.max_batch_size
        self._steps = 0
        self._write_slot = 0

    def seed_window(self, layer: int, request: int, queries: torch.Tensor) -> None:
        """Seed the window with the trailing prompt queries (HF behavior).

        ``queries``: ``[n, num_q_heads, head_dim]`` post-RoPE, temporal order,
        with ``n <= window_size`` (the last ``window_size`` prompt queries, or
        fewer for very short prompts). Only valid before the first decode step
        (the circular layout assumes seeds occupy the trailing slots).
        """
        if self._steps:
            raise RuntimeError("seed_window after decode began; call reset() first")
        window_size = self.config.window_size
        n = min(queries.shape[0], window_size)
        self.window[layer, request, :, window_size - n :, :] = queries[-n:].transpose(
            0, 1
        )
        self.window_len[request] = n

    def begin_step(self, requests: Sequence[int]) -> None:
        """Advance the write cursor and window-fill counters; once per step."""
        window_size = self.config.window_size
        self._write_slot = self._steps % window_size
        self._steps += 1
        for r in requests:
            self.window_len[r] = min(self.window_len[r] + 1, window_size)

    def push_queries(
        self, layer: int, rows: torch.Tensor, queries: torch.Tensor
    ) -> None:
        """Push one decode step's post-RoPE queries for ``layer``.

        ``rows``: ``[n]`` long tensor of request rows; ``queries``:
        ``[n, num_q_heads, head_dim]``, one query per row. Overwrites the
        step's circular slot in place (no shift).
        """
        self.window[layer, rows, :, self._write_slot] = queries

    def should_compact(self, phys_len: int) -> bool:
        """Trigger check, applied per request after each decode append."""
        return phys_len == self.config.budget + self.config.buffer

    def compact(
        self, request: int, layer: int, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress one (request, layer)'s KV region via ``algo.update_kv``.

        ``keys`` / ``values``: ``[seq_len, num_kv_heads, head_dim]`` in slot
        order (post-RoPE keys). Returns the kept
        ``([budget, num_kv_heads, head_dim], [budget, num_kv_heads, head_dim])``
        with per-kv-head token selection (slot i may hold different logical
        tokens per head; only the count is shared).
        """
        cfg = self.config
        if self.window_len[request] < cfg.window_size:
            raise RuntimeError(
                f"request {request}: observation window not full "
                f"({self.window_len[request]}/{cfg.window_size}); with "
                "buffer >= window_size this should be unreachable"
            )

        # Rotate the circular slots back into temporal order (oldest first).
        # After `_steps` pushes the oldest live slot is `_steps % window_size`;
        # a pure permutation, so scoring stays bit-identical to a shifted
        # window holding the same queries.
        window_queries = self.window[layer, request]
        start = self._steps % cfg.window_size
        if start:
            window_queries = torch.roll(window_queries, -start, dims=1)
        window_queries = window_queries.unsqueeze(0)
        k, v = algo.update_kv(
            keys.transpose(0, 1).unsqueeze(0),
            window_queries,
            values.transpose(0, 1).unsqueeze(0),
            budget=cfg.budget,
            window_size=cfg.window_size,
            kernel_size=cfg.kernel_size,
            mix_lambda=cfg.mix_lambda,
            retain_ratio=cfg.retain_ratio,
            retain_direction=cfg.retain_direction,
        )
        # One request-level compaction spans all layers; count it once.
        if layer == 0:
            self.num_compactions[request] += 1
        return k.squeeze(0).transpose(0, 1), v.squeeze(0).transpose(0, 1)

    # compact_batch sizes its chunks so the transient [pairs, kv_heads, seq,
    # seq] scoring buffers (cosine matrix + bool mask + int32 indices, ~7B or
    # ~9B per element depending on dtype) stay under this budget — peak memory
    # is a headline metric for R-KV, so the launch amortization must not eat
    # the pool savings. 32 pairs is the cap even when the budget allows more.
    _BATCH_CHUNK_BYTES = 512 << 20
    _BATCH_CHUNK_MAX_PAIRS = 32

    def compact_batch(
        self, requests: Sequence[int], keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress every layer of ``requests`` in batched ``update_kv`` calls.

        ``keys`` / ``values``: ``[num_layers, len(requests), seq_len,
        num_kv_heads, head_dim]`` in slot order — rows compacting on the same
        step always share ``seq_len`` because the trigger is an equality test.
        Returns kept ``(k, v)`` shaped ``[num_layers, len(requests), budget,
        num_kv_heads, head_dim]``.

        Batched GEMMs are not provably bit-identical to per-pair bsz=1 calls,
        so the first call A/B-checks the batched result against the per-pair
        reference on the same data (``torch.equal``) and permanently falls
        back to the per-pair path if they differ.
        """
        cfg = self.config
        rows = list(requests)
        for r in rows:
            if self.window_len[r] < cfg.window_size:
                raise RuntimeError(
                    f"request {r}: observation window not full "
                    f"({self.window_len[r]}/{cfg.window_size}); with "
                    "buffer >= window_size this should be unreachable"
                )
        num_layers, num_rows, seq_len = keys.shape[0], keys.shape[1], keys.shape[2]
        pairs = num_layers * num_rows

        # [L, R, H_q, W, D] -> temporal order -> [L*R, H_q, W, D]
        window = self.window[:, rows]
        start = self._steps % cfg.window_size
        if start:
            window = torch.roll(window, -start, dims=3)
        window = window.reshape(pairs, *window.shape[2:])

        # [L, R, S, H_kv, D] -> [L*R, H_kv, S, D] (strided views, same layout
        # the per-layer path fed update_kv via transpose)
        k_pairs = keys.reshape(pairs, seq_len, -1, keys.shape[-1]).transpose(1, 2)
        v_pairs = values.reshape(pairs, seq_len, -1, values.shape[-1]).transpose(1, 2)

        params = dict(
            budget=cfg.budget,
            window_size=cfg.window_size,
            kernel_size=cfg.kernel_size,
            mix_lambda=cfg.mix_lambda,
            retain_ratio=cfg.retain_ratio,
            retain_direction=cfg.retain_direction,
        )

        def run(chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
            ks, vs = [], []
            for c in range(0, pairs, chunk):
                k_c, v_c = algo.update_kv(
                    k_pairs[c : c + chunk],
                    window[c : c + chunk],
                    v_pairs[c : c + chunk],
                    **params,
                )
                ks.append(k_c)
                vs.append(v_c)
            return torch.cat(ks), torch.cat(vs)

        # Chunk size from the scoring-transient budget: cosine matrix in the
        # key dtype (x2 counting the softmax read/write) + bool mask + int32
        # indices per element of [kv_heads, seq, seq].
        kv_heads = keys.shape[3]
        per_pair = (2 * keys.element_size() + 1 + 4) * kv_heads * seq_len * seq_len
        chunk_pairs = max(
            1, min(self._BATCH_CHUNK_MAX_PAIRS, self._BATCH_CHUNK_BYTES // per_pair)
        )

        if self._batched_ok is None:
            kept_k, kept_v = run(1)  # per-pair reference on the real data
            batched_k, batched_v = run(chunk_pairs)
            self._batched_ok = torch.equal(batched_k, kept_k) and torch.equal(
                batched_v, kept_v
            )
        else:
            kept_k, kept_v = run(chunk_pairs if self._batched_ok else 1)

        for r in rows:
            self.num_compactions[r] += 1
        # [L*R, H_kv, budget, D] -> [L, R, budget, H_kv, D]
        kept_k = kept_k.transpose(1, 2).reshape(
            num_layers, num_rows, cfg.budget, -1, kept_k.shape[-1]
        )
        kept_v = kept_v.transpose(1, 2).reshape(
            num_layers, num_rows, cfg.budget, -1, kept_v.shape[-1]
        )
        return kept_k, kept_v
