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
    (temporal order along the window axis, newest last) plus per-request
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

    def reset(self) -> None:
        """Clear all per-request state for a new ``generate()`` call."""
        self.window.zero_()
        self.window_len = [0] * self.max_batch_size
        self.num_compactions = [0] * self.max_batch_size

    def seed_window(self, layer: int, request: int, queries: torch.Tensor) -> None:
        """Seed the window with the trailing prompt queries (HF behavior).

        ``queries``: ``[n, num_q_heads, head_dim]`` post-RoPE, temporal order,
        with ``n <= window_size`` (the last ``window_size`` prompt queries, or
        fewer for very short prompts).
        """
        window_size = self.config.window_size
        n = min(queries.shape[0], window_size)
        self.window[layer, request, :, window_size - n :, :] = queries[-n:].transpose(
            0, 1
        )
        self.window_len[request] = n

    def begin_step(self, requests: Sequence[int]) -> None:
        """Advance window-fill counters; call once per decode step."""
        window_size = self.config.window_size
        for r in requests:
            self.window_len[r] = min(self.window_len[r] + 1, window_size)

    def push_queries(
        self, layer: int, rows: torch.Tensor, queries: torch.Tensor
    ) -> None:
        """Push one decode step's post-RoPE queries for ``layer``.

        ``rows``: ``[n]`` long tensor of request rows; ``queries``:
        ``[n, num_q_heads, head_dim]``, one query per row.
        """
        window = self.window[layer, rows]
        self.window[layer, rows] = torch.cat(
            [window[:, :, 1:], queries.unsqueeze(2)], dim=2
        )

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

        window_queries = self.window[layer, request].unsqueeze(0)
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
