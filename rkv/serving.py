"""Request-local serving interface for the R-KV compression algorithm.

This module owns serving configuration, query history, observation timing,
and retained-position decisions. The legacy R1KV algorithm remains separate.
"""

from collections.abc import Mapping
from typing import Any, Literal, Protocol

import torch

from .compression.r1_kv import R1KV


class KVView(Protocol):
    """LMCache-supplied read-only GPU KV view, without an LMCache import."""

    def get_keys(self) -> torch.Tensor: ...
    def get_values(self) -> torch.Tensor: ...


class RKVServing(R1KV):
    """R-KV algorithm with LMCache's per-request serving lifecycle."""

    def __init__(self, *, buffer: int = 128, **rkv_settings: Any) -> None:
        super().__init__(**rkv_settings)
        if buffer < self.window_size:
            raise ValueError("buffer must be >= window_size")
        self.buffer = buffer
        self._serving_query_history: dict[str, torch.Tensor] = {}
        self._serving_layer_order: tuple[str, ...] | None = None

    @classmethod
    def from_serving_config(cls, config: Mapping[str, Any]) -> "RKVServing":
        """Create a request-local algorithm using validated serving defaults."""
        return cls(**_validated_serving_config(config))

    def should_observe_token_queries(
        self, phase: Literal["prefill", "decode"], decoded_tokens_before_step: int
    ) -> int:
        """Return how many Q rows to capture from this forward."""
        if phase == "prefill":
            return self.window_size
        if phase != "decode":
            raise ValueError(f"Unknown token-drop phase: {phase!r}")
        if decoded_tokens_before_step < 0:
            raise ValueError("decoded_tokens_before_step must be nonnegative")
        return int(
            decoded_tokens_before_step % self.buffer >= self.buffer - self.window_size
        )

    def observe_token_queries(
        self, queries_by_layer: Mapping[str, torch.Tensor]
    ) -> None:
        """Store the most recent post-RoPE Q rows for this request."""
        if not queries_by_layer:
            raise ValueError("R-KV observation requires non-empty layer inputs")
        layer_order = tuple(queries_by_layer)
        if self._serving_layer_order is None:
            self._serving_layer_order = layer_order
        elif layer_order != self._serving_layer_order:
            raise RuntimeError("R-KV serving layer order changed")

        for layer_name, query_rows in queries_by_layer.items():
            if query_rows.ndim != 3 or query_rows.shape[0] == 0:
                raise ValueError("R-KV observation expects [tokens, q_heads, head_dim]")
            recent_query_rows = query_rows[-self.window_size :].detach()
            previous_query_rows = self._serving_query_history.get(layer_name)
            if previous_query_rows is not None:
                recent_query_rows = torch.cat(
                    [previous_query_rows, recent_query_rows], dim=0
                )[-self.window_size :]
            self._serving_query_history[layer_name] = recent_query_rows

    def should_compact_kv(
        self,
        phase: Literal["prefill", "decode"],
        resident_kv_tokens: int,
        decoded_tokens_before_step: int,
    ) -> bool:
        """Run end-of-prefill or buffer-boundary R-KV compaction."""
        if phase == "prefill":
            return resident_kv_tokens > self.budget
        if phase != "decode":
            raise ValueError(f"Unknown token-drop phase: {phase!r}")
        return (
            decoded_tokens_before_step >= 0
            and (decoded_tokens_before_step + 1) % self.buffer == 0
            and resident_kv_tokens >= self.budget + self.buffer
        )

    def select_kept_token_positions(
        self, kv_by_layer: Mapping[str, KVView]
    ) -> Mapping[str, torch.Tensor]:
        """Return independent, algorithm-ordered positions per layer/KV head."""
        if not kv_by_layer:
            raise ValueError("R-KV selection requires non-empty KV views")
        layer_order = tuple(kv_by_layer)
        if layer_order != self._serving_layer_order:
            raise RuntimeError("R-KV serving layer order changed")

        kept_positions_by_layer = {}
        for layer_name, kv_view in kv_by_layer.items():
            observed_queries = self._serving_query_history.get(layer_name)
            if observed_queries is None or observed_queries.shape[0] < self.window_size:
                raise RuntimeError("R-KV does not have a full query window")
            resident_keys = kv_view.get_keys()
            if resident_keys.ndim != 4 or resident_keys.shape[0] != 1:
                raise ValueError("R-KV KVView keys must be [1, kv_heads, tokens, dim]")
            if resident_keys.shape[2] < self.budget:
                raise ValueError("R-KV cannot compact fewer than budget tokens")
            query_window = (
                observed_queries[-self.window_size :].permute(1, 0, 2).unsqueeze(0)
            )
            per_head_scores = self.score_kv(query_window, resident_keys)
            kept_past_positions = per_head_scores.topk(
                self.budget - self.window_size, dim=-1
            ).indices
            recent_positions = (
                torch.arange(
                    resident_keys.shape[2] - self.window_size,
                    resident_keys.shape[2],
                    device=resident_keys.device,
                    dtype=torch.long,
                )
                .view(1, 1, -1)
                .expand(1, resident_keys.shape[1], -1)
            )
            kept_positions_by_layer[layer_name] = torch.cat(
                [kept_past_positions, recent_positions], dim=-1
            )[0]

        self._serving_query_history.clear()
        return kept_positions_by_layer


def _validated_serving_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve R-KV serving defaults and validate algorithm-specific settings."""
    if not isinstance(config, Mapping):
        raise ValueError("R-KV serving config must be a mapping")

    allowed_keys = {
        "budget",
        "buffer",
        "window_size",
        "kernel_size",
        "mix_lambda",
        "retain_ratio",
        "retain_direction",
    }
    unknown_keys = set(config) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f"Unsupported R-KV serving config keys: {sorted(unknown_keys)}"
        )

    resolved_config = {
        "budget": 128,
        "buffer": 128,
        "window_size": 8,
        "kernel_size": 7,
        "mix_lambda": 0.1,
        "retain_ratio": 0.1,
        "retain_direction": "last",
    }
    resolved_config.update(config)

    budget = resolved_config["budget"]
    buffer = resolved_config["buffer"]
    window_size = resolved_config["window_size"]
    kernel_size = resolved_config["kernel_size"]
    mix_lambda = resolved_config["mix_lambda"]
    retain_ratio = resolved_config["retain_ratio"]
    retain_direction = resolved_config["retain_direction"]

    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not isinstance(buffer, int) or isinstance(buffer, bool) or buffer <= 0:
        raise ValueError("buffer must be a positive integer")
    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size <= 0
    ):
        raise ValueError("window_size must be a positive integer")
    if budget <= window_size:
        raise ValueError("budget must be greater than window_size")
    if buffer < window_size:
        raise ValueError("buffer must be >= window_size")
    if (
        not isinstance(kernel_size, int)
        or isinstance(kernel_size, bool)
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be a positive odd integer")
    if not isinstance(mix_lambda, (int, float)) or isinstance(mix_lambda, bool):
        raise ValueError("mix_lambda must be numeric")
    if not 0.0 <= float(mix_lambda) <= 1.0:
        raise ValueError("mix_lambda must be in [0, 1]")
    if not isinstance(retain_ratio, (int, float)) or isinstance(retain_ratio, bool):
        raise ValueError("retain_ratio must be numeric")
    if not 0.0 < float(retain_ratio) <= 1.0:
        raise ValueError("retain_ratio must be in (0, 1]")
    if retain_direction not in ("last", "first", "last_percent", "first_percent"):
        raise ValueError("Unsupported retain_direction")

    return dict(
        budget=budget,
        buffer=buffer,
        window_size=window_size,
        kernel_size=kernel_size,
        mix_lambda=float(mix_lambda),
        retain_ratio=float(retain_ratio),
        retain_direction=retain_direction,
    )
