"""Serving integration for R-KV, separate from the legacy compression algorithm."""

from collections.abc import Mapping
from typing import Any, Literal

import torch

from .compression.r1_kv import R1KV


class RKVServing(R1KV):
    """Request-local R-KV policy; serving methods are added separately."""

    @classmethod
    def from_serving_config(cls, config: Mapping[str, Any]) -> "RKVServing":
        """Create an R-KV instance using serving defaults and validated settings."""
        if not isinstance(config, Mapping):
            raise ValueError("R-KV serving config must be a mapping")

        # These are opt-in serving defaults; R1KV's legacy defaults are unchanged.
        resolved_config = {
            "budget": 128,
            "buffer": 128,
            "window_size": 8,
            "kernel_size": 7,
            "mix_lambda": 0.1,
            "retain_ratio": 0.1,
            "retain_direction": "last",
        }
        unknown_keys = set(config) - resolved_config.keys()
        if unknown_keys:
            raise ValueError(
                f"Unsupported R-KV serving config keys: {sorted(unknown_keys)}"
            )
        resolved_config.update(config)

        budget = resolved_config["budget"]
        buffer = resolved_config["buffer"]
        window_size = resolved_config["window_size"]
        kernel_size = resolved_config["kernel_size"]
        mix_lambda = resolved_config["mix_lambda"]
        retain_ratio = resolved_config["retain_ratio"]
        retain_direction = resolved_config["retain_direction"]

        if type(budget) is not int or budget <= 0:
            raise ValueError("budget must be a positive integer")
        if type(buffer) is not int or buffer <= 0:
            raise ValueError("buffer must be a positive integer")
        if type(window_size) is not int or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if budget <= window_size:
            raise ValueError("budget must be greater than window_size")
        if buffer < window_size:
            raise ValueError("buffer must be >= window_size")
        if type(kernel_size) is not int or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if type(mix_lambda) not in (int, float):
            raise ValueError("mix_lambda must be numeric")
        if not 0.0 <= mix_lambda <= 1.0:
            raise ValueError("mix_lambda must be in [0, 1]")
        if type(retain_ratio) not in (int, float):
            raise ValueError("retain_ratio must be numeric")
        if not 0.0 < retain_ratio <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1]")
        if retain_direction not in ("last", "first"):
            raise ValueError("Unsupported retain_direction")

        instance = cls(
            budget=budget,
            window_size=window_size,
            kernel_size=kernel_size,
            mix_lambda=float(mix_lambda),
            retain_ratio=float(retain_ratio),
            retain_direction=retain_direction,
        )
        instance.buffer = buffer
        instance._serving_query_history = {}
        instance._serving_layer_order = None
        return instance

    def should_observe_token_queries(
        self, phase: Literal["prefill", "decode"], decoded_tokens_before_step: int
    ) -> int:
        """Capture the last prefill queries or the final steps of each decode buffer."""
        if phase == "prefill":
            return self.window_size
        step_index_in_buffer = decoded_tokens_before_step % self.buffer
        if step_index_in_buffer >= self.buffer - self.window_size:
            return 1
        return 0

    def observe_token_queries(
        self, queries_by_layer: Mapping[str, torch.Tensor]
    ) -> None:
        """Keep the last window of post-RoPE Q rows per layer."""
        if self._serving_layer_order is None:
            self._serving_layer_order = tuple(queries_by_layer)

        for layer, queries in queries_by_layer.items():
            recent_queries = queries[-self.window_size :].detach()
            previous_queries = self._serving_query_history.get(layer)
            if previous_queries is None:
                recent_queries = recent_queries.clone()
            else:
                recent_queries = torch.cat(
                    (previous_queries, recent_queries), dim=0
                )[-self.window_size :]
            self._serving_query_history[layer] = recent_queries

    def should_compact_kv(
        self,
        phase: Literal["prefill", "decode"],
        resident_kv_tokens: int,
        decoded_tokens_before_step: int,
    ) -> bool:
        """Compact after prefill or at a full decode buffer boundary."""
        if phase == "prefill":
            return resident_kv_tokens > self.budget

        # When: only at the end of each decode buffer.
        if (decoded_tokens_before_step + 1) % self.buffer != 0:
            return False

        # How much: only after enough KV has accumulated.
        return resident_kv_tokens >= self.budget + self.buffer

    def select_kept_token_positions(
        self, kv_by_layer: Mapping[str, Any]
    ) -> Mapping[str, torch.Tensor]:
        """Select per-head positions from views exposing get_keys()."""
        kept_by_layer = {}
        for layer, view in kv_by_layer.items():
            queries = self._serving_query_history[layer]
            keys = view.get_keys()
            query_window = queries.permute(1, 0, 2).unsqueeze(0)
            scores = self.score_kv(query_window, keys)
            past = scores.topk(self.budget - self.window_size, dim=-1).indices[0]
            recent = torch.arange(
                keys.shape[2] - self.window_size, keys.shape[2], device=keys.device
            ).expand(keys.shape[1], -1)
            kept_by_layer[layer] = torch.cat((past, recent), dim=-1)

        self._serving_query_history.clear()
        return kept_by_layer
