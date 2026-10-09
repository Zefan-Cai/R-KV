"""Serving integration for R-KV, separate from the legacy compression algorithm."""

from collections.abc import Mapping
from typing import Any

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
        if not isinstance(mix_lambda, (int, float)) or isinstance(mix_lambda, bool):
            raise ValueError("mix_lambda must be numeric")
        if not 0.0 <= float(mix_lambda) <= 1.0:
            raise ValueError("mix_lambda must be in [0, 1]")
        if not isinstance(retain_ratio, (int, float)) or isinstance(
            retain_ratio, bool
        ):
            raise ValueError("retain_ratio must be numeric")
        if not 0.0 < float(retain_ratio) <= 1.0:
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
