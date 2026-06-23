from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    # R-KV decode-time KV cache compression (see minisgl.compress).
    # Defaults match the HuggingFace reference. The attention layer
    # reads ``rkv_config`` to instantiate an ``RKVCompressor``; see
    # ``docs/RKV.md`` for the wiring contract.
    rkv_enabled: bool = False
    rkv_budget: int = 1024
    rkv_buffer: int = 128
    rkv_window_size: int = 8
    rkv_kernel_size: int = 7
    rkv_mix_lambda: float = 0.07
    rkv_retain_ratio: float = 0.1
    rkv_retain_direction: str = "last"

    @cached_property
    def rkv_config(self):
        from minisgl.compress import RKVConfig

        return RKVConfig(
            enabled=self.rkv_enabled,
            budget=self.rkv_budget,
            buffer=self.rkv_buffer,
            window_size=self.rkv_window_size,
            kernel_size=self.rkv_kernel_size,
            mix_lambda=self.rkv_mix_lambda,
            retain_ratio=self.rkv_retain_ratio,
            retain_direction=self.rkv_retain_direction,
        )

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
