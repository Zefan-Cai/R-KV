import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    rkv_enabled: bool = False
    rkv_budget: int = 1024
    rkv_buffer: int = 128
    rkv_window_size: int = 8
    rkv_kernel_size: int = 7
    rkv_mix_lambda: float = 0.07
    rkv_retain_ratio: float = 0.1
    rkv_retain_direction: str = "last"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.rkv_budget > self.rkv_window_size
        assert self.rkv_buffer >= 1
        if self.rkv_enabled:
            self.enforce_eager = True
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
