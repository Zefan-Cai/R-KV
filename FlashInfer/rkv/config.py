"""R-KV configuration and public result types (importable without flashinfer)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RKVConfig:
    """Hyper-parameters for R-KV decode-time KV cache compression.

    ``mix_lambda=0.1`` follows the CLI convention of the reference eval
    scripts; the reference class default is ``0.07`` (same note as the SGLang
    backend). ``mix_lambda=1.0`` gives attention-only (SnapKV-style) scoring.
    The similarity threshold stays hardcoded at 0.5 inside the algorithm.
    """

    budget: int = 1024
    buffer: int = 128
    window_size: int = 8
    kernel_size: int = 7
    mix_lambda: float = 0.1
    retain_ratio: float = 0.1
    retain_direction: str = "last"
    compress_prefill: bool = True

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.budget <= self.window_size:
            raise ValueError("budget must be greater than window_size")
        # buffer >= window_size guarantees the observation window holds only
        # real queries by the time the first decode compaction fires.
        if self.buffer < self.window_size:
            raise ValueError("buffer must be >= window_size")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 < self.mix_lambda <= 1.0:
            raise ValueError("mix_lambda must be in (0, 1]")
        if self.retain_direction not in ("last", "first"):
            raise ValueError("retain_direction must be 'last' or 'first'")


@dataclass
class GenOutput:
    """Per-request result of ``FlashInferEngine.generate``.

    Lives here (not in engine.py) so CPU CI can import it without flashinfer.
    """

    token_ids: list[int]
    num_prefill_tokens: int
    num_output_tokens: int
    num_compactions: int
    finish_reason: str  # "stop" | "length"
