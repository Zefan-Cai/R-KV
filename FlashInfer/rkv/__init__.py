"""R-KV decode-time KV cache compression on FlashInfer.

Config and the pure-torch algorithm import eagerly so CPU CI runs without
flashinfer installed; the engine and compaction driver load lazily on first
attribute access.
"""

from importlib import import_module

from .algo import (
    cal_similarity,
    compute_attention_scores,
    compute_scores,
    select_indices,
    update_kv,
)
from .config import GenOutput, RKVConfig

__all__ = [
    "FA3Engine",
    "FlashInferEngine",
    "GenOutput",
    "R1KV",
    "RKVConfig",
    "cal_similarity",
    "compute_attention_scores",
    "compute_scores",
    "select_indices",
    "update_kv",
]

_LAZY_ATTRS = {
    "FA3Engine": "engine_fa3",
    "FlashInferEngine": "engine",
    "R1KV": "compressor",
}


def __getattr__(name: str):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module_name}", __name__), name)
