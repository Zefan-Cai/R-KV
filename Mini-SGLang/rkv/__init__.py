"""R-KV decode-time KV cache compression for Mini-SGLang.

This package ports the R-KV algorithm (https://arxiv.org/abs/2505.24133)
into the Mini-SGLang attention path. The pure-Python algorithm itself
lives in :mod:`minisgl.rkv.algo`, and the per-sequence integration
helper that drives it against a paged KV cache lives in
:mod:`minisgl.rkv.integration`.
"""

from .integration import RKVCompressor, RKVConfig
from .algo import R1KV, cal_similarity, compute_attention_scores

__all__ = [
    "R1KV",
    "RKVCompressor",
    "RKVConfig",
    "cal_similarity",
    "compute_attention_scores",
]
