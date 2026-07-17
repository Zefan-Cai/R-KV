"""R-KV: Redundancy-aware KV Cache Compression for reasoning models.

Decoding-time KV cache compression ported onto vLLM v1 (v0.25.1), following the
same patch-style layout as the SGLang R-KV port.

This package is split into two layers:

* ``algo`` -- the device-agnostic pure algorithm (a faithful port of the
  reference ``R1KV``). It has no vLLM dependencies and can be unit-tested on
  CPU against the original.
* ``integration`` -- wires the algorithm into vLLM's paged KV cache and the
  FlashAttention v1 backend (physical eviction, logical/physical position
  bookkeeping).
"""

from vllm.rkv.algo import R1KV, cal_similarity, compute_attention_scores
from vllm.rkv.integration import RKVCompressor, RKVConfig

__all__ = [
    "R1KV",
    "cal_similarity",
    "compute_attention_scores",
    "RKVConfig",
    "RKVCompressor",
]
