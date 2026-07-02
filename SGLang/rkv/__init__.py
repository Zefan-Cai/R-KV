"""R-KV: Redundancy-aware KV Cache Compression for reasoning models.

Decoding-time KV cache compression ported from the reference HuggingFace
implementation (``rkv/compression/r1_kv.py`` in the R-KV repository) onto
SGLang v0.5.14.

This package is split into two layers:

* ``algo`` -- the device-agnostic pure algorithm (a faithful port of the
  reference ``R1KV``). It has no SGLang dependencies and can be unit-tested
  on CPU against the original.
* (integration layer -- added later) wires the algorithm into SGLang's paged
  KV pool and the FlashInfer attention backend.
"""

from sglang.srt.mem_cache.rkv.algo import (
    R1KV,
    cal_similarity,
    compute_attention_scores,
)

__all__ = ["R1KV", "cal_similarity", "compute_attention_scores"]
