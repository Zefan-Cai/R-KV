"""Serving integration for R-KV, separate from the legacy compression algorithm."""

from .compression.r1_kv import R1KV


class RKVServing(R1KV):
    """Request-local R-KV policy; serving methods are added separately."""
