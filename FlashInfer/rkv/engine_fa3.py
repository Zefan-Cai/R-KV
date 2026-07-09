"""FlashAttention-3 attention line for the R-KV engine (H100 comparison).

Same engine, KV pool, compressor, sampling, and non-attention kernels as
:class:`FlashInferEngine` — only the two attention calls swap to FA3:

* prefill: ``flash_attn_varlen_func`` over the ragged prompt;
* decode: ``flash_attn_with_kvcache`` over per-request regions — the paged
  pool with ``page_size=1`` and fixed contiguous regions *is* FA3's
  ``[batch, seqlen, heads, head_dim]`` cache layout when viewed per layer,
  so no data movement is needed.

Everything else (RoPE, norms, silu_and_mul, top-p sampling, compaction) is
shared code, which isolates the attention kernel for a clean
FlashInfer-vs-FA3 A/B. Requires ``flash_attn_interface`` (FlashAttention-3,
built from ``flash-attention/hopper``, SM90).
"""

from __future__ import annotations

import inspect

import torch

from .engine import FlashInferEngine


def _first(result):
    """FA3 entry points return ``out`` or ``(out, softmax_lse)`` depending on
    version; normalize to ``out``."""
    return result[0] if isinstance(result, tuple) else result


class FA3Engine(FlashInferEngine):
    """FlashInferEngine with FA3 prefill/decode attention kernels."""

    def __init__(self, *args, **kwargs) -> None:
        try:
            import flash_attn_interface
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "FA3Engine requires flash_attn_interface (FlashAttention-3, "
                "built from flash-attention/hopper on SM90)"
            ) from exc
        # Fail before loading model weights / allocating the KV pool when the
        # optional backend is missing or exposes the wrong API.
        required = ("flash_attn_varlen_func", "flash_attn_with_kvcache")
        missing = [name for name in required if not hasattr(flash_attn_interface, name)]
        if missing:  # pragma: no cover - env-dependent
            raise ImportError(
                "FA3Engine requires the Hopper flash_attn_interface API; "
                f"missing: {', '.join(missing)}"
            )
        self._fa3 = flash_attn_interface
        self._fa3_has_batch_idx = "cache_batch_idx" in inspect.signature(
            flash_attn_interface.flash_attn_with_kvcache
        ).parameters
        super().__init__(*args, **kwargs)
        # Full-batch cache_seqlens for the padded fallback path; inactive
        # rows use length 1 so the kernel reads one stale slot and the
        # discarded output stays finite.
        self._fa3_pad_seqlens = torch.ones(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

    # ------------------------------------------------------------------
    # Prefill: ragged varlen attention
    # ------------------------------------------------------------------
    def _plan_prefill(self, indptr: list[int], prompt_lens: list[int]) -> None:
        self._fa3_cu_seqlens = torch.tensor(
            indptr, dtype=torch.int32, device=self.device
        )
        self._fa3_max_seqlen = max(prompt_lens)

    def _run_prefill_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return _first(
            self._fa3.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=self._fa3_cu_seqlens,
                cu_seqlens_k=self._fa3_cu_seqlens,
                max_seqlen_q=self._fa3_max_seqlen,
                max_seqlen_k=self._fa3_max_seqlen,
                causal=True,
            )
        )

    # ------------------------------------------------------------------
    # Decode: kv-cache attention over contiguous regions
    # ------------------------------------------------------------------
    def _plan_decode(self, active: list[int], lens: list[int]) -> None:
        self._fa3_all_active = active == list(range(self.max_batch_size))
        if self._fa3_all_active:
            self._fa3_seqlens = torch.tensor(
                lens, dtype=torch.int32, device=self.device
            )
        elif self._fa3_has_batch_idx:
            packed = torch.tensor([lens, active], dtype=torch.int32).to(
                self.device, non_blocking=True
            )
            self._fa3_seqlens, self._fa3_batch_idx = packed.unbind(0)
        else:
            # Padded fallback: full-batch q with length-1 dummies for
            # inactive rows; outputs for those rows are discarded.
            seqlens = self._fa3_pad_seqlens.clone()
            seqlens[torch.tensor(active, device=self.device)] = torch.tensor(
                lens, dtype=torch.int32, device=self.device
            )
            self._fa3_seqlens = seqlens
            self._fa3_active_rows = torch.tensor(
                active, dtype=torch.long, device=self.device
            )

    def _layer_cache(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        pool = self.kv_pool[layer]  # [slots, 2, 1, kv_heads, head_dim]
        shape = (
            self.max_batch_size,
            self.region_len,
            self.model.num_kv_heads,
            self.model.head_dim,
        )
        return pool[:, 0, 0].view(shape), pool[:, 1, 0].view(shape)

    def _run_decode_attention(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        k_cache, v_cache = self._layer_cache(layer)
        if self._fa3_all_active:
            out = self._fa3.flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=self._fa3_seqlens,
                causal=True,
            )
            return _first(out).squeeze(1)
        if self._fa3_has_batch_idx:
            out = self._fa3.flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=self._fa3_seqlens,
                cache_batch_idx=self._fa3_batch_idx,
                causal=True,
            )
            return _first(out).squeeze(1)
        q_full = torch.zeros(
            self.max_batch_size,
            1,
            q.shape[1],
            q.shape[2],
            dtype=q.dtype,
            device=q.device,
        )
        q_full[self._fa3_active_rows, 0] = q
        out = _first(
            self._fa3.flash_attn_with_kvcache(
                q_full,
                k_cache,
                v_cache,
                cache_seqlens=self._fa3_seqlens,
                causal=True,
            )
        )
        return out[self._fa3_active_rows, 0]
