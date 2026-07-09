"""Llama / Qwen2 / Qwen3 decoder-only models on FlashInfer kernels.

Module shape follows Nano-vLLM's model code; per-family quirks (Qwen2 qkv
bias, Qwen3 per-head q/k rmsnorm and config ``head_dim``) follow the
HuggingFace reference implementations. Attention itself is injected by the
engine as a per-layer callable so the same forward serves both the ragged
prefill path and the paged decode path, and the engine can append KV and
collect window queries per layer.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from flashinfer.activation import silu_and_mul
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

# attention_fn(layer_idx, q, k, v) -> o, all shaped [num_tokens, heads, head_dim]
AttentionFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _family_quirks(config) -> tuple[bool, bool, bool]:
    """(qkv_bias, o_bias, qk_norm) from ``config.architectures[0]``."""
    arch = config.architectures[0]
    attention_bias = bool(getattr(config, "attention_bias", False))
    if arch == "LlamaForCausalLM":
        return attention_bias, attention_bias, False
    if arch == "Qwen2ForCausalLM":
        return True, False, False
    if arch == "Qwen3ForCausalLM":
        return attention_bias, attention_bias, True
    raise ValueError(f"unsupported architecture: {arch!r}")


def _llama3_scaled_inv_freq(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    original_max_pos = scaling["original_max_position_embeddings"]

    low_freq_wavelen = original_max_pos / low_freq_factor
    high_freq_wavelen = original_max_pos / high_freq_factor
    wavelen = 2 * torch.pi / inv_freq
    smooth = (original_max_pos / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * inv_freq / factor + smooth * inv_freq
    return torch.where(
        wavelen > low_freq_wavelen,
        inv_freq / factor,
        torch.where(wavelen < high_freq_wavelen, inv_freq, smoothed),
    )


def _rope_parameters(config) -> dict:
    """RoPE parameters as one flat dict across transformers versions.

    transformers >= 5 deletes the top-level ``rope_theta`` attribute and
    standardizes theta plus scaling fields into ``config.rope_parameters``
    (always holding ``rope_theta`` and ``rope_type``); reading the legacy
    attributes there silently yields the 10000.0 default. On 4.x the same
    shape is rebuilt from ``rope_theta`` / ``rope_scaling``.
    """
    params = getattr(config, "rope_parameters", None)
    if params is not None:
        params = dict(params)
    else:
        params = dict(getattr(config, "rope_scaling", None) or {})
        params.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
    params.setdefault("rope_type", params.get("type", "default"))
    return params


def _build_cos_sin_cache(
    config, head_dim: int, max_position: int, device: torch.device
) -> torch.Tensor:
    """FlashInfer ``cos_sin_cache``: fp32 ``[max_position, head_dim]``, cos‖sin."""
    params = _rope_parameters(config)
    if float(params.get("partial_rotary_factor") or 1.0) != 1.0:
        raise NotImplementedError("partial_rotary_factor != 1.0 not supported")
    theta = float(params["rope_theta"])
    inv_freq = 1.0 / theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    )
    rope_type = params["rope_type"]
    if rope_type == "llama3":
        inv_freq = _llama3_scaled_inv_freq(inv_freq, params)
    elif rope_type != "default":
        raise NotImplementedError(f"rope_type {rope_type!r} not supported")
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


class Linear(nn.Module):
    """Bare ``F.linear`` over pre-allocated (uninitialized) parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(
        self, size: int, eps: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(size, device=device, dtype=dtype), requires_grad=False
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        head_dim: int,
        qkv_bias: bool,
        o_bias: bool,
        qk_norm: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim

        hidden_size = config.hidden_size
        self.q_proj = Linear(
            hidden_size, self.num_q_heads * head_dim, qkv_bias, device, dtype
        )
        self.k_proj = Linear(
            hidden_size, self.num_kv_heads * head_dim, qkv_bias, device, dtype
        )
        self.v_proj = Linear(
            hidden_size, self.num_kv_heads * head_dim, qkv_bias, device, dtype
        )
        self.o_proj = Linear(
            self.num_q_heads * head_dim, hidden_size, o_bias, device, dtype
        )
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, config.rms_norm_eps, device, dtype)
            self.k_norm = RMSNorm(head_dim, config.rms_norm_eps, device, dtype)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attention_fn: AttentionFn,
    ) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        q = self.q_proj(hidden).view(num_tokens, self.num_q_heads, self.head_dim)
        k = self.k_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # Rotate q/k once at their logical positions; keys are never re-rotated
        # after compaction.
        apply_rope_with_cos_sin_cache_inplace(
            positions, q, k, self.head_dim, cos_sin_cache, is_neox=True
        )
        o = attention_fn(self.layer_idx, q, k, v)
        return self.o_proj(o.view(num_tokens, -1))


class MLP(nn.Module):
    def __init__(self, config, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = Linear(
            config.hidden_size, config.intermediate_size, bias, device, dtype
        )
        self.up_proj = Linear(
            config.hidden_size, config.intermediate_size, bias, device, dtype
        )
        self.down_proj = Linear(
            config.intermediate_size, config.hidden_size, bias, device, dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(silu_and_mul(gate_up))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        head_dim: int,
        qkv_bias: bool,
        o_bias: bool,
        qk_norm: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(
            config, layer_idx, head_dim, qkv_bias, o_bias, qk_norm, device, dtype
        )
        self.mlp = MLP(config, device, dtype)
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, device, dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, device, dtype
        )

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attention_fn: AttentionFn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            fused_add_rmsnorm(
                hidden, residual, self.input_layernorm.weight, self.input_layernorm.eps
            )
        hidden = self.self_attn(hidden, positions, cos_sin_cache, attention_fn)
        fused_add_rmsnorm(
            hidden,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        hidden = self.mlp(hidden)
        return hidden, residual


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class Model(nn.Module):
    def __init__(
        self,
        config,
        head_dim: int,
        qkv_bias: bool,
        o_bias: bool,
        qk_norm: bool,
        max_position: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, device, dtype
        )
        self.layers = nn.ModuleList(
            DecoderLayer(
                config, layer_idx, head_dim, qkv_bias, o_bias, qk_norm, device, dtype
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.register_buffer(
            "cos_sin_cache",
            _build_cos_sin_cache(config, head_dim, max_position, device),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_fn: AttentionFn,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(
                hidden, residual, positions, self.cos_sin_cache, attention_fn
            )
        fused_add_rmsnorm(hidden, residual, self.norm.weight, self.norm.eps)
        return hidden


class CausalLM(nn.Module):
    """Decoder-only causal LM over an HF ``AutoConfig`` (Llama/Qwen2/Qwen3)."""

    def __init__(
        self,
        config,
        max_position: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        qkv_bias, o_bias, qk_norm = _family_quirks(config)
        # Qwen3 sets head_dim != hidden_size // num_attention_heads.
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads

        self.num_layers = config.num_hidden_layers
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = config.hidden_size

        self.model = Model(
            config, head_dim, qkv_bias, o_bias, qk_norm, max_position, device, dtype
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = None
        else:
            self.lm_head = Linear(
                config.hidden_size, config.vocab_size, False, device, dtype
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_fn: AttentionFn,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, attention_fn)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = (
            self.model.embed_tokens.weight
            if self.lm_head is None
            else self.lm_head.weight
        )
        return F.linear(hidden, weight)
