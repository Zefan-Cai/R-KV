"""Native Transformers hybrid attention with R-KV on full attention only.

This adapter leaves projections, normalization, RoPE, attention gates, linear
attention, and sliding attention in their native forwards. It intercepts the
post-RoPE attention backend on full layers, preserving absolute positions when
each KV head selects a different subset. Prefill is uncompressed; decoding is
compressed every ``compression_interval`` single-token calls, as in the original
``step_length`` evaluation. Use an eager or SDPA model and a dynamic cache.
"""

import copy
import inspect
import math
import weakref
from dataclasses import dataclass

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .compression.r1_kv import R1KV


def _attention_backend(module, query, key, value, attention_mask, **kwargs):
    return module._rkv_hybrid_adapter._attention(
        module, query, key, value, attention_mask, **kwargs
    )


@dataclass
class _LayerState:
    seen: int
    positions: torch.Tensor
    queries: torch.Tensor
    decode_calls: int = 0


class HybridRKVAdapter:
    """Install decoding-time compression on a loaded native hybrid text model.

    ``budget=None`` installs an exact pass-through adapter for parity checks.
    New generation caches own independent compression/query state. Stats can be
    reset between samples without changing active cache state. Gemma4 shared full
    layers consume their producer's selected KV; their SWA counterparts remain
    entirely native. Removing the adapter requires starting a fresh cache.
    """

    def __init__(
        self,
        model,
        budget=None,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        compression_interval=128,
    ):
        if compression_interval < 1:
            raise ValueError("compression_interval must be positive")
        self.budget = budget
        self.window_size = window_size
        self.compression_interval = compression_interval
        self.selector = None if budget is None else R1KV(
            budget=budget, window_size=window_size, kernel_size=kernel_size,
            mix_lambda=mix_lambda, retain_ratio=retain_ratio,
            retain_direction=retain_direction,
        )
        text_config = model.config.get_text_config()
        self.layer_types = list(text_config.layer_types)
        self._modules = {}
        self._original = {}
        self._hooks = []
        self._calls = {}
        self._sources = {}
        # Only text attention modules have both a decoder index and q_proj.
        # Vision attention modules are excluded by their config's layer types.
        for module in model.modules():
            if not hasattr(module, "q_proj") or not hasattr(module, "layer_idx"):
                continue
            index = module.layer_idx
            config = getattr(module, "config", None)
            if getattr(config, "model_type", None) != text_config.model_type:
                continue
            if self.layer_types[index] != "full_attention":
                continue
            if hasattr(module, "_rkv_hybrid_adapter"):
                raise ValueError("an adapter is already installed on this model")
            backend = config._attn_implementation
            if backend not in ("sdpa", "eager"):
                raise ValueError("the hybrid adapter requires sdpa or eager attention")
            native_module = inspect.getmodule(type(module))
            default = native_module.eager_attention_forward
            interface = ALL_ATTENTION_FUNCTIONS.get_interface(backend, default)
            self._modules[index] = module
            self._original[index] = (config, interface)
            if getattr(module, "is_kv_shared_layer", False):
                first_shared = config.num_hidden_layers - config.num_kv_shared_layers
                self._sources[index] = max(
                    i for i in range(first_shared)
                    if self.layer_types[i] == "full_attention"
                )
            else:
                self._sources[index] = index
        if not self._modules:
            raise ValueError("no native full attention layers were found")
        ALL_ATTENTION_FUNCTIONS.register("rkv_hybrid", _attention_backend)
        for index, module in self._modules.items():
            # A private attention config avoids changing the model's mask backend
            # or any shared-config sliding and recurrent attention modules.
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "rkv_hybrid"
            module._rkv_hybrid_adapter = self
            self._hooks.append(module.register_forward_pre_hook(self._capture, with_kwargs=True))
        self.reset_stats()

    def _capture(self, module, args, kwargs):
        bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
        self._calls[module.layer_idx] = (
            bound.arguments.get("past_key_values"),
            bound.arguments.get("shared_kv_states"),
        )

    def reset_stats(self):
        self._stats = {
            index: {
                "calls": 0, "compression_count": 0, "tokens_evicted": 0,
                "logical_length": 0, "stored_length": 0, "max_stored_length": 0,
                "shared_from": self._sources[index],
            }
            for index in self._modules
        }

    def stats(self):
        return {
            "budget": self.budget,
            "compression_interval": self.compression_interval,
            "layer_types": self.layer_types,
            "full_attention_layers": sorted(self._modules),
            "full_cache_layers": sorted(i for i, source in self._sources.items() if i == source),
            "layers": {str(i): dict(stats) for i, stats in self._stats.items()},
            "compression_count": sum(s["compression_count"] for s in self._stats.values()),
            "tokens_evicted": sum(s["tokens_evicted"] for s in self._stats.values()),
        }

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        for index, module in self._modules.items():
            module.config = self._original[index][0]
            del module._rkv_hybrid_adapter
        self._hooks.clear()

    def _attention(self, module, query, key, value, attention_mask, **kwargs):
        index = module.layer_idx
        original_interface = self._original[index][1]
        self._stats[index]["calls"] += 1
        cache, shared_kv_states = self._calls.pop(index)
        if self.budget is None or cache is None:
            return original_interface(module, query, key, value, attention_mask, **kwargs)
        if query.shape[0] != 1:
            raise ValueError("the hybrid evaluation adapter supports batch size 1")
        if not hasattr(cache, "layers"):
            raise TypeError("the hybrid adapter requires a modern Transformers dynamic cache")
        if not hasattr(cache, "_rkv_hybrid_states"):
            cache._rkv_hybrid_states = {}
        states = cache._rkv_hybrid_states
        source = self._sources[index]
        self._stats[index]["max_stored_length"] = max(
            self._stats[index]["max_stored_length"], key.shape[-2]
        )
        if source == index:
            state = states.get(index)
            q_len = query.shape[-2]
            if state is None or key.shape[-2] == q_len:
                positions = torch.arange(key.shape[-2], device=key.device)
                positions = positions.view(1, 1, -1).expand(key.shape[:3])
                state = _LayerState(key.shape[-2], positions, query[:, :, -self.window_size:])
                states[index] = state
                cache.layers[index]._rkv_hybrid_state = state
            else:
                new_positions = torch.arange(state.seen, state.seen + q_len, device=key.device)
                new_positions = new_positions.view(1, 1, -1).expand(*key.shape[:2], -1)
                state.positions = torch.cat((state.positions, new_positions), dim=-1)
                state.seen += q_len
                state.queries = torch.cat((state.queries, query), dim=-2)[:, :, -self.window_size:]
                if q_len != 1:
                    raise ValueError("after prefill, hybrid compression requires single-token decoding")
                state.decode_calls += 1
                if state.decode_calls % self.compression_interval == 0 and key.shape[-2] > self.budget:
                    key, value = self._compress(cache, index, state, key, value)
                    if getattr(module, "store_full_length_kv", False):
                        shared_kv_states["full_attention"] = key, value
        else:
            state = states[source]
        stats = self._stats[index]
        stats["logical_length"] = state.seen
        stats["stored_length"] = key.shape[-2]
        if attention_mask is not None and state.seen != key.shape[-2]:
            attention_mask = self._select_mask(attention_mask, state.positions, query.shape[1])
        return original_interface(module, query, key, value, attention_mask, **kwargs)

    def _compress(self, cache, index, state, key, value):
        layer = cache.layers[index]
        # Static and sliding caches have different update semantics. Fail before
        # replacing any storage, rather than silently corrupting their state.
        if type(layer).__name__ != "DynamicLayer":
            raise TypeError("only native DynamicLayer full caches can be compressed")
        before = key.shape[-2]
        # Values are opaque to the original selector. Integer slots recover its
        # exact head-specific top-k choices without duplicating scoring logic or
        # enabling the selector's expensive CPU visualization recording.
        slots = torch.arange(before, device=key.device).view(1, 1, -1, 1)
        slots = slots.expand(*key.shape)
        # The original selector normalizes logits by sqrt(head_dim). Gemma uses
        # a model-specific scalar (Gemma4 uses 1.0), so scale only the scoring
        # queries to recover native attention logits; model queries stay intact.
        scoring_scale = self._modules[index].scaling * math.sqrt(state.queries.shape[-1])
        scoring_queries = state.queries * scoring_scale
        key, selected_slots = self.selector.update_kv(key, scoring_queries, slots)
        selected = selected_slots[..., 0]
        value = value.gather(-2, selected.unsqueeze(-1).expand(*selected.shape, value.shape[-1]))
        state.positions = state.positions.gather(-1, selected)
        layer.keys, layer.values = key, value
        layer._rkv_hybrid_state = state
        # Native mask construction and RoPE need the uncompressed logical length.
        # DynamicLayer.get_mask_sizes calls this getter too. SWA and recurrent
        # caches retain their original methods and tensors.
        layer_ref = weakref.ref(layer)

        def logical_length():
            current_layer = layer_ref()
            if current_layer is None or not current_layer.is_initialized or current_layer.keys.numel() == 0:
                return 0
            return current_layer._rkv_hybrid_state.seen

        layer.get_seq_length = logical_length
        stats = self._stats[index]
        stats["compression_count"] += 1
        stats["tokens_evicted"] += before - key.shape[-2]
        return key, value

    @staticmethod
    def _select_mask(mask, positions, query_heads):
        if mask.ndim != 4:
            raise ValueError("compressed attention requires a native 4D mask or no mask")
        kv_heads = positions.shape[1]
        expanded_positions = positions.repeat_interleave(query_heads // kv_heads, dim=1)
        gather_indices = expanded_positions.unsqueeze(-2).expand(-1, -1, mask.shape[-2], -1)
        mask = mask.expand(positions.shape[0], query_heads, -1, -1)
        return mask.gather(-1, gather_indices)
