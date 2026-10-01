"""CPU checks using native tiny Qwen3.5, Gemma3 and Gemma4 models."""

import sys
import weakref
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)
from transformers.cache_utils import DynamicCache

from rkv.hybrid import HybridRKVAdapter, _LayerState
from rkv.utils import compute_attention_scores


torch.set_num_threads(1)


def tiny_model(kind, backend="sdpa"):
    options = dict(
        vocab_size=96, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256,
    )
    if kind == "qwen3_5":
        options.update(
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"] * 2,
            rope_parameters={
                "rope_type": "default", "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2],
            },
        )
        config_class, model_class = Qwen3_5TextConfig, Qwen3_5ForCausalLM
    else:
        options.update(
            layer_types=["sliding_attention", "full_attention"] * 2,
            sliding_window=4,
        )
        if kind == "gemma3":
            config_class, model_class = Gemma3TextConfig, Gemma3ForCausalLM
        else:
            options.update(hidden_size_per_layer_input=0, num_kv_shared_layers=2)
            config_class, model_class = Gemma4TextConfig, Gemma4ForCausalLM
    config = config_class(**options)
    config._attn_implementation = backend
    torch.manual_seed(7)
    return model_class(config).eval()


@torch.inference_mode()
def trace(model, decode_length=5):
    output = model(torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]]), use_cache=True)
    logits = [output.logits.clone()]
    lengths = [output.past_key_values.get_seq_length()]
    for step in range(decode_length):
        output = model(
            torch.tensor([[11 + step]]), past_key_values=output.past_key_values,
            use_cache=True,
        )
        logits.append(output.logits.clone())
        lengths.append(output.past_key_values.get_seq_length())
    return logits, lengths, output.past_key_values


@pytest.mark.parametrize("kind", ["qwen3_5", "gemma3", "gemma4"])
@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@pytest.mark.parametrize("budget", [None, 64])
def test_no_eviction_exact_native_logits_and_tokens(kind, backend, budget):
    model = tiny_model(kind, backend)
    native_logits, native_lengths, _ = trace(model)
    adapter = HybridRKVAdapter(
        model, budget=budget, window_size=2, kernel_size=3,
        compression_interval=2,
    )
    logits, lengths, _ = trace(model)
    assert lengths == native_lengths == list(range(8, 14))
    for expected, actual in zip(native_logits, logits):
        assert torch.equal(expected, actual)
        assert torch.equal(expected.argmax(-1), actual.argmax(-1))
    assert adapter.stats()["tokens_evicted"] == 0
    adapter.remove()
    restored_logits, _, _ = trace(model)
    assert all(torch.equal(a, b) for a, b in zip(native_logits, restored_logits))


@pytest.mark.parametrize("kind", ["qwen3_5", "gemma3", "gemma4"])
def test_only_full_cache_compressed_and_absolute_length_preserved(kind):
    model = tiny_model(kind)
    untouched_modules = {
        name: (module, module.config, module.forward.__func__)
        for name, module in model.named_modules()
        if hasattr(module, "layer_idx") and hasattr(module, "config")
        and model.config.layer_types[module.layer_idx] != "full_attention"
    }
    adapter = HybridRKVAdapter(
        model, budget=6, window_size=2, kernel_size=3,
        compression_interval=2,
    )
    native_prefill, _, _ = trace(tiny_model(kind), decode_length=0)
    logits, lengths, cache = trace(model)
    assert torch.equal(native_prefill[0], logits[0])
    assert lengths == list(range(8, 14))
    stats = adapter.stats()
    assert stats["compression_count"] > 0
    assert stats["layers"]["1"]["max_stored_length"] == 10
    assert stats["full_cache_layers"] == ([1] if kind == "gemma4" else [1, 3])
    for index, layer in enumerate(cache.layers):
        layer_type = model.config.layer_types[index]
        if layer_type == "full_attention":
            assert layer.keys.shape[-2] == 7  # budget plus one buffered token
            assert layer.get_seq_length() == 13
            assert layer.get_mask_sizes(1) == (14, 0)
            state = cache._rkv_hybrid_states[index]
            assert state.positions.shape == layer.keys.shape[:-1]
            assert torch.equal(state.positions[..., -3:], torch.tensor([10, 11, 12]).view(1, 1, 3).expand(1, 2, 3))
        else:
            assert "get_seq_length" not in layer.__dict__
            assert index not in cache._rkv_hybrid_states
            if layer_type == "sliding_attention":
                assert layer.keys.shape[-2] == model.config.sliding_window - 1
    for module, config, native_forward in untouched_modules.values():
        assert module.config is config
        assert module.forward.__func__ is native_forward
        assert not hasattr(module, "_rkv_hybrid_adapter")
    if kind == "gemma4":
        producer, consumer = stats["layers"]["1"], stats["layers"]["3"]
        assert consumer["shared_from"] == 1
        assert consumer["compression_count"] == 0
        assert consumer["stored_length"] == producer["stored_length"]


@pytest.mark.parametrize("kind", ["qwen3_5", "gemma3", "gemma4"])
def test_repeated_generate_uses_new_cache_state(kind):
    model = tiny_model(kind)
    adapter = HybridRKVAdapter(
        model, budget=6, window_size=2, kernel_size=3,
        compression_interval=2,
    )
    prompt = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    with torch.inference_mode():
        first = model.generate(prompt, max_new_tokens=6, do_sample=False, eos_token_id=None)
        first_stats = adapter.stats()
        adapter.reset_stats()
        second = model.generate(prompt, max_new_tokens=6, do_sample=False, eos_token_id=None)
    assert torch.equal(first, second)
    assert adapter.stats() == first_stats
    assert first_stats["tokens_evicted"] > 0


def test_selector_is_original_rkv_and_keeps_head_specific_positions():
    model = tiny_model("gemma3")
    adapter = HybridRKVAdapter(
        model, budget=6, window_size=2, kernel_size=3,
        compression_interval=2,
    )
    torch.manual_seed(13)
    key = torch.randn(1, 2, 12, 8)
    value = torch.randn_like(key)
    query = torch.randn(1, 4, 2, 8)
    positions = torch.arange(20, 32).view(1, 1, 12).expand(1, 2, 12)
    state = _LayerState(32, positions, query)
    cache = DynamicCache(config=model.config)
    cache.update(key, value, 1)
    cache._rkv_hybrid_states = {1: state}
    scoring_query = query * adapter._modules[1].scaling * math.sqrt(query.shape[-1])
    expected_key, expected_value = adapter.selector.update_kv(key, scoring_query, value)
    actual_key, actual_value = adapter._compress(cache, 1, state, key, value)
    assert torch.equal(actual_key, expected_key)
    assert torch.equal(actual_value, expected_value)
    assert torch.equal(state.positions[..., -2:], torch.tensor([30, 31]).view(1, 1, 2).expand(1, 2, 2))
    assert not torch.equal(state.positions[:, 0, :-2], state.positions[:, 1, :-2])
    assert cache.get_seq_length(1) == 32


def test_mask_selection_follows_absolute_position_for_each_gqa_head():
    mask = torch.arange(12).view(1, 1, 1, 12).float()
    positions = torch.tensor([[[7, 1, 10, 11], [3, 8, 10, 11]]])
    selected = HybridRKVAdapter._select_mask(mask, positions, query_heads=4)
    expected = positions.repeat_interleave(2, dim=1).unsqueeze(-2).float()
    assert torch.equal(selected, expected)


def test_modern_import_does_not_load_legacy_monkeypatch():
    assert "rkv.monkeypatch" not in sys.modules
    assert "rkv.modeling" not in sys.modules


@pytest.mark.parametrize("kind", ["qwen3_5", "gemma3", "gemma4"])
def test_adapter_does_not_retain_completed_generation_cache(kind):
    model = tiny_model(kind)
    adapter = HybridRKVAdapter(
        model, budget=6, window_size=2, kernel_size=3,
        compression_interval=2,
    )
    _, _, cache = trace(model)
    cache_ref = weakref.ref(cache)
    layer_ref = weakref.ref(cache.layers[1])
    assert adapter._calls == {}
    del cache
    # CPython refcounting releases the cache and tensors immediately; neither
    # adapter hooks nor logical length getters require a later cyclic GC pass.
    assert cache_ref() is None
    assert layer_ref() is None


def test_native_cache_reset_reuses_absolute_length_state():
    model = tiny_model("gemma3")
    adapter = HybridRKVAdapter(
        model, budget=6, window_size=2, kernel_size=3,
        compression_interval=2,
    )
    _, _, cache = trace(model)
    cache.reset()
    assert cache.get_seq_length(1) == 0
    with torch.inference_mode():
        output = model(torch.tensor([[3, 4, 5, 6]]), past_key_values=cache, use_cache=True)
        assert cache.get_seq_length(1) == 4
        output = model(torch.tensor([[7]]), past_key_values=cache, use_cache=True)
        assert cache.get_seq_length(1) == 5
    assert adapter._calls == {}


@pytest.mark.parametrize("kind", ["qwen3_5", "gemma3", "gemma4"])
def test_selector_attention_logits_match_native_model_scaling(kind, monkeypatch):
    model = tiny_model(kind)
    adapter = HybridRKVAdapter(
        model, budget=6, window_size=2, kernel_size=3,
        compression_interval=2,
    )
    torch.manual_seed(13)
    key = torch.randn(1, 2, 12, 8)
    value = torch.randn_like(key)
    query = torch.randn(1, 4, 2, 8)
    state = _LayerState(12, torch.arange(12).view(1, 1, 12).expand(1, 2, 12), query)
    cache = DynamicCache(config=model.config)
    cache.update(key, value, 1)
    cache._rkv_hybrid_states = {1: state}
    original_selector = adapter.selector.update_kv
    captured = {}

    def capture_scores(key_states, query_states, value_states):
        captured["scores"] = compute_attention_scores(query_states, key_states)
        return original_selector(key_states, query_states, value_states)

    monkeypatch.setattr(adapter.selector, "update_kv", capture_scores)
    adapter._compress(cache, 1, state, key, value)
    grouped_query = query.reshape(1, 2, 2, 2, 8)
    expected = (grouped_query @ key.unsqueeze(2).transpose(-1, -2))
    expected = (expected * adapter._modules[1].scaling).max(dim=2).values
    torch.testing.assert_close(captured["scores"], expected)
    assert torch.equal(state.queries, query)
