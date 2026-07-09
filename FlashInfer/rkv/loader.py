"""Safetensors weight loading into pre-allocated model parameters.

Nano-vLLM-style streaming: shards are opened one tensor at a time and copied
straight into the model's parameters (dtype/device conversion happens in
``copy_``), so the full checkpoint is never materialized twice.
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def _shard_files(model_path: str) -> list[str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return sorted({os.path.join(model_path, shard) for shard in weight_map.values()})
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint under {model_path!r}")
    return files


@torch.no_grad()
def load_weights(model: nn.Module, model_path: str) -> None:
    """Stream all checkpoint tensors into ``model``'s named parameters.

    Checkpoint keys with no matching parameter are skipped (e.g. a redundant
    ``lm_head.weight`` in tied-embedding checkpoints, or rotary ``inv_freq``
    buffers); every model parameter must be covered or a ``ValueError`` is
    raised.
    """
    params = dict(model.named_parameters())
    loaded: set[str] = set()
    for file in _shard_files(model_path):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                param = params.get(name)
                if param is None:
                    continue
                tensor = f.get_tensor(name)
                if tensor.shape != param.shape:
                    raise ValueError(
                        f"shape mismatch for {name!r}: checkpoint "
                        f"{tuple(tensor.shape)} vs model {tuple(param.shape)}"
                    )
                param.copy_(tensor)
                loaded.add(name)
    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError(f"checkpoint {model_path!r} is missing weights: {missing}")
