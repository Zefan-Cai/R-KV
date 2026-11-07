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
        # The index is untrusted input: a shard entry must be a bare file name
        # inside ``model_path``. The check is lexical on purpose -- hub snapshots
        # symlink their files to ``../../blobs/<sha>``, so resolve()-based
        # containment would reject legitimate ``snapshot_download`` checkpoints.
        shards = set()
        for shard in weight_map.values():
            if (
                not isinstance(shard, str)
                or shard in ("", ".", "..")
                or os.path.basename(shard) != shard
            ):
                raise ValueError(
                    f"invalid shard name {shard!r} in {index_path!r}: "
                    "index entries must be bare file names under the model directory"
                )
            shards.add(os.path.join(model_path, shard))
        return sorted(shards)
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
    buffers). Keys listed in the model's ``packed_map`` (checkpoint name ->
    (fused parameter name, row offset)) are copied into the row slice of the
    fused parameter; a fused parameter counts as loaded only once every one
    of its source tensors has landed. Every model parameter must be covered
    or a ``ValueError`` is raised.
    """
    params = dict(model.named_parameters())
    packed: dict[str, tuple[str, int]] = dict(getattr(model, "packed_map", {}) or {})
    outstanding: dict[str, set[str]] = {}
    for source, (target, _) in packed.items():
        outstanding.setdefault(target, set()).add(source)
    loaded: set[str] = set()
    for file in _shard_files(model_path):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                param = params.get(name)
                if param is not None:
                    tensor = f.get_tensor(name)
                    if tensor.shape != param.shape:
                        raise ValueError(
                            f"shape mismatch for {name!r}: checkpoint "
                            f"{tuple(tensor.shape)} vs model {tuple(param.shape)}"
                        )
                    param.copy_(tensor)
                    loaded.add(name)
                elif name in packed:
                    target, offset = packed[name]
                    param = params[target]
                    tensor = f.get_tensor(name)
                    rows = tensor.shape[0]
                    if (
                        tensor.shape[1:] != param.shape[1:]
                        or offset + rows > param.shape[0]
                    ):
                        raise ValueError(
                            f"packed shape mismatch for {name!r}: checkpoint "
                            f"{tuple(tensor.shape)} into rows "
                            f"[{offset}, {offset + rows}) of {target!r} "
                            f"{tuple(param.shape)}"
                        )
                    param[offset : offset + rows].copy_(tensor)
                    outstanding[target].discard(name)
                    if not outstanding[target]:
                        loaded.add(target)
    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError(f"checkpoint {model_path!r} is missing weights: {missing}")
