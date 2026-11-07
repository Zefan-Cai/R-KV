"""CPU-only tests for safetensors shard discovery in the R-KV loader.

The index file is untrusted input: shard names must be bare file names inside
the model directory, so ``../x.safetensors`` or an absolute path cannot make
``load_weights`` read tensors from outside it. The check is lexical, and the
symlinked-snapshot test pins down why: a Hugging Face hub cache stores
``snapshots/<rev>/*.safetensors`` as symlinks to ``../../blobs/<sha>``, which a
``realpath``/``resolve`` containment check would wrongly reject.

CI does not install safetensors, so the module is stubbed when it is missing
and only the end-to-end load is skipped; the path checks still run.

Run: python FlashInfer/tests/test_loader.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import traceback
import types

import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOADER_PATH = os.path.abspath(os.path.join(_HERE, "..", "rkv", "loader.py"))
_MODULE_NAME = "_rkv_loader_contract"

try:
    from safetensors import safe_open as _safe_open  # noqa: F401

    HAS_SAFETENSORS = True
except ImportError:
    stub = types.ModuleType("safetensors")

    def safe_open(*args, **kwargs):
        raise RuntimeError("safetensors is not installed")

    stub.safe_open = safe_open
    sys.modules["safetensors"] = stub
    HAS_SAFETENSORS = False


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_loader = _load_module()
_shard_files = _loader._shard_files
load_weights = _loader.load_weights


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(2, 3))
        self.w1 = nn.Parameter(torch.zeros(4))


def _write_index(directory: str, shards: list[str]) -> None:
    weight_map = {f"w{i}": shard for i, shard in enumerate(shards)}
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


def test_bare_names_ok() -> None:
    with tempfile.TemporaryDirectory() as model_path:
        shards = [
            "model-00002-of-00002.safetensors",
            "model-00001-of-00002.safetensors",
        ]
        _write_index(model_path, shards)
        for shard in shards:
            open(os.path.join(model_path, shard), "wb").close()

        assert _shard_files(model_path) == [
            os.path.join(model_path, "model-00001-of-00002.safetensors"),
            os.path.join(model_path, "model-00002-of-00002.safetensors"),
        ]


def test_rejects_non_bare_names() -> None:
    cases = [
        "../x.safetensors",
        "..",
        ".",
        "",
        os.path.join(os.sep, "tmp", "x.safetensors"),
        "sub/x.safetensors",
        123,
        None,
    ]
    for bad in cases:
        with tempfile.TemporaryDirectory() as model_path:
            _write_index(model_path, ["model.safetensors", bad])
            try:
                _shard_files(model_path)
            except ValueError as exc:
                assert repr(bad) in str(exc), (bad, str(exc))
            else:
                raise AssertionError(f"{bad!r} should be rejected")


def test_hf_symlink_snapshot() -> None:
    with tempfile.TemporaryDirectory() as root:
        blobs = os.path.join(root, "blobs")
        snapshot = os.path.join(root, "snapshots", "rev")
        os.makedirs(blobs)
        os.makedirs(snapshot)
        tensors = {
            "w0": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "w1": torch.tensor([1.5, -2.5, 3.5, -4.5]),
        }
        weight_map = {}
        for sha, (name, tensor) in enumerate(tensors.items()):
            shard = f"model-0000{sha + 1}-of-00002.safetensors"
            blob = os.path.join(blobs, f"blob{sha}")
            if HAS_SAFETENSORS:
                from safetensors.torch import save_file

                save_file({name: tensor}, blob)
            else:
                open(blob, "wb").close()
            os.symlink(
                os.path.join("..", "..", "blobs", f"blob{sha}"),
                os.path.join(snapshot, shard),
            )
            weight_map[name] = shard
        with open(os.path.join(snapshot, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)

        # resolve()-based containment would reject this legitimate layout.
        base = os.path.realpath(snapshot)
        for shard in weight_map.values():
            resolved = os.path.realpath(os.path.join(snapshot, shard))
            assert not resolved.startswith(base + os.sep), resolved

        assert _shard_files(snapshot) == sorted(
            os.path.join(snapshot, shard) for shard in weight_map.values()
        )

        if not HAS_SAFETENSORS:
            print("       (load skipped: safetensors not installed)")
            return
        model = _TinyModel()
        load_weights(model, snapshot)
        assert torch.equal(model.w0, tensors["w0"])
        assert torch.equal(model.w1, tensors["w1"])


TESTS = [
    test_bare_names_ok,
    test_rejects_non_bare_names,
    test_hf_symlink_snapshot,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
        except Exception:
            failures += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
        else:
            print(f"[ OK ] {test.__name__}")
    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
