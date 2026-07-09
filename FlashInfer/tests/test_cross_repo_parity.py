"""Cross-repo parity: the FlashInfer port's R1KV vs this repo's reference R1KV.

Feeds identical random tensors through the port (``FlashInfer/rkv/algo.py``)
and the repo-root reference (``rkv/compression/r1_kv.py``) and compares
bit-for-bit, across MHA/GQA shapes (including real Qwen2.5 head layouts),
batch > 1, budgets, and dtypes. Both sides are loaded by file path — no
package imports, no flashinfer dependency (CPU-only torch).

Run from anywhere (needs torch only):

    python FlashInfer/tests/test_cross_repo_parity.py
"""
import importlib.util
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str, search_path: str) -> types.ModuleType:
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.abspath(search_path)]
    sys.modules[name] = pkg
    return pkg


# Port: algo.py has zero flashinfer deps — load the module file directly,
# under a synthetic parent so rkv/__init__.py (flashinfer engine) never runs.
_package("fi_rkv", os.path.join(_HERE, "..", "rkv"))
port_algo = _load("fi_rkv.algo", os.path.join(_HERE, "..", "rkv", "algo.py"))

# Reference: rkv/compression/r1_kv.py uses ``from . import ...``, so give it a
# synthetic parent package exposing the utils functions instead of executing
# the real package __init__ (which drags in every compression method).
_package("rkv_ref", os.path.join(_REPO_ROOT, "rkv"))
ref_utils = _load("rkv_ref.utils", os.path.join(_REPO_ROOT, "rkv", "utils.py"))
_ref_comp = _package("rkv_ref.compression", os.path.join(_REPO_ROOT, "rkv", "compression"))
_ref_comp.cal_similarity = ref_utils.cal_similarity
_ref_comp.compute_attention_scores = ref_utils.compute_attention_scores
RefR1KV = _load(
    "rkv_ref.compression.r1_kv",
    os.path.join(_REPO_ROOT, "rkv", "compression", "r1_kv.py"),
).R1KV
ref_cal_similarity = ref_utils.cal_similarity
ref_attn = ref_utils.compute_attention_scores

CASES = [
    # (bsz, q_heads, kv_heads, kv_len, head_dim, budget, window, dtype)
    (1, 8, 8, 300, 64, 128, 8, torch.float32),     # MHA
    (1, 28, 4, 700, 128, 512, 8, torch.bfloat16),  # Qwen2.5-Math-7B GQA shape
    (2, 16, 2, 1024, 64, 256, 8, torch.float32),   # batch>1, GQA 8x
    (1, 14, 2, 640, 64, 512, 8, torch.bfloat16),   # Qwen2.5-0.5B GQA shape
    (1, 8, 8, 100, 64, 128, 8, torch.float32),     # below budget -> noop
]


def main() -> int:
    failures = 0
    for i, (bsz, qh, kvh, kvlen, hd, budget, window, dtype) in enumerate(CASES):
        torch.manual_seed(1234 + i)
        q = torch.randn(bsz, qh, kvlen, hd, dtype=dtype)
        k = torch.randn(bsz, kvh, kvlen, hd, dtype=dtype)
        v = torch.randn(bsz, kvh, kvlen, hd, dtype=dtype)

        kwargs = dict(budget=budget, window_size=window, kernel_size=7,
                      mix_lambda=0.07, retain_ratio=0.1, retain_direction="last")
        ref = RefR1KV(**kwargs)
        port = port_algo.R1KV(**kwargs)

        rk, rv = ref.update_kv(k.clone(), q.clone(), v.clone())
        pk, pv = port.update_kv(k.clone(), q.clone(), v.clone())
        ok = torch.equal(rk, pk) and torch.equal(rv, pv)

        ok_attn = torch.equal(
            ref_attn(q.clone(), k.clone()),
            port_algo.compute_attention_scores(q.clone(), k.clone()),
        )
        ok_sim = torch.equal(
            ref_cal_similarity(k.clone()), port_algo.cal_similarity(k.clone())
        )

        # select_indices must pick exactly the token set update_kv keeps
        sel = port.select_indices(k.clone(), q.clone(), sort=True)
        if kvlen < budget:
            ok_sel = sel is None
        else:
            scores = port._scores(k.clone(), q.clone())
            past = scores.topk(budget - window, dim=-1).indices
            winidx = (
                torch.arange(kvlen - window, kvlen)
                .view(1, 1, -1)
                .expand(bsz, kvh, -1)
            )
            expect = torch.sort(torch.cat([past, winidx], dim=-1), dim=-1)[0]
            ok_sel = torch.equal(sel, expect)

        status = "OK " if (ok and ok_attn and ok_sim and ok_sel) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] case{i}: bsz={bsz} qh={qh} kvh={kvh} len={kvlen} hd={hd} "
              f"budget={budget} dtype={dtype} | update_kv={ok} attn={ok_attn} "
              f"sim={ok_sim} select={ok_sel}")

    if failures:
        print(f"\nRESULT: {failures} FAILURES")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
