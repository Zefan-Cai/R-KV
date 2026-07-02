"""Cross-repo parity: the SGLang port's R1KV vs this repo's reference R1KV.

Feeds identical random tensors through the port (``SGLang/rkv/algo.py``) and
the repo-root reference (``rkv/compression/r1_kv.py``) and compares
bit-for-bit, across MHA/GQA shapes (including real Qwen2.5 head layouts),
batch > 1, budgets, and dtypes.

Run from anywhere (needs torch only):

    python SGLang/tests/test_cross_repo_parity.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# Port: algo.py has zero sglang deps — import the module file directly.
sys.path.insert(0, os.path.join(_HERE, "..", "rkv"))
import algo as port_algo  # noqa: E402

# Reference: the rkv package at the repo root.
sys.path.insert(0, _REPO_ROOT)
from rkv.compression.r1_kv import R1KV as RefR1KV  # noqa: E402
from rkv.utils import cal_similarity as ref_cal_similarity  # noqa: E402
from rkv.utils import compute_attention_scores as ref_attn  # noqa: E402

CASES = [
    # (bsz, q_heads, kv_heads, kv_len, head_dim, budget, window, dtype)
    (1, 8, 8, 300, 64, 128, 8, torch.float32),     # MHA
    (1, 28, 4, 700, 128, 512, 8, torch.bfloat16),  # Qwen2.5-Math-7B GQA shape
    (2, 16, 2, 1024, 64, 256, 8, torch.float32),   # batch>1, GQA 8x
    (1, 14, 2, 640, 64, 512, 8, torch.bfloat16),   # Qwen2.5-0.5B GQA shape
    (1, 8, 8, 100, 64, 128, 8, torch.float32),     # below budget -> noop
]


def main():
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

    print("\nRESULT:", "ALL PARITY CHECKS PASSED (bit-for-bit)" if failures == 0
          else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
