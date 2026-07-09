"""Teacher-forced FA3-vs-FlashInfer logits parity on one H100.

Uses ragged synthetic prompts, drops the middle request before decode so active
rows are non-contiguous ``[0, 2]``, and (in R-KV mode) drives far enough to
compact after prefill. This targets the adapter branches that the batch=max,
forced-length throughput benchmark cannot cover.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FlashInfer"))

from rkv import FA3Engine, FlashInferEngine, RKVConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--max-argmax-mismatches", type=int, default=2)
    return parser.parse_args()


def vocab_size(model: str) -> int:
    with (Path(model) / "config.json").open() as fin:
        return int(json.load(fin)["vocab_size"])


def build_prompts(model: str) -> list[list[int]]:
    rng = random.Random(20260709)
    vocab = vocab_size(model)
    return [[rng.randrange(vocab) for _ in range(length)] for length in (48, 56, 64)]


def run_mode(args: argparse.Namespace, mode: str) -> tuple[int, int, float]:
    prompts = build_prompts(args.model)
    rkv = RKVConfig(budget=64, buffer=16, window_size=8) if mode == "rkv" else None
    engine_args = {
        "max_batch_size": 3,
        "max_seq_len": max(map(len, prompts)) + args.steps + 4,
        "rkv": rkv,
    }
    fi = FlashInferEngine(args.model, **engine_args)
    fa3 = FA3Engine(args.model, **engine_args)
    for engine in (fi, fa3):
        engine._phys_len = [0] * len(prompts)
        engine._logical_len = [0] * len(prompts)
        if engine.compressor is not None:
            engine.compressor.reset()
    active = [0, 2]  # request 1 is treated as if it stopped after prefill
    total = 0
    agree = 0
    worst = 0.0

    def compare(left: torch.Tensor, right: torch.Tensor) -> None:
        nonlocal total, agree, worst
        left = left.float()
        right = right.float()
        worst = max(worst, (left - right).abs().max().item())
        total += left.shape[0]
        agree += int((left.argmax(-1) == right.argmax(-1)).sum().item())

    with torch.inference_mode():
        lens = list(map(len, prompts))
        fi_logits = fi._prefill(prompts, lens)
        fa3_logits = fa3._prefill(prompts, lens)
        compare(fi_logits, fa3_logits)
        last = fi_logits.argmax(-1).tolist()

        for _ in range(args.steps):
            fi_logits = fi._decode_step(active, last)
            fa3_logits = fa3._decode_step(active, last)
            compare(fi_logits, fa3_logits)
            next_tokens = fi_logits.argmax(-1).tolist()
            for index, row in enumerate(active):
                last[row] = next_tokens[index]

            assert fi._phys_len == fa3._phys_len
            if rkv is not None:
                rows = [row for row in active if fi._phys_len[row] == fi.region_len]
                if rows:
                    fi._compact_rows(rows)
                    fa3._compact_rows(rows)
                    assert fi._phys_len == fa3._phys_len

    compactions = [0] * len(prompts)
    if rkv is not None:
        compactions = torch.as_tensor(fi.compressor.num_compactions).tolist()
        assert compactions == torch.as_tensor(fa3.compressor.num_compactions).tolist()
        assert sum(compactions) > 0
    print(
        f"{mode}: argmax_agree={agree}/{total} mismatches={total - agree} "
        f"worst_maxdelta={worst:.4f} phys_len={fi._phys_len} "
        f"compactions={compactions}"
    )
    del fi, fa3
    gc.collect()
    torch.cuda.empty_cache()
    return agree, total, worst


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("fa3_parity.py needs an H100 CUDA device")
    results = [run_mode(args, mode) for mode in ("fullkv", "rkv")]
    mismatches = sum(total - agree for agree, total, _ in results)
    passed = mismatches <= args.max_argmax_mismatches
    print("FA3_PARITY_PASS" if passed else "FA3_PARITY_SUSPECT")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
