"""Decode throughput + peak-memory benchmark for the FlashInfer R-KV backend.

Follows the offline harness shape of Mini-SGLang/benchmark/bench_rkv.py:
seeded synthetic prompts, engine constructed directly, FullKV vs R-KV via
--mode. Prints a per-trial human table, then exactly one machine-readable
JSON object as the final stdout line so sweep scripts can `tail -n 1`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

# Prefer this backend's `rkv` package over the repo-root reference package
# (or any installed distribution) that shares the name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rkv import FlashInferEngine, RKVConfig

TEMPERATURE = 0.6
TOP_P = 0.95
PROMPT_SEED = 1234
SAMPLING_SEED = 0
WARMUP_TOKENS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF repo dir (config.json + safetensors)")
    parser.add_argument("--mode", choices=("fullkv", "rkv"), default="rkv")
    parser.add_argument("--budget", type=int, default=1024, help="R-KV budget (rkv mode only)")
    parser.add_argument("--buffer", type=int, default=128, help="R-KV buffer (rkv mode only)")
    parser.add_argument(
        "--mix-lambda", type=float, default=0.1,
        help="R-KV score mix; 1.0 = attention-only (SnapKV-style)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gen-len", type=int, default=2048, help="forced decode length per request")
    parser.add_argument("--prompt-len", type=int, default=512, help="synthetic prompt length per request")
    parser.add_argument("--trials", type=int, default=3)
    return parser.parse_args()


def stats_dict(stats: object) -> dict:
    """Normalize engine.stats (dict, dataclass, or attribute object) to a dict."""
    if isinstance(stats, dict):
        return dict(stats)
    if dataclasses.is_dataclass(stats):
        return dataclasses.asdict(stats)
    return dict(vars(stats))


def stat_seconds(stats: dict, key: str) -> float:
    if stats.get(key) is None:
        raise KeyError(
            f"engine.stats has no '{key}' (available: {sorted(stats)}); "
            "bench_rkv.py expects the DESIGN.md §5.5 stats contract"
        )
    return float(stats[key])


def read_vocab_size(model_path: str) -> int:
    config = json.loads((Path(model_path) / "config.json").read_text())
    return int(config["vocab_size"])


def build_prompts(batch_size: int, prompt_len: int, vocab_size: int) -> list[list[int]]:
    rng = random.Random(PROMPT_SEED)
    return [[rng.randrange(vocab_size) for _ in range(prompt_len)] for _ in range(batch_size)]


def run_trial(engine: FlashInferEngine, prompts: list[list[int]], gen_len: int) -> dict:
    torch.cuda.synchronize()
    start = time.perf_counter()
    outs = engine.generate(
        prompts,
        max_new_tokens=gen_len,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop_token_ids=(),  # forced length: random-token prompts must never stop early
        seed=SAMPLING_SEED,
    )
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - start

    stats = stats_dict(engine.stats)  # snapshot of the generate() call above
    prefill_s = stat_seconds(stats, "prefill_seconds")
    decode_s = stat_seconds(stats, "decode_seconds")
    new_tokens = sum(out.num_output_tokens for out in outs)
    return {
        "wall_s": wall_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "new_tokens": new_tokens,
        "decode_tok_s": new_tokens / decode_s,
        "decode_tok_s_per_seq": new_tokens / decode_s / len(prompts),
        "num_compactions": sum(out.num_compactions for out in outs),
        "compaction_s": float(stats.get("compaction_seconds") or 0.0),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench_rkv.py needs a CUDA GPU")

    rkv_cfg = (
        RKVConfig(budget=args.budget, buffer=args.buffer, mix_lambda=args.mix_lambda)
        if args.mode == "rkv"
        else None
    )
    engine = FlashInferEngine(
        args.model,
        max_batch_size=args.batch_size,
        max_seq_len=args.prompt_len + args.gen_len,
        rkv=rkv_cfg,
    )
    prompts = build_prompts(args.batch_size, args.prompt_len, read_vocab_size(args.model))

    engine.generate(
        prompts,
        max_new_tokens=min(WARMUP_TOKENS, args.gen_len),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop_token_ids=(),
        seed=SAMPLING_SEED,
    )
    torch.cuda.synchronize()
    # Peak from here on covers weights + KV pool (already resident) + trial activations.
    torch.cuda.reset_peak_memory_stats()

    trials = [run_trial(engine, prompts, args.gen_len) for _ in range(args.trials)]
    peak_mem_gib = torch.cuda.max_memory_allocated() / 2**30

    def mean(key: str) -> float:
        return statistics.fmean(trial[key] for trial in trials)

    mode_label = f"rkv(budget={args.budget}, buffer={args.buffer})" if rkv_cfg else "fullkv"
    print(
        f"model={args.model} mode={mode_label} batch={args.batch_size} "
        f"prompt_len={args.prompt_len} gen_len={args.gen_len} trials={args.trials}"
    )
    print(
        f"{'trial':>5} {'wall_s':>9} {'prefill_s':>10} {'decode_s':>9} "
        f"{'decode_tok/s':>13} {'tok/s/seq':>10} {'compactions':>12} {'compaction_s':>13}"
    )
    for i, trial in enumerate(trials, start=1):
        print(
            f"{i:>5} {trial['wall_s']:>9.2f} {trial['prefill_s']:>10.3f} {trial['decode_s']:>9.2f} "
            f"{trial['decode_tok_s']:>13.1f} {trial['decode_tok_s_per_seq']:>10.1f} "
            f"{trial['num_compactions']:>12d} {trial['compaction_s']:>13.3f}"
        )
    print(
        f"{'mean':>5} {mean('wall_s'):>9.2f} {mean('prefill_s'):>10.3f} {mean('decode_s'):>9.2f} "
        f"{mean('decode_tok_s'):>13.1f} {mean('decode_tok_s_per_seq'):>10.1f} "
        f"{mean('num_compactions'):>12.1f} {mean('compaction_s'):>13.3f}"
    )
    print(f"peak torch.cuda.max_memory_allocated: {peak_mem_gib:.2f} GiB")

    summary = {
        "script": "bench_rkv",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "model": args.model,
        "mode": args.mode,
        "budget": args.budget if rkv_cfg else None,
        "buffer": args.buffer if rkv_cfg else None,
        "mix_lambda": args.mix_lambda if rkv_cfg else None,
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "trials": args.trials,
        "prefill_s_mean": round(mean("prefill_s"), 4),
        "decode_s_mean": round(mean("decode_s"), 4),
        "decode_tok_s_mean": round(mean("decode_tok_s"), 2),
        "decode_tok_s_per_seq_mean": round(mean("decode_tok_s_per_seq"), 2),
        "num_compactions_mean": round(mean("num_compactions"), 2),
        "compaction_s_mean": round(mean("compaction_s"), 4),
        "peak_mem_gib": round(peak_mem_gib, 3),
        "per_trial": trials,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
