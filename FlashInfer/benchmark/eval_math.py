"""GSM8K / MATH-500 / AIME24 accuracy + throughput harness for the FlashInfer R-KV backend.

Prompting follows HuggingFace/run_math.py: its math prompt template wrapped in
tokenizer.apply_chat_template. Scoring reuses the in-repo HuggingFace/evaluation
pipeline (parser.run_execute + evaluate.evaluate, prompt_type "cot"). Writes
id-preserving jsonl records, prints a human summary, then exactly one
machine-readable JSON object as the final stdout line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = REPO_ROOT / "HuggingFace" / "data"
EVAL_DIR = REPO_ROOT / "HuggingFace" / "evaluation"

# Same instruct prompt as HuggingFace/run_math.py (verbatim, incl. spacing).
PROMPT_TEMPLATE = (
    "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the"
    " problem step by step. First, you need to provide the chain-of-thought, then"
    " provide the final answer.\n\n Provide the final answer in the format:"
    " Final answer:  \\boxed{{}}"
)

DATASET_QUESTION_KEY = {"gsm8k": "question", "aime24": "question", "math": "problem"}
DATASET_MAX_NEW_TOKENS = {"gsm8k": 8192, "math": 16384, "aime24": 32768}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="HF repo dir; required unless --score-only")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_QUESTION_KEY))
    parser.add_argument("--data-path", default=None, help=f"defaults to {DATA_DIR}/<dataset>.jsonl")
    parser.add_argument("--mode", choices=("fullkv", "rkv"), default="rkv")
    parser.add_argument("--budget", type=int, default=1024, help="R-KV budget (rkv mode only)")
    parser.add_argument("--buffer", type=int, default=128, help="R-KV buffer (rkv mode only)")
    parser.add_argument(
        "--mix-lambda", type=float, default=0.1,
        help="R-KV score mix; 1.0 = attention-only (SnapKV-style)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="per-dataset defaults: gsm8k 8192, math 16384, aime24 32768",
    )
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="records jsonl; defaults to eval_<dataset>_<mode>[_shardIofN].jsonl next to this script",
    )
    parser.add_argument(
        "--shard",
        default="0/1",
        help="'i/N': this process handles records[i::N] (one process per GPU)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="skip generation; score an existing --output jsonl (e.g. merged shards)",
    )
    return parser.parse_args()


def parse_shard(spec: str) -> tuple[int, int]:
    try:
        idx_str, count_str = spec.split("/")
        idx, count = int(idx_str), int(count_str)
    except ValueError as exc:
        raise SystemExit(f"--shard must look like 'i/N', got {spec!r}") from exc
    if count < 1 or not 0 <= idx < count:
        raise SystemExit(f"--shard index out of range: {spec!r}")
    return idx, count


def default_output(args: argparse.Namespace, shard: tuple[int, int]) -> Path:
    mode_tag = f"rkv{args.budget}" if args.mode == "rkv" else "fullkv"
    idx, count = shard
    suffix = f"_shard{idx}of{count}" if count > 1 else ""
    return SCRIPT_DIR / f"eval_{args.dataset}_{mode_tag}{suffix}.jsonl"


def load_records(path: Path, max_samples: int | None, shard: tuple[int, int]) -> list[dict]:
    with path.open() as fin:
        records = [{"idx": idx, **json.loads(line)} for idx, line in enumerate(fin)]
    if max_samples is not None:
        records = records[:max_samples]
    idx, count = shard
    # Strided slicing balances load across shards; the global 'idx' keeps
    # merged shard outputs id-stable for --score-only and the evaluator sort.
    return records[idx::count]


def score_records(records: list[dict], dataset: str) -> dict | None:
    """Score with the in-repo evaluator; returns None when its deps are absent."""
    sys.path.insert(0, str(EVAL_DIR))
    try:
        from evaluate import evaluate
        from parser import run_execute
    except ImportError as exc:
        print(f"[score] evaluator deps missing ({exc}); generation records are saved.")
        print("[score] install the deps and re-score offline:")
        print(f"  pip install -r {EVAL_DIR / 'requirements.txt'}")
        print(f"  python {SCRIPT_DIR / 'eval_math.py'} --dataset {dataset} --output <records.jsonl> --score-only")
        return None

    samples = []
    for rec in records:
        pred, _ = run_execute(None, rec["output"], "cot", dataset)
        samples.append({**rec, "pred": [pred]})
    _, result = evaluate(data_name=dataset, prompt_type="cot", samples=samples)
    return {
        "acc": float(result["acc"]),
        "num_scored": int(result["num_samples"]),
        "empty_samples": int(result["empty_samples"]),
        "timeout_samples": int(result["timeout_samples"]),
    }


def generate(args: argparse.Namespace, records: list[dict], out_path: Path) -> tuple[list[dict], dict]:
    # Imported here so --score-only works on machines without GPU deps.
    import torch
    from transformers import AutoTokenizer

    # Prefer this backend's `rkv` package over the repo-root reference package.
    sys.path.insert(0, str(BACKEND_DIR))
    from rkv import FlashInferEngine, RKVConfig

    if not torch.cuda.is_available():
        raise SystemExit("eval_math.py generation needs a CUDA GPU (--score-only scores without one)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    question_key = DATASET_QUESTION_KEY[args.dataset]
    prompt_ids: list[list[int]] = []
    for rec in records:
        prompt = PROMPT_TEMPLATE.format(question=rec[question_key])
        # R1-distill models only enter their trained <think> reasoning format
        # inside the chat template (see HuggingFace/run_math.py).
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # The chat template already inserts BOS/special tokens.
        prompt_ids.append(tokenizer.encode(prompt, add_special_tokens=False))

    rkv_cfg = (
        RKVConfig(budget=args.budget, buffer=args.buffer, mix_lambda=args.mix_lambda)
        if args.mode == "rkv"
        else None
    )
    engine = FlashInferEngine(
        args.model,
        max_batch_size=args.bsz,
        max_seq_len=max(len(ids) for ids in prompt_ids) + args.max_new_tokens,
        rkv=rkv_cfg,
    )

    outputs: list[dict] = []
    totals = {
        "prefill_tokens": 0,
        "output_tokens": 0,
        "num_compactions": 0,
        "gen_wall_s": 0.0,
        "gpu": torch.cuda.get_device_name(0),
    }
    num_batches = (len(records) + args.bsz - 1) // args.bsz
    with out_path.open("w") as fout:
        for b in range(num_batches):
            batch = records[b * args.bsz : (b + 1) * args.bsz]
            batch_ids = prompt_ids[b * args.bsz : (b + 1) * args.bsz]
            start = time.perf_counter()
            outs = engine.generate(
                batch_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_token_ids=(tokenizer.eos_token_id,),
                seed=args.seed,
            )
            wall_s = time.perf_counter() - start
            totals["gen_wall_s"] += wall_s
            for rec, out in zip(batch, outs):
                row = {
                    **rec,
                    "output": tokenizer.decode(out.token_ids, skip_special_tokens=True),
                    "prefill_tokens": out.num_prefill_tokens,
                    "output_tokens": out.num_output_tokens,
                    "num_compactions": out.num_compactions,
                    "finish_reason": out.finish_reason,
                }
                totals["prefill_tokens"] += row["prefill_tokens"]
                totals["output_tokens"] += row["output_tokens"]
                totals["num_compactions"] += row["num_compactions"]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                outputs.append(row)
            fout.flush()
            print(
                f"[gen] batch {b + 1}/{num_batches}: {wall_s:.1f}s, "
                f"output tokens {sum(out.num_output_tokens for out in outs)}",
                flush=True,
            )
    return outputs, totals


def main() -> None:
    args = parse_args()
    shard = parse_shard(args.shard)
    if args.max_new_tokens is None:
        args.max_new_tokens = DATASET_MAX_NEW_TOKENS[args.dataset]
    out_path = Path(args.output) if args.output else default_output(args, shard)

    if args.score_only:
        with out_path.open() as fin:
            records = [json.loads(line) for line in fin]
        if not records:
            raise SystemExit(f"no records in {out_path}")
        totals = {
            "prefill_tokens": sum(rec.get("prefill_tokens", 0) for rec in records),
            "output_tokens": sum(rec.get("output_tokens", 0) for rec in records),
            "num_compactions": sum(rec.get("num_compactions", 0) for rec in records),
            "gen_wall_s": None,
            "gpu": None,
        }
    else:
        if not args.model:
            raise SystemExit("--model is required unless --score-only")
        data_path = Path(args.data_path) if args.data_path else DATA_DIR / f"{args.dataset}.jsonl"
        records = load_records(data_path, args.max_samples, shard)
        if not records:
            raise SystemExit(f"no records in shard {args.shard} of {data_path}")
        records, totals = generate(args, records, out_path)

    score = score_records(records, args.dataset)

    mode_label = f"rkv(budget={args.budget}, buffer={args.buffer})" if args.mode == "rkv" else "fullkv"
    print("== eval_math summary ==")
    print(
        f"model={args.model} dataset={args.dataset} mode={mode_label} "
        f"shard={args.shard} samples={len(records)}"
    )
    if score is not None:
        shard_note = " (shard-local; merge shards + --score-only for the full split)" if shard[1] > 1 else ""
        print(f"acc={score['acc']}{shard_note} empty={score['empty_samples']} timeout={score['timeout_samples']}")
    mean_output_tokens = totals["output_tokens"] / len(records)
    print(
        f"tokens: prefill={totals['prefill_tokens']} output={totals['output_tokens']} "
        f"(mean {mean_output_tokens:.1f}/seq) compactions={totals['num_compactions']}"
    )
    output_tok_s = None
    if totals["gen_wall_s"]:
        output_tok_s = totals["output_tokens"] / totals["gen_wall_s"]
        print(f"generate wall {totals['gen_wall_s']:.1f}s -> {output_tok_s:.1f} output tok/s")
    print(f"records: {out_path}")

    summary = {
        "script": "eval_math",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": totals["gpu"],
        "model": args.model,
        "dataset": args.dataset,
        "mode": args.mode,
        "budget": args.budget if args.mode == "rkv" else None,
        "buffer": args.buffer if args.mode == "rkv" else None,
        "mix_lambda": args.mix_lambda if args.mode == "rkv" else None,
        "shard": args.shard,
        "num_samples": len(records),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "acc": score["acc"] if score else None,
        "empty_samples": score["empty_samples"] if score else None,
        "timeout_samples": score["timeout_samples"] if score else None,
        "prefill_tokens": totals["prefill_tokens"],
        "output_tokens": totals["output_tokens"],
        "mean_output_tokens": round(mean_output_tokens, 1),
        "num_compactions": totals["num_compactions"],
        "gen_wall_s": round(totals["gen_wall_s"], 2) if totals["gen_wall_s"] else None,
        "output_tok_s": round(output_tok_s, 2) if output_tok_s else None,
        "output": str(out_path),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
