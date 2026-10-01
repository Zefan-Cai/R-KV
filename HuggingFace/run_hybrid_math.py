"""Evaluate full-attention-only R-KV on native hybrid Transformers models."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from rkv.hybrid import HybridRKVAdapter


ROOT = Path(__file__).resolve().parent
PROMPT = (
    "You are given a math problem.\n\nProblem: {question}\n\n"
    "Solve the problem step by step. Provide the final answer in the format: "
    "Final answer: \\boxed{{}}"
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def final_response(output, model_type, thinking):
    """Keep final text only; an unfinished reasoning channel has no answer."""
    model_type = (model_type or "").lower()
    is_qwen = "qwen3_5" in model_type or "qwen3.5" in model_type or "qwen3.6" in model_type
    if "</think>" in output:
        output = output.rsplit("</think>", 1)[1]
    elif "<think>" in output or (is_qwen and thinking):
        # Qwen's generation prompt already opens <think>, so the generated
        # continuation does not necessarily contain the opening marker.
        return "", False
    thought_marker = "<|channel>thought"
    while thought_marker in output:
        start = output.index(thought_marker)
        end = output.find("<channel|>", start + len(thought_marker))
        if end == -1:
            return "", False
        output = output[:start] + output[end + len("<channel|>"):]
    output = output.replace("<|channel>final", "").replace("<channel|>", "")
    for stop in ("<|im_end|>", "<|endoftext|>", "<eos>", "<end_of_turn>", "<turn|>"):
        output = output.split(stop, 1)[0]
    return output.strip(), True


def gsm8k_exact_match(prediction, reference):
    """Exact normalized numeric equality, without tolerance or percent scaling."""
    try:
        pred = Decimal(prediction.replace(",", "").strip())
        gold = Decimal(reference.replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return False
    return pred.is_finite() and gold.is_finite() and pred == gold


def score(records, dataset, model_type=None):
    # MATH uses the repository's symbolic grader after final-channel extraction.
    sys.path.insert(0, str(ROOT / "evaluation"))
    from evaluate import evaluate
    from parser import extract_answer, parse_ground_truth

    samples = []
    for record in records:
        final, complete = final_response(
            record["output"], model_type or record.get("model_type") or record.get("model"),
            record.get("thinking", False),
        )
        # The generic parser can return numbers from incomplete \boxed{... or
        # ordinary reasoning. Require a complete box or explicit answer phrase.
        closed_box = True
        if "boxed{" in final:
            tail = final.rsplit("boxed{", 1)[1]
            depth = 1
            for char in tail:
                depth += (char == "{") - (char == "}")
                if depth == 0:
                    break
            closed_box = depth == 0
        pred = extract_answer(final, dataset, use_last_number=False) if complete and closed_box else ""
        samples.append({**record, "final_response": final, "reasoning_complete": complete,
                        "pred": [pred]})
    if dataset == "gsm8k":
        for sample in samples:
            sample["gt_cot"], sample["gt"] = parse_ground_truth(sample, dataset)
            sample["score"] = [gsm8k_exact_match(sample["pred"][0], sample["gt"])]
        result = {
            "num_samples": len(samples), "num_scores": len(samples), "timeout_samples": 0,
            "empty_samples": sum(not s["pred"][0] for s in samples),
            "acc": round(100 * sum(s["score"][0] for s in samples) / len(samples), 1),
        }
        scored = samples
    else:
        scored, result = evaluate(data_name=dataset, prompt_type="cot", samples=samples)
    result["scoring_protocol"] = (
        "final channel; complete box or explicit answer phrase; no last-number fallback; "
        + ("exact numeric equality" if dataset == "gsm8k" else "repository symbolic equivalence")
    )
    result["unfinished_reasoning"] = sum(not s["reasoning_complete"] for s in samples)
    return scored, result


def inputs_for(tokenizer, record, dataset, thinking):
    question = record["question" if dataset == "gsm8k" else "problem"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(question=question)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")


def validation(model, tokenizer, record, args):
    inputs = inputs_for(tokenizer, record, "gsm8k", args.thinking)
    kwargs = dict(max_new_tokens=32, min_new_tokens=32, do_sample=False)
    with torch.inference_mode():
        native = model.generate(**inputs, **kwargs)
        adapter = HybridRKVAdapter(model, budget=None)
        try:
            disabled = model.generate(**inputs, **kwargs)
            parity = torch.equal(native, disabled)
            disabled_stats = adapter.stats()
        finally:
            adapter.remove()
        if not parity:
            raise AssertionError("Disabled adapter changes native generated tokens")
        adapter = HybridRKVAdapter(
            model, budget=32, compression_interval=16,
            window_size=args.window_size, mix_lambda=args.mix_lambda,
            retain_ratio=args.retain_ratio,
        )
        try:
            model.generate(**inputs, max_new_tokens=48, min_new_tokens=48, do_sample=False)
            compressed_stats = adapter.stats()
        finally:
            adapter.remove()
    result = {"native_disabled_token_parity": parity,
              "disabled_stats": disabled_stats, "compressed_stats": compressed_stats}
    if compressed_stats["compression_count"] == 0:
        raise AssertionError("Forced compression validation did not evict any KV")
    if any(compressed_stats["layer_types"][int(i)] != "full_attention"
           for i in compressed_stats["layers"]):
        raise AssertionError("Adapter touched a non-full attention layer")
    write_json(Path(args.output_dir) / "validation.json", result)
    print("VALIDATION " + json.dumps(result), flush=True)
    return result


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(args.model)
    text_config = config.get_text_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=text_config, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    metadata = {
        "args": vars(args), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config": text_config.to_dict(),
        "protocol": "BF16; native SDPA; batch=1; greedy; identical seeded sample IDs; "
                    f"full-attention-only decode compression every {args.compression_interval} steps; "
                    "native SWA/linear state and absolute positions preserved",
    }
    write_json(out_dir / "metadata.json", metadata)
    gsm_record = json.loads((ROOT / "data/gsm8k.jsonl").read_text().splitlines()[0])
    if args.validate or args.validation_only:
        validation(model, tokenizer, gsm_record, args)
    if args.validation_only:
        return
    summaries = []
    for mode in args.modes:
        adapter = None if mode == "fullkv" else HybridRKVAdapter(
            model, budget=int(mode), compression_interval=args.compression_interval,
            window_size=args.window_size, mix_lambda=args.mix_lambda,
            retain_ratio=args.retain_ratio,
        )
        try:
            # The same unscored warmup for every arm; excludes initial kernels.
            warmup = inputs_for(tokenizer, gsm_record, "gsm8k", args.thinking)
            with torch.inference_mode():
                model.generate(**warmup, max_new_tokens=8, do_sample=False)
            for dataset in args.datasets:
                path = ROOT / "data" / (dataset + ".jsonl")
                records = [{"idx": i, **json.loads(line)}
                           for i, line in enumerate(path.read_text().splitlines())]
                if args.sample_limit and args.sample_limit < len(records):
                    ids = sorted(random.Random(args.seed).sample(
                        range(len(records)), args.sample_limit))
                    records = [records[i] for i in ids]
                records = records[args.shard_index::args.num_shards]
                max_tokens = args.max_new_tokens or (8192 if dataset == "gsm8k" else 16384)
                tag = "fullkv" if mode == "fullkv" else "rkv" + mode
                suffix = f"_shard{args.shard_index}of{args.num_shards}"
                output_path = out_dir / (dataset + "_" + tag + suffix + ".jsonl")
                if output_path.exists():
                    outputs = [json.loads(line) for line in output_path.read_text().splitlines()]
                else:
                    outputs = []
                completed = {r["idx"] for r in outputs}
                with output_path.open("a") as stream:
                    for record in records:
                        if record["idx"] in completed:
                            continue
                        inputs = inputs_for(tokenizer, record, dataset, args.thinking)
                        if adapter is not None:
                            adapter.reset_stats()
                        # Reseed per example so a resumed run uses the same state.
                        torch.manual_seed(args.seed + record["idx"])
                        baseline_memory = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        with torch.inference_mode():
                            generated = model.generate(
                                **inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - start
                        generated_ids = generated[0, inputs.input_ids.shape[1]:]
                        result = {
                            **record, "model": args.model, "mode": tag,
                            "model_type": text_config.model_type,
                            "thinking": args.thinking,
                            "output": tokenizer.decode(generated_ids, skip_special_tokens=False),
                            "prefill_tokens": inputs.input_ids.shape[1],
                            "output_tokens": generated_ids.numel(),
                            "generation_seconds": wall,
                            "tokens_per_second": generated_ids.numel() / wall,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "baseline_allocated_gib": baseline_memory / 2**30,
                            "incremental_peak_gib": (torch.cuda.max_memory_allocated() - baseline_memory) / 2**30,
                            "truncated": generated_ids.numel() == max_tokens,
                            "adapter_stats": adapter.stats() if adapter is not None else None,
                        }
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stream.flush()
                        outputs.append(result)
                        del generated, generated_ids, inputs
                        print(json.dumps({"dataset": dataset, "mode": tag,
                                          "completed": len(outputs), "total": len(records),
                                          "idx": result["idx"], "tokens": result["output_tokens"],
                                          "seconds": round(wall, 2)}), flush=True)
                scored, accuracy = score(outputs, dataset, text_config.model_type)
                with output_path.open("w") as stream:
                    for record in scored:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary = {
                    "model": args.model, "dataset": dataset, "mode": tag,
                    "thinking": args.thinking, "score": accuracy,
                    "num_truncated": sum(r["truncated"] for r in outputs),
                    "output_tokens": sum(r["output_tokens"] for r in outputs),
                    "generation_seconds": sum(r["generation_seconds"] for r in outputs),
                    "peak_allocated_gib": max(r["peak_allocated_gib"] for r in outputs),
                    "incremental_peak_gib": max(r["incremental_peak_gib"] for r in outputs),
                    "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sample_ids": [r["idx"] for r in outputs],
                }
                summary["tokens_per_second"] = summary["output_tokens"] / summary["generation_seconds"]
                write_json(output_path.with_suffix(".summary.json"), summary)
                summaries.append(summary)
                print("RESULT " + json.dumps(summary), flush=True)
        finally:
            if adapter is not None:
                adapter.remove()
    write_json(out_dir / f"summary_shard{args.shard_index}of{args.num_shards}.json", summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", choices=["gsm8k", "math"], default=["gsm8k", "math"])
    parser.add_argument("--modes", nargs="+", default=["fullkv", "128", "512", "1024"])
    parser.add_argument("--sample-limit", type=int, default=100, help="0 = full benchmark")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression-interval", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mix-lambda", type=float, default=0.1)
    parser.add_argument("--retain-ratio", type=float, default=0.2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard index/count")
    main(args)
