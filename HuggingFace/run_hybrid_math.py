"""Evaluate full-attention-only R-KV on native hybrid Transformers models."""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    default=lambda value: sorted(value) if isinstance(value, set) else str(value)) + "\n")
    temporary.replace(path)


def load_dataset(path, dataset, sample_limit, seed, shard_index, num_shards, require_full):
    records = []
    for idx, line in enumerate(path.read_text().splitlines()):
        record = json.loads(line)
        question = record["question" if dataset == "gsm8k" else "problem"]
        if not question or not record.get("answer" if dataset == "gsm8k" else "solution"):
            raise ValueError(f"Missing question/reference in {dataset} example {idx}")
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        records.append({**record, "idx": idx, "example_sha256": fingerprint})
    if require_full and (sample_limit != 0 or len(records) != {"gsm8k": 1319, "math": 5000}[dataset]):
        raise ValueError(f"Full {dataset} requires all test examples and sample-limit=0; found {len(records)}")
    if require_full and len({r["example_sha256"] for r in records}) != len(records):
        raise ValueError(f"Duplicate source records in full {dataset}")
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    total = len(records)
    if sample_limit and sample_limit < total:
        ids = sorted(random.Random(seed).sample(range(total), sample_limit))
        records = [records[i] for i in ids]
    records = records[shard_index::num_shards]
    if not records:
        raise ValueError("Empty dataset shard")
    manifest = {"dataset": dataset, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dataset_count": total, "expected_shard_count": len(records),
                "sample_ids": [r["idx"] for r in records], "full_benchmark": require_full}
    return records, manifest


def resume_outputs(path, records, protocol):
    if not path.exists():
        return []
    expected = {record["idx"]: record for record in records}
    outputs, seen = [], set()
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    valid_bytes = 0
    for position, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if position != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt completed record in {path}")
            path.with_suffix(path.suffix + ".partial").write_bytes(line)
            with path.open("r+b") as stream:
                stream.truncate(valid_bytes)
            break
        idx = record["idx"]
        if idx in seen or idx not in expected or record.get("example_sha256") != expected[idx]["example_sha256"]:
            raise ValueError(f"Duplicate, foreign, or changed example {idx} in {path}")
        required = {"output", "prefill_tokens", "output_tokens", "generation_seconds", "tokens_per_second",
                    "peak_allocated_gib", "baseline_allocated_gib", "incremental_peak_gib", "truncated", "adapter_stats"}
        if not required <= record.keys() or any(record.get(key) != value for key, value in protocol.items()):
            raise ValueError(f"Wrong generation protocol or incomplete example {idx} in {path}")
        outputs.append(record)
        seen.add(idx)
        valid_bytes += len(line)
    if outputs and lines and not lines[-1].endswith(b"\n") and valid_bytes == len(raw):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return outputs


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
    def number(text):
        text = text.replace(",", "").strip()
        clock = re.fullmatch(r"(\d{1,2})(:00)?\s*(AM|PM)?", text, flags=re.IGNORECASE)
        if clock and (clock.group(2) or clock.group(3)):
            hour = int(clock.group(1))
            lower, upper = (1, 12) if clock.group(3) else (0, 23)
            if lower <= hour <= upper:
                text = clock.group(1)
        numeric = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        latex_fraction = re.fullmatch(
            rf"([+-]?)\\(?:d|t)?frac\s*\{{\s*({numeric})\s*\}}\s*\{{\s*({numeric})\s*\}}", text
        )
        fraction = re.fullmatch(rf"({numeric})\s*/\s*({numeric})", text)
        if latex_fraction:
            sign, numerator, denominator = latex_fraction.groups()
            return (-1 if sign == "-" else 1) * Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        if fraction:
            numerator, denominator = fraction.groups()
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(text))
    try:
        pred, gold = number(prediction), number(reference)
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return False
    return pred == gold


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
        parse_text = re.sub(r"(?i)final\s+answer\s*(?::|is)\s*", "final answer is ", final)
        parse_text = re.sub(r"boxed\s+\{", "boxed{", parse_text)
        # An unfinished answer block must not fall through to last-number
        # extraction. Completed responses can use the repository fallback below.
        closed_box = True
        if "boxed{" in parse_text:
            tail = parse_text.rsplit("boxed{", 1)[1]
            depth = 1
            for char in tail:
                depth += (char == "{") - (char == "}")
                if depth == 0:
                    break
            closed_box = depth == 0
        format_pred = extract_answer(parse_text, dataset, use_last_number=False) if complete and closed_box else ""
        pred = format_pred
        fallback = complete and closed_box and not record.get("truncated", False) and not format_pred
        if fallback:
            # Standard repository extraction accepts an unboxed final number.
            # Only completed responses can use it; an intermediate number from
            # capped generation or an open reasoning/answer block receives no credit.
            pred = extract_answer(parse_text, dataset, use_last_number=True)
        samples.append({**record, "final_response": final, "reasoning_complete": complete,
                        "format_pred": format_pred, "answer_format_compliant": bool(format_pred),
                        "answer_extraction": "last_number_finished_response" if fallback and pred
                        else "explicit_answer" if format_pred else "none", "pred": [pred]})
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
        try:
            scored, result = evaluate(data_name=dataset, prompt_type="cot", samples=samples)
        except SystemExit as error:
            # The legacy grader calls exit() on worker failures. A zero exit
            # must never certify that a full experiment completed successfully.
            raise RuntimeError("MATH grading exited before completion") from error
    result["scoring_protocol"] = (
        "final channel; explicit answer or repository last-number extraction for completed, "
        "non-truncated responses only; open reasoning/answer blocks rejected; "
        + ("exact numeric equality" if dataset == "gsm8k" else "repository symbolic equivalence")
    )
    result["unfinished_reasoning"] = sum(not s["reasoning_complete"] for s in samples)
    result["answer_format_failures"] = sum(not s["answer_format_compliant"] for s in samples)
    result["fallback_answers"] = sum(s["answer_extraction"] == "last_number_finished_response" for s in samples)
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
    data = {dataset: load_dataset(Path(args.data_dir) / (dataset + ".jsonl"), dataset,
                                 args.sample_limit, args.seed, args.shard_index, args.num_shards,
                                 args.require_full_benchmarks) for dataset in args.datasets}
    generation_args = {key: value for key, value in vars(args).items()
                       if key not in {"output_dir", "data_dir", "validate", "validation_only"}}
    experiment = {"args": generation_args, "datasets": {key: value[1] for key, value in data.items()},
                  "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "torch": torch.__version__, "transformers": transformers.__version__}
    experiment_path = out_dir / "experiment.json"
    if experiment_path.exists() and json.loads(experiment_path.read_text()) != experiment:
        raise ValueError("Resume experiment differs in data, generation settings, code, or environment")
    if not experiment_path.exists() and list(out_dir.glob("*_shard*.jsonl")):
        raise ValueError("Existing outputs lack the verified resume manifest; use a fresh output directory")
    write_json(experiment_path, experiment)
    torch.set_num_threads(8)
    # cuDNN SDPA rebuilds an execution plan for each growing decode length on
    # Torch 2.11. Keep native Flash/Efficient/Math SDPA backends for all arms.
    torch.backends.cuda.enable_cudnn_sdp(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(args.model)
    text_config = config.get_text_config()
    loading_kwargs = {}
    if text_config.model_type == "gemma4_text":
        loading_kwargs["key_mapping"] = {r"^model\.language_model\.": "model."}
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.model, config=text_config, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
        output_loading_info=True, **loading_kwargs,
    )
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise RuntimeError("Incomplete text weights: " + str(loading_info))
    model.eval()
    metadata = {
        "args": vars(args), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config": text_config.to_dict(),
        "loading_info": loading_info,
        "sdpa_backends": {
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "protocol": "BF16; native SDPA; batch=1; greedy; identical seeded sample IDs; "
                    f"full-attention-only decode compression every {args.compression_interval} steps; "
                    "native SWA/linear state and absolute positions preserved",
    }
    metadata["datasets"] = experiment["datasets"]
    write_json(out_dir / "metadata.json", metadata)
    gsm_record = json.loads((Path(args.data_dir) / "gsm8k.jsonl").read_text().splitlines()[0])
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
                records, data_manifest = data[dataset]
                max_tokens = args.max_new_tokens or (8192 if dataset == "gsm8k" else 16384)
                tag = "fullkv" if mode == "fullkv" else "rkv" + mode
                suffix = f"_shard{args.shard_index}of{args.num_shards}"
                output_path = out_dir / (dataset + "_" + tag + suffix + ".jsonl")
                outputs = resume_outputs(output_path, records,
                                         {"model": args.model, "mode": tag, "thinking": args.thinking})
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
                        os.fsync(stream.fileno())
                        outputs.append(result)
                        del generated, generated_ids, inputs
                        print(json.dumps({"dataset": dataset, "mode": tag,
                                          "completed": len(outputs), "total": len(records),
                                          "idx": result["idx"], "tokens": result["output_tokens"],
                                          "seconds": round(wall, 2)}), flush=True)
                if {r["idx"] for r in outputs} != set(data_manifest["sample_ids"]):
                    raise AssertionError("Dataset shard is incomplete")
                outputs.sort(key=lambda record: record["idx"])
                scored, accuracy = score(outputs, dataset, text_config.model_type)
                scored_path = output_path.with_suffix(".scored.tmp")
                with scored_path.open("w") as stream:
                    for record in scored:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                scored_path.replace(output_path)
                summary = {
                    "model": args.model, "dataset": dataset, "mode": tag,
                    "thinking": args.thinking, "score": accuracy,
                    "num_correct": sum(bool(r["score"][0]) for r in scored),
                    "num_truncated": sum(r["truncated"] for r in outputs),
                    "output_tokens": sum(r["output_tokens"] for r in outputs),
                    "generation_seconds": sum(r["generation_seconds"] for r in outputs),
                    "peak_allocated_gib": max(r["peak_allocated_gib"] for r in outputs),
                    "incremental_peak_gib": max(r["incremental_peak_gib"] for r in outputs),
                    **data_manifest, "complete": True,
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
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--require-full-benchmarks", action="store_true",
                        help="Reject any dataset other than GSM8K test 1319 / MATH test 5000")
    parser.add_argument("--sample-limit", type=int, default=0, help="0 = all examples in each input file")
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
