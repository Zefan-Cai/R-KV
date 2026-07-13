#!/usr/bin/env python3
"""Small, resumable OpenAI-compatible smoke and AIME evaluator."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any], timeout: int = 1800) -> dict[str, Any]:
    """POST JSON with bounded retries for transient server failures."""
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(5):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
            if exc.code < 500 and exc.code != 429:
                raise last_error
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"request failed after retries: {last_error}")


def write_json(path: Path, value: Any) -> None:
    """Atomically write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    **extra: Any,
) -> dict[str, Any]:
    """Call the OpenAI chat-completions endpoint and retain latency metadata."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
    }
    payload.update(extra)
    started = time.perf_counter()
    response = request_json(f"{base_url.rstrip('/')}/chat/completions", payload)
    response["_client_latency_s"] = time.perf_counter() - started
    return response


def smoke(args: argparse.Namespace) -> None:
    """Exercise short chat, native tool calling, and a forced long decode."""
    short = chat(
        args.base_url,
        args.model,
        [{"role": "user", "content": "Return exactly the string RKV_SMOKE_OK and nothing else."}],
        32,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]
    tool = chat(
        args.base_url,
        args.model,
        [{"role": "user", "content": "Use the multiply tool to compute 37 times 41."}],
        128,
        tools=tools,
        tool_choice="auto",
    )
    long_payload = {
        "text": (
            "Write a numbered sequence forever. Each line must contain the next "
            "integer and a short explanation. Begin at 1."
        ),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.long_max_tokens,
            "ignore_eos": True,
        },
    }
    started = time.perf_counter()
    long_response = request_json(f"{args.base_url.removesuffix('/v1').rstrip('/')}/generate", long_payload)
    long_response["_client_latency_s"] = time.perf_counter() - started
    write_json(
        args.output,
        {
            "model": args.model,
            "short": short,
            "tool": tool,
            "forced_long": long_response,
            "forced_long_max_new_tokens": args.long_max_tokens,
        },
    )


def extract_aime_answer(text: str) -> str | None:
    """Extract the last integer answer from a boxed/final-answer response."""
    boxed = re.findall(r"\\boxed\s*\{\s*([0-9]{1,6})\s*\}", text)
    if boxed:
        return boxed[-1]
    final = re.findall(r"(?:final answer|answer is)\D{0,20}([0-9]{1,6})", text, flags=re.IGNORECASE)
    if final:
        return final[-1]
    numbers = re.findall(r"(?<![.\d])([0-9]{1,6})(?![.\d])", text)
    return numbers[-1] if numbers else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aime(args: argparse.Namespace) -> None:
    """Run a deterministic, resumable AIME slice."""
    samples = load_jsonl(args.data)
    if args.limit:
        samples = samples[: args.limit]
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        existing = {str(row["id"]): row for row in load_jsonl(args.output)}

    def run_one(sample: dict[str, Any]) -> dict[str, Any]:
        question = sample.get("problem") or sample.get("question")
        prompt = (
            "Solve the following AIME problem carefully. Show a complete step-by-step derivation, "
            "then end with the integer answer in the exact form \\boxed{N}.\n\n"
            f"Problem: {question}"
        )
        response = chat(
            args.base_url,
            args.model,
            [{"role": "user", "content": prompt}],
            args.max_tokens,
        )
        content = response["choices"][0]["message"].get("content") or ""
        prediction = extract_aime_answer(content)
        answer = str(sample["answer"]).strip()
        return {
            "id": sample["id"],
            "answer": answer,
            "prediction": prediction,
            "correct": prediction == answer,
            "response": response,
        }

    pending = [sample for sample in samples if str(sample["id"]) not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, sample): sample for sample in pending}
        with args.output.open("a", encoding="utf-8") as handle:
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                existing[str(row["id"])] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    selected = [existing[str(sample["id"])] for sample in samples if str(sample["id"]) in existing]
    correct = sum(bool(row["correct"]) for row in selected)
    usage = [row["response"].get("usage", {}) for row in selected]
    summary = {
        "count": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected) if selected else None,
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in usage),
        "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in usage),
        "model": args.model,
    }
    write_json(args.output.with_suffix(".summary.json"), summary)


def parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    root = argparse.ArgumentParser()
    root.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    root.add_argument("--model", default="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")
    sub = root.add_subparsers(dest="command", required=True)

    smoke_cmd = sub.add_parser("smoke")
    smoke_cmd.add_argument("--output", type=Path, required=True)
    smoke_cmd.add_argument("--long-max-tokens", type=int, default=4608)
    smoke_cmd.set_defaults(func=smoke)

    aime_cmd = sub.add_parser("aime")
    aime_cmd.add_argument("--data", type=Path, required=True)
    aime_cmd.add_argument("--output", type=Path, required=True)
    aime_cmd.add_argument("--limit", type=int, default=0)
    aime_cmd.add_argument("--concurrency", type=int, default=2)
    aime_cmd.add_argument("--max-tokens", type=int, default=8192)
    aime_cmd.set_defaults(func=aime)
    return root


if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.func(parsed)
