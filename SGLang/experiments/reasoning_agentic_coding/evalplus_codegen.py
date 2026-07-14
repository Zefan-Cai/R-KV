#!/usr/bin/env python3
"""Concurrent, resumable EvalPlus code generation via an OpenAI endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, Callable

from api_eval import request_json

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
SYSTEM_MESSAGE = "You are a helpful assistant good at coding."


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    """Load completed records and repair only a malformed trailing write."""
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            raise ValueError(f"blank JSONL record at {path}:{index + 1}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            print(f"repairing malformed trailing JSONL record at {path}:{index + 1}")
            tmp_path = path.with_name(f".{path.name}.repair-{os.getpid()}")
            tmp_path.write_text("".join(valid_lines), encoding="utf-8")
            os.replace(tmp_path, path)
            break
        rows.append(row)
        valid_lines.append(line if line.endswith("\n") else f"{line}\n")

    existing: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in existing:
            raise ValueError(f"duplicate task_id in {path}: {task_id}")
        existing[task_id] = row
    if content and not content.endswith("\n"):
        write_rows(path, rows)
    return existing


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file with complete records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.rewrite-{os.getpid()}")
    tmp_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def request_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    """Match EvalPlus 0.3.1's deterministic OpenAI request payload."""
    return request_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "n": 1,
            "top_p": 0.95,
        },
    )


def generate(
    *,
    base_url: str,
    model: str,
    output: Path,
    concurrency: int,
    max_tokens: int,
    dataset: dict[str, dict[str, Any]],
    sanitizer: Callable[..., str],
) -> dict[str, int]:
    """Resume a deterministic EvalPlus sample file with concurrent requests."""
    raw_output = output.with_name(output.name.replace(".jsonl", ".raw.jsonl"))
    existing = load_existing(output)
    raw_existing = load_existing(raw_output)
    orphan_raw_ids = set(raw_existing) - set(existing)
    if orphan_raw_ids:
        print(f"dropping uncommitted raw records: {sorted(orphan_raw_ids)}")
        write_rows(
            raw_output,
            [row for task_id, row in raw_existing.items() if task_id in existing],
        )
        raw_existing = {task_id: row for task_id, row in raw_existing.items() if task_id in existing}
    missing_raw_ids = set(existing) - set(raw_existing)
    if missing_raw_ids:
        print(f"warning: committed samples lack legacy raw records: {sorted(missing_raw_ids)}")

    pending = [(task_id, task) for task_id, task in dataset.items() if task_id not in existing]
    output.parent.mkdir(parents=True, exist_ok=True)

    def run_one(item: tuple[str, dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
        task_id, task = item
        prompt = str(task["prompt"]).strip() + "\n"
        user_message = f"{INSTRUCTION_PREFIX}\n```python\n{prompt.strip()}\n```"
        response = request_completion(
            base_url,
            model,
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            max_tokens,
        )
        content = response["choices"][0]["message"].get("content") or ""
        sanitized = sanitizer(content, entrypoint=task["entry_point"])
        return (
            {"task_id": task_id, "solution": sanitized},
            {"task_id": task_id, "solution": content},
        )

    with (
        output.open("a", encoding="utf-8") as sample_handle,
        raw_output.open("a", encoding="utf-8") as raw_handle,
        concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):
        futures = {pool.submit(run_one, item): item[0] for item in pending}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            sample, raw = future.result()
            # Raw is written first; the sanitized sample is the commit marker.
            # A reclaim between the writes leaves an orphan raw row that the
            # next invocation removes before regenerating the task.
            raw_handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
            raw_handle.flush()
            sample_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            sample_handle.flush()
            print(
                f"EvalPlus codegen {sample['task_id']} "
                f"({len(existing) + completed}/{len(dataset)})",
                flush=True,
            )

    return {
        "existing": len(existing),
        "generated": len(pending),
        "legacy_missing_raw": len(missing_raw_ids),
        "total": len(dataset),
    }


def main() -> None:
    """Run concurrent HumanEval+ code generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()

    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize

    summary = generate(
        base_url=args.base_url,
        model=args.model,
        output=args.output,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        dataset=get_human_eval_plus(),
        sanitizer=sanitize,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
