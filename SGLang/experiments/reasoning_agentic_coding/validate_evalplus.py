#!/usr/bin/env python3
"""Validate that an EvalPlus generation and score artifact is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL artifact and reject blank or malformed records."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            rows.append(json.loads(line))
    return rows


def validate(root: Path, expected_tasks: int = 164) -> dict[str, Any]:
    """Return a summary or raise when samples/scores are incomplete."""
    sample_paths = sorted(
        path
        for path in root.glob("**/*.jsonl")
        if not path.name.endswith(".raw.jsonl")
    )
    if len(sample_paths) != 1:
        raise ValueError(f"expected one sanitized sample JSONL, found {sample_paths}")

    sample_path = sample_paths[0]
    rows = load_jsonl(sample_path)
    task_ids = [str(row["task_id"]) for row in rows]
    unique_task_ids = set(task_ids)
    if len(rows) != expected_tasks or len(unique_task_ids) != expected_tasks:
        raise ValueError(
            "incomplete or duplicate samples: "
            f"rows={len(rows)} unique_tasks={len(unique_task_ids)} "
            f"expected={expected_tasks}"
        )

    result_path = sample_path.with_name(
        f"{sample_path.name.removesuffix('.jsonl')}_eval_results.json"
    )
    if not result_path.is_file():
        raise ValueError(f"missing EvalPlus result: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    evaluated = result.get("eval")
    if not isinstance(evaluated, dict):
        raise ValueError("EvalPlus result is missing the eval mapping")
    evaluated_ids = set(map(str, evaluated))
    if len(evaluated_ids) != expected_tasks or evaluated_ids != unique_task_ids:
        raise ValueError(
            "score coverage mismatch: "
            f"evaluated={len(evaluated_ids)} samples={len(unique_task_ids)} "
            f"expected={expected_tasks}"
        )

    entries = [
        entry
        for task_entries in evaluated.values()
        if isinstance(task_entries, list)
        for entry in task_entries
        if isinstance(entry, dict)
    ]
    if len(entries) != expected_tasks:
        raise ValueError(
            "unexpected EvalPlus result cardinality: "
            f"entries={len(entries)} expected={expected_tasks}"
        )
    if entries and all(
        entry.get("base_status") == "timeout"
        and entry.get("plus_status") == "timeout"
        for entry in entries
    ):
        raise ValueError(
            "all EvalPlus tasks timed out; evaluator infrastructure failure"
        )

    base_passes = sum(entry.get("base_status") == "pass" for entry in entries)
    # EvalPlus reports HumanEval+ as passing both the original base tests and
    # the extra tests. `plus_status` alone only represents the extra-test side
    # and can be `pass` even when `base_status` is `fail`.
    plus_passes = sum(
        entry.get("base_status") == "pass"
        and entry.get("plus_status") == "pass"
        for entry in entries
    )

    return {
        "valid": True,
        "expected_tasks": expected_tasks,
        "sample_count": len(rows),
        "unique_task_count": len(unique_task_ids),
        "evaluated_task_count": len(evaluated_ids),
        "base_passes": base_passes,
        "plus_passes": plus_passes,
        "base_pass_at_1": base_passes / expected_tasks,
        "plus_pass_at_1": plus_passes / expected_tasks,
        "samples": str(sample_path),
        "results": str(result_path),
    }


def main() -> None:
    """Parse CLI arguments, validate artifacts, and write a summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=164)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = validate(args.root, args.expected_tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
