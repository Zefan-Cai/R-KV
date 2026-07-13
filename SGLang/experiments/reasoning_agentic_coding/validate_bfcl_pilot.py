#!/usr/bin/env python3
"""Reject incomplete or error-contaminated BFCL pilot artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read BFCL's newline-delimited JSON artifact format."""
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        entries.append(value)
    return entries


def find_one(root: Path, name: str) -> Path:
    """Return the unique recursively matched BFCL artifact."""
    matches = list(root.rglob(name)) if root.is_dir() else []
    if len(matches) != 1:
        raise ValueError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def validate(root: Path) -> dict[str, Any]:
    """Validate manifest coverage, inference health, and score counts."""
    manifest_path = root / "test_case_ids_to_generate.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("BFCL manifest must be a non-empty category mapping")

    summary: dict[str, Any] = {"valid": True, "categories": {}}
    for category, raw_ids in manifest.items():
        if not isinstance(category, str) or not isinstance(raw_ids, list):
            raise ValueError("BFCL manifest categories must map to ID lists")
        expected = [str(value) for value in raw_ids]
        expected_set = set(expected)
        if len(expected) != len(expected_set):
            raise ValueError(f"duplicate manifest IDs in {category}")

        result_path = find_one(root / "result", f"BFCL_v4_{category}_result.json")
        result_entries = read_jsonl(result_path)
        result_ids = [str(entry.get("id")) for entry in result_entries]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError(f"duplicate generated result IDs in {category}")
        if set(result_ids) != expected_set:
            missing = sorted(expected_set - set(result_ids))
            unexpected = sorted(set(result_ids) - expected_set)
            raise ValueError(
                f"BFCL result coverage mismatch for {category}: "
                f"missing={missing} unexpected={unexpected}"
            )

        inference_errors = [
            entry["id"]
            for entry in result_entries
            if "error during inference:" in json.dumps(entry).lower()
        ]
        if inference_errors:
            raise ValueError(
                f"BFCL inference errors in {category}: {sorted(inference_errors)}"
            )

        score_path = find_one(root / "score", f"BFCL_v4_{category}_score.json")
        score_entries = read_jsonl(score_path)
        if not score_entries:
            raise ValueError(f"empty BFCL score file for {category}")
        header = score_entries[0]
        target_count = sum("prereq" not in entry_id for entry_id in expected)
        total_count = header.get("total_count")
        correct_count = header.get("correct_count")
        if total_count != target_count:
            raise ValueError(
                f"BFCL score count mismatch for {category}: "
                f"expected={target_count} actual={total_count}"
            )
        if not isinstance(correct_count, int) or not 0 <= correct_count <= target_count:
            raise ValueError(f"invalid BFCL correct_count for {category}: {correct_count}")

        summary["categories"][category] = {
            "generated_count": len(result_entries),
            "scored_count": total_count,
            "correct_count": correct_count,
            "accuracy": header.get("accuracy"),
        }
    return summary


def main() -> None:
    """Validate one BFCL project root and persist a compact audit summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = validate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
