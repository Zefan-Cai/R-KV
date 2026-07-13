#!/usr/bin/env python3
"""Create a deterministic BFCL V4 long-context/memory pilot manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bfcl_eval.utils import load_dataset_entry


def numeric_id(entry: dict) -> int:
    """Return BFCL's leading numeric case id for stable corpus ordering."""
    match = re.search(r"_(\d+)(?:-|$)", str(entry["id"]))
    if match is None:
        raise ValueError(f"cannot parse numeric BFCL id: {entry['id']}")
    return int(match.group(1))


def ids(category: str, limit: int) -> list[str]:
    """Select evenly across a category and include the dependency closure."""
    entries = load_dataset_entry(category, include_prereq=True)
    by_id = {str(entry["id"]): entry for entry in entries}
    targets = sorted(
        (entry for entry in entries if "prereq" not in str(entry["id"])),
        key=numeric_id,
    )
    limit = min(limit, len(targets))
    if limit <= 0:
        return []
    selected = [
        targets[((2 * index + 1) * len(targets)) // (2 * limit)]
        for index in range(limit)
    ]

    required: set[str] = set()
    pending = [str(entry["id"]) for entry in selected]
    while pending:
        entry_id = pending.pop()
        if entry_id in required:
            continue
        required.add(entry_id)
        pending.extend(str(value) for value in by_id[entry_id].get("depends_on", []))

    return [str(entry["id"]) for entry in entries if str(entry["id"]) in required]


def main() -> None:
    """Write the manifest consumed by ``bfcl generate --run-ids``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-context", type=int, default=10)
    parser.add_argument("--memory-each", type=int, default=4)
    args = parser.parse_args()
    manifest = {
        "multi_turn_long_context": ids("multi_turn_long_context", args.long_context),
        "memory_kv": ids("memory_kv", args.memory_each),
        "memory_vector": ids("memory_vector", args.memory_each),
        "memory_rec_sum": ids("memory_rec_sum", args.memory_each),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
