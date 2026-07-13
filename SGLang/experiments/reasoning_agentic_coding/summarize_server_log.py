#!/usr/bin/env python3
"""Extract stable R-KV evidence counters from an SGLang server log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def summarize(text: str) -> dict[str, Any]:
    """Summarize compactions, freed slots, readiness, and hard failures."""
    freed = [int(value) for value in re.findall(r"freed(?:_slots)?[=: ]+(\d+)", text)]
    return {
        "ready": "The server is fired up and ready to roll" in text,
        "decode_compactions": len(re.findall(r"R-KV compacted", text)),
        "prefill_compactions": len(
            re.findall(r"R-KV[- ]prefill.*compact", text, flags=re.IGNORECASE)
        ),
        "freed_slots_observed": sum(freed),
        "fused_kernel_adopted": "fused-redundancy gate: OK" in text,
        "tp_mismatch": bool(
            re.search(
                r"(?:TP.*diverg|mismatch.*kept|kept.*(?:mismatch|diverg))",
                text,
                flags=re.IGNORECASE,
            )
        ),
        "oom_count": len(re.findall(r"out of memory|CUDA OOM", text, flags=re.IGNORECASE)),
        "traceback_count": text.count("Traceback (most recent call last)"),
    }


def validate(summary: dict[str, Any], arm: str) -> None:
    """Require healthy server evidence and arm-specific R-KV activity."""
    failures: list[str] = []
    if not summary["ready"]:
        failures.append("server readiness marker missing")
    if summary["tp_mismatch"]:
        failures.append("TP mismatch/divergence detected")
    if summary["oom_count"]:
        failures.append(f"OOM count={summary['oom_count']}")
    if summary["traceback_count"]:
        failures.append(f"traceback count={summary['traceback_count']}")
    if arm in {"d-4k", "d-8k"} and summary["decode_compactions"] <= 0:
        failures.append("decode R-KV arm produced no compaction")
    if arm in {"p-4k", "p-2k", "p-o4k"} and summary["prefill_compactions"] <= 0:
        failures.append("prefill R-KV arm produced no compaction")
    if failures:
        raise ValueError("invalid server evidence: " + "; ".join(failures))


def main() -> None:
    """Write a server summary and optionally enforce pilot acceptance gates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-arm")
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    summary = summarize(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.require_arm:
        validate(summary, args.require_arm)


if __name__ == "__main__":
    main()
