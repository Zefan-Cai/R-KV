#!/usr/bin/env python3
"""Extract stable R-KV evidence counters from an SGLang server log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    """Summarize compactions, freed slots, readiness, and hard failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    freed = [int(value) for value in re.findall(r"freed(?:_slots)?[=: ]+(\d+)", text)]
    summary = {
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
