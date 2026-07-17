#!/usr/bin/env bash
# Verify the bundled eval dataset is present.
#
# The GSM8K-style few-shot set (1319 items, format: {"request": {...}, "answer":
# "... #### N"}) is committed at benchmark/data/gsm8k_fewshot.jsonl, so the
# benchmark is fully reproducible with no external download. It is the same file
# the SGLang port ships (SGLang/benchmark/data/gsm8k_fewshot.jsonl).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
F="$DIR/data/gsm8k_fewshot.jsonl"

if [[ -f "$F" ]]; then
  echo "OK: dataset present -> $F ($(wc -l < "$F" | tr -d ' ') items)"
else
  echo "ERROR: dataset missing at $F" >&2
  echo "       It ships with this repo; restore it from git if deleted." >&2
  exit 1
fi
