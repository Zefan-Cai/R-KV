#!/usr/bin/env bash
# Fetch the GSM8K-style few-shot eval set (the same set the SGLang port uses)
# into ./data/gsm8k_fewshot.jsonl.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"     # the R-KV repo root (has origin/dev)
OUT="$DIR/data/gsm8k_fewshot.jsonl"

mkdir -p "$DIR/data"
cd "$REPO"
git fetch origin dev
git show origin/dev:evaluation/data/test.jsonl > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
