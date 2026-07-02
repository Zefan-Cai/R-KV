#!/usr/bin/env bash
# Launch an SGLang server for the R-KV benchmark.
#
# Usage:
#   ./launch_server.sh baseline               # R-KV OFF, eager config (fair compare)
#   ./launch_server.sh rkv 512                 # R-KV ON,  budget=512 (256/512/1024 ...)
#   ./launch_server.sh baseline-cudagraph      # R-KV OFF, CUDA graph on (production-like)
#
# Env overrides: MODEL, PORT, WINDOW, BUFFER, MEM_FRAC
set -euo pipefail

MODE="${1:-rkv}"
BUDGET="${2:-512}"
MODEL="${MODEL:-/data/model/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-30000}"
WINDOW="${WINDOW:-8}"
BUFFER="${BUFFER:-16}"
MEM_FRAC="${MEM_FRAC:-0.6}"

# Locate the patched SGLang source produced by scripts/apply_rkv.sh.
# Override with RKV_SGLANG_SRC=/path/to/sglang if you cloned it elsewhere.
SGLANG_SRC="${RKV_SGLANG_SRC:-$(cd "$(dirname "$0")/.." && pwd)/sglang-src}"
if [[ ! -d "$SGLANG_SRC/python/sglang" ]]; then
  echo "ERROR: patched SGLang not found at: $SGLANG_SRC" >&2
  echo "       Run  scripts/apply_rkv.sh  first (see README.md), or set" >&2
  echo "       RKV_SGLANG_SRC=/path/to/your/sglang checkout." >&2
  exit 1
fi
export PYTHONPATH="$SGLANG_SRC/python"
export HF_HUB_DISABLE_XET=1  # HF Xet transfer can hang on large files

# R-KV requires: eager decode (no captured CUDA graph), radix/prefix cache OFF
# (R-KV frees slots the radix tree still references -> pool double-count crash),
# overlap OFF (simple timing), page_size=1 (clean per-slot free).
RKV_FLAGS=(--disable-decode-cuda-graph --disable-prefill-cuda-graph
           --disable-overlap-schedule --disable-radix-cache --page-size 1)

COMMON=(--model-path "$MODEL" --attention-backend flashinfer
        --mem-fraction-static "$MEM_FRAC" --host 127.0.0.1 --port "$PORT")

case "$MODE" in
  rkv)
    echo ">> R-KV ON  | budget=$BUDGET window=$WINDOW buffer=$BUFFER"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${RKV_FLAGS[@]}" \
      --enable-rkv \
      --rkv-config "{\"budget\":$BUDGET,\"window_size\":$WINDOW,\"buffer_size\":$BUFFER}"
    ;;
  baseline)
    echo ">> BASELINE (eager, same flags as R-KV, no --enable-rkv)"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${RKV_FLAGS[@]}"
    ;;
  baseline-cudagraph)
    echo ">> BASELINE (production-like: CUDA graph on)"
    exec python3 -m sglang.launch_server "${COMMON[@]}" --disable-prefill-cuda-graph
    ;;
  *)
    echo "unknown mode: $MODE (use: baseline | rkv <budget> | baseline-cudagraph)" >&2
    exit 1
    ;;
esac
