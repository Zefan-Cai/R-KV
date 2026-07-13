#!/usr/bin/env bash
# Launch an SGLang server for the R-KV benchmark.
#
# Usage:
#   ./launch_server.sh rkv 256            # R-KV ON, budget=256 (fastest: CUDA graphs ON)
#   ./launch_server.sh fullkv             # Full-KV, production stack (radix + overlap + graphs)
#   ./launch_server.sh constrained        # Full-KV under R-KV's flags (radix/overlap OFF, page 1)
#
# Parallelism (optional, mutually exclusive -- TP wins if both are set):
#   DP=N ./launch_server.sh rkv 256       # N plain data-parallel replicas (tp=1)
#   TP=N ./launch_server.sh rkv 256       # N-way tensor parallel (R-KV all-reduces the score)
#
# Evaluate with SGLang's GSM8K harness (5-shot, standard GSM8K test set):
#   PYTHONPATH=../sglang-src/python python3 ../sglang-src/benchmark/gsm8k/bench_sglang.py \
#       --num-questions 200 --num-shots 5 --parallel 32 --port 30000
#
# Env overrides: MODEL, PORT, WINDOW, BUFFER, MEM_FRAC, DP, TP
set -euo pipefail

MODE="${1:-rkv}"
BUDGET="${2:-256}"
MODEL="${MODEL:-/data/model/Qwen2.5-Math-7B-Instruct}"
PORT="${PORT:-30000}"
WINDOW="${WINDOW:-8}"
BUFFER="${BUFFER:-128}"
MEM_FRAC="${MEM_FRAC:-0.85}"
DP="${DP:-1}"
TP="${TP:-1}"

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

# R-KV requires: radix/prefix cache OFF (R-KV frees slots the radix tree still
# references -> pool double-count crash), overlap OFF (avoids an allocator
# free-list race during compaction), page_size=1 (clean per-slot free).
# Decode AND prefill CUDA graphs are SUPPORTED for decode R-KV and left ON here
# (fastest path: in-graph observation-query collection + a hybrid eager path only
# for the compaction steps). These are the exact flags behind the RESULTS numbers.
CONSTRAINED_FLAGS=(--disable-overlap-schedule --disable-radix-cache --page-size 1)

COMMON=(--model-path "$MODEL" --attention-backend flashinfer
        --mem-fraction-static "$MEM_FRAC" --host 127.0.0.1 --port "$PORT")

# Parallelism (mutually exclusive; TP preferred when both >1). R-KV supports
# tensor parallelism (the per-token eviction score is all-reduced across the
# attention-TP group so every rank evicts identical tokens; see RESULTS_tp.md)
# and plain data parallelism (each replica runs its own independent R-KV; see
# RESULTS_dp.md).
PAR_FLAGS=()
if [[ "$TP" -gt 1 ]]; then
  PAR_FLAGS=(--tp-size "$TP")
elif [[ "$DP" -gt 1 ]]; then
  PAR_FLAGS=(--dp-size "$DP" --tp-size 1)
fi

case "$MODE" in
  rkv)
    echo ">> R-KV ON  | budget=$BUDGET window=$WINDOW buffer=$BUFFER dp=$DP tp=$TP (CUDA graphs ON)"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${CONSTRAINED_FLAGS[@]}" "${PAR_FLAGS[@]}" \
      --enable-rkv \
      --rkv-config "{\"budget\":$BUDGET,\"window_size\":$WINDOW,\"buffer_size\":$BUFFER}"
    ;;
  fullkv)
    # Production Full-KV: radix/prefix cache, overlap schedule and CUDA graphs all
    # ON -- the fastest Full-KV baseline (best case for Full-KV).
    echo ">> FULL-KV (production: radix + overlap + CUDA graphs ON) dp=$DP tp=$TP"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${PAR_FLAGS[@]}"
    ;;
  constrained|baseline)
    # Full-KV under R-KV's required flags (radix/overlap OFF, page_size 1), no
    # compression -- the FAIR A/B baseline: the throughput delta to `rkv` is purely
    # R-KV's compression cost, with the radix prefix-cache advantage removed from
    # both sides.
    echo ">> FULL-KV constrained (R-KV flags: radix/overlap OFF, page 1; no --enable-rkv) dp=$DP tp=$TP"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${CONSTRAINED_FLAGS[@]}" "${PAR_FLAGS[@]}"
    ;;
  *)
    echo "unknown mode: $MODE (use: rkv <budget> | fullkv | constrained)" >&2
    exit 1
    ;;
esac
