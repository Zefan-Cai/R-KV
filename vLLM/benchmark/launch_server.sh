#!/usr/bin/env bash
# Launch a vLLM (v0.25.1 + R-KV) OpenAI server for the R-KV benchmark.
#
# Usage:
#   ./launch_server.sh rkv 256         # R-KV ON, budget=256 (PIECEWISE cudagraph + async)
#   ./launch_server.sh fullkv          # Full-KV, production (prefix caching + full cudagraph)
#   ./launch_server.sh constrained     # Full-KV under R-KV's flags (prefix caching OFF)
#
# Parallelism (optional, mutually exclusive -- TP wins if both are set):
#   DP=N ./launch_server.sh rkv 256    # N data-parallel replicas (tp=1)
#   TP=N ./launch_server.sh rkv 256    # N-way tensor parallel (R-KV all-reduces the score)
#
# NOTE ON METHODOLOGY: the numbers in RESULTS*.md were produced with the OFFLINE
# driver (eval.py -> LLM.generate over all prompts at once), NOT with this
# server. This launcher is for real serving / OpenAI-client benchmarking and
# uses the exact same R-KV knobs; drive it with `vllm bench serve` or any
# OpenAI-compatible client.
#
# Best-throughput R-KV path (see ../docs/OPTIMIZATIONS.md):
#   * NO --enforce-eager -> R-KV auto-selects PIECEWISE cudagraph (attention
#     stays eager so the in-forward hooks fire; the rest of the layer is graphed).
#   * VLLM_V1_R_KV_ASYNC=1 -> async scheduling (runner-authoritative num_dropped).
#   * VLLM_V1_R_KV_FREE_BLOCKS=1 -> free evicted blocks (bounded KV footprint).
#     Opt-in / experimental (no scheduler<->worker block-table version handshake
#     yet -- see ../docs/IMPLEMENTATION.md 6.5); drop it for the default-safe path.
#
# Env overrides: MODEL, PORT, HOST, BUFFER, WINDOW, MEM_FRAC, DP, TP, EXTRA
#   EXTRA is appended verbatim.
set -euo pipefail

MODE="${1:-rkv}"
BUDGET="${2:-256}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
BUFFER="${BUFFER:-128}"
WINDOW="${WINDOW:-8}"
MEM_FRAC="${MEM_FRAC:-0.85}"
DP="${DP:-1}"
TP="${TP:-1}"
EXTRA="${EXTRA:-}"

COMMON=(--host "$HOST" --port "$PORT" --gpu-memory-utilization "$MEM_FRAC")

# Parallelism (mutually exclusive; TP preferred when both >1). R-KV supports
# tensor parallelism (the per-token eviction score is all-reduced across the
# attention-TP group so every rank evicts identical tokens; see RESULTS_tp.md)
# and plain data parallelism (each replica runs its own independent R-KV; see
# RESULTS_dp.md).
PAR_FLAGS=()
if [[ "$TP" -gt 1 ]]; then
  PAR_FLAGS=(--tensor-parallel-size "$TP")
elif [[ "$DP" -gt 1 ]]; then
  PAR_FLAGS=(--data-parallel-size "$DP" --tensor-parallel-size 1)
fi

case "$MODE" in
  rkv)
    export VLLM_V1_R_KV_BUDGET="$BUDGET"
    export VLLM_V1_R_KV_BUFFER="$BUFFER"
    export VLLM_V1_R_KV_WINDOW="$WINDOW"
    export VLLM_V1_R_KV_ASYNC=1        # async scheduling (runner-authoritative)
    export VLLM_V1_R_KV_FREE_BLOCKS=1  # free evicted blocks (opt-in; see note above)
    echo ">> R-KV ON  | budget=$BUDGET buffer=$BUFFER window=$WINDOW dp=$DP tp=$TP (PIECEWISE cudagraph + async)"
    # shellcheck disable=SC2086
    exec vllm serve "$MODEL" "${COMMON[@]}" "${PAR_FLAGS[@]}" $EXTRA
    ;;
  fullkv)
    # Production Full-KV: prefix caching + full cudagraph + async, all upstream
    # defaults -- the fastest Full-KV baseline (best case for Full-KV).
    echo ">> FULL-KV (production: prefix caching + full cudagraph + async, upstream defaults) dp=$DP tp=$TP"
    # shellcheck disable=SC2086
    exec vllm serve "$MODEL" "${COMMON[@]}" "${PAR_FLAGS[@]}" $EXTRA
    ;;
  constrained|baseline)
    # Full-KV under R-KV's required constraint (prefix caching OFF), no
    # compression -- the FAIR A/B baseline: the throughput delta to `rkv` is
    # purely R-KV's compression cost, with the shared-prefix prefill-dedup
    # advantage (which R-KV structurally cannot use) removed from both sides.
    echo ">> FULL-KV constrained (prefix caching OFF, matching R-KV; no compression) dp=$DP tp=$TP"
    # shellcheck disable=SC2086
    exec vllm serve "$MODEL" "${COMMON[@]}" --no-enable-prefix-caching "${PAR_FLAGS[@]}" $EXTRA
    ;;
  *)
    echo "unknown mode: $MODE (use: rkv <budget> | fullkv | constrained)" >&2
    exit 1
    ;;
esac
