#!/usr/bin/env bash
# Launch a vLLM (v0.25.1 + R-KV) OpenAI server at BEST THROUGHPUT.
#
# Usage:
#   ./launch_server.sh rkv 256       # R-KV ON, budget=256 (fastest R-KV path)
#   ./launch_server.sh fullkv        # Full-KV baseline (upstream defaults)
#
# BEST-THROUGHPUT R-KV path (all levers measured on 8x H100, Qwen2.5-Math-7B;
# see RESULTS_H100.md / ../docs/OPTIMIZATIONS.md):
#   * NO  --enforce-eager  -> R-KV auto-selects PIECEWISE cudagraph (attention
#     stays eager so the in-forward hooks fire, the rest of the layer is graphed).
#     +30-40% decode tok/s vs eager. FULL cudagraph would capture attention and
#     silently no-op R-KV, so PIECEWISE is auto-forced -- do NOT pass
#     --enforce-eager.
#   * VLLM_V1_R_KV_ASYNC=1 -> async scheduling. R-KV normally force-disables it
#     (compaction's evicted-token feedback would be one step stale under async);
#     with this flag the runner applies the drop itself right after compaction
#     (runner-authoritative num_dropped), so async is safe. +16.7% decode tok/s
#     at budget=256/buffer=64 sustained (gap to Full-KV -26% -> -13%).
#   * VLLM_V1_R_KV_FREE_BLOCKS=1 (default) -> R-KV frees the KV blocks it evicts,
#     so the per-request KV footprint is bounded at budget+buffer. Under memory
#     pressure this fits many more concurrent requests (e.g. +76% at a tight KV
#     budget); harmless when not memory-bound.
#   * Batched cross-layer scoring + batched compaction (in the patch) keep the
#     per-compaction cost low, so small buffers stay fast.
#
# Env overrides: MODEL, PORT, BUFFER, WINDOW, MEM_FRAC, HOST, EXTRA
#   EXTRA is appended verbatim. R-KV is correct under tensor & data parallelism
#   (it all-reduces its eviction scores across each replica's TP group), e.g.
#   EXTRA="--tensor-parallel-size 2" or EXTRA="--data-parallel-size 2".
#
# Evaluate with any OpenAI-compatible client, or:
#   vllm bench serve --model "$MODEL" --port "$PORT" ...
set -euo pipefail

MODE="${1:-rkv}"
BUDGET="${2:-256}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
BUFFER="${BUFFER:-64}"
WINDOW="${WINDOW:-8}"
MEM_FRAC="${MEM_FRAC:-0.9}"
EXTRA="${EXTRA:-}"

COMMON=(--host "$HOST" --port "$PORT" --gpu-memory-utilization "$MEM_FRAC")

case "$MODE" in
  rkv)
    # Best-throughput R-KV: PIECEWISE cudagraph (no --enforce-eager) + async.
    export VLLM_V1_R_KV_BUDGET="$BUDGET"
    export VLLM_V1_R_KV_BUFFER="$BUFFER"
    export VLLM_V1_R_KV_WINDOW="$WINDOW"
    export VLLM_V1_R_KV_ASYNC=1        # async scheduling (+16.7%), runner-authoritative
    export VLLM_V1_R_KV_FREE_BLOCKS=1  # free evicted blocks (default); bounded KV footprint
    echo ">> R-KV  budget=$BUDGET buffer=$BUFFER window=$WINDOW  async=ON  cudagraph=PIECEWISE"
    # shellcheck disable=SC2086
    exec vllm serve "$MODEL" "${COMMON[@]}" $EXTRA
    ;;
  fullkv)
    echo ">> Full-KV baseline (async + FULL cudagraph + prefix caching, all upstream defaults)"
    # shellcheck disable=SC2086
    exec vllm serve "$MODEL" "${COMMON[@]}" $EXTRA
    ;;
  *)
    echo "usage: $0 {rkv [budget]|fullkv}   (env: MODEL PORT BUFFER WINDOW MEM_FRAC HOST EXTRA)" >&2
    exit 1
    ;;
esac
