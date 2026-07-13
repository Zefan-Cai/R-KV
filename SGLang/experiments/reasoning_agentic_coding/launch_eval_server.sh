#!/usr/bin/env bash
# Launch one reproducible FullKV/R-KV evaluation arm on the pinned SGLang tree.
set -euo pipefail

ARM="${1:-}"
if [[ -z "$ARM" ]]; then
  echo "usage: $0 <d-prod|d-full|d-4k|d-8k|p-full|p-4k|p-2k|p-o4k> [extra sglang args...]" >&2
  exit 2
fi
shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd "$HERE/../.." && pwd)"
SGLANG_SRC="${RKV_SGLANG_SRC:-$SGLANG_DIR/sglang-src}"
if [[ ! -d "$SGLANG_SRC/python/sglang" ]]; then
  echo "patched SGLang tree missing at $SGLANG_SRC; run SGLang/scripts/apply_rkv.sh" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8}"
MODEL_REVISION="${MODEL_REVISION:-003f183a92fbe5b9a8325aaa8b2ae797c91dd90f}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
EP_SIZE="${EP_SIZE:-$TP}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRAC="${MEM_FRAC:-0.90}"
MAX_ACTIVE_REQUESTS="${MAX_ACTIVE_REQUESTS:-16}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"

export PYTHONPATH="$SGLANG_SRC/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

COMMON=(
  --model-path "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --attention-backend flashinfer
  --tp-size "$TP"
  --ep-size "$EP_SIZE"
  --kv-cache-dtype auto
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRAC"
  --host "$HOST"
  --port "$PORT"
  --tool-call-parser qwen3_coder
)
if [[ ! -d "$MODEL" ]]; then
  COMMON+=(--revision "$MODEL_REVISION")
fi
CONSTRAINED=(--disable-radix-cache --disable-overlap-schedule --page-size 1)

if [[ "${DISABLE_GRAPHS:-0}" == "1" ]]; then
  COMMON+=(--disable-decode-cuda-graph --disable-prefill-cuda-graph)
fi

case "$ARM" in
  d-prod)
    MODE_ARGS=()
    ;;
  d-full)
    MODE_ARGS=("${CONSTRAINED[@]}")
    ;;
  d-4k)
    MODE_ARGS=(
      "${CONSTRAINED[@]}"
      --rkv-max-active-requests "$MAX_ACTIVE_REQUESTS"
      --enable-rkv
      --rkv-config '{"budget":4096,"window_size":8,"buffer_size":128}'
    )
    ;;
  d-8k)
    MODE_ARGS=(
      "${CONSTRAINED[@]}"
      --rkv-max-active-requests "$MAX_ACTIVE_REQUESTS"
      --enable-rkv
      --rkv-config '{"budget":8192,"window_size":8,"buffer_size":128}'
    )
    ;;
  p-full)
    MODE_ARGS=(
      "${CONSTRAINED[@]}"
      --disable-prefill-cuda-graph
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    )
    ;;
  p-4k)
    MODE_ARGS=(
      "${CONSTRAINED[@]}"
      --disable-prefill-cuda-graph
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
      --enable-rkv-prefill
      --rkv-prefill-config '{"mode":"buffered","budget":4096,"window_size":32,"buffer":512,"row_block":2048}'
    )
    ;;
  p-2k)
    MODE_ARGS=(
      "${CONSTRAINED[@]}"
      --disable-prefill-cuda-graph
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
      --enable-rkv-prefill
      --rkv-prefill-config '{"mode":"buffered","budget":2048,"window_size":32,"buffer":512,"row_block":2048}'
    )
    ;;
  p-o4k)
    MODE_ARGS=(
      "${CONSTRAINED[@]}"
      --disable-prefill-cuda-graph
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
      --enable-rkv-prefill
      --rkv-prefill-config '{"mode":"oneshot","budget":4096,"window_size":32,"row_block":2048}'
    )
    ;;
  *)
    echo "unknown arm: $ARM" >&2
    exit 2
    ;;
esac

echo "arm=$ARM model=$MODEL revision=$MODEL_REVISION tp=$TP ep=$EP_SIZE context=$CONTEXT_LENGTH kv_cache=auto"
exec python3 -m sglang.launch_server "${COMMON[@]}" "${MODE_ARGS[@]}" "$@"
