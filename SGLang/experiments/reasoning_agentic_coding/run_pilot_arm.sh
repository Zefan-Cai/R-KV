#!/usr/bin/env bash
# Start one arm, run paired smoke/AIME prompts, and preserve all raw evidence.
set -euo pipefail

ARM="${1:?arm required}"
OUT_DIR="${2:?output directory required}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PORT="${PORT:-30000}"
MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8}"
mkdir -p "$OUT_DIR"

SERVER_LOG="$OUT_DIR/server.log"
PID_FILE="$OUT_DIR/server.pid"

cleanup() {
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

SGLANG_RKV_TP_CHECK="${SGLANG_RKV_TP_CHECK:-1}" \
  setsid bash "$HERE/launch_eval_server.sh" "$ARM" >"$SERVER_LOG" 2>&1 &
server_pid=$!
echo "$server_pid" >"$PID_FILE"

ready=0
for _ in $(seq 1 "${SERVER_READY_POLLS:-540}"); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
if [[ "$ready" != "1" ]]; then
  echo "server did not become ready" >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  exit 1
fi

python "$HERE/api_eval.py" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL_NAME" \
  smoke \
  --long-max-tokens "${LONG_MAX_TOKENS:-4608}" \
  --output "$OUT_DIR/smoke.json"

case "$ARM" in
  d-*)
    python "$HERE/api_eval.py" \
      --base-url "http://127.0.0.1:$PORT/v1" \
      --model "$MODEL_NAME" \
      aime \
      --data "$REPO_ROOT/HuggingFace/data/aime24.jsonl" \
      --limit "${AIME_LIMIT:-5}" \
      --concurrency "${AIME_CONCURRENCY:-2}" \
      --max-tokens "${AIME_MAX_TOKENS:-8192}" \
      --output "$OUT_DIR/aime24.jsonl"

    if [[ "${RUN_EVALPLUS:-1}" == "1" ]]; then
      mkdir -p "$OUT_DIR/evalplus"
      OPENAI_API_KEY=EMPTY evalplus.evaluate \
        --model "$MODEL_NAME" \
        --dataset humaneval \
        --backend openai \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --root "$OUT_DIR/evalplus" \
        --parallel "${EVALPLUS_PARALLEL:-8}" \
        --greedy \
        --mini 2>&1 | tee "$OUT_DIR/evalplus.log"
    fi
    ;;
  p-*)
    if [[ "${RUN_BFCL:-1}" == "1" ]]; then
      bash "$HERE/run_bfcl_pilot.sh" "$OUT_DIR/bfcl"
    fi
    ;;
esac

sleep 5
python "$HERE/summarize_server_log.py" \
  "$SERVER_LOG" \
  --output "$OUT_DIR/server_summary.json" \
  --require-arm "$ARM"
touch "$OUT_DIR/PILOT_COMPLETE"
