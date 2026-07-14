#!/usr/bin/env bash
# Run deterministic BFCL V4 long-context and memory slices against an existing server.
set -euo pipefail

OUT_DIR="${1:?output directory required}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-/mnt/localssd/rkv-eval-tools}"
GORILLA_DIR="$WORK_ROOT/gorilla"
GORILLA_REVISION="${GORILLA_REVISION:-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8}"
BFCL_VENV="${BFCL_VENV:-/mnt/localssd/rkv-bfcl-venv}"
MODEL_REGISTRY_NAME="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-FC"
MODEL_DIR="${MODEL:?MODEL must point at the local Qwen checkpoint}"
PORT="${PORT:-30000}"

mkdir -p "$WORK_ROOT" "$OUT_DIR"
if [[ ! -d "$GORILLA_DIR/.git" ]]; then
  git clone https://github.com/ShishirPatil/gorilla.git "$GORILLA_DIR"
fi
git -C "$GORILLA_DIR" fetch origin "$GORILLA_REVISION"
git -C "$GORILLA_DIR" checkout --detach "$GORILLA_REVISION"
if ! git -C "$GORILLA_DIR" apply --reverse --check "$HERE/bfcl-qwen3-coder-480b.patch" >/dev/null 2>&1; then
  git -C "$GORILLA_DIR" apply --check "$HERE/bfcl-qwen3-coder-480b.patch"
  git -C "$GORILLA_DIR" apply "$HERE/bfcl-qwen3-coder-480b.patch"
fi
if [[ ! -x "$BFCL_VENV/bin/python" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 --seed "$BFCL_VENV"
  else
    python3 -m venv "$BFCL_VENV"
  fi
fi
bfcl_ready_marker="$BFCL_VENV/.bfcl-$GORILLA_REVISION-runtime-v2-ready"
if [[ ! -f "$bfcl_ready_marker" ]]; then
  # BFCL imports every registered Qwen handler at CLI startup. qwen-agent's
  # handler imports soundfile, but the BFCL editable dependency set does not
  # install it, so generation otherwise exits before its first request.
  "$BFCL_VENV/bin/python" -m pip install \
    -e "$GORILLA_DIR/berkeley-function-call-leaderboard" \
    "soundfile==0.13.1"
  touch "$bfcl_ready_marker"
fi

export BFCL_PROJECT_ROOT="$OUT_DIR"
export REMOTE_OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
export REMOTE_OPENAI_TOKENIZER_PATH="$MODEL_DIR"

"$BFCL_VENV/bin/python" "$HERE/make_bfcl_pilot_manifest.py" \
  --output "$BFCL_PROJECT_ROOT/test_case_ids_to_generate.json" \
  --long-context "${BFCL_LONG_CONTEXT_LIMIT:-10}" \
  --memory-each "${BFCL_MEMORY_EACH_LIMIT:-4}"

"$BFCL_VENV/bin/bfcl" generate \
  --model "$MODEL_REGISTRY_NAME" \
  --run-ids \
  --skip-server-setup \
  --temperature 0 \
  --num-threads "${BFCL_THREADS:-2}" \
  --include-input-log

"$BFCL_VENV/bin/bfcl" evaluate \
  --model "$MODEL_REGISTRY_NAME" \
  --test-category multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum \
  --partial-eval

"$BFCL_VENV/bin/python" "$HERE/validate_bfcl_pilot.py" \
  "$OUT_DIR" \
  --output "$OUT_DIR/validation.json"

touch "$OUT_DIR/BFCL_PILOT_COMPLETE"
