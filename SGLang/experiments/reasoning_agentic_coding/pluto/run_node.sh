#!/usr/bin/env bash
# Idempotent Pluto node setup followed by a paired crossover pilot.
set -euo pipefail

ROLE="${1:?role p1 or p2 required}"
case "$ROLE" in
  p1) ARMS=(d-full d-4k p-full p-4k) ;;
  p2) ARMS=(d-4k d-full p-4k p-full) ;;
  *) echo "unknown role: $ROLE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP_DIR="$REPO_ROOT/SGLang/experiments/reasoning_agentic_coding"
CAMPAIGN="rkv-sglang-eval-20260713"
SHARED_ROOT="${SHARED_ROOT:-/sensei-fs/users/zcai/$CAMPAIGN}"
ATTEMPT_ID="${ATTEMPT_ID:-pilot-v1}"
RESULT_ROOT="$SHARED_ROOT/results/$ROLE/$ATTEMPT_ID"
MODEL_ID="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
MODEL_REVISION="003f183a92fbe5b9a8325aaa8b2ae797c91dd90f"
MODEL_ROOT="${MODEL_ROOT:-/mnt/localssd/rkv-models}"
MODEL_DIR="$MODEL_ROOT/Qwen3-Coder-480B-A35B-Instruct-FP8-$MODEL_REVISION"
VENV="/mnt/localssd/rkv-sglang-venv"
mkdir -p "$RESULT_ROOT" "$MODEL_ROOT"
exec > >(tee -a "$RESULT_ROOT/node.log") 2>&1

echo "role=$ROLE arms=${ARMS[*]} host=$(hostname) started_utc=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

available_kib="$(df -Pk "$MODEL_ROOT" | awk 'NR==2 {print $4}')"
required_kib="${MODEL_REQUIRED_KIB:-629145600}"
if (( available_kib < required_kib )); then
  echo "insufficient local NVMe for checkpoint: available_kib=$available_kib required_kib=$required_kib" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --upgrade uv
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 --seed "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip

cd "$REPO_ROOT/SGLang"
bash scripts/apply_rkv.sh --force

# Editable SGLang builds its bundled gRPC extension from Rust. Use upstream's
# idempotent installer so both its pinned toolchain and protoc are available.
bash sglang-src/scripts/ci/utils/install_rust_protoc.sh
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
rustc --version
cargo --version
protoc --version
command -v cc

python -m pip install -e sglang-src/python --extra-index-url https://docs.sglang.ai/whl/cu129/
python -m pip install -r requirements-rkv.txt --extra-index-url https://docs.sglang.ai/whl/cu129/
python -m pip install "huggingface_hub[cli]" "evalplus==0.3.1"

python tests/test_rkv_algo.py
python tests/test_rkv_integration.py
python tests/test_rkv_prefill.py
python tests/test_rkv_prefill_integration.py
python tests/test_cross_repo_parity.py
python tests/test_rkv_redundancy_fused.py

(
  flock -x 9
  if [[ ! -f "$MODEL_DIR/MODEL_READY" ]]; then
    mkdir -p "$MODEL_DIR"
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    hf download "$MODEL_ID" \
      --revision "$MODEL_REVISION" \
      --local-dir "$MODEL_DIR"
    test -s "$MODEL_DIR/config.json"
    test -s "$MODEL_DIR/tokenizer_config.json"
    touch "$MODEL_DIR/MODEL_READY"
  fi
) 9>"$MODEL_ROOT/qwen3-coder-480b.lock"

export MODEL="$MODEL_DIR"
export MODEL_REVISION="$MODEL_REVISION"
export SERVED_MODEL_NAME="$MODEL_ID"
export TP=8
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
export MEM_FRAC="${MEM_FRAC:-0.90}"
export MAX_ACTIVE_REQUESTS="${MAX_ACTIVE_REQUESTS:-16}"
export HF_HOME="${HF_HOME:-/mnt/localssd/hf-home}"

cat >"$RESULT_ROOT/provenance.txt" <<EOF
rkv_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
model=$MODEL_ID
model_revision=$MODEL_REVISION
arms=${ARMS[*]}
role=$ROLE
attempt=$ATTEMPT_ID
sglang_upstream=49e384ce9d304648e9959666ecb8ce8cd98d0deb
EOF
python -m pip freeze >"$RESULT_ROOT/pip-freeze.txt"

if command -v aws >/dev/null 2>&1 && aws sts get-caller-identity >/dev/null 2>&1; then
  (
    while true; do
      aws s3 sync "$SHARED_ROOT/results" "s3://phidias/zcai/codex/$CAMPAIGN/results" --quiet || true
      sleep 300
    done
  ) &
  echo $! >"$RESULT_ROOT/s3-backup.pid"
fi

for arm in "${ARMS[@]}"; do
  if [[ "$arm" == p-* ]]; then
    export LONG_MAX_TOKENS="${PREFILL_SMOKE_LONG_MAX_TOKENS:-128}"
  else
    unset LONG_MAX_TOKENS || true
  fi
  bash "$EXP_DIR/run_pilot_arm.sh" "$arm" "$RESULT_ROOT/$arm"
done
touch "$RESULT_ROOT/NODE_PILOT_COMPLETE"

while true; do
  sleep 300
done
