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
# Reclaimable P1/P2 allocations cannot reliably finish a 450 GiB checkpoint
# transfer on node-local NVMe. Sensei FS preserves partial shards across runs;
# the shared flock lets either lane resume the single writer safely.
MODEL_ROOT="${MODEL_ROOT:-$SHARED_ROOT/models}"
MODEL_DIR="$MODEL_ROOT/Qwen3-Coder-480B-A35B-Instruct-FP8-$MODEL_REVISION"
VENV="/mnt/localssd/rkv-sglang-venv"
SGLANG_INSTALL_MODE="${SGLANG_INSTALL_MODE:-wheel}"
export HF_HOME="${HF_HOME:-/mnt/localssd/.cache/huggingface}"
mkdir -p "$RESULT_ROOT" "$MODEL_ROOT" "$HF_HOME"
default_hf_token="$HOME/.cache/huggingface/token"
if [[ "$HF_HOME/token" != "$default_hf_token" && -s "$default_hf_token" && ! -s "$HF_HOME/token" ]]; then
  install -m 600 "$default_hf_token" "$HF_HOME/token"
fi
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
case "$SGLANG_INSTALL_MODE" in
  wheel)
    # Use the release wheel for dependencies/native artifacts, then put the
    # exact patched 49e384ce Python tree first. The campaign is HTTP-only;
    # gRPC's wheel-only nested extension is intentionally not imported.
    python -m pip install --only-binary=sglang "sglang==0.5.14" \
      --extra-index-url https://docs.sglang.ai/whl/cu129/
    ;;
  source)
    # Conservative full-source fallback, including the bundled gRPC extension.
    bash sglang-src/scripts/ci/utils/install_rust_protoc.sh
    export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
    rustc --version
    cargo --version
    protoc --version
    command -v cc
    python -m pip install -e sglang-src/python \
      --extra-index-url https://docs.sglang.ai/whl/cu129/
    ;;
  *)
    echo "unknown SGLANG_INSTALL_MODE: $SGLANG_INSTALL_MODE" >&2
    exit 2
    ;;
esac
python -m pip install -r requirements-rkv.txt --extra-index-url https://docs.sglang.ai/whl/cu129/
python -m pip install "huggingface_hub[cli]" "evalplus==0.3.1"

export RKV_SGLANG_SRC="$REPO_ROOT/SGLang/sglang-src"
export PYTHONPATH="$RKV_SGLANG_SRC/python${PYTHONPATH:+:$PYTHONPATH}"
unset SGLANG_ENABLE_GRPC SGLANG_GRPC_PORT
python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path

import sglang
from sglang.srt.environ import envs
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs

source = (Path(os.environ["RKV_SGLANG_SRC"]) / "python").resolve()
assert Path(sglang.__file__).resolve().is_relative_to(source)
assert metadata.version("sglang") in {"0.5.14", "0.0.0.dev1+g49e384ce9.d20260713"}
assert hasattr(ServerArgs, "enable_rkv")
assert hasattr(ServerArgs, "enable_rkv_prefill")
assert callable(launch_server)
assert envs.SGLANG_ENABLE_GRPC.get() is False
assert importlib.util.find_spec("sgl_kernel") is not None
assert importlib.util.find_spec("flashinfer") is not None
print("HTTP_OVERLAY_OK", sglang.__file__, metadata.version("sglang"))
PY
python -m sglang.launch_server --help >"$RESULT_ROOT/sglang-launch-help.txt"
grep -q -- '--enable-rkv' "$RESULT_ROOT/sglang-launch-help.txt"
grep -q -- '--enable-rkv-prefill' "$RESULT_ROOT/sglang-launch-help.txt"

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
    if [[ -n "${MODEL_RSYNC_HOST:-}" ]]; then
      rsync -a --partial --whole-file --no-compress \
        --contimeout=30 --timeout=300 --info=progress2 \
        --exclude='.cache/' --exclude='MODEL_READY' \
        "rsync://${MODEL_RSYNC_HOST}:${MODEL_RSYNC_PORT:-18731}/model/" \
        "$MODEL_DIR/"
    else
      export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
      hf download "$MODEL_ID" \
        --revision "$MODEL_REVISION" \
        --local-dir "$MODEL_DIR"
    fi
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

cat >"$RESULT_ROOT/provenance.txt" <<EOF
rkv_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
model=$MODEL_ID
model_revision=$MODEL_REVISION
sglang_install_mode=$SGLANG_INSTALL_MODE
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
