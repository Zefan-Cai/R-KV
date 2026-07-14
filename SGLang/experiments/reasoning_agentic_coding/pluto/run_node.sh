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
MODEL_SECONDARY_ROOT="${MODEL_SECONDARY_ROOT:-/sensei-fs-3/users/zcai/$CAMPAIGN/models}"
MODEL_SECONDARY_DIR="$MODEL_SECONDARY_ROOT/Qwen3-Coder-480B-A35B-Instruct-FP8-$MODEL_REVISION"
VENV="/mnt/localssd/rkv-sglang-venv"
SGLANG_INSTALL_MODE="${SGLANG_INSTALL_MODE:-wheel}"
export HF_HOME="${HF_HOME:-/mnt/localssd/.cache/huggingface}"
# Keep package caches node-local. The two persistent project quotas are reserved
# for the roughly 450 GiB checkpoint and its small result artifacts.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/localssd/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/localssd/.cache/uv}"
mkdir -p \
  "$RESULT_ROOT" "$MODEL_ROOT" "$MODEL_SECONDARY_ROOT" "$HF_HOME" \
  "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
default_hf_token="$HOME/.cache/huggingface/token"
if [[ "$HF_HOME/token" != "$default_hf_token" && -s "$default_hf_token" && ! -s "$HF_HOME/token" ]]; then
  install -m 600 "$default_hf_token" "$HF_HOME/token"
fi
exec > >(tee -a "$RESULT_ROOT/node.log") 2>&1

echo "role=$ROLE arms=${ARMS[*]} host=$(hostname) started_utc=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

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
      echo "MODEL_RSYNC_HOST is incompatible with the quota-striped checkpoint" >&2
      exit 2
    else
      hf_token_path="${HF_TOKEN_PATH:-$HF_HOME/token}"
      for _ in $(seq 1 "${HF_TOKEN_WAIT_POLLS:-120}"); do
        if [[ -n "${HF_TOKEN:-}" || -s "$hf_token_path" ]]; then
          break
        fi
        echo "waiting for private HF token file at $hf_token_path"
        sleep 5
      done
      if [[ -z "${HF_TOKEN:-}" && ! -s "$hf_token_path" ]]; then
        echo "HF token unavailable; refusing an unauthenticated 450 GiB transfer" >&2
        exit 1
      fi
      export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
      if [[ -n "$MODEL_SECONDARY_ROOT" ]]; then
        mkdir -p "$MODEL_SECONDARY_DIR"
        # Each Sensei project quota holds about half this model. Keep shards
        # 1-24 on the primary mount and 25-49 on the independent secondary
        # mount, leaving enough headroom for metadata and final markers.
        find "$MODEL_DIR/.cache/huggingface/download" \
          -type f -name '*.incomplete' -delete 2>/dev/null || true

        for index in $(seq 25 49); do
          shard="$(printf 'model-%05d-of-00049.safetensors' "$index")"
          primary_shard="$MODEL_DIR/$shard"
          secondary_shard="$MODEL_SECONDARY_DIR/$shard"
          if [[ -f "$primary_shard" && ! -L "$primary_shard" ]]; then
            primary_size="$(stat -c %s "$primary_shard")"
            # Always rerun rsync: an interrupted prior migration may leave a
            # non-empty but incomplete destination that must be resumed.
            rsync -a --partial "$primary_shard" "$MODEL_SECONDARY_DIR/"
            secondary_size="$(stat -c %s "$secondary_shard")"
            if [[ "$primary_size" != "$secondary_size" ]]; then
              echo "shard migration size mismatch: $shard" >&2
              exit 1
            fi
            rm -f "$primary_shard"
            ln -s "$secondary_shard" "$primary_shard"
          fi
        done

        hf download "$MODEL_ID" \
          --revision "$MODEL_REVISION" \
          --exclude 'model-*.safetensors' \
          --local-dir "$MODEL_SECONDARY_DIR"

        missing_primary_shards=()
        missing_secondary_shards=()
        for index in $(seq 1 49); do
          shard="$(printf 'model-%05d-of-00049.safetensors' "$index")"
          if [[ ! -s "$MODEL_DIR/$shard" && ! -s "$MODEL_SECONDARY_DIR/$shard" ]]; then
            if (( index <= 24 )); then
              missing_primary_shards+=("$shard")
            else
              missing_secondary_shards+=("$shard")
            fi
          fi
        done
        if (( ${#missing_primary_shards[@]} > 0 )); then
          printf 'downloading %d missing shards on primary Sensei mount\n' \
            "${#missing_primary_shards[@]}"
          hf download "$MODEL_ID" "${missing_primary_shards[@]}" \
            --revision "$MODEL_REVISION" \
            --local-dir "$MODEL_DIR"
        fi
        if (( ${#missing_secondary_shards[@]} > 0 )); then
          printf 'downloading %d missing shards on secondary Sensei mount\n' \
            "${#missing_secondary_shards[@]}"
          hf download "$MODEL_ID" "${missing_secondary_shards[@]}" \
            --revision "$MODEL_REVISION" \
            --local-dir "$MODEL_SECONDARY_DIR"
        fi

        while IFS= read -r -d '' source_path; do
          relative_path="${source_path#"$MODEL_SECONDARY_DIR/"}"
          destination_path="$MODEL_DIR/$relative_path"
          if [[ ! -e "$destination_path" ]]; then
            mkdir -p "$(dirname "$destination_path")"
            [[ -L "$destination_path" ]] && rm -f "$destination_path"
            ln -s "$source_path" "$destination_path"
          fi
        done < <(
          find "$MODEL_SECONDARY_DIR" \
            -path "$MODEL_SECONDARY_DIR/.cache" -prune -o \
            -type f -print0
        )
      else
        hf download "$MODEL_ID" \
          --revision "$MODEL_REVISION" \
          --local-dir "$MODEL_DIR"
      fi
    fi
    test -s "$MODEL_DIR/config.json"
    test -s "$MODEL_DIR/tokenizer_config.json"
    for index in $(seq 1 49); do
      test -s "$MODEL_DIR/$(printf 'model-%05d-of-00049.safetensors' "$index")"
    done
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
  if [[ -f "$RESULT_ROOT/$arm/PILOT_COMPLETE" ]]; then
    echo "arm=$arm already complete; preserving artifacts and skipping rerun"
    continue
  fi
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
