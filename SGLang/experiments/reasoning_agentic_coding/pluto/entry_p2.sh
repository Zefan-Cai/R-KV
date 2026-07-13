#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/localssd/R-KV
BRANCH=eval/rkv-sglang-reasoning-agentic-coding-20260713
if [[ ! -d "$REPO/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch https://github.com/Zefan-Cai/R-KV.git "$REPO"
else
  git -C "$REPO" pull --ff-only origin "$BRANCH"
fi
exec bash "$REPO/SGLang/experiments/reasoning_agentic_coding/pluto/run_node.sh" p2
