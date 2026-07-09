#!/usr/bin/env bash
# Dynamic-queue sweep worker. Usage: sweep_worker.sh <gpu_id>
# Pulls the next un-claimed line from the runlist (flock-serialized), runs it
# with CUDA_VISIBLE_DEVICES pinned, logs to results/logs/<name>.log, appends
# one status line per run to results/progress.log.
set -u
GPU="$1"
ROOT=/mnt/localssd/zefan/rkv-fi
export M="$ROOT/models"
export B="$ROOT/R-KV/FlashInfer/benchmark"
export R="$ROOT/results"
export PY="$ROOT/venv/bin/python"
RUNLIST="$ROOT/sweep_runlist.txt"
CLAIMS="$R/claims.txt"
PROGRESS="$R/progress.log"
mkdir -p "$R/logs" "$R/records"
touch "$CLAIMS"

while true; do
  NAME=""
  exec 9>>"$CLAIMS"
  flock 9
  while IFS='|' read -r name timeout cmd; do
    case "$name" in ''|'#'*) continue ;; esac
    if ! grep -qx "$name" "$CLAIMS"; then
      echo "$name" >> "$CLAIMS"
      NAME="$name"; TIMEOUT="$timeout"; CMD="$cmd"
      break
    fi
  done < "$RUNLIST"
  flock -u 9
  exec 9>&-

  [ -z "$NAME" ] && break

  echo "START gpu$GPU $NAME $(date -u +%FT%TZ)" >> "$PROGRESS"
  start=$(date +%s)
  if CUDA_VISIBLE_DEVICES="$GPU" timeout "$TIMEOUT" bash -c "cd $ROOT && $CMD" > "$R/logs/$NAME.log" 2>&1; then
    status=OK
  else
    status="FAIL($?)"
  fi
  dur=$(( $(date +%s) - start ))
  summary=$(grep -E '^\{' "$R/logs/$NAME.log" | tail -1 | head -c 400)
  echo "DONE gpu$GPU $NAME $status ${dur}s ${summary}" >> "$PROGRESS"
done
echo "WORKER gpu$GPU EXHAUSTED $(date -u +%FT%TZ)" >> "$PROGRESS"
