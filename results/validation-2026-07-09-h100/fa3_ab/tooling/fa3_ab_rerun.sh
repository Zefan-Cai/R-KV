#!/usr/bin/env bash
set -uo pipefail

ROOT=/mnt/localssd/zefan/rkv-fi
TEST_ROOT=/mnt/localssd/zefan/rkv-fi-opt-test
PY="$ROOT/venv/bin/python"
MODEL="$ROOT/models/DeepSeek-R1-Distill-Qwen-1.5B"
OUT="$TEST_ROOT/validation_rerun"
LOG=/mnt/localssd/zefan/fa3_ab_rerun.log

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "[rerun] start $(date -u +%FT%TZ)"
overall=0

# Reverse the first run's backend order to expose warmup/thermal order bias.
for attn in fa3 flashinfer; do
  for mode in fullkv rkv; do
    name="bench_${attn}_${mode}"
    extra=()
    if [[ "$mode" == rkv ]]; then
      extra=(--budget 1024 --buffer 128)
    fi
    (
      cd "$TEST_ROOT" || exit 125
      CUDA_VISIBLE_DEVICES=0 timeout 3600 "$PY" FlashInfer/benchmark/bench_rkv.py \
        --model "$MODEL" --mode "$mode" --attention "$attn" \
        --batch-size 16 --gen-len 2048 --prompt-len 512 --trials 2 \
        "${extra[@]}" > "$OUT/${name}.log" 2>&1
    )
    rc=$?
    json=$(grep -E '^\{' "$OUT/${name}.log" 2>/dev/null | tail -1 | head -c 300)
    echo "$name: exit=$rc $json"
    if [[ $rc -ne 0 ]] || [[ -z "$json" ]]; then
      overall=1
    fi
  done
done

echo "FA3_AB_RERUN_DONE rc=$overall $(date -u +%FT%TZ)"
exit "$overall"
