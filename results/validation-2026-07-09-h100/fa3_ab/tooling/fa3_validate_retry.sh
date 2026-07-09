#!/usr/bin/env bash
set -uo pipefail

ROOT=/mnt/localssd/zefan/rkv-fi
TEST_ROOT=/mnt/localssd/zefan/rkv-fi-opt-test
PY="$ROOT/venv/bin/python"
MODEL="$ROOT/models/DeepSeek-R1-Distill-Qwen-1.5B"
OUT="$TEST_ROOT/validation"
LOG=/mnt/localssd/zefan/fa3_retry.log

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "[retry] start $(date -u +%FT%TZ)"
overall=0

echo "== fa3 introspection =="
"$PY" - <<'PY'
import inspect
import flash_attn_interface as fa
print("module:", fa.__file__)
for name in ("flash_attn_with_kvcache", "flash_attn_varlen_func"):
    fn = getattr(fa, name, None)
    print(name, "->", "MISSING" if fn is None else inspect.signature(fn))
PY
if [[ $? -ne 0 ]]; then
  overall=1
fi

echo "== smokes =="
for attn in fa3 flashinfer; do
  for on in 1 0; do
    name="smoke_${attn}_on${on}"
    (
      cd "$TEST_ROOT" || exit 125
      RKV_SMOKE_MODEL="$MODEL" RKV_ON="$on" RKV_ATTN="$attn" \
        CUDA_VISIBLE_DEVICES=0 timeout 900 "$PY" tests/smoke/flashinfer_smoke.py \
        > "$OUT/${name}.log" 2>&1
    )
    rc=$?
    tail_line=$(tail -1 "$OUT/${name}.log" 2>/dev/null || true)
    echo "$name: exit=$rc $tail_line"
    if [[ $rc -ne 0 ]] || ! grep -q 'FLASHINFER_SMOKE_PASS' "$OUT/${name}.log"; then
      overall=1
    fi
  done
done

echo "== bench A/B (same node, same code, attention only) =="
for attn in flashinfer fa3; do
  for mode in rkv fullkv; do
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

echo "FA3_RETRY_DONE rc=$overall $(date -u +%FT%TZ)"
exit "$overall"
