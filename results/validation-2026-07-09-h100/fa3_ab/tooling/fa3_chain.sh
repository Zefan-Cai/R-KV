#!/usr/bin/env bash
# rkv-fi-bench-1: wait for env -> build FA3 -> validate FA3 vs FlashInfer.
set -u
ROOT=/mnt/localssd/zefan/rkv-fi
T=/mnt/localssd/zefan/rkv-fi-opt-test
PY=$ROOT/venv/bin/python
M=$ROOT/models
OUT=$T/validation
mkdir -p "$OUT"

echo "[chain] waiting for venv torch ($(date -u +%FT%TZ))"
until grep -q "cuda_available True" $ROOT/setup.log 2>/dev/null; do sleep 30; done
echo "[chain] torch ready; building FA3 ($(date -u +%FT%TZ))"

cd /mnt/localssd/zefan
rm -rf flash-attention
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git 2>&1 | tail -1
cd flash-attention/hopper
MAX_JOBS=48 timeout 10800 $PY setup.py install 2>&1 | tail -3
if ! $PY -c "import flash_attn_interface" 2>/dev/null; then
  echo "FA3_INSTALL_FAILED"; echo "FA3_CHAIN_DONE rc=1"; exit 1
fi
echo "FA3_INSTALL_DONE ($(date -u +%FT%TZ))"

echo "[chain] waiting for models ($(date -u +%FT%TZ))"
until grep -q "setup done" $ROOT/setup.log 2>/dev/null; do sleep 30; done

echo "== fa3 introspection =="
$PY - <<'PYEOF'
import inspect
import flash_attn_interface as fa
print("module:", fa.__file__)
for fn in ("flash_attn_with_kvcache", "flash_attn_varlen_func"):
    f = getattr(fa, fn, None)
    print(fn, "->", "MISSING" if f is None else str(inspect.signature(f))[:400])
PYEOF

echo "== smokes =="
for attn in fa3 flashinfer; do
  for on in 1 0; do
    ( cd "$T" && RKV_SMOKE_MODEL=$M/DeepSeek-R1-Distill-Qwen-1.5B RKV_ON=$on RKV_ATTN=$attn \
      CUDA_VISIBLE_DEVICES=0 timeout 900 $PY tests/smoke/flashinfer_smoke.py \
      > "$OUT/smoke_${attn}_on$on.log" 2>&1 )
    echo "smoke_${attn}_on$on: exit=$? $(tail -1 "$OUT/smoke_${attn}_on$on.log")"
  done
done

echo "== bench A/B (same node, same code, attention only) =="
for attn in flashinfer fa3; do
  for mode in rkv fullkv; do
    extra=""; [ "$mode" = rkv ] && extra="--budget 1024 --buffer 128"
    ( cd "$T" && CUDA_VISIBLE_DEVICES=0 timeout 3600 \
      $PY FlashInfer/benchmark/bench_rkv.py --model $M/DeepSeek-R1-Distill-Qwen-1.5B \
        --mode $mode --attention $attn --batch-size 16 --gen-len 2048 --prompt-len 512 \
        --trials 2 $extra > "$OUT/bench_${attn}_${mode}.log" 2>&1 )
    echo "bench_${attn}_${mode}: exit=$? $(grep -E '^\{' "$OUT/bench_${attn}_${mode}.log" | tail -1 | head -c 200)"
  done
done
echo "FA3_VALIDATION_DONE $(date -u +%FT%TZ)"
echo "FA3_CHAIN_DONE rc=0"
