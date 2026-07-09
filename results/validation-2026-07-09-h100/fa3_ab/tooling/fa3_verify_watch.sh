#!/usr/bin/env bash
set -u

ROOT=/mnt/localssd/zefan/rkv-fi
TEST_ROOT=/mnt/localssd/zefan/rkv-fi-opt-test
CHAIN_LOG=/mnt/localssd/zefan/fa3_chain.log
VERIFY_LOG=/mnt/localssd/zefan/fa3_verify.log
PY="$ROOT/venv/bin/python"
OUT="$TEST_ROOT/validation"

exec > >(tee -a "$VERIFY_LOG") 2>&1
echo "[verify] start $(date -u +%FT%TZ)"

deadline=$((SECONDS + 14400))
while ! grep -q '^FA3_CHAIN_DONE ' "$CHAIN_LOG" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "FA3_VERIFIED_DONE rc=124 reason=chain_timeout"
    exit 124
  fi
  sleep 30
done

rc=0
if ! "$PY" -c 'import flash_attn_interface' >/dev/null 2>&1; then
  echo "[verify] import flash_attn_interface: FAIL"
  rc=1
else
  echo "[verify] import flash_attn_interface: PASS"
fi

for attn in fa3 flashinfer; do
  for on in 1 0; do
    name="smoke_${attn}_on${on}"
    if grep -q "^${name}: exit=0 " "$CHAIN_LOG" && [[ -s "$OUT/${name}.log" ]]; then
      echo "[verify] $name: PASS"
    else
      echo "[verify] $name: FAIL"
      rc=1
    fi
  done
done

for attn in flashinfer fa3; do
  for mode in rkv fullkv; do
    name="bench_${attn}_${mode}"
    if grep -q "^${name}: exit=0 " "$CHAIN_LOG" && grep -q '^{' "$OUT/${name}.log" 2>/dev/null; then
      echo "[verify] $name: PASS"
    else
      echo "[verify] $name: FAIL"
      rc=1
    fi
  done
done

if ! grep -q '^FA3_VALIDATION_DONE ' "$CHAIN_LOG"; then
  echo "[verify] completion marker: FAIL"
  rc=1
fi

echo "FA3_VERIFIED_DONE rc=$rc $(date -u +%FT%TZ)"
exit "$rc"
