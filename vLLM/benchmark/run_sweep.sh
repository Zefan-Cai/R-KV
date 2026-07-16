#!/usr/bin/env bash
# R-KV budget x buffer sweep on GSM8K, one config per GPU, waves of 8 GPUs.
# Produces one JSON per config in $OUTDIR; aggregate into RESULTS_H100.md.
#
# Usage:  bash run_sweep.sh            # full 13-config sweep, 200 questions
#         SWEEP_N=100 bash run_sweep.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"           # vLLM/benchmark
VLLM_ROOT="$(cd "$HERE/.." && pwd)"
cd "$VLLM_ROOT"
# shellcheck disable=SC1091
source .venv-rkv/bin/activate

OUTDIR="${OUTDIR:-/tmp/sweep_results}"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.json "$OUTDIR"/*.log

# budget:buffer  (first entry = Full-KV baseline)
configs=(
  "0:0"
  "128:16" "128:64" "128:128" "128:256"
  "256:16" "256:64" "256:128" "256:256"
  "512:16" "512:64" "512:128" "512:256"
)

NGPU="${NGPU:-8}"
N="${SWEEP_N:-200}"
MAXTOK="${SWEEP_MAXTOK:-512}"

i=0
for cfg in "${configs[@]}"; do
  budget="${cfg%:*}"
  buffer="${cfg#*:}"
  gpu=$((i % NGPU))
  name="b${budget}_buf${buffer}"
  echo ">> launch $name on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
    VLLM_V1_R_KV_BUDGET=$budget VLLM_V1_R_KV_BUFFER=$buffer \
    RKV_N=$N RKV_MAXTOK=$MAXTOK RKV_OUT="$OUTDIR/$name.json" \
    python "$HERE/bench_sweep.py" > "$OUTDIR/$name.log" 2>&1 &
  i=$((i + 1))
  # End of a wave: wait so the next wave reuses GPUs without collision.
  if [ $((i % NGPU)) -eq 0 ]; then
    echo "   ... waiting for wave to finish"
    wait
  fi
done
wait
echo "ALL DONE ($i configs) -> $OUTDIR"

python - "$OUTDIR" << 'PY'
import json, glob, sys
rows = [json.load(open(f)) for f in glob.glob(f"{sys.argv[1]}/*.json")]
rows.sort(key=lambda r: (-1 if r["budget"] == 0 else r["budget"], r["buffer"]))
print(f'{"config":<14}{"acc":>8}{"decode tok/s":>14}{"wall s":>9}{"avg gen":>9}')
for r in rows:
    print(f'{r["tag"]:<14}{r["accuracy"]*100:>7.1f}%{r["decode_tok_s"]:>14}'
          f'{r["wall_s"]:>9}{r["avg_gen_len"]:>9}')
PY
