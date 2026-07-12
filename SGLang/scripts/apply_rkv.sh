#!/usr/bin/env bash
#
# apply_rkv.sh — reproducibly build a patched SGLang tree with R-KV enabled.
#
# What it does (all pinned, no surprises):
#   1. Clones upstream SGLang at the EXACT commit R-KV was ported against.
#   2. Copies the standalone R-KV package (rkv/) into the SGLang source tree.
#   3. Applies the small wiring patch (9 files) that hooks R-KV into the runtime.
#
# After this, run the server with benchmark/launch_server.sh (it points
# PYTHONPATH at the tree produced here).
#
# Usage:
#   scripts/apply_rkv.sh                 # clone into ./sglang-src
#   RKV_SGLANG_SRC=/abs/path scripts/apply_rkv.sh   # clone elsewhere
#   scripts/apply_rkv.sh --force         # overwrite an existing checkout
#
set -euo pipefail

# --- Pinned upstream reference (release/v0.5.14). Do not change casually: the
#     patch is generated against this exact commit and applies cleanly to it. ---
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
SGLANG_COMMIT="${SGLANG_COMMIT:-49e384ce9d304648e9959666ecb8ce8cd98d0deb}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"        # the SGLang/ folder of this repo
SGLANG_SRC="${RKV_SGLANG_SRC:-$HERE/sglang-src}"
PATCH="$HERE/patch/rkv-sglang-0.5.14.patch"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if [[ -e "$SGLANG_SRC" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    echo ">> removing existing $SGLANG_SRC (--force)"
    rm -rf "$SGLANG_SRC"
  else
    echo "ERROR: $SGLANG_SRC already exists. Re-run with --force to overwrite," >&2
    echo "       or set RKV_SGLANG_SRC to a different path." >&2
    exit 1
  fi
fi

echo ">> [1/3] Cloning SGLang @ $SGLANG_COMMIT"
echo "         $SGLANG_REPO"
git clone "$SGLANG_REPO" "$SGLANG_SRC"
git -C "$SGLANG_SRC" checkout --quiet "$SGLANG_COMMIT"
echo "         checked out $(git -C "$SGLANG_SRC" rev-parse --short HEAD) (detached)"

echo ">> [2/3] Installing R-KV package into the SGLang tree"
DEST="$SGLANG_SRC/python/sglang/srt/mem_cache/rkv"
mkdir -p "$DEST"
cp "$HERE"/rkv/*.py "$DEST"/
echo "         copied $(ls "$HERE"/rkv/*.py | wc -l | tr -d ' ') files -> $DEST"

echo ">> [3/3] Applying R-KV wiring patch (9 files)"
git -C "$SGLANG_SRC" apply --check "$PATCH"     # fail loudly if it won't apply cleanly
git -C "$SGLANG_SRC" apply --whitespace=nowarn "$PATCH"
echo "         patch applied cleanly"

cat <<EOF

Done. Patched SGLang is at:
  $SGLANG_SRC

Next steps (see README.md for the full guide):
  1. Install the verified dependency stack:
       pip install -e "$SGLANG_SRC/python" \\
         --extra-index-url https://docs.sglang.ai/whl/cu129/
       pip install -r "$HERE/requirements-rkv.txt" \\
         --extra-index-url https://docs.sglang.ai/whl/cu129/
  2. Launch a server with R-KV on:
       MODEL=/path/to/Qwen2.5-Math-7B-Instruct benchmark/launch_server.sh rkv 512
  3. Run the eval (batch>1):
       python3 benchmark/eval.py --n 20 --concurrency 8 --label rkv_b512
EOF
