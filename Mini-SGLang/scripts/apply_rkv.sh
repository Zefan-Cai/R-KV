#!/usr/bin/env bash
#
# apply_rkv.sh — reproducibly build a patched Mini-SGLang tree with R-KV enabled.
#
# What it does (all pinned, no surprises):
#   1. Clones upstream Mini-SGLang at the EXACT commit R-KV was ported against.
#   2. Copies the standalone R-KV package (rkv/) into the Mini-SGLang source tree
#      as `python/minisgl/rkv/`.
#   3. Applies the small wiring patch (9 files) that hooks R-KV into the runtime.
#
# After this, run the server / benchmark with benchmark/launch_server.sh (it
# points PYTHONPATH at the tree produced here).
#
# Usage:
#   scripts/apply_rkv.sh                 # clone into ./mini-sglang-src
#   RKV_MINISGL_SRC=/abs/path scripts/apply_rkv.sh   # clone elsewhere
#   scripts/apply_rkv.sh --force         # overwrite an existing checkout
#
set -euo pipefail

# --- Pinned upstream reference. Do not change casually: the patch is generated
#     against this exact commit and applies cleanly to it. ---
MINISGL_REPO="${MINISGL_REPO:-https://github.com/sgl-project/mini-sglang.git}"
MINISGL_COMMIT="${MINISGL_COMMIT:-9a91cfafe754aa85daee49998176275667eb58f2}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"        # the Mini-SGLang/ folder of this repo
MINISGL_SRC="${RKV_MINISGL_SRC:-$HERE/mini-sglang-src}"
PATCH="$HERE/patch/rkv-mini-sglang-9a91cfa.patch"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if [[ -e "$MINISGL_SRC" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    echo ">> removing existing $MINISGL_SRC (--force)"
    rm -rf "$MINISGL_SRC"
  else
    echo "ERROR: $MINISGL_SRC already exists. Re-run with --force to overwrite," >&2
    echo "       or set RKV_MINISGL_SRC to a different path." >&2
    exit 1
  fi
fi

echo ">> [1/3] Cloning Mini-SGLang @ $MINISGL_COMMIT"
echo "         $MINISGL_REPO"
git clone "$MINISGL_REPO" "$MINISGL_SRC"
git -C "$MINISGL_SRC" checkout --quiet "$MINISGL_COMMIT"
echo "         checked out $(git -C "$MINISGL_SRC" rev-parse --short HEAD) (detached)"

echo ">> [2/3] Installing R-KV package into the Mini-SGLang tree"
DEST="$MINISGL_SRC/python/minisgl/rkv"
mkdir -p "$DEST"
cp "$HERE"/rkv/*.py "$DEST"/
echo "         copied $(ls "$HERE"/rkv/*.py | wc -l | tr -d ' ') files -> $DEST"

echo ">> [3/3] Applying R-KV wiring patch (9 files)"
git -C "$MINISGL_SRC" apply --check "$PATCH"     # fail loudly if it won't apply cleanly
git -C "$MINISGL_SRC" apply --whitespace=nowarn "$PATCH"
echo "         patch applied cleanly"

cat <<EOF

Done. Patched Mini-SGLang is at:
  $MINISGL_SRC

Next steps (see README.md for the full guide):
  1. Install Mini-SGLang from the patched tree (see its own README; e.g. uv):
       cd "$MINISGL_SRC" && uv venv --python=3.12 && source .venv/bin/activate
       uv pip install -e .
  2. Run the offline R-KV benchmark:
       python3 benchmark/bench_rkv.py
  3. Or the CPU-only algorithm test (no GPU / install needed):
       python3 tests/test_rkv_algorithm.py
EOF
