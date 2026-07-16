#!/usr/bin/env bash
#
# apply_rkv.sh — reproducibly build a patched vLLM tree with R-KV enabled.
#
# What it does (all pinned, no surprises):
#   1. Clones upstream vLLM at the EXACT tag R-KV was ported against (v0.25.1).
#   2. Copies the standalone R-KV package (rkv/) into the vLLM source tree
#      (as vllm/rkv/).
#   3. Applies the small wiring patch (13 files) that hooks R-KV into the v1
#      runtime.
#
# After this, install the tree (see README.md) and run with R-KV enabled by
# setting VLLM_V1_R_KV_BUDGET / VLLM_V1_R_KV_BUFFER.
#
# Usage:
#   scripts/apply_rkv.sh                 # clone into ./vllm-src
#   RKV_VLLM_SRC=/abs/path scripts/apply_rkv.sh     # clone elsewhere
#   scripts/apply_rkv.sh --force         # overwrite an existing checkout
#
set -euo pipefail

# --- Pinned upstream reference (release tag v0.25.1). Do not change casually:
#     the patch is generated against this exact commit and applies cleanly to
#     it. ---
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_TAG="${VLLM_TAG:-v0.25.1}"
VLLM_COMMIT="${VLLM_COMMIT:-752a3a504485790a2e8491cacbb35c137339ad34}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"        # the vLLM/ folder of this repo
VLLM_SRC="${RKV_VLLM_SRC:-$HERE/vllm-src}"
PATCH="$HERE/patch/rkv-vllm-0.25.1.patch"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

if [[ -e "$VLLM_SRC" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    echo ">> removing existing $VLLM_SRC (--force)"
    rm -rf "$VLLM_SRC"
  else
    echo "ERROR: $VLLM_SRC already exists. Re-run with --force to overwrite," >&2
    echo "       or set RKV_VLLM_SRC to a different path." >&2
    exit 1
  fi
fi

echo ">> [1/3] Cloning vLLM @ $VLLM_TAG ($VLLM_COMMIT)"
echo "         $VLLM_REPO"
git clone --depth 1 --branch "$VLLM_TAG" "$VLLM_REPO" "$VLLM_SRC"
# Assert we are on the exact pinned commit the patch was generated against.
GOT="$(git -C "$VLLM_SRC" rev-parse HEAD)"
if [[ "$GOT" != "$VLLM_COMMIT" ]]; then
  echo "ERROR: tag $VLLM_TAG resolved to $GOT, expected $VLLM_COMMIT." >&2
  echo "       The upstream tag may have moved; set VLLM_COMMIT to override." >&2
  exit 1
fi
echo "         checked out $(git -C "$VLLM_SRC" rev-parse --short HEAD)"

echo ">> [2/3] Installing R-KV package into the vLLM tree"
DEST="$VLLM_SRC/vllm/rkv"
mkdir -p "$DEST"
cp "$HERE"/rkv/*.py "$DEST"/
echo "         copied $(ls "$HERE"/rkv/*.py | wc -l | tr -d ' ') files -> $DEST"

echo ">> [3/3] Applying R-KV wiring patch (13 files)"
git -C "$VLLM_SRC" apply --check "$PATCH"     # fail loudly if it won't apply cleanly
git -C "$VLLM_SRC" apply --whitespace=nowarn "$PATCH"
echo "         patch applied cleanly"

cat <<EOF

Done. Patched vLLM is at:
  $VLLM_SRC

Next steps (see README.md for the full guide):
  1. Install the patched tree (a source build; needs CUDA + a GPU):
       pip install -e "$VLLM_SRC"
     (or follow vLLM's build-from-source instructions for your platform)
  2. Serve a model with R-KV at best throughput (decode-time compression):
       VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=64 VLLM_V1_R_KV_ASYNC=1 \\
         vllm serve Qwen/Qwen2.5-Math-7B-Instruct
     (or just: benchmark/launch_server.sh rkv 256 -- best-throughput flags baked
      in. Do NOT pass --enforce-eager: R-KV auto-selects PIECEWISE cudagraph.)
  3. R-KV activates automatically once BUDGET and BUFFER are > 0.
EOF
