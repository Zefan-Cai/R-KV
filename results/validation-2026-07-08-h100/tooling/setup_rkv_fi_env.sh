#!/usr/bin/env bash
# One-shot environment bootstrap for R-KV FlashInfer benchmarks on a Pluto H100 pod.
# Idempotent: safe to re-run. Everything lives under /mnt/localssd/zefan/rkv-fi.
set -euo pipefail

ROOT=/mnt/localssd/zefan/rkv-fi
VENV="$ROOT/venv"
MODELS="$ROOT/models"
LOG="$ROOT/setup.log"
mkdir -p "$ROOT" "$MODELS"
exec > >(tee -a "$LOG") 2>&1
echo "=== setup start $(date -u +%FT%TZ) ==="

# Tokens (HUGGINGFACE_TOKEN etc.) without printing them
if [ -f "$HOME/.codex/configs.bash" ]; then
  set +x; source "$HOME/.codex/configs.bash"; set -x >/dev/null 2>&1 || true
fi

# uv for fast, reproducible python management
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$VENV/bin/python" ]; then
  uv venv -p 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python - <<'EOF' 2>/dev/null && SKIP_INSTALL=1 || SKIP_INSTALL=0
import torch, flashinfer, transformers, safetensors
assert torch.__version__.startswith("2.11.0"), torch.__version__
assert flashinfer.__version__ == "0.6.12", flashinfer.__version__
print("env already satisfied")
EOF
if [ "${SKIP_INSTALL}" != "1" ]; then
  uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
  uv pip install flashinfer-python==0.6.12 flashinfer-cubin==0.6.12
  uv pip install "transformers==5.8.1" safetensors accelerate datasets "huggingface_hub[hf_transfer]" hf_transfer
fi

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
EOF

# Model downloads (skip if present). hf_transfer for speed.
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
dl() {
  local repo="$1" dest="$MODELS/$(basename "$1")"
  if [ -f "$dest/config.json" ]; then echo "have $dest"; return 0; fi
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$dest', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','*.jinja'])
print('downloaded $repo')
"
}
dl deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
dl deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
dl Qwen/Qwen3-0.6B
dl Qwen/Qwen3-8B
dl meta-llama/Llama-3.2-1B-Instruct
dl meta-llama/Llama-3.1-8B-Instruct

echo "=== setup done $(date -u +%FT%TZ) ==="
