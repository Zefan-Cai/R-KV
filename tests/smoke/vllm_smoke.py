"""vLLM R-KV smoke: Qwen3-0.6B, V1 engine, FLASH_ATTN backend.

R-KV is on when VLLM_V1_R_KV_BUDGET and VLLM_V1_R_KV_BUFFER are both > 0 (the port
requires both-or-neither); set both to 0 (or unset) to disable. Build the v0.25.1
patch-style port first (``cd vLLM && scripts/apply_rkv.sh && pip install -e
vllm-src``). Do NOT force enforce_eager — the port auto-selects PIECEWISE cudagraph
(attention stays eager so the R-KV hooks fire), which is the default this smoke
exercises. When R-KV is on it also asserts that physical compaction actually
fired (read from every worker via collective_rpc), so a silently-Full-KV build
fails instead of passing the lexical check.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from health_check import report

MODEL = os.environ.get("RKV_SMOKE_MODEL", "Qwen/Qwen3-0.6B")

# read_compactions() below sends a closure to every worker via collective_rpc;
# vLLM 0.25.1 refuses to serialize a callable across the engine-core process
# boundary unless this is set (mirrors vLLM/benchmark/eval.py). Must precede the
# vllm import.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def read_compactions(llm):
    """Max physical R-KV compactions across TP ranks, read from every worker's
    compactor via collective_rpc (mirrors vLLM/benchmark/eval.py). Returns None
    when unavailable — lets the smoke fail a build that silently ran Full-KV."""

    def _get(self):
        mr = getattr(self, "model_runner", None)
        comp = getattr(mr, "rkv_compactor", None)
        return int(getattr(comp, "_n_compactions", 0)) if comp is not None else 0

    try:
        counts = llm.collective_rpc(_get)
        return max(counts) if counts else 0
    except Exception:
        return None


prompts = [
    "Hello, my name is",
    "Solve step by step: A train travels 60 km in 45 minutes, then 80 km in "
    "75 minutes. What is its average speed for the entire journey in km/h?",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=800,
    stop_token_ids=[tokenizer.eos_token_id],
)

t0 = time.time()
llm = LLM(
    model=MODEL,
    # No enforce_eager: the v0.25.1 port auto-selects PIECEWISE cudagraph (attention
    # stays eager so the R-KV hooks still fire) — that is the default path to smoke.
    enable_prefix_caching=False,  # R-KV frees KV slots the prefix cache would reference
    gpu_memory_utilization=0.55,
)
load_s = time.time() - t0

t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
gen_s = time.time() - t0

tag = f"budget={os.environ.get('VLLM_V1_R_KV_BUDGET','?')},buffer={os.environ.get('VLLM_V1_R_KV_BUFFER','?')}"
all_ok = True
for i, output in enumerate(outputs):
    text = output.outputs[0].text
    print(f"\n===== PROMPT {i} ({tag}) =====")
    print(f"Prompt: {output.prompt!r}")
    print(f"Output({len(text)} chars): {text[:600]!r}")
    all_ok &= report(f"vllm[{tag}] prompt{i}", text)

# When R-KV is on, prove physical compaction actually fired — otherwise a build
# that silently runs Full-KV would still pass the lexical health check.
rkv_on = (int(os.environ.get("VLLM_V1_R_KV_BUDGET", "0") or 0) > 0
          and int(os.environ.get("VLLM_V1_R_KV_BUFFER", "0") or 0) > 0)
if rkv_on:
    n_comp = read_compactions(llm)
    print(f"RKV_COMPACTIONS {n_comp}")
    if not n_comp:
        print("RKV_NEVER_COMPACTED")
        all_ok = False

print(f"LOAD_SECONDS {load_s:.3f}")
print(f"GENERATE_SECONDS {gen_s:.3f}")
print("VLLM_SMOKE_PASS" if all_ok else "VLLM_SMOKE_FAIL")
sys.exit(0 if all_ok else 1)
