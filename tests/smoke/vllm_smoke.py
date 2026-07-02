"""vLLM R-KV smoke: Qwen3-0.6B, V1 engine, FLASH_ATTN backend.

R-KV is switched on/off via VLLM_V1_R_KV_BUFFER (0 disables).
Run under envs/vllm with the checked-in vLLM/vllm package overlaid.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from health_check import report

MODEL = os.environ.get("RKV_SMOKE_MODEL", "Qwen/Qwen3-0.6B")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

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
    enforce_eager=True,
    enable_prefix_caching=False,
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

print(f"LOAD_SECONDS {load_s:.3f}")
print(f"GENERATE_SECONDS {gen_s:.3f}")
print("VLLM_SMOKE_PASS" if all_ok else "VLLM_SMOKE_FAIL")
sys.exit(0 if all_ok else 1)
