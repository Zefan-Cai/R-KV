"""Nano-vLLM R-KV smoke: Qwen3-0.6B local dir, eager, small budget so
compression fires well before max_tokens. Toggle with RKV_ON=0/1."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from health_check import report

MODEL = os.environ.get("RKV_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
RKV_ON = os.environ.get("RKV_ON", "1") == "1"

from transformers import AutoTokenizer  # noqa: E402
from nanovllm import LLM, SamplingParams  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
kwargs = dict(enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.55)
if RKV_ON:
    kwargs.update(rkv_enabled=True, rkv_budget=256, rkv_buffer=32,
                  rkv_window_size=8)

t0 = time.time()
llm = LLM(MODEL, **kwargs)
load_s = time.time() - t0

prompt = tok.apply_chat_template(
    [{"role": "user", "content":
      "Solve step by step: A train travels 60 km in 45 minutes, then 80 km "
      "in 75 minutes. What is its average speed for the entire journey in "
      "km/h?"}],
    tokenize=False, add_generation_prompt=True)

sp = SamplingParams(temperature=0.6, max_tokens=1024)
t0 = time.time()
outputs = llm.generate([prompt], sp)
gen_s = time.time() - t0

text = outputs[0]["text"]
tag = "rkv_on" if RKV_ON else "rkv_off"
print(f"\n===== NANO ({tag}) =====")
print(f"Output({len(text)} chars): {text[:800]!r}")
ok = report(f"nano[{tag}]", text)
print(f"LOAD_SECONDS {load_s:.3f}")
print(f"GENERATE_SECONDS {gen_s:.3f}")
print("NANO_SMOKE_PASS" if ok else "NANO_SMOKE_FAIL")
sys.exit(0 if ok else 1)
