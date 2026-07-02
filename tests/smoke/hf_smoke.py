"""HuggingFace monkeypatch smoke (issue #26/#17 repro check).

Runs DeepSeek-R1-Distill-Qwen-1.5B on one GSM8K-style problem with:
  MODE=fullkv  -> stock transformers forward (no patch)
  MODE=rkv     -> replace_qwen2 + R-KV compression
Checks the output stays coherent under the pinned transformers range.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from health_check import report

MODEL = os.environ.get("RKV_SMOKE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
MODE = os.environ.get("MODE", "rkv")

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

print(f"transformers={transformers.__version__} torch={torch.__version__} mode={MODE}")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "HuggingFace"))

prompt = (
    "You are given a math problem.\n\nProblem: Natalia sold clips to 48 of "
    "her friends in April, and then she sold half as many clips in May. How "
    "many clips did Natalia sell altogether in April and May?\n\nYou need to "
    "solve the problem step by step. Provide the final answer in the format: "
    "Final answer: \\boxed{}"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

if MODE == "rkv":
    from rkv.monkeypatch import replace_qwen2

    compression_config = {
        "method": "rkv",
        "method_config": {
            "budget": 512,
            "window_size": 8,
            "mix_lambda": 0.1,
            "retain_ratio": 0.2,
            "retain_direction": "last",
            "record_kept_token_indices": False,
        },
        "compression": None,
        "update_kv": True,
    }
    replace_qwen2(compression_config)

attn_impl = os.environ.get("ATTN", "flash_attention_2")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation=attn_impl
).eval()

if MODE == "rkv":
    model.config.update({
        "divide_method": "step_length",
        "divide_length": 128,
        "compression_content": "all",
    })
    model.newline_token_ids = [
        tokenizer.encode("\n")[-1],
        tokenizer.encode(".\n")[-1],
        tokenizer.encode(")\n")[-1],
        tokenizer.encode("\n\n")[-1],
        tokenizer.encode(".\n\n")[-1],
        tokenizer.encode(")\n\n")[-1],
    ]
    model.after_think_token_ids = [tokenizer.encode("</think>")[-1]]

model.to("cuda")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

t0 = time.time()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=1500,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
gen_s = time.time() - t0

text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\n===== HF ({MODE}, transformers {transformers.__version__}) =====")
print(f"Output({len(text)} chars): {text[:700]!r}")
ok = report(f"hf[{MODE}]", text)
print(f"GENERATE_SECONDS {gen_s:.3f}")
print("HF_SMOKE_PASS" if ok else "HF_SMOKE_FAIL")
sys.exit(0 if ok else 1)
