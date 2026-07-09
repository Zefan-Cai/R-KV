"""FlashInfer R-KV smoke: standalone FlashInferEngine (paged KV, eager),
Qwen3-0.6B, small budget so compaction fires well before max_new_tokens.
GenOutput reports num_compactions directly, so no monkeypatching is needed
to prove compression ran. Toggle with RKV_ON=0/1."""
import os
import sys
import time

_SMOKE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SMOKE_DIR)
# FlashInfer/rkv (the engine) must shadow the repo-root rkv/ reference package.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(_SMOKE_DIR)), "FlashInfer")
)
from health_check import report

MODEL = os.environ.get("RKV_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
RKV_ON = os.environ.get("RKV_ON", "1") == "1"
# RKV_ATTN=fa3 swaps the attention kernels to FlashAttention-3 (H100 line).
ATTN = os.environ.get("RKV_ATTN", "flashinfer")
if ATTN not in {"flashinfer", "fa3"}:
    raise SystemExit(f"RKV_ATTN must be flashinfer or fa3, got {ATTN!r}")

from transformers import AutoTokenizer  # noqa: E402
from rkv import FlashInferEngine, RKVConfig  # noqa: E402

if ATTN == "fa3":
    from rkv import FA3Engine as EngineCls
else:
    EngineCls = FlashInferEngine

if not os.path.isdir(MODEL):
    from huggingface_hub import snapshot_download

    MODEL = snapshot_download(MODEL)

tok = AutoTokenizer.from_pretrained(MODEL)
prompt_ids = tok.apply_chat_template(
    [{"role": "user", "content":
      "Solve step by step: A train travels 60 km in 45 minutes, then 80 km "
      "in 75 minutes. What is its average speed for the entire journey in "
      "km/h?"}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True)

# Budget small enough that even a short instruct-style answer (~150 tokens for
# non-reasoning models like Llama-3.2-1B-Instruct) crosses budget+buffer and
# actually exercises compaction; RKV_NEVER_TRIGGERED below depends on it.
RKV_BUDGET = int(os.environ.get("RKV_SMOKE_BUDGET", "64"))
RKV_BUFFER = int(os.environ.get("RKV_SMOKE_BUFFER", "16"))
rkv = RKVConfig(budget=RKV_BUDGET, buffer=RKV_BUFFER, window_size=8) if RKV_ON else None
t0 = time.time()
engine = EngineCls(MODEL, max_batch_size=1, max_seq_len=4096, rkv=rkv)
load_s = time.time() - t0

t0 = time.time()
# temperature 0.6: greedy decoding makes the 0.6B model fall into repetition
# loops regardless of R-KV, which would mask compression bugs.
outs = engine.generate([prompt_ids], max_new_tokens=1024, temperature=0.6,
                       top_p=0.95, stop_token_ids=(tok.eos_token_id,), seed=42)
gen_s = time.time() - t0

out = outs[0]
text = tok.decode(out.token_ids, skip_special_tokens=True)
tag = ("rkv_on" if RKV_ON else "rkv_off") + ("+fa3" if ATTN == "fa3" else "")
print(f"\n===== FLASHINFER ({tag}) =====")
print(f"Output({len(text)} chars): {text[:800]!r}")
print(f"RKV_COMPACTIONS {out.num_compactions} finish={out.finish_reason}")
ok = report(f"flashinfer[{tag}]", text)
if RKV_ON and out.num_compactions == 0:
    print("RKV_NEVER_TRIGGERED")
    ok = False
print(f"LOAD_SECONDS {load_s:.3f}")
print(f"GENERATE_SECONDS {gen_s:.3f}")
print("FLASHINFER_SMOKE_PASS" if ok else "FLASHINFER_SMOKE_FAIL")
sys.exit(0 if ok else 1)
