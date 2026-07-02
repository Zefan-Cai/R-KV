"""Mini-SGLang R-KV smoke: offline LLM engine, CUDA graphs OFF
(cuda_graph_max_bs=0 — None silently auto-enables graphs and skips R-KV).
Instruments RKVCompressor.maybe_compress to prove compression fires.
Toggle with RKV_ON=0/1."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from health_check import report

MODEL = os.environ.get("RKV_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
RKV_ON = os.environ.get("RKV_ON", "1") == "1"

from minisgl.rkv.integration import RKVCompressor  # noqa: E402
from minisgl.core import SamplingParams  # noqa: E402
from minisgl.llm import LLM  # noqa: E402

stats = {"calls": 0}
_orig = RKVCompressor.maybe_compress

def _patched(self, *a, **k):
    r = _orig(self, *a, **k)
    if r is not None:
        stats["calls"] += 1
    return r

RKVCompressor.maybe_compress = _patched

kwargs = dict(
    max_seq_len_override=8192,
    cuda_graph_max_bs=0,
    page_size=1,
    cache_type="naive",
    attention_backend="fi",
)
if RKV_ON:
    kwargs.update(rkv_enabled=True, rkv_budget=512, rkv_buffer=64,
                  rkv_window_size=8, rkv_kernel_size=7, rkv_mix_lambda=0.07,
                  rkv_retain_ratio=0.1, rkv_retain_direction="last")

t0 = time.time()
llm = LLM(MODEL, **kwargs)
load_s = time.time() - t0

prompt = ("Solve this problem step by step, showing all reasoning: "
          "A train travels 60 km in 45 minutes, then 80 km in 75 minutes. "
          "What is its average speed for the entire journey in km/h? "
          "Answer: Let's think step by step.")

t0 = time.time()
# temperature 0.6: greedy decoding makes the 0.6B base model fall into
# repetition loops regardless of R-KV, which would mask compression bugs.
out = llm.generate([prompt], SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1500))
gen_s = time.time() - t0

text = out[0]["text"] if isinstance(out[0], dict) else str(out[0])
tag = "rkv_on" if RKV_ON else "rkv_off"
print(f"\n===== MINISGL ({tag}) =====")
print(f"Output({len(text)} chars): {text[:800]!r}")
print(f"RKV_COMPRESS_CALLS {stats['calls']}")
ok = report(f"minisgl[{tag}]", text)
if RKV_ON and stats["calls"] == 0:
    print("RKV_NEVER_TRIGGERED")
    ok = False
print(f"LOAD_SECONDS {load_s:.3f}")
print(f"GENERATE_SECONDS {gen_s:.3f}")
print("MINISGL_SMOKE_PASS" if ok else "MINISGL_SMOKE_FAIL")
sys.exit(0 if ok else 1)
