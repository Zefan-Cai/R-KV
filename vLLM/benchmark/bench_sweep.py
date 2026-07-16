"""One R-KV config (budget,buffer from env) on SGLang's few-shot GSM8K harness.

Measures accuracy + offline batched throughput. budget=0/buffer=0 => Full-KV.
Dumps a JSON summary to RKV_OUT. Designed to be launched one-per-GPU in parallel
by run_sweep.sh.

Env knobs: VLLM_V1_R_KV_BUDGET, VLLM_V1_R_KV_BUFFER, RKV_N (200), RKV_MAXTOK (512),
RKV_CONC (0=default max_num_seqs), RKV_NOASYNC (1 => sync sched for Full-KV),
RKV_OUT, RKV_MODEL, RKV_DATA.
"""
import json
import os
import re
import time

from vllm import LLM, SamplingParams

DATA = os.environ.get(
    "RKV_DATA", "/home/sigma/github/r-kv/SGLang/benchmark/data/gsm8k_fewshot.jsonl"
)
MODEL = os.environ.get("RKV_MODEL", "/data/model/Qwen2.5-Math-7B-Instruct")
N = int(os.environ.get("RKV_N", "200"))
MAXTOK = int(os.environ.get("RKV_MAXTOK", "512"))
CONC = int(os.environ.get("RKV_CONC", "0"))  # 0 = unbounded (default max_num_seqs)
OUT = os.environ.get("RKV_OUT", "/tmp/bench_result.json")
budget = int(os.environ.get("VLLM_V1_R_KV_BUDGET", "0"))
buffer = int(os.environ.get("VLLM_V1_R_KV_BUFFER", "0"))
tag = f"b{budget}/buf{buffer}" if (budget and buffer) else "full-kv"


def extract_gold(a):
    m = re.search(r"####\s*([\-0-9\.,/]+)", a)
    return m.group(1).replace(",", "").rstrip(".") if m else None


_PATS = [r"\\boxed\{([^{}]*)\}",
         r"[Tt]he final answer is[:\s]*\$?\\?\(?\$?([\-0-9\.,/]+)",
         r"[Tt]he answer is[:\s]*\$?([\-0-9\.,/]+)"]


def extract_pred(t):
    for pat in _PATS:
        f = re.findall(pat, t)
        if f:
            return f[-1].replace(",", "").replace("$", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "") if nums else None


def to_num(x):
    if x is None:
        return None
    x = x.strip()
    try:
        if "/" in x:
            a, b = x.split("/"); return float(a) / float(b)
        return float(x)
    except Exception:
        return None


data = [json.loads(l) for l in open(DATA)][:N]
prompts = [d["request"]["messages"][0]["content"] for d in data]
sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, stop=["\nProblem"])

kw = {}
if CONC > 0:
    kw["max_num_seqs"] = CONC
if os.environ.get("RKV_NOASYNC") == "1":
    kw["async_scheduling"] = False
# RKV_EAGER=1 (default) forces eager; RKV_EAGER=0 lets R-KV run under PIECEWISE
# cudagraph (the config auto-forces PIECEWISE when R-KV is on and not eager).
_eager = os.environ.get("RKV_EAGER", "1") == "1"
llm = LLM(model=MODEL, enforce_eager=_eager, gpu_memory_utilization=0.85,
          max_model_len=4096, disable_log_stats=True, seed=0, block_size=16, **kw)
tok = llm.get_tokenizer()
in_toks = sum(len(tok(p).input_ids) for p in prompts)

t0 = time.time()
outs = llm.generate(prompts, sp)
dt = time.time() - t0

correct = 0
out_toks = 0
for d, o in zip(data, outs):
    out_toks += len(o.outputs[0].token_ids)
    g = to_num(extract_gold(d["answer"]))
    p = to_num(extract_pred(o.outputs[0].text))
    correct += int(g is not None and p is not None and abs(g - p) < 1e-4)

res = {
    "tag": tag, "budget": budget, "buffer": buffer, "n": len(data),
    "maxtok": MAXTOK, "max_num_seqs": (CONC or "default"),
    "accuracy": round(correct / len(data), 4), "correct": correct,
    "wall_s": round(dt, 1),
    "out_tokens": out_toks, "in_tokens": in_toks,
    "decode_tok_s": round(out_toks / dt, 1),
    "total_tok_s": round((out_toks + in_toks) / dt, 1),
    "avg_gen_len": round(out_toks / len(data), 1),
}
json.dump(res, open(OUT, "w"))
print(f"DONE {tag}: acc={res['accuracy']} decode={res['decode_tok_s']}tok/s "
      f"wall={res['wall_s']}s -> {OUT}", flush=True)
