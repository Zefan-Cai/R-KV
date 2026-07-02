"""Offline GSM8K accuracy + throughput for Mini-SGLang R-KV (Math-7B).

Mirrors the SGLang benchmark (few-shot MATH prompts, numeric judging) but drives
the *offline* LLM engine (Mini-SGLang has no R-KV server flags). Batched
generation is the concurrency proxy.

Usage (inside the patched-tree venv):
    python3 eval_math.py --budget 512 --n 20     # 0 = full KV (R-KV off)

SamplingParams has no stop strings, so we truncate each completion at the next
"\\nProblem" to emulate the SGLang harness's stop=["\\nProblem"].
"""
import argparse
import json
import os
import re
import time

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from minisgl.rkv.integration import RKVCompressor


def extract_gold(answer: str):
    m = re.search(r"####\s*([\-0-9\.,/]+)", answer)
    return m.group(1).replace(",", "").rstrip(".") if m else None


_PATS = [
    r"\\boxed\{([^{}]*)\}",
    r"####\s*([\-0-9\.,/]+)",
    r"[Tt]he final answer is[:\s]*\$?\\?\(?\$?([\-0-9\.,/]+)",
    r"[Tt]he answer is[:\s]*\$?([\-0-9\.,/]+)",
]


def extract_pred(text: str):
    # Emulate stop=["\nProblem"]: only judge the answer to THIS problem.
    text = re.split(r"\nProblem", text)[0]
    for pat in _PATS:
        found = re.findall(pat, text)
        if found:
            return found[-1].replace(",", "").replace("$", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def to_num(x):
    if x is None:
        return None
    x = x.strip()
    try:
        if "/" in x:
            a, b = x.split("/")
            return float(a) / float(b)
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=0, help="0 = full KV (R-KV off)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="/data/model/Qwen2.5-Math-7B-Instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "gsm8k_fewshot.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--buffer", type=int, default=64)
    args = ap.parse_args()

    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= args.n:
                break
    prompts = [d["request"]["messages"][0]["content"] for d in data]
    golds = [extract_gold(d["answer"]) for d in data]

    # Count real compaction events.
    _orig = RKVCompressor.maybe_compress
    stats = {"n": 0}

    def _wrap(self, *a, **k):
        r = _orig(self, *a, **k)
        if r is not None:
            stats["n"] += 1
        return r

    RKVCompressor.maybe_compress = _wrap

    kwargs = dict(page_size=1, cuda_graph_max_bs=0, max_seq_len_override=4096,
                  memory_ratio=0.85, attention_backend="fi")
    label = "full KV (R-KV off)"
    if args.budget > 0:
        kwargs.update(rkv_enabled=True, rkv_budget=args.budget,
                      rkv_window_size=args.window, rkv_buffer=args.buffer)
        label = f"R-KV budget={args.budget}"

    llm = LLM(args.model, **kwargs)
    llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=4))
    sp = [SamplingParams(temperature=0.0, max_tokens=args.max_tokens) for _ in prompts]
    t = time.time()
    res = llm.generate(prompts, sp)
    dt = time.time() - t

    correct, toks = 0, 0
    for gold, r in zip(golds, res):
        nt = len(r["token_ids"]); toks += nt
        pred = extract_pred(r["text"])
        g, p = to_num(gold), to_num(pred)
        ok = g is not None and p is not None and abs(g - p) < 1e-4
        correct += ok

    print(
        f"RESULT | {label} | n={len(prompts)} | acc={correct}/{len(prompts)}="
        f"{correct/len(prompts):.3f} | avg_tok={toks/len(prompts):.0f} | "
        f"wall={dt:.1f}s | throughput={toks/dt:.1f} tok/s | compactions={stats['n']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
