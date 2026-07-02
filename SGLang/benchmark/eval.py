"""R-KV accuracy + throughput driver.

Reuses the dev-branch eval dataset (few-shot MATH-style prompts with a GSM8K
`#### N` gold answer) but talks directly to our SGLang server over /generate and
does simple numeric judging, so we can compare R-KV on vs off under identical
judging without the old vllm/transformers-4.x harness.

Usage:
    python3 eval.py --n 100 --port 30000 --max-tokens 512 --label rkv_b512
"""

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_HERE, "data", "gsm8k_fewshot.jsonl")


def extract_gold(answer: str):
    m = re.search(r"####\s*([\-0-9\.,/]+)", answer)
    return m.group(1).replace(",", "").rstrip(".") if m else None


_PATS = [
    r"\\boxed\{([^{}]*)\}",
    r"[Tt]he final answer is[:\s]*\$?\\?\(?\$?([\-0-9\.,/]+)",
    r"[Tt]he answer is[:\s]*\$?([\-0-9\.,/]+)",
]


def extract_pred(text: str):
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


def generate(port: int, prompt: str, max_tokens: int):
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_tokens,
                "stop": ["\nProblem"],
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    return out["text"], out.get("meta_info", {}).get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=_DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of in-flight requests. >1 forces the server to batch "
        "(exercises the R-KV batch>=2 per-request path).",
    )
    args = ap.parse_args()

    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= args.n:
                break

    n = len(data)

    def run_one(item):
        prompt = item["request"]["messages"][0]["content"]
        gold = extract_gold(item["answer"])
        gen, ntok = generate(args.port, prompt, args.max_tokens)
        pred = extract_pred(gen)
        g, p = to_num(gold), to_num(pred)
        ok = g is not None and p is not None and abs(g - p) < 1e-4
        return gold, pred, ntok, ok

    t0 = time.time()
    if args.concurrency <= 1:
        results = [run_one(d) for d in data]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            results = list(ex.map(run_one, data))
    dt = time.time() - t0

    correct = 0
    tot_tokens = 0
    for i, (gold, pred, ntok, ok) in enumerate(results):
        tot_tokens += ntok
        correct += ok
        if i < 5:
            print(f"[{i}] gold={gold} pred={pred} ok={ok} ntok={ntok}")

    mode = (
        "serial batch=1"
        if args.concurrency <= 1
        else f"concurrency={args.concurrency}"
    )
    print(
        f"\n=== {args.label} ===\n"
        f"accuracy   : {correct}/{n} = {correct/n:.3f}\n"
        f"avg_tokens : {tot_tokens/n:.0f}\n"
        f"wall_time  : {dt:.1f}s\n"
        f"throughput : {tot_tokens/dt:.1f} tok/s (end-to-end, {mode})"
    )


if __name__ == "__main__":
    main()
