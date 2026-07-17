"""R-KV accuracy + throughput driver for vLLM (offline batched).

Mirrors ``SGLang/benchmark/eval.py``, but because vLLM's R-KV path is a drop-in
serving change gated by env vars, this driver runs the model **offline**
(``LLM.generate`` over all prompts at once) instead of over HTTP. That keeps the
numeric judging identical to the SGLang harness while measuring vLLM's own
batched decode throughput. R-KV on/off is selected by the
``VLLM_V1_R_KV_BUDGET`` / ``VLLM_V1_R_KV_BUFFER`` env vars (0/unset => Full-KV).

Reported throughput is offline batched (all ``--n`` prompts submitted at once),
not a served-request rate; see the RESULTS_*.md reports for the methodology note.

Usage:
    # R-KV budget=256 buffer=128, 200 questions, one GPU:
    VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
      python eval.py --n 200 --label rkv_b256_buf128 --out /tmp/r.json

    # Full-KV production baseline (prefix caching on, upstream defaults):
    python eval.py --n 200 --label fullkv_production

    # Full-KV constrained (fair A/B) baseline (prefix caching off, like R-KV):
    python eval.py --n 200 --no-prefix --label fullkv_constrained

    # 4-way tensor parallel:
    VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
      python eval.py --n 500 --tp 4 --label rkv_tp4
"""

import argparse
import json
import os
import re
import time

# The compaction counter is read from each worker via collective_rpc with a
# small closure; vLLM gates callable RPCs behind this flag. It is safe here --
# a local, offline benchmark run by the same user. Set before importing vllm.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from vllm import LLM, SamplingParams

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_HERE, "data", "gsm8k_fewshot.jsonl")
# Portable default (resolved from the HF hub / local cache). Point RKV_MODEL at a
# local checkout to avoid a download, e.g. RKV_MODEL=/path/to/Qwen2.5-Math-7B-Instruct.
_DEFAULT_MODEL = os.environ.get(
    "RKV_MODEL", "Qwen/Qwen2.5-Math-7B-Instruct"
)


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


def read_compactions(llm: LLM):
    """Total physical R-KV compactions during the run (max across TP ranks).

    Read from every worker's compactor via ``collective_rpc``; returns ``None``
    when the count is unavailable (e.g. R-KV off).
    """

    def _get(self):
        mr = getattr(self, "model_runner", None)
        comp = getattr(mr, "rkv_compactor", None)
        return int(getattr(comp, "_n_compactions", 0)) if comp is not None else 0

    try:
        counts = llm.collective_rpc(_get)
        return max(counts) if counts else 0
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=_DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="")
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument(
        "--no-prefix",
        action="store_true",
        help="disable prefix caching (fair Full-KV A/B; R-KV forces this itself)",
    )
    ap.add_argument(
        "--eager",
        action="store_true",
        help="force eager (default: cudagraph -- PIECEWISE when R-KV is on)",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip the first OFFSET questions (used to shard across DP replicas)",
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="generate exactly --max-tokens per request (stress test: fixed-length "
        "sequences to load the KV cache regardless of the model's natural stop)",
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="enable vLLM engine stat logging (so peak 'Running: N reqs' -- the "
        "achieved concurrency -- is printed)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    budget = int(os.environ.get("VLLM_V1_R_KV_BUDGET", "0"))
    buffer = int(os.environ.get("VLLM_V1_R_KV_BUFFER", "0"))
    rkv_on = budget > 0 and buffer > 0
    label = args.label or (f"rkv_b{budget}_buf{buffer}" if rkv_on else "fullkv")

    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    data = data[args.offset : args.offset + args.n]
    n = len(data)
    prompts = [d["request"]["messages"][0]["content"] for d in data]

    kw = {}
    if args.no_prefix:
        kw["enable_prefix_caching"] = False
    llm = LLM(
        model=_DEFAULT_MODEL,
        tensor_parallel_size=args.tp,
        enforce_eager=args.eager,
        gpu_memory_utilization=args.mem_frac,
        max_model_len=args.max_model_len,
        disable_log_stats=not args.stats,
        seed=0,
        block_size=16,
        **kw,
    )
    tok = llm.get_tokenizer()
    in_toks = sum(len(tok(p).input_ids) for p in prompts)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        # ignore_eos forces fixed-length generation (stress test); otherwise stop
        # at the next few-shot problem boundary for accurate GSM8K judging.
        ignore_eos=args.ignore_eos,
        stop=None if args.ignore_eos else ["\nProblem"],
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0

    correct = 0
    out_toks = 0
    for i, (d, o) in enumerate(zip(data, outs)):
        out_toks += len(o.outputs[0].token_ids)
        g = to_num(extract_gold(d["answer"]))
        p = to_num(extract_pred(o.outputs[0].text))
        ok = g is not None and p is not None and abs(g - p) < 1e-4
        correct += int(ok)
        if i < 5:
            print(f"[{i}] gold={g} pred={p} ok={ok} ntok={len(o.outputs[0].token_ids)}")

    compactions = read_compactions(llm) if rkv_on else 0
    res = {
        "label": label,
        "budget": budget,
        "buffer": buffer,
        "n": n,
        "tp": args.tp,
        "max_tokens": args.max_tokens,
        "accuracy": round(correct / n, 4),
        "correct": correct,
        "wall_s": round(dt, 1),
        "out_tokens": out_toks,
        "in_tokens": in_toks,
        "decode_tok_s": round(out_toks / dt, 1),
        "total_tok_s": round((out_toks + in_toks) / dt, 1),
        "avg_gen_len": round(out_toks / n, 1),
        "compactions": compactions,
    }
    print(
        f"\n=== {label} ===\n"
        f"accuracy    : {correct}/{n} = {correct / n:.3f}\n"
        f"avg_tokens  : {out_toks / n:.0f}\n"
        f"wall_time   : {dt:.1f}s\n"
        f"decode_tput : {out_toks / dt:.1f} tok/s (offline batched, {n} prompts in flight)\n"
        f"compactions : {compactions}"
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
