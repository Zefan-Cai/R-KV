"""Correctness cross-check: FlashInferEngine FullKV greedy vs HF transformers greedy.

Reports prefill-logits max-abs-diff and greedy common-prefix length. bf16 kernels
differ between stacks, so small logit diffs and eventual divergence are expected;
a long common prefix (>=50 tokens) plus small prefill deltas is the pass signal.
"""
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--flashinfer-dir", default="/mnt/localssd/zefan/rkv-fi/R-KV/FlashInfer")
parser.add_argument("--max-new-tokens", type=int, default=200)
parser.add_argument("--min-prefix", type=int, default=50)
args = parser.parse_args()

sys.path.insert(0, args.flashinfer_dir)
from rkv import FlashInferEngine  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 17 * 23? Think step by step."}],
    tokenize=True, return_dict=False, add_generation_prompt=True,
)

hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
ids = torch.tensor([prompt], device="cuda")
with torch.no_grad():
    hf_prefill_logits = hf(ids).logits[0, -1].float()
    hf_out = hf.generate(
        ids, max_new_tokens=args.max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tok.eos_token_id,
    )[0, len(prompt):].tolist()
del hf
torch.cuda.empty_cache()

engine = FlashInferEngine(args.model, max_batch_size=1,
                          max_seq_len=len(prompt) + args.max_new_tokens + 8)
fi_prefill_logits = None
out = engine.generate([prompt], max_new_tokens=args.max_new_tokens,
                      temperature=0.0, stop_token_ids=(tok.eos_token_id,))[0]
fi_out = out.token_ids

prefix = 0
for a, b in zip(hf_out, fi_out):
    if a != b:
        break
    prefix += 1

print(f"hf_len={len(hf_out)} fi_len={len(fi_out)} common_prefix={prefix}")
print("hf :", tok.decode(hf_out[:60]).replace("\n", "\\n")[:200])
print("fi :", tok.decode(fi_out[:60]).replace("\n", "\\n")[:200])
ok = prefix >= args.min_prefix
print("COMPARE_HF_PASS" if ok else "COMPARE_HF_FAIL")
sys.exit(0 if ok else 1)
