"""Teacher-forced logits parity probe: HF transformers vs FlashInferEngine (FullKV).

Three-way comparison on the identical forced token sequence:
  fi_decode  — this engine, step-by-step decode path
  hf_decode  — HF generate() step-by-step logits (output_logits=True)
  hf_prefill — HF single full forward over the whole sequence
The hf_decode-vs-hf_prefill delta is HF's own prefill/decode kernel-path noise
floor; fi is judged against hf_decode with that floor as the yardstick.
"""
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--flashinfer-dir", default="/mnt/localssd/zefan/rkv-fi/R-KV/FlashInfer")
parser.add_argument("--steps", type=int, default=64)
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
    gen = hf.generate(ids, max_new_tokens=args.steps, do_sample=False,
                      temperature=None, top_p=None, top_k=None,
                      pad_token_id=tok.eos_token_id,
                      output_logits=True, return_dict_in_generate=True)
    forced = gen.sequences[0, len(prompt):].tolist()
    hf_decode = [l[0].float() for l in gen.logits]          # per-step decode-path logits
    hf_prefill = hf(gen.sequences).logits[0].float()        # full-forward logits
del hf
torch.cuda.empty_cache()

engine = FlashInferEngine(args.model, max_batch_size=1,
                          max_seq_len=len(prompt) + len(forced) + 8)
with torch.inference_mode():
    engine._phys_len = [0]
    engine._logical_len = [0]
    fi = [engine._prefill([prompt], [len(prompt)])[0].float()]
    for t in forced[:-1]:
        fi.append(engine._decode_step([0], [t])[0].float())

pos = len(prompt) - 1
n = min(len(fi), len(hf_decode))
stats = {"fi_vs_hfdec": [0.0, 0], "hfdec_vs_hfpre": [0.0, 0], "fi_vs_hfpre": [0.0, 0]}


def upd(key: str, a: torch.Tensor, b: torch.Tensor) -> None:
    d = (a - b).abs().max().item()
    if d > stats[key][0]:
        stats[key][0] = d
    if a.argmax().item() == b.argmax().item():
        stats[key][1] += 1


profile = []
for i in range(n):
    hp = hf_prefill[pos + i]
    upd("fi_vs_hfdec", fi[i], hf_decode[i])
    upd("hfdec_vs_hfpre", hf_decode[i], hp)
    upd("fi_vs_hfpre", fi[i], hp)
    profile.append((fi[i] - hf_decode[i]).abs().max().item())

marks = [i for i in (0, 1, 2, 4, 8, 16, 24, 32, 48, n - 1) if i < n]
print("delta profile:", " ".join(f"s{i}={profile[i]:.3f}" for i in marks))

# Long-prefill isolation: run OUR prefill over the whole forced sequence at once;
# if this matches hf_prefill closely, the prefill path is clean and the drift
# comes from the decode/KV-append path.
full_seq = prompt + forced
engine2 = FlashInferEngine(args.model, max_batch_size=1, max_seq_len=len(full_seq) + 8)
with torch.inference_mode():
    engine2._phys_len = [0]
    engine2._logical_len = [0]
    fi_full = engine2._prefill([full_seq], [len(full_seq)])[0].float()
d = (fi_full - hf_prefill[-1]).abs().max().item()
same = fi_full.argmax().item() == hf_prefill[-1].argmax().item()
print(f"long_prefill_vs_hfpre: maxdelta={d:.4f} argmax_same={same}")

for key, (worst, agree) in stats.items():
    print(f"{key}: argmax_agree={agree}/{n} worst_maxdelta={worst:.4f}")

fi_agree = stats["fi_vs_hfdec"][1]
base_agree = stats["hfdec_vs_hfpre"][1]
print("PROBE_OK" if fi_agree >= base_agree - 2 else "PROBE_SUSPECT")
