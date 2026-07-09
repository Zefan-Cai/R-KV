"""FullKV baseline decode with the FlashInfer R-KV engine.

Usage: python examples/example.py [model_path]   (or set RKV_MODEL)
"""

import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkv import FlashInferEngine  # noqa: E402


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "RKV_MODEL", "Qwen/Qwen3-0.6B"
    )
    if not os.path.isdir(model_path):
        # The engine loads safetensors from a local directory only; resolve
        # hub ids by downloading a snapshot first (same as the smoke test).
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engine = FlashInferEngine(model_path, max_batch_size=2, max_seq_len=8192)

    questions = [
        "Solve step by step: an integer is chosen uniformly at random from 1 "
        "to 1000. What is the probability it is divisible by 7 but not by 11?",
        "Prove that the sum of the first n odd positive integers equals n^2.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]

    outputs = engine.generate(
        prompts,
        max_new_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        stop_token_ids=(tokenizer.eos_token_id,),
        seed=42,
    )

    for question, out in zip(questions, outputs):
        print(f"\n=== {question}")
        print(tokenizer.decode(out.token_ids, skip_special_tokens=True))
        print(
            f"--- prefill={out.num_prefill_tokens} tok, "
            f"output={out.num_output_tokens} tok, finish={out.finish_reason}"
        )
    print(f"\nstats: {engine.stats}")


if __name__ == "__main__":
    main()
