import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        rkv_enabled=True,
        rkv_budget=1024,
        rkv_buffer=128,
        rkv_window_size=8,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=4096)
    prompts = [
        "Solve step by step: An integer is chosen uniformly at random "
        "from 1 to 1000. What is the probability it is divisible by 7 "
        "but not by 11?",
        "Prove that the sum of the first n odd positive integers equals n^2.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
