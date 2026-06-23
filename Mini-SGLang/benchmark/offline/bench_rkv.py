"""R-KV enabled Mini-SGLang offline benchmark.

Mirrors ``benchmark/offline/bench.py`` but flips on R-KV via the
LLM constructor kwargs. The R-KV knobs are forwarded through
``SchedulerConfig(EngineConfig)`` → ``EngineConfig.rkv_*`` fields.

R-KV requires variable per-step ``cache_seqlens`` so CUDA graph capture
is disabled here (``cuda_graph_bs=None``, ``cuda_graph_max_bs=None``).
See ``docs/RKV.md`` for the wiring contract and limitations.
"""

import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    seed(0)
    num_seqs = 32
    max_input_len = 1024
    max_output_len = 4096

    llm = LLM(
        "Qwen/Qwen3-0.6B",
        max_seq_len_override=8192,
        max_extend_tokens=16384,
        # R-KV needs variable cache_seqlens per step -> no CUDA graphs.
        cuda_graph_bs=None,
        cuda_graph_max_bs=None,
        page_size=1,
        # R-KV configuration mirroring the HuggingFace reference.
        rkv_enabled=True,
        rkv_budget=1024,
        rkv_buffer=128,
        rkv_window_size=8,
        rkv_kernel_size=7,
        rkv_mix_lambda=0.07,
        rkv_retain_ratio=0.1,
        rkv_retain_direction="last",
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(1024, max_output_len),
        )
        for _ in range(num_seqs)
    ]
    llm.generate(["Warmup: "], SamplingParams(temperature=0.1))
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
