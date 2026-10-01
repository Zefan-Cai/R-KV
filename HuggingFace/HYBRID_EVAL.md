# Hybrid model GSM8K and MATH evaluation

`run_hybrid_math.py` uses native Transformers 5.18.0 text models and the original
`R1KV` selector. Only layers declared `full_attention` are compressed. Qwen's
linear states and Gemma's sliding attention caches retain their native behavior.
Gemma4 shared full layers reuse the compressed producing layer once per step.
Absolute RoPE positions and head-specific attention masks remain correct.

This is a batch-one correctness and accuracy evaluation using BF16, native SDPA,
greedy decoding, each checkpoint's chat template, and an explicit thinking flag.
cuDNN SDPA is disabled for every arm to avoid per-length execution-plan building
overhead observed on Torch 2.11; native Flash/Efficient/Math SDPA remain enabled.
The selected backend flags are saved in metadata.
It is not a production throughput benchmark. FullKV and compressed arms share
the same prompts and seeded sample IDs. Prefill remains uncompressed. The default
compression interval is 128 decoding steps; storage grows between compactions,
so the budget is the retained size after a compaction rather than a hard cap.

Use a clean environment with a CUDA-enabled torch, transformers==5.18.0,
accelerate, and the requirements from `evaluation/requirements.txt`. Do not
install the old `HuggingFace/requirements.txt` Transformers pin for this adapter.

```bash
CUDA_VISIBLE_DEVICES=6 python HuggingFace/run_hybrid_math.py \
  --model /path/to/Qwen3.5-0.8B --output-dir /path/to/results/qwen35-08b \
  --validation-only --thinking

CUDA_VISIBLE_DEVICES=6 python HuggingFace/run_hybrid_math.py \
  --model /path/to/Qwen3.5-0.8B --output-dir /path/to/results/qwen35-08b \
  --datasets gsm8k math --modes fullkv 128 512 1024 --sample-limit 100 --thinking
```

The repository datasets contain the full GSM8K test (1,319 examples) and
MATH-500 (500 examples). `--sample-limit 0` uses the entire dataset. Independent
GPU workers can use `--shard-index i --num-shards N`; IDs remain stable. The output
token caps default to 8,192 (GSM8K) and 16,384 (MATH). Always report truncation
and actual eviction counts beside accuracy. Small checkpoints can enter thinking
loops; increase caps or run a separately labeled non-thinking arm if needed.

Each record retains raw output, scoring, generation time, token count, absolute
and incremental peak allocated GPU memory, and full-layer compression counters.
One identical eight-token warmup per arm is excluded. Validation compares native
and disabled-adapter tokens and forces evictions with a small budget. CPU tests
also compare exact native logits, untouched non-full cache storage, shared KV,
head-specific masks, new-generation reset, and immediate cache release.

```bash
cd HuggingFace
python -m pytest tests/test_hybrid.py -q
```
