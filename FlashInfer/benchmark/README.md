# R-KV Benchmark (FlashInfer)

Offline harnesses for the standalone FlashInfer R-KV engine. Both scripts
construct `FlashInferEngine` directly (no server), flip R-KV on via
`--mode rkv`, and print exactly **one machine-readable JSON object as the
final stdout line** — sweep scripts harvest with `tail -n 1`.

| File | Purpose |
| --- | --- |
| `bench_rkv.py` | Decode throughput + peak `torch.cuda.max_memory_allocated`, FullKV vs R-KV, synthetic seeded prompts, per-trial table |
| `eval_math.py` | GSM8K / MATH-500 / AIME24 accuracy + throughput; id-preserving jsonl records; scores via the in-repo `HuggingFace/evaluation` pipeline; `--shard i/N` data parallelism |
| `RESULTS_*.md` | Dated, hardware-tagged measurements (filled from GPU validation) |

## Run

Environment: the pinned stack from [`../README.md`](../README.md) quick start
(`pip install -r ../requirements-rkv.txt`), plus the scoring deps for
`eval_math.py` (sympy / latex2sympy2 / pebble / ...; **Python ≤ 3.12** —
`latex2sympy2` breaks on 3.13):

```bash
pip install -r ../../HuggingFace/evaluation/requirements.txt   # from this directory
```

Eval data is the committed jsonl under `../../HuggingFace/data/` — no fetch step.

```bash
cd R-KV/FlashInfer

# throughput + peak memory (DESIGN §8 perf matrix)
python benchmark/bench_rkv.py --model /path/DeepSeek-R1-Distill-Qwen-1.5B \
    --mode fullkv --batch-size 8 --prompt-len 512 --gen-len 2048 --trials 3
python benchmark/bench_rkv.py --model /path/DeepSeek-R1-Distill-Qwen-1.5B \
    --mode rkv --budget 1024 --buffer 128 --batch-size 8 --prompt-len 512 --gen-len 2048

# accuracy (per-dataset --max-new-tokens defaults: gsm8k 8192, math 16384, aime24 32768)
python benchmark/eval_math.py --model /path/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset gsm8k --mode rkv --budget 1024 --max-samples 100 --bsz 8

# multi-GPU data-parallel sweep: one process per GPU, disjoint strided shards
CUDA_VISIBLE_DEVICES=0 python benchmark/eval_math.py --model M --dataset math --mode rkv --shard 0/2 &
CUDA_VISIBLE_DEVICES=1 python benchmark/eval_math.py --model M --dataset math --mode rkv --shard 1/2 &
wait
cat benchmark/eval_math_rkv1024_shard*.jsonl > merged.jsonl   # records carry a global 'idx'
python benchmark/eval_math.py --dataset math --output merged.jsonl --score-only
```

## Config notes

- **Sampling** — `eval_math.py`: `--temperature 0.6 --top-p 0.95 --seed 42`
  (paper-style reproduction). `bench_rkv.py` hardcodes the same
  temperature/top-p; prompts are seeded random token ids and stop tokens are
  disabled, so every request decodes exactly `--gen-len` tokens.
- **Prompt convention** (`eval_math.py`) — the `HuggingFace/run_math.py` math
  template wrapped in `tokenizer.apply_chat_template` (R1-distill models only
  enter their trained `<think>` format inside the chat template), encoded with
  `add_special_tokens=False`.
- **Scoring** — in-process via `HuggingFace/evaluation`
  (`parser.run_execute` + `evaluate.evaluate`, `prompt_type="cot"`). If the
  scoring deps are missing, generation records are still written and the
  script prints the exact `--score-only` command to score them later.
  Per-shard accuracy is shard-local; merge shards and `--score-only` for the
  full split.
- **`engine.stats` contract** (`bench_rkv.py`) — reads
  `prefill_seconds` / `decode_seconds` / `compaction_seconds`, assumed to be a
  snapshot of the most recent `generate()` call (DESIGN.md §5.5); missing keys
  fail loudly. Compaction counts come from `GenOutput.num_compactions`.
- **Peak memory** (`bench_rkv.py`) — `torch.cuda.reset_peak_memory_stats()`
  after a warmup `generate()`, read after all trials: covers weights, the
  static KV pool (this is where R-KV's saving shows up vs FullKV), and
  steady-state activations.

## Status

**Pending GPU validation.** `RESULTS_*.md` will be filled from the H100 runs
of the DESIGN.md §8 accuracy/perf matrices. CPU-verifiable pieces (shard
slicing, prompt-template parity with `run_math.py`, in-repo scoring on all
three datasets, `--score-only` CLI) have been exercised.
