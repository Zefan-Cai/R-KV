# Validation artifacts — 2026-07-02, A100-40G

Raw outputs from the serving-integration validation pass (repo commit
`e9f54c45`, node: 2×A100-SXM4-40GB, torch 2.6.0+cu124, transformers
4.55.4 for the HF runs). Scored with the fixed evaluator (`eval_math.py`
preferring the current run's `output` field — see the commit message of
`e9f54c45` for the stale-`generation` scoring bug these runs uncovered).

## gsm8k_n50/ — HuggingFace path accuracy sanity

DeepSeek-R1-Distill-Qwen-1.5B, first 50 GSM8K problems, greedy-free
sampling (temperature 0.6, top-p 0.95, seed 42), `eval_batch_size=1`,
single candidate, pass@1. R-KV: `budget=1024, window=8, mix_lambda=0.1,
divide_method=step_length, divide_length=128`.

| file | prompt style | method | max_length | acc |
|---|---|---|---:|---:|
| `fullkv.jsonl` | plain template | FullKV | 4096 | 80.0 |
| `rkv.jsonl` | plain template | R-KV budget=1024 | 4096 | **84.0** |
| `fullkv_ct.jsonl` | `--use_chat_template` | FullKV | 8192 | 78.0 |
| `rkv_ct.jsonl` | `--use_chat_template` | R-KV budget=1024 | 8192 | 78.0 |

Reading notes:

- With a fixed seed, FullKV and R-KV generations are token-identical
  until the first compression event (trigger ≈ budget + buffer ≈ 1152
  tokens). In the chat-template runs the longest sequence was 995
  tokens, so **compression never fired and `rkv_ct.jsonl` is
  byte-identical to `fullkv_ct.jsonl`** — a trivial "lossless" case,
  not an interesting comparison.
- In the plain-prompt runs 7/50 sequences crossed the trigger; only
  those diverged. R-KV's +4 points come from rescuing two of the long,
  meandering generations — consistent with the paper's observation that
  pruning redundant tokens can improve reasoning quality.
- Before the evaluator fix, all four configurations scored exactly
  40.0% because the stale `generation` field shipped inside the old
  `data/gsm8k.jsonl` was being scored instead of these outputs.

## vllm_bench/ — vLLM V1 throughput (issue #8)

Qwen/Qwen3-0.6B, single A100-40G, checked-in vLLM 0.8.5 V1 overlay,
`enforce_eager`, prefix caching off, FLASH_ATTN, 2048 forced decode
tokens per sequence (`ignore_eos`), temperature 0.8 / top-p 0.95.
R-KV `budget=512, buffer=128` vs FullKV (`buffer=0`). Raw logs.

| batch | FullKV tok/s | R-KV tok/s | ratio |
|---:|---:|---:|---:|
| 1 | 50.5 | 51.6 | 1.02× |
| 4 | 183.1 | 187.7 | 1.03× |
| 16 | 597.2 | 721.9 | 1.21× |
| 64 | 2065.2 | 2431.4 | 1.18× |

## aime24_n30/ — AIME 2024 reference (pipeline sanity)

DeepSeek-R1-Distill-Qwen-1.5B, all 30 AIME-2024 problems, chat
template, max_length 16384, temperature 0.6 / top-p 0.95, seed 42,
**single candidate per problem** — a pipeline-sanity reference, not a
paper reproduction (the paper reports 8B/14B models with 64 candidates
averaged). See the metrics JSONs for scores.
