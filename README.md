<h1 align="center">R-KV</h1>


> **Shrink the cache, keep the brains.**  
> R-KV discards repetitive tokens on-the-fly, delivering *full-accuracy* reasoning with only a fraction of the memory.


<p align="center">
    ｜<a href="https://zefan-cai.github.io/R-KV.page/"><b>🌏 Page</b></a> |
    <a href="https://arxiv.org/abs/2505.24133"><b>📖 Paper</b></a>｜
</p>

## TODO
- Apply R-KV in GPT-OSS
- Integrate R-KV with VeRL
- ~~Harden R-KV benchmark coverage in vLLM, Nano-vLLM and SGLang~~ (GPU smokes in `tests/smoke/`, throughput bench + raw artifacts in `results/`, CPU tests in CI)
- Extend dataset to GPQA, liveCodeBench
- Apply R-KV in QwQ
- Expand Qwen-3 evaluation coverage

## 🔥 News

- 🚀 [26/07/02] **The SGLang (v0.5.14) R-KV port is complete.** Decoding-time KV-cache compression with true physical eviction on the FlashInfer decode path, validated on Qwen2.5-Math-7B (H100): accuracy on par with the baseline (95% GSM8K), server-side batching (`eval.py --concurrency N`), and **plain data parallelism** (`DP=N ./launch_server.sh rkv 512`, throughput scales up to 5.2× on 8× H100). Code in `SGLang/rkv/` (`algo.py`, `integration.py`) plus the wiring patch `SGLang/patch/rkv-sglang-0.5.14.patch`; design/implementation notes in `SGLang/docs/DESIGN.md` and `SGLang/docs/IMPLEMENTATION.md`; benchmarks + reproduce steps in `SGLang/benchmark/` (`RESULTS_math7b.md`, `RESULTS_dp.md`, `RESULTS_a100_n100.md`).
- ⚠️ [26/07/02] **If you computed GSM8K numbers with this repo before commit `e9f54c45`, please rerun them.** The shipped `data/gsm8k.jsonl` carried a stale `generation` field from an old experiment and `eval_math.py` preferred it over the fresh run's `output`, freezing GSM8K scores at ~40% regardless of method or budget (issues #26/#23). MATH and AIME24 were unaffected. All four serving integrations (vLLM, SGLang, Nano-vLLM, Mini-SGLang) plus the HuggingFace path are now GPU-validated on A100 with R-KV on/off — see `tests/smoke/` and `results/validation-2026-07-02-a100/`.
- 🚀 [25/05/29] We are pleased to announce the release of **R-KV**, a highly efficient decoding time KV Cache compression method to serve reasoning Models.


## Setup

### Install Dependencies
Use the following command to install the minimal required dependencies for the HuggingFace implementation:
```bash
pip install -r requirements.txt
```

For the SGLang implementation, use the pinned environment note in `SGLang/requirements-rkv.txt`.

For lightweight reference ports, see `Nano-vLLM/` and `Mini-SGLang/`. `Nano-vLLM/` is a ~1.2k-line paged-KV inference engine with R-KV plugged into the attention path (`enforce_eager=True` required); `Mini-SGLang/` carries the R-KV algorithm and a paged-cache integration helper under `python/minisgl/compress/`, with the attention-layer wiring documented in `Mini-SGLang/docs/RKV.md`.

For the vLLM implementation, use the checked-in `vLLM/` tree as one source tree. The R-KV changes touch coupled vLLM V1 files such as the scheduler, engine core, platform helpers, tokenizer utilities, model definitions, and FlashAttention backend, so do not copy only a few Python files onto an arbitrary PyPI `vllm` wheel. A typical source install is:

```bash
cd vLLM
SETUPTOOLS_SCM_PRETEND_VERSION=0.8.5 pip install -e . --no-build-isolation

export VLLM_USE_V1=1
export VLLM_V1_R_KV_BUDGET=64
export VLLM_V1_R_KV_BUFFER=8
```

If you must reuse a prebuilt `vllm==0.8.5` wheel for the compiled CUDA extensions, overlay the entire checked-in `vLLM/vllm/` Python package, not individual files.

### Install FlashAttention
If you're using Hugging Face, we default to flash attention to speed up attention computation:

```python
model = AutoModelForCausalLM.from_pretrained(
    "model_name_or_path",
    attn_implementation="flash_attention_2",
)
```

### Evaluation

You need to build the dependencies of the evaluation toolkit separately:
```bash
cd HuggingFace
cd evaluation/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt
```

## Quick Start
Before running the scripts, you need to build the rkv package:
```bash
cd HuggingFace
pip install -r requirements.txt
pip install -e .
```

Use the following command to run R1-like models with R-KV on math benchmarks:
```bash
bash scripts/run.sh
```

Or you could use the code scripts:

```bash
export CUDA_VISIBLE_DEVICES=0
cd HuggingFace

python3 ./run_math.py \
--dataset_path ./data/aime24.jsonl \
--save_path ./outputs/output.jsonl \
--model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
--max_length 32768 \
--eval_batch_size 1 \
--method rkv \
--kv_budget 1024 \
--window_size 8 \
--mix_lambda 0.1 \
--divide_method step_length \
--divide_length 128 \
--do_sample \
--temperature 0.6 \
--top_p 0.95

```

For paper-style candidate generation, additionally set `--num_return_sequences 64`.
Add `--use_chat_template` when evaluating DeepSeek-R1-distill (or other chat)
models: they only enter their trained `<think>` reasoning format inside the
chat template.

To evaluate benchmark results, simply run:
```bash
bash examples/eval.sh
```
The results will be saved in the `outputs` directory.

## Tests

CPU-only unit tests cover the R-KV algorithm port in the lightweight reference implementations:

```bash
# Nano-vLLM port — pre-trigger no-op, post-trigger shape + trailing-window
# preservation, and cross-port equality against Mini-SGLang.
python Nano-vLLM/tests/test_rkv_algorithm.py

# Mini-SGLang port — same algorithm tests plus disabled-compressor /
# drop_request / drain_pending_free_slots integration coverage.
PYTHONPATH=Mini-SGLang/python python Mini-SGLang/tests/misc/test_rkv_algorithm.py

# SGLang port — algorithm parity, compaction/lifecycle/batch integration, and
# cross-repo bit-level parity against this repo's rkv/ reference.
python SGLang/tests/test_rkv_algo.py
python SGLang/tests/test_rkv_integration.py
python SGLang/tests/test_cross_repo_parity.py
```

All run on CPU without a GPU and exercise the exact `R1KV` class used at
runtime. The Nano-vLLM and Mini-SGLang tests also run in CI on every push
(`.github/workflows/cpu-tests.yml`).

GPU smoke tests for all four serving integrations (vLLM, Nano-vLLM,
Mini-SGLang, SGLang) plus the HuggingFace monkeypatch live in
`tests/smoke/` — each generates with R-KV on/off and applies a lexical
health check that catches garbled or looping output. Raw outputs, accuracy
metrics, and vLLM throughput benchmarks from the validated A100 runs are
checked in under `results/`.

## Overview

Large language models that rely on chain-of-thought (CoT) or self-reflection can crack tough reasoning tasks—but at the cost of **very long outputs that bloat the key–value (KV) cache** during inference.  
Traditional cache-compression schemes, tuned for long *prompts*, tumble on these generated traces, keeping only ~60 % of the original accuracy when restricted to 10 % of the cache.

**R-KV** — **R**edundancy-aware KV-cache compression for **R**easoning models — solves this by ranking tokens on-the-fly for both **importance** *and* **non-redundancy**, retaining only the informative, diverse ones.

- **10 % cache → ≈ 100 % accuracy**
- **16 % cache → 105 % accuracy**<br>
  *(noise reduction even nudges performance above the full cache)*  
- **90 % memory saved** and **6.6 × throughput** during long CoT generation  
- Consistent wins over all prior baselines on two math-reasoning benchmarks  

## 🌟 Highlights

| Metric | Full KV | **R-KV (ours)** |
|-------:|--------:|----------------:|
| KV cache kept | 100 % | **10 %** |
| Accuracy | 100 % | **≈100 %** |
| Throughput ↑ | 1× | **6.6×** |
| Memory saved ↓ | – | **90 %** |

*At 16 % cache, noise removal even boosts accuracy to **105 %** of the full baseline.*

## ✨ Why R-KV?

- **Up to 90 % KV-cache memory savings** with zero—sometimes negative—accuracy loss  
- **Plug-and-play:** a lightweight wrapper for any autoregressive LLM  
- **Training-free:** drop straight into inference or RL roll-outs—no finetuning required



## Motivation

Chain-of-thought (CoT) and self-reflection unlock impressive reasoning, but they **explode the key–value (KV) cache**.  
A single DeepSeek-R1-Distill-8B run on a tough math problem can:

* Generate **32 000 tokens**  
* Load **15.5 GB** of weights  
* Allocate **4.1 GB** of KV just to remember its own musings

Existing compression tools focus on *long prompts* and falter on *long generations*—often pruning the wrong tokens because redundant self-checks still attend heavily to themselves.





## 🔍 Method — Redundancy-aware KV Cache Compression (R-KV)

Redundancy-aware KV Cache Compression for R1-Style models (i.e., **R-KV**) is the first production-level KV Cache Compression Method for serving R1-like models.

<div align="center">
  <img src="./assets/method.png" alt="R-KV Main Results" width="100%">
</div>

Large-language models (LLMs) waste a surprising amount of memory on **redundant key/value (KV) tokens** during long-form reasoning.  
**R-KV** tackles this by *compressing the KV cache on-the-fly while the model is decoding*, keeping only tokens that are **important** *and* **non-redundant**.

### How it works
| Stage | What happens | Key idea |
|-------|--------------|----------|
| 1. **Decoding-time KV staging** | Newly generated tokens are first written to a **buffer** (`B_buffer`). | Separate buffer lets us decide what to keep *after* seeing a chunk of text. |
| 2. **Importance scoring** | Use attention weights from the last `α` **observation tokens** to score each candidate token. | High attention ⇒ token is critical for future predictions. |
| 3. **Redundancy estimation** | Compute cosine similarity between key vectors. For each token keep at most `β` *most-recent* highly similar neighbors; older near-duplicates are marked redundant. | Keeps semantics while pruning repetition. |
| 4. **Joint selection** | Final score `Z = λ·Importance − (1-λ)·Redundancy`. Top-`k` tokens + `α` observation tokens are retained in the **budgeted cache** (`B_budget`). | One knob (`λ`) trades memory vs. quality. |




## Performance Results

Our method surpassed baselines by a large margin in challenging math benchmarks, and surprisingly, even outperformed full KV.

## Experimental Setup

### Models
We benchmark two distilled DeepSeek-R1 variants:

| Nickname | Checkpoint |
|----------|------------|
| **R1-Llama-8B**  | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| **R1-Qwen-14B**  | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |

### Datasets
| Benchmark | Tokens / sol. (avg) | Max gen len |
|-----------|--------------------:|------------:|
| **MATH-500** | 2 979 | 16 384 |
| **AIME 2024** | 15 536 | 32 768 |


### Default Hyper-parameters
| Symbol | Meaning | Value |
|--------|---------|------:|
| `B_buffer` | staging-buffer size | **128** |
| `α` | # observation tokens for attention probe | **8** |
| `λ` | importance / redundancy trade-off | **0.1** (see § λ-sweep) |

Sampling: *temperature 0.6*, *top-p 0.95*, *64 candidates* per problem; we report **pass@1**.

For the HuggingFace runner, `B_buffer = 128` maps to `--divide_method step_length --divide_length 128`. `--kv_budget` is the retained KV budget (`B_budget`). `divide_method=newline` is a convenience heuristic and was not the fixed-buffer setting used for the main reported results.

The released HuggingFace implementation performs decoding-time compression. It should not be used as a prefill-stage long-prompt compressor or TTFT optimization without adding a separate prefill compression pass.

### Baselines
* **FullKV** – no compression, upper-bound quality.  
* **SnapKV** – adapted from prompt-time to *decode-time* by compressing every 128 generated tokens with the same budgets as R-KV.  
Head-layer-budget schedule (e.g. PyramidKV, HeadKV) are orthogonal and omitted here.

---

## Results

### Main Accuracy Curves

**Pass@1 vs. KV-cache budget** on MATH-500 and AIME-24.  
R-KV attains parity with **10–34 %** cache and even beats the FullKV baseline at *10 %*.

<div align="center">
  <img src="./assets/main_results.png" alt="R-KV Main Results" width="100%">
</div>




### Budget at a Glance

| Model | Dataset | **Lossless @ Ratio** | **Lossless @ Fixed tokens** |
|-------|---------|---------------------:|----------------------------:|
| R1-Llama-8B | MATH-500 | 34 % | 1 024 |
| R1-Llama-8B | AIME-24 | **10 %** | 1 536 |
| R1-Qwen-14B | MATH-500 | 54 % | 1 536 |
| R1-Qwen-14B | AIME-24 | 25 % | 3 072 |

At *16 %* (Llama-8B) or *33 %* (Qwen-14B) cache, **R-KV reaches 105 % of FullKV accuracy**—evidence that trimming redundant tokens can *improve* reasoning quality.



## ⚡️ Efficiency Results

### 🧮 Memory Saving
**R-KV** allocates two fixed-size buffers—one for the retained KV cache and one for freshly generated tokens.  
Because these buffers stay the same size regardless of sequence length, memory usage remains *constant*, unlike **FullKV**, whose memory grows linearly with the sequence.

R-KV keeps **two small, fixed-size buffers**:

| Buffer          | Purpose            | Shape                                    |
|-----------------|--------------------|------------------------------------------|
| **KV budget**   | Retained tokens    | `b × B_budget × L × H × d`               |
| **KV buffer**   | Fresh tokens       | `b × B_buffer × L × H × d`               |

> `b` = batch · `L` = #layers · `H` = #heads · `d` = head-dim

An optional **query cache** keeps only the last `α` query states.


### 🚀 Computation Overhead
R-KV adds a lightweight scoring pass for *importance* and *redundancy*, but the extra FLOPs are quickly offset by having to attend over a much smaller, compressed KV cache.  
As sequences get longer, the cost/benefit curve tips even further in R-KV’s favor (details in the same appendix).

| Stage                     | Complexity                                |
|---------------------------|-------------------------------------------|
| Importance scoring        | **O(α × B_budget)**                       |
| Redundancy scoring        | **O(B_budget²)**                          |
| Attention (compressed)    | **O((B_budget + B_buffer) × B_buffer)**   |
| Attention (FullKV)        | **O(B_full × B_buffer)**                  |

For long sequences (`B_full ≫ B_budget`), the tiny cache more than offsets scoring overhead.

### 📈 Real-Time Results
We measured both memory savings and end-to-end throughput (see **Table&nbsp;`efficiency`** in the paper):

- **Batch = 1:** R-KV already edges out FullKV in tokens-per-second, showing that reduced attention cost outweighs the scoring overhead.
- **Larger batches:** The real win comes from compression—smaller caches let us pack far more sequences into GPU memory, multiplying throughput.


#### 3.1 Ratio-Based Budget

| Output Len | Compression | Batch Gain | TPS Gain |
|-----------:|------------:|-----------:|---------:|
| **8 K**    | 54 % | 1.7 × | 1.5 × |
|            | 10 % | 7.7 × | 4.5 × |
| **16 K**   | 54 % | 1.5 × | 1.7 × |
|            | 10 % | 9.0 × | 6.6 × |

#### 3.2 Fixed KV Budget

| Output Len | `B_budget` | Batch Gain | TPS Gain |
|-----------:|-----------:|-----------:|---------:|
| **8 K**    | 1024 | 6.5 × | 3.8 × |
|            | 1536 | 4.6 × | 3.0 × |
| **16 K**   | 1024 | 13.4 × | 9.2 × |
|            | 1536 | 9.6 × | 7.1 × |

> *TPS = tokens per second.* Throughput scales almost linearly with batch until compute saturation (~128 sequences on an A100).


> Full benchmarking scripts and raw logs are provided in below table. 


| Gen. Length | Method | Budget | Mem. Saving (%) | Batch | Throughput (tok/s) | Tokens Gen. | Dec. Time (s) |
|-------------|--------|--------|-----------------|-------|--------------------|-------------|---------------|
| 8 K | **FullKV** | – | – | 1 | 75.44 | 8,094 | 107.30 |
| 8 K | **FullKV** | – | – | 62 (max) | 849.13 | 501,828 | 590.99 |
| 8 K | **R-KV** | Fixed – 1024 | 87.50 | 1 | 80.46 | 8,094 | 100.60 |
| 8 K | **R-KV** | Fixed – 1024 | 87.50 | 402 (max) | 3,251.52 | 3,253,788 | 1,000.70 |
| 8 K | **R-KV** | Fixed – 1536 | 81.25 | 287 (max) | 2,525.75 | 6,546,972 | 919.72 |
| 8 K | **R-KV** | Fixed – 3072 | 62.50 | 150 (max) | 1,520.99 | 1,214,100 | 798.23 |
| 8 K | **R-KV** | Ratio – 10 % – 819 | 90.00 | 479 (max) | 3,809.15 | 3,877,026 | 1,017.82 |
| 8 K | **R-KV** | Ratio – 34 % – 2,785 | 66.00 | 167 (max) | 1,608.01 | 1,351,698 | 840.61 |
| 8 K | **R-KV** | Ratio – 54 % – 4,423 | 46.00 | 105 (max) | 1,257.83 | 849,870 | 675.66 |
| 16 K | **FullKV** | – | – | 1 | 69.41 | 16,286 | 234.65 |
| 16 K | **FullKV** | – | – | 30 (max) | 347.03 | 488,580 | 1,407.89 |
| 16 K | **R-KV** | Fixed – 1024 | 93.75 | 1 | 80.95 | 16,286 | 201.18 |
| 16 K | **R-KV** | Fixed – 1024 | 93.75 | 402 (max) | 3,188.82 | 6,546,972 | 2,053.10 |
| 16 K | **R-KV** | Fixed – 1536 | 90.63 | 287 (max) | 2,447.61 | 4,674,082 | 1,909.65 |
| 16 K | **R-KV** | Fixed – 3072 | 81.25 | 150 (max) | 1,406.28 | 2,442,900 | 1,737.13 |
| 16 K | **R-KV** | Ratio – 10 % – 1,638 | 90.00 | 271 (max) | 2,300.28 | 4,413,506 | 1,918.68 |
| 16 K | **R-KV** | Ratio – 34 % – 5,570 | 66.00 | 82 (max) | 797.43 | 1,335,452 | 1,674.70 |
| 16 K | **R-KV** | Ratio – 54 % – 8,847 | 46.00 | 46 (max) | 584.77 | 749,156 | 1,281.12 |

### Key Takeaways

* **Constant memory** → run much longer sequences without OOM.  
* **Tiny attention window** → lower FLOPs despite a lightweight scoring pass.  
* **Bigger batches** → up to **13 ×** more sequences in parallel and **9 ×** higher tokens/s than FullKV.





## 🔍 R-KV vs. SnapKV: Token-Selection Comparison

> The figure below shows which tokens are picked by **R-KV** and the pure-attention baseline **SnapKV** at the same decoding step.  
> **Grey** = not selected &nbsp;|&nbsp; **Light orange → Dark red** = selected tokens (deeper red = chosen by more attention heads)

<div align="center">
  <img src="./assets/comparison.png" alt="R-KV Main Results" width="100%">
</div>

### Key Findings
- **Broader Coverage**  
  With its hybrid score (attention × redundancy suppression), *R-KV* selects tokens that are spread across the whole output, preserving critical context cues.

- **Higher Information Diversity**  
  Unlike SnapKV—which prefers tokens near the query and often selects the same local tokens repeatedly—*R-KV* captures valuable pieces of information at diverse positions.

- **Significantly Less Redundancy**  
  SnapKV still picks distant, low-value segments (e.g., *“3 students are leaving early.”*, *“But in the initial”*).  
  *R-KV*’s redundancy check filters most of these out.

### Takeaway
By combining **attention strength with redundancy filtering**, **R-KV** retains the important context and removes noise, successfully completing the task.  
In this example, the pure attention strategy of **SnapKV** fails due to limited coverage and excess redundancy.

## Visualization

We implement visualization functions to help illustrate the multi-step token eviction pattern.

Run `analysis_scripts/analysis.ipynb` to see which tokens are kept at each compression step.

# Citation

If you find R-KV useful in your research, please cite us:

```
@article{cai2025r,
  title={R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration},
  author={Cai, Zefan and Xiao, Wen and Sun, Hanshi and Luo, Cheng and Zhang, Yikai and Wan, Ke and Li, Yucheng and Zhou, Yeyang and Chang, Li-Wen and Gu, Jiuxiang and others},
  journal={arXiv preprint arXiv:2505.24133},
  year={2025}
}
```
