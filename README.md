<h1 align="center">R1-KV</h1>


> **Shrink the cache, keep the brains.**  
> R-KV discards repetitive tokens on-the-fly, delivering *full-accuracy* reasoning with only a fraction of the memory.


<p align="center">
    ｜<a href="WIP"><b>🌏 Page</b></a> |
    <a href="WIP"><b>📖 Paper</b></a>｜
</p>


## 🔥 News

- 🚀 [25/05/29] We are pleased to announce the release of **R-KV**, a highly efficient decoding time KV Cache compression method to serve reasoning Models.

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



## Setup

### Install Dependencies
Use the following command to install the minimal required dependencies:
```bash
pip install -r requirements.txt
```

### Install FlashAttention
If you're using Hugging Face, we default to flash attention to speed up attention computation:

```python
model = AutoModelForCausalLM.from_pretrained(
    "model_name_or_path",
    attn_implementation="flash_attention_2",
)
```

## Quick Start
Before running the scripts, you need to build the rkv package:
```bash
pip install -e .
```

Use the following command to run R1-like models with R1-KV on math benchmarks:
```bash
bash examples/run.sh
```

Or you could use the code scripts:

```bash
export CUDA_VISIBLE_DEVICES=0

python3 ./run_math.py \
--dataset_path ./data/aime24.jsonl \
--save_path ./outputs/output.jsonl \
--model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
--max_length 16484 \
--eval_batch_size 1 \
--method rkv \
--kv_budget 128

```

To evaluate benchmark results, simply run:
```bash
bash examples/eval.sh
```
The results will be saved in the `outputs` directory.

### Evaluation

You need to build the dependencies of the evaluation toolkit separately:
```bash
cd evaluation/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt
```

## Visualization

We implement visualization functions to help illustrate the multi-step token eviction pattern.

Run `analysis_scripts/analysis.ipynb` to see which tokens are kept at each compression step.


## Motivation

Chain-of-thought (CoT) and self-reflection unlock impressive reasoning, but they **explode the key–value (KV) cache**.  
A single DeepSeek-R1-Distill-8B run on a tough math problem can:

* Generate **32 000 tokens**  
* Load **15.5 GB** of weights  
* Allocate **4.1 GB** of KV just to remember its own musings

Existing compression tools focus on *long prompts* and falter on *long generations*—often pruning the wrong tokens because redundant self-checks still attend heavily to themselves.




## 🔍 Method — Redundancy-aware KV Cache Compression (R-KV)

Redundancy-aware KV Cache Compression for R1-Style models (i.e., **R1-KV**) is the first production-level KV Cache Compression Method for serving R1-like models.

<div align="center">
  <img src="./assets/method.png" alt="R1-KV Main Results" width="100%">
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
  <img src="./assets/main_results.png" alt="R1-KV Main Results" width="100%">
</div>

### Main Accuracy Numerical Table

| Model       | Benchmark | Method | 128   | 256   | 512   | 768   | 1024  | 1536  | 2048  | 2560 | 3072 | 4096 |
|-------------|-----------|--------|-------|-------|-------|-------|-------|-------|-------|------|------|------|
| **Llama3-8B** | MATH| FullKV | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | —    | —    | —    |
| **Llama3-8B** | MATH | R-KV   | 51.08 | 67.39 | 76.92 | 80.21 | 81.34 | 82.34 | 82.65 | —    | —    | —    |
| **Llama3-8B** | MATH | SnapKV | 32.53 | 50.07 | 64.03 | 70.81 | 74.43 | 78.43 | 80.50 | —    | —    | —    |
| **Llama3-8B** | AIME24    | FullKV | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | —    |
| **Llama3-8B** | AIME24    | R-KV   | 0.42  | 10.21 | 29.48 | 40.31 | 45.26 | 51.56 | 52.29 | 53.85 | 53.13 | —    |
| **Llama3-8B** | AIME24    | SnapKV | 0.16  | 0.94  | 4.53  | 11.20 | 15.73 | 26.04 | 32.76 | 39.43 | 41.93 | —    |
| **Qwen-14B**  | MATH      | FullKV | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | —    | —    | —    |
| **Qwen-14B** | MATH     | R-KV   | 56.21 | 73.33 | 84.77 | 88.79 | 90.72 | 92.72 | 93.62 | —    | —    | —    |
| **Qwen-14B** | MATH      | SnapKV | 26.32 | 43.93 | 77.93 | 82.52 | 86.63 | 90.86 | 92.73 | —    | —    | —    |
| **Qwen-14B** | AIME24    | FullKV | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | —    | 65.68 | 65.68 |
| **Qwen-14B** | AIME24    | R-KV   | 0.57  | 7.92  | 24.53 | 36.25 | 42.66 | 55.00 | 56.09 | —    | 64.32 | 67.45 |
| **Qwen-14B** | AIME24    | SnapKV | 0.26  | 2.86  | 12.86 | 16.30 | 25.00 | 36.41 | 46.56 | —    | 52.86 | 54.32 |



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
Because these buffers stay the same size regardless of sequence length, memory usage remains *constant*, unlike **FullKV**, whose memory grows linearly with the sequence. See the full breakdown in the [Complexity appendix](./docs/appendix_complexity.md).

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
| 8 K | **SnapKV** | Fixed – 1024 | 87.50 | 1 | 81.26 | 8,094 | 99.60 |
| 8 K | **SnapKV** | Fixed – 1024 | 87.50 | 402 (max) | 3,253.93 | 3,253,788 | 999.96 |
| 8 K | **SnapKV** | Fixed – 1536 | 81.25 | 287 (max) | 2,525.25 | 2,322,978 | 919.90 |
| 8 K | **SnapKV** | Fixed – 3072 | 62.50 | 150 (max) | 1,527.67 | 1,214,100 | 794.74 |
| 8 K | **SnapKV** | Ratio – 10 % – 819 | 90.00 | 479 (max) | 3,808.81 | 3,877,026 | 1,017.91 |
| 8 K | **SnapKV** | Ratio – 34 % – 2,785 | 66.00 | 167 (max) | 1,625.46 | 1,351,698 | 831.58 |
| 8 K | **SnapKV** | Ratio – 54 % – 4,423 | 46.00 | 105 (max) | 1,269.68 | 849,870 | 669.36 |
| 8 K | **R-KV** | Fixed – 1024 | 87.50 | 1 | 80.46 | 8,094 | 100.60 |
| 8 K | **R-KV** | Fixed – 1024 | 87.50 | 402 (max) | 3,251.52 | 3,253,788 | 1,000.70 |
| 8 K | **R-KV** | Fixed – 1536 | 81.25 | 287 (max) | 2,525.75 | 6,546,972 | 919.72 |
| 8 K | **R-KV** | Fixed – 3072 | 62.50 | 150 (max) | 1,520.99 | 1,214,100 | 798.23 |
| 8 K | **R-KV** | Ratio – 10 % – 819 | 90.00 | 479 (max) | 3,809.15 | 3,877,026 | 1,017.82 |
| 8 K | **R-KV** | Ratio – 34 % – 2,785 | 66.00 | 167 (max) | 1,608.01 | 1,351,698 | 840.61 |
| 8 K | **R-KV** | Ratio – 54 % – 4,423 | 46.00 | 105 (max) | 1,257.83 | 849,870 | 675.66 |
| 16 K | **FullKV** | – | – | 1 | 69.41 | 16,286 | 234.65 |
| 16 K | **FullKV** | – | – | 30 (max) | 347.03 | 488,580 | 1,407.89 |
| 16 K | **SnapKV** | Fixed – 1024 | 87.50 | 1 | 81.03 | 16,286 | 200.99 |
| 16 K | **SnapKV** | Fixed – 1024 | 87.50 | 402 (max) | 3,202.17 | 6,546,972 | 2,044.54 |
| 16 K | **SnapKV** | Fixed – 1536 | 81.25 | 287 (max) | 2,449.02 | 4,674,082 | 1,908.56 |
| 16 K | **SnapKV** | Fixed – 3072 | 81.25 | 150 (max) | 1,413.84 | 2,442,900 | 1,727.84 |
| 16 K | **SnapKV** | Ratio – 10 % – 1,638 | 90.00 | 271 (max) | 2,306.26 | 4,413,506 | 1,913.71 |
| 16 K | **SnapKV** | Ratio – 34 % – 5,570 | 66.00 | 82 (max) | 798.42 | 1,335,452 | 1,672.61 |
| 16 K | **SnapKV** | Ratio – 54 % – 8,847 | 46.00 | 46 (max) | 586.43 | 749,156 | 1,277.48 |
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
  <img src="./assets/comparison.png" alt="R1-KV Main Results" width="100%">
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

> For additional experiments and implementation details, see [docs/RKV_details.md](./docs/RKV_details.md).
