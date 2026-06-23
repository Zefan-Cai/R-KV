# R-KV Cluster (Sglang Implementation)

This repository provides the **Sglang implementation** of the R-KV cluster.  
It extends Sglang with support for **R-KV compression clusters**, enabling experimentation with compression-aware scheduling, attention, and memory pooling.

> **Note**: This implementation is based on **Sglang v0.4.3**.

---

## 📦 Installation

Follow the [Sglang v0.4.3 installation guideline](https://github.com/sgl-project/sglang/tree/v0.4.3#installation).  
After installing Sglang, clone this repository and use the modified modules as described below.

For a concrete starting environment, install the pinned R-KV dependency set:

```bash
pip install -r requirements-rkv.txt
pip install -e python
```

The checked-in SGLang package is `0.4.3.post1`; the requirements file keeps `vllm`, `sgl-kernel`, `flashinfer-python`, `torch`, and `triton` in the version range expected by this tree.

---

## 🚀 Getting Started

1. Open and run [`test.ipynb`](test.ipynb).  
2. Configure the parameters inside the notebook.  
3. Run the cells to execute the R-KV compression cluster and generate the expected outputs.

---

## 🔧 Main Changes

The following modifications were introduced to integrate R-KV into Sglang:

- **Cluster Parameters**
  - `python/sglang/srt/server_args.py`: added parameters for configuring compression cluster.

- **Testing**
  - `test.ipynb`: notebook to test R-KV compression cluster.

- **Utilities**
  - `python/sglang/compress/r1kv_utils.py`: utility functions for R-KV.

- **Core Logic**
  - `python/sglang/srt/layers/radix_attention.py`: core implementation for compression condition logic.

- **Schedulers & Executors**
  - `python/sglang/srt/managers/schedule_batch.py`  
  - `python/sglang/srt/managers/scheduler.py`  
  - `python/sglang/srt/managers/tp_worker.py`  
  - `python/sglang/srt/model_executor/forward_batch_info.py`  
  - `python/sglang/srt/model_executor/model_runner.py`  
  - *Added support for R-KV compression cluster parameters.*

- **Memory Management**
  - `python/sglang/srt/mem_cache/memory_pool.py`: added Q cache memory pool.  
  - `python/sglang/srt/mem_cache/chunk_cache.py`: added metadata for requests using R-KV compression.

---

## ⚠️ Limitations

- For **large batches**, the current Sglang implementation may not work as expected.  
- Further fixes and optimizations are needed for stability and scalability.
