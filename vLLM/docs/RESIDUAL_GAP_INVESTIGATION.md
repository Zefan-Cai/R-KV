# R-KV on vLLM — Accuracy-Gap Investigation (async-scheduling KV corruption)

This document records the end-to-end investigation that closed the accuracy gap
between the vLLM R-KV port and the SGLang reference at a tight budget/buffer, and
the root-cause bug it uncovered. It is the narrative companion to the bug list in
[OPTIMIZATIONS.md](OPTIMIZATIONS.md).

## TL;DR

At `budget=256 buffer=64` (decode-only, Qwen2.5-Math-7B, few-shot GSM8K, greedy,
200 questions) the port trailed SGLang by ~6.5 points (82.0% vs 88.5%) even after
a faithful two-phase cross-layer refactor. The residual was **not** numerics — it
was a **batch-dependent KV-cache corruption** caused by vLLM V1 **async
scheduling** racing R-KV's post-compaction `num_dropped_tokens` feedback. Forcing
synchronous scheduling when R-KV is enabled lifted `buffer=64` to **87.5%**
(SGLang 88.5%, within n=200 noise) and restored `buffer=128` to the lossless
90.5% Full-KV ceiling.

## 1. Starting point

- Full-KV parity was already established: vLLM Full-KV = SGLang Full-KV = **90.5%**
  on the identical harness. So the two engines are equivalent for full attention.
- R-KV degradation was asymmetric: vLLM `−8.5` (82.0%) vs SGLang `−2.0` (88.5%) at
  `budget=256 buffer=64`.
- A conspicuous clue: **buffer sensitivity**. vLLM `buffer=64 ≈ 82%`,
  `buffer=128 ≈ 88–90%` (lossless); SGLang was flat (`88.5%` → `89.0%`). vLLM was
  ~6× more buffer-sensitive than SGLang.

## 2. What was ruled out (differential testing)

Each of these was verified **identical/correct** and is *not* the cause:

| Component | Method | Result |
| --- | --- | --- |
| Compaction algorithm | Read both `algo.py`; feed identical K/Q | Bit-identical scores |
| Redundancy kernel | vLLM `cal_similarity` vs SGLang fused Triton | ≤ 1e-5 |
| Integration arithmetic | Ran SGLang algo on vLLM's captured keys | Kept set matches |
| KV persistence | Traced 4 consecutive compactions | Relocated keys maxabs 0.00, 256/256 exact |
| Compaction cadence | Traced seq_len at each compaction | vLLM 728,320,320 vs SGLang 727,320,320 |
| RoPE policy | Logical positions after eviction | Continuous (729,730,731), physical slot 257,258,259 |
| RoPE precision | Forced vLLM cos/sin cache to fp32 | Caches bit-identical; R-KV accuracy unchanged |
| K/Q vs HF reference | Compared post-RoPE keys | vLLM matches HF (mean 0.009); SGLang deviates (0.656) |

The K/Q tensors differ ~1% on Qwen's massive-activation dims (cosine 0.998) purely
from FlashAttention-vs-FlashInfer kernel numerics. This *looked* like the answer
("irreducible bf16 numerics amplified by discrete top-k") — but it was a red
herring, because it could not explain the buffer-sensitivity asymmetry.

## 3. The breakthrough — per-prompt analysis

Two observations reframed the problem:

1. **Full-KV greedy output is byte-identical** between vLLM and SGLang for the
   first 60 tokens (below the first compaction). So the engine kernels do **not**
   flip token decisions — every divergence is introduced by the compaction's
   kept-set selection.

2. Comparing vLLM R-KV against **vLLM's *own* Full-KV** (not against SGLang) at
   `buffer=64`: R-KV lost **12 points** from its own 90.5% baseline, while SGLang
   R-KV loses ~0 from its own baseline. A cross-engine numeric difference would
   affect each engine *relative to its own baseline* equally — so this asymmetry
   pointed to a **real R-KV bug in vLLM**, independent of SGLang.

The victim prompts (Full-KV correct, R-KV wrong) had two signatures:

- **Concentrated on long generations** (median gen-len 422 vs 165 overall; several
  prompts that finish in 72 tokens with Full-KV ran to the 512-token cap producing
  garbage — negative numbers — with R-KV). A slightly-different token selection
  cannot do that; this is **KV corruption**, and it **compounds per compaction**.
- **Batch-dependent**: prompt idx 26 was *correct* run in isolation and in a
  40-prompt batch, but *broke* in a 100-prompt batch. Full-KV is batch-independent;
  R-KV was not. That non-determinism-by-batch is the fingerprint of a bug that
  couples a request to its batch neighbours.

## 4. Root cause — async scheduling races the eviction feedback

R-KV compaction runs **inside the forward pass** and reports each request's
evicted-token count as a **model output** (`num_dropped_tokens`). The scheduler
accumulates it (`request.num_dropped_tokens += …`), and the model runner uses it
to compute the next step's **physical** KV positions:

```
physical_position = logical_position − num_dropped_tokens
seq_len(physical)  = num_computed − num_dropped + num_scheduled
slot_mapping       = block_table[physical_position // block_size] · block_size + …
```

vLLM V1 **async scheduling** prepares step *N+1* **before** step *N*'s output is
processed. So for exactly one step after every compaction, `num_dropped_tokens` is
**stale** (too small), the physical position is computed one block too high, and
the new decode token is written to a slot that still holds a **surviving** kept
token — overwriting live KV. The corrupted token then poisons all subsequent
attention, and the generation drifts (often running to the token cap).

Why it hid so well:

- **Single-request / small-batch runs never trip it** — the race only manifests
  under the memory pressure and pipeline overlap of many concurrent requests, so
  every isolated repro passed.
- **It compounds with the number of compactions**, so long generations and tight
  buffers (more frequent compaction) are hit hardest — which is exactly the
  "buffer sensitivity" that was misattributed to numerics.

Falsified alternative hypotheses along the way: preemption reset (`num_dropped`
not reset — but preemption never fired at this memory setting), cross-request slot
collisions (none), score/seq-len misalignment between `observe_layer` and
`compact_step` (none), and a suspected per-layer observation-window bug (a no-op:
each attention layer owns its own `RKVCompressor`, so keying `_qwin` by request id
is already per-layer).

## 5. The fix

Force-disable async scheduling whenever R-KV is enabled, in
`VllmConfig.__post_init__` — right beside the existing R-KV prefix-caching
disable, so the eviction accounting is always current before the next step:

```python
if (
    envs.VLLM_V1_R_KV_BUDGET > 0
    and envs.VLLM_V1_R_KV_BUFFER > 0
    and self.scheduler_config is not None
    and self.scheduler_config.async_scheduling is not False
):
    logger.info_once(
        "R-KV enabled: disabling async scheduling (compaction's "
        "dropped-token feedback must be applied before the next step's "
        "KV positions are computed)."
    )
    self.scheduler_config.async_scheduling = False
```

The `is not False` guard also catches the default (`None`, which otherwise
resolves to enabled). It applies automatically — no user flag — and logs the
reason at startup.

## 6. Result

Same few-shot GSM8K harness, Qwen2.5-Math-7B-Instruct, 200 questions, greedy,
decode-only, `budget=256`:

| Config | Before fix | After fix | SGLang |
| --- | --- | --- | --- |
| buffer=64 | 82.0% | **87.5%** | 88.5% |
| buffer=128 | 88.0% | **90.5%** (lossless) | 89.0% |

87.5% vs 88.5% is a one-question difference — within n=200 noise. Both engines now
degrade only ~2–3 points from the identical 90.5% Full-KV ceiling, and the
buffer-sensitivity asymmetry is gone. The remaining ~1-point spread at `buffer=64`
is genuine FlashAttention-vs-FlashInfer K/Q numerics near the top-k score cutoff
(vLLM's post-RoPE keys match the HuggingFace reference bit-closely); it is a
property of aggressive compression, not a bug. Recommended: `buffer ≈ budget`.

## 7. Lessons

- **Compare a compression method against *its own* uncompressed baseline**, not
  only against a reference implementation. The within-engine degradation
  (12 pts vs ~0) exposed the bug that the cross-engine comparison masked as
  "numerics".
- **Batch-dependent output is a correctness bug**, full stop — greedy decoding must
  be deterministic per prompt regardless of batch neighbours. Reproducing at
  N=1 / N=40 / N=100 localised it fast.
- **Any per-request state that a compression pass feeds back through a model output
  is incompatible with async scheduling** unless it is applied before the next
  step's inputs are built. This is the same class of hazard as the SGLang port's
  overlap-scheduling ban.
