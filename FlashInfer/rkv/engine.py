"""FlashInfer serving engine for R-KV decode-time KV cache compression.

Paged pool with static per-request regions and a logical/physical length
split (scheme follows ``SGLang/rkv/integration.py``); prefill compression and
window-query seeding follow ``HuggingFace/rkv/modeling.py``. One decode plan
per step over active rows; compaction runs eagerly on the main stream.
"""

from __future__ import annotations

import time
from itertools import accumulate
from typing import Sequence

import torch
from transformers import AutoConfig

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.sampling import top_p_sampling_from_probs
from flashinfer.utils import device_support_pdl

from . import algo
from .compressor import R1KV
from .config import GenOutput, RKVConfig
from .loader import load_weights
from .models import CausalLM

_WORKSPACE_BYTES = 128 * 1024 * 1024


class FlashInferEngine:
    """Single-GPU, eager, static-batch generation engine.

    ``rkv=None`` runs the FullKV baseline (per-request region of
    ``max_seq_len`` slots); with an :class:`RKVConfig` each request's region is
    ``budget + buffer`` slots, which is where the memory saving physically
    comes from.
    """

    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 8,
        max_seq_len: int = 32768,
        rkv: RKVConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.rkv = rkv

        hf_config = AutoConfig.from_pretrained(model_path)
        self.model = CausalLM(
            hf_config, max_position=max_seq_len, device=self.device, dtype=dtype
        )
        load_weights(self.model, model_path)

        model = self.model
        self.region_len = max_seq_len if rkv is None else rkv.budget + rkv.buffer
        # page_size = 1 (slot == token), layout NHD: per layer
        # [max_slots, 2, 1, num_kv_heads, head_dim]; request r owns slots
        # [r * region_len, (r + 1) * region_len).
        self.kv_pool = torch.zeros(
            (
                model.num_layers,
                max_batch_size * self.region_len,
                2,
                1,
                model.num_kv_heads,
                model.head_dim,
            ),
            device=self.device,
            dtype=dtype,
        )

        workspace = torch.zeros(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )
        group_size = model.num_q_heads // model.num_kv_heads
        self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=group_size >= 4
        )
        self._prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

        # Static per-step planning inputs. Slot ids per request are always
        # `r * region_len + [0, n)`, so decode kv_indices are cheap slices of
        # one precomputed arange instead of per-request arange launches.
        # indptr / last_page_len stay on CPU: flashinfer 0.6.x `plan()` reads
        # them host-side (`.to("cpu")`), so passing device tensors would force
        # a D2H sync copy inside every decode step.
        self._pool_indices = torch.arange(
            max_batch_size * self.region_len, dtype=torch.int32, device=self.device
        )
        self._last_page_len_cpu = torch.ones(max_batch_size, dtype=torch.int32)
        # run() otherwise allocates a fresh out tensor and re-detects PDL
        # support num_layers times per decode step.
        self._attn_out = torch.empty(
            max_batch_size,
            model.num_q_heads,
            model.head_dim,
            dtype=dtype,
            device=self.device,
        )
        self._enable_pdl = (
            device_support_pdl(self.device) if self.device.type == "cuda" else False
        )

        self.compressor: R1KV | None = None
        if rkv is not None:
            self.compressor = R1KV(
                rkv,
                num_layers=model.num_layers,
                max_batch_size=max_batch_size,
                num_q_heads=model.num_q_heads,
                head_dim=model.head_dim,
                device=self.device,
                dtype=dtype,
            )

        self.stats: dict[str, float] = {}

    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        stop_token_ids: Sequence[int] = (),
        seed: int = 42,
    ) -> list[GenOutput]:
        batch = len(prompts)
        prompt_lens = [len(p) for p in prompts]
        self._validate_request(batch, prompt_lens, max_new_tokens)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        self._phys_len = [0] * batch
        self._logical_len = [0] * batch
        if self.compressor is not None:
            self.compressor.reset()

        start = time.perf_counter()
        logits = self._prefill(prompts, prompt_lens)
        self._sync()
        prefill_seconds = time.perf_counter() - start

        token_ids: list[list[int]] = [[] for _ in range(batch)]
        finish_reason = [""] * batch
        last_token = [0] * batch
        stop_ids = set(stop_token_ids)

        def record(rows: list[int], sampled: list[int]) -> None:
            for row, token in zip(rows, sampled):
                token_ids[row].append(token)
                last_token[row] = token
                if token in stop_ids:
                    finish_reason[row] = "stop"
                elif len(token_ids[row]) >= max_new_tokens:
                    finish_reason[row] = "length"
                elif self._logical_len[row] >= self.max_seq_len:
                    finish_reason[row] = "length"

        start = time.perf_counter()
        compaction_seconds = 0.0
        # Compaction time is attributed with CUDA event pairs read after the
        # final sync, instead of bracketing each round with two host syncs
        # that would stall the decode pipeline.
        compaction_events: list[tuple] = []
        use_events = self.device.type == "cuda"
        sampled = self._sample(logits, temperature, top_p, generator).tolist()
        record(list(range(batch)), sampled)

        while True:
            active = [r for r in range(batch) if not finish_reason[r]]
            if not active:
                break
            logits = self._decode_step(active, last_token)
            sampled = self._sample(logits, temperature, top_p, generator).tolist()
            record(active, sampled)
            if self.compressor is not None:
                rows = [
                    r
                    for r in active
                    if not finish_reason[r]
                    and self.compressor.should_compact(self._phys_len[r])
                ]
                if rows:
                    if use_events:
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                    else:
                        compact_start = time.perf_counter()
                    self._compact_rows(rows)
                    if use_events:
                        end.record()
                        compaction_events.append((begin, end))
                    else:
                        compaction_seconds += time.perf_counter() - compact_start
        self._sync()
        decode_seconds = time.perf_counter() - start
        compaction_seconds += (
            sum(begin.elapsed_time(end) for begin, end in compaction_events) / 1e3
        )

        num_compactions = (
            self.compressor.num_compactions[:batch]
            if self.compressor is not None
            else [0] * batch
        )
        self.stats = {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "compaction_seconds": compaction_seconds,
            "peak_memory_bytes": (
                float(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0.0
            ),
            "total_compactions": float(sum(num_compactions)),
        }
        return [
            GenOutput(
                token_ids=token_ids[r],
                num_prefill_tokens=prompt_lens[r],
                num_output_tokens=len(token_ids[r]),
                num_compactions=num_compactions[r],
                finish_reason=finish_reason[r],
            )
            for r in range(batch)
        ]

    def _validate_request(
        self, batch: int, prompt_lens: list[int], max_new_tokens: int
    ) -> None:
        if not 0 < batch <= self.max_batch_size:
            raise ValueError(f"batch size {batch} not in [1, {self.max_batch_size}]")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        for i, length in enumerate(prompt_lens):
            if length < 1:
                raise ValueError(f"prompt {i} is empty")
            if length >= self.max_seq_len:
                raise ValueError(
                    f"prompt {i} has {length} tokens; needs < max_seq_len "
                    f"({self.max_seq_len}) to generate"
                )
            if (
                self.rkv is not None
                and not self.rkv.compress_prefill
                and length >= self.region_len
            ):
                raise ValueError(
                    f"prompt {i} has {length} tokens but the R-KV region holds "
                    f"{self.region_len}; enable compress_prefill or raise "
                    "budget + buffer"
                )

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------
    def _prefill(
        self, prompts: Sequence[Sequence[int]], prompt_lens: list[int]
    ) -> torch.Tensor:
        model = self.model
        device = self.device
        indptr = [0, *accumulate(prompt_lens)]
        input_ids = torch.tensor(
            [t for p in prompts for t in p], dtype=torch.long, device=device
        )
        positions = torch.cat(
            [
                torch.arange(length, dtype=torch.int32, device=device)
                for length in prompt_lens
            ]
        )
        self._plan_prefill(indptr, prompt_lens)
        self._prefill_bounds = list(zip(indptr[:-1], indptr[1:]))
        for row, length in enumerate(prompt_lens):
            # Physical length after prefill: update_kv keeps exactly `budget`
            # tokens once the prompt reaches it, otherwise the whole prompt.
            if self.rkv is not None and self.rkv.compress_prefill:
                self._phys_len[row] = min(length, self.rkv.budget)
            else:
                self._phys_len[row] = length
            self._logical_len[row] = length

        hidden = model(input_ids, positions, self._prefill_attention)
        last = torch.tensor(
            [end - 1 for end in indptr[1:]], dtype=torch.long, device=device
        )
        return model.compute_logits(hidden[last])

    def _plan_prefill(self, indptr: list[int], prompt_lens: list[int]) -> None:
        # CPU tensor: flashinfer plan() reads indptr host-side (see __init__).
        qo_indptr = torch.tensor(indptr, dtype=torch.int32)
        self._prefill_wrapper.plan(
            qo_indptr,
            qo_indptr,
            self.model.num_q_heads,
            self.model.num_kv_heads,
            self.model.head_dim,
            causal=True,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def _run_prefill_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return self._prefill_wrapper.run(q, k, v)

    def _prefill_attention(
        self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        rkv = self.rkv
        for row, (start, end) in enumerate(self._prefill_bounds):
            keys, values = k[start:end], v[start:end]
            if self.compressor is not None:
                window_q = q[max(start, end - rkv.window_size) : end]
                self.compressor.seed_window(layer, row, window_q)
                if rkv.compress_prefill:
                    kept_k, kept_v = algo.update_kv(
                        keys.transpose(0, 1).unsqueeze(0),
                        window_q.transpose(0, 1).unsqueeze(0),
                        values.transpose(0, 1).unsqueeze(0),
                        budget=rkv.budget,
                        window_size=rkv.window_size,
                        kernel_size=rkv.kernel_size,
                        mix_lambda=rkv.mix_lambda,
                        retain_ratio=rkv.retain_ratio,
                        retain_direction=rkv.retain_direction,
                    )
                    keys = kept_k.squeeze(0).transpose(0, 1)
                    values = kept_v.squeeze(0).transpose(0, 1)
            base = row * self.region_len
            kept = keys.shape[0]
            self.kv_pool[layer, base : base + kept, 0, 0] = keys
            self.kv_pool[layer, base : base + kept, 1, 0] = values
        # Prefill attention runs over the full (uncompressed) ragged prompt
        # K/V, exactly like the HF reference; only the cache is compressed.
        return self._run_prefill_attention(q, k, v)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def _plan_decode(self, active: list[int], lens: list[int]) -> None:
        # Plan over post-append KV lengths: the step's k/v are written into
        # slot phys_len before the wrapper runs. indptr / last_page_len are
        # CPU tensors by design (see __init__); kv_indices are device slices
        # of the precomputed pool arange.
        model = self.model
        kv_indptr = torch.tensor([0, *accumulate(lens)], dtype=torch.int32)
        regions = [
            self._pool_indices[r * self.region_len : r * self.region_len + n]
            for r, n in zip(active, lens)
        ]
        kv_indices = regions[0] if len(regions) == 1 else torch.cat(regions)
        self._decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            self._last_page_len_cpu[: len(active)],
            model.num_q_heads,
            model.num_kv_heads,
            model.head_dim,
            1,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def _decode_step(self, active: list[int], last_token: list[int]) -> torch.Tensor:
        model = self.model
        device = self.device
        lens = [self._phys_len[r] + 1 for r in active]
        self._plan_decode(active, lens)

        # One H2D transfer for all per-step scalars instead of four.
        staged = torch.tensor(
            [
                [r * self.region_len + self._phys_len[r] for r in active],
                active,
                [last_token[r] for r in active],
                [self._logical_len[r] for r in active],
            ],
            dtype=torch.long,
        ).to(device, non_blocking=True)
        self._decode_slots, self._decode_rows, input_ids, positions64 = staged.unbind(0)
        positions = positions64.to(torch.int32)
        if self.compressor is not None:
            self.compressor.begin_step(active)

        hidden = model(input_ids, positions, self._decode_attention)
        for r in active:
            self._phys_len[r] += 1
            self._logical_len[r] += 1
        return model.compute_logits(hidden)

    def _decode_attention(
        self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        self.kv_pool[layer, self._decode_slots, 0, 0] = k
        self.kv_pool[layer, self._decode_slots, 1, 0] = v
        if self.compressor is not None:
            self.compressor.push_queries(layer, self._decode_rows, q)
        return self._run_decode_attention(layer, q)

    def _run_decode_attention(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        return self._decode_wrapper.run(
            q,
            self.kv_pool[layer],
            out=self._attn_out[: q.shape[0]],
            enable_pdl=self._enable_pdl,
        )

    def _compact_rows(self, rows: list[int]) -> None:
        # Rows compacting on the same step share phys_len (equality trigger),
        # so all layers of all rows batch into few update_kv calls; the
        # per-pair bit-parity gate lives in ``compressor.compact_batch``.
        budget = self.rkv.budget
        phys = self._phys_len[rows[0]]
        keys = torch.stack(
            [
                self.kv_pool[:, r * self.region_len : r * self.region_len + phys, 0, 0]
                for r in rows
            ],
            dim=1,
        )
        values = torch.stack(
            [
                self.kv_pool[:, r * self.region_len : r * self.region_len + phys, 1, 0]
                for r in rows
            ],
            dim=1,
        )
        kept_k, kept_v = self.compressor.compact_batch(rows, keys, values)
        for i, row in enumerate(rows):
            base = row * self.region_len
            self.kv_pool[:, base : base + budget, 0, 0] = kept_k[:, i]
            self.kv_pool[:, base : base + budget, 1, 0] = kept_v[:, i]
            self._phys_len[row] = budget

    # ------------------------------------------------------------------
    # Sampling / misc
    # ------------------------------------------------------------------
    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if temperature == 0.0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return top_p_sampling_from_probs(
            probs, top_p, deterministic=True, generator=generator
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
