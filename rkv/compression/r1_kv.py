from collections.abc import Mapping
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cal_similarity, compute_attention_scores



class KVView(Protocol):
    """LMCache-supplied read-only GPU KV view, without an LMCache import."""

    def get_keys(self) -> torch.Tensor: ...
    def get_values(self) -> torch.Tensor: ...


class R1KV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        record_kept_token_indices=False,
        buffer=128,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.buffer = buffer
        if self.buffer < self.window_size:
            raise ValueError("buffer must be >= window_size")
        self._serving_query_history = {}
        self._serving_layer_order = None

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

    def _compute_scores(self, query_states, key_states):
        attn_weights = compute_attention_scores(query_states, key_states)

        attn_weights_sum = (
            nn.functional.softmax(
                attn_weights[:, :, -self.window_size :, : -self.window_size],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)
            .to(query_states.dtype)
        )
        # TODO: Softmax then reduce head

        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        similarity_cos = cal_similarity(
            key_states,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )[:, :, : -self.window_size]

        final_score = attn_cache * self.mix_lambda - similarity_cos * (
            1 - self.mix_lambda
        )
        return final_score, attn_weights

    def score_kv(self, query_states, key_states):
        """Compute per-KV-head R-KV scores for tokens preceding the observation window, using Q and K without modifying either input."""
        scores, _ = self._compute_scores(query_states, key_states)
        return scores

    def should_observe_token_queries(
        self, phase: Literal["prefill", "decode"], decoded_tokens_before_step: int
    ) -> int:
        """Return how many Q rows to capture from this forward."""
        if phase == "prefill":
            return self.window_size
        if phase != "decode":
            raise ValueError(f"Unknown token-drop phase: {phase!r}")
        if decoded_tokens_before_step < 0:
            raise ValueError("decoded_tokens_before_step must be nonnegative")
        return int(
            decoded_tokens_before_step % self.buffer
            >= self.buffer - self.window_size
        )

    def observe_token_queries(
        self, queries_by_layer: Mapping[str, torch.Tensor]
    ) -> None:
        """Store the most recent post-RoPE Q rows for this request."""
        if not queries_by_layer:
            raise ValueError("R-KV observation requires non-empty layer inputs")
        layer_order = tuple(queries_by_layer)
        if self._serving_layer_order is None:
            self._serving_layer_order = layer_order
        elif layer_order != self._serving_layer_order:
            raise RuntimeError("R-KV serving layer order changed")

        for layer_name, query in queries_by_layer.items():
            if query.ndim != 3 or query.shape[0] == 0:
                raise ValueError(
                    "R-KV observation expects [tokens, q_heads, head_dim]"
                )
            recent = query[-self.window_size:].detach()
            previous = self._serving_query_history.get(layer_name)
            if previous is not None:
                recent = torch.cat([previous, recent], dim=0)[-self.window_size:]
            self._serving_query_history[layer_name] = recent

    def should_compact_kv(
        self,
        phase: Literal["prefill", "decode"],
        resident_kv_tokens: int,
        decoded_tokens_before_step: int,
    ) -> bool:
        """Run end-of-prefill or buffer-boundary R-KV compaction."""
        if phase == "prefill":
            return resident_kv_tokens > self.budget
        if phase != "decode":
            raise ValueError(f"Unknown token-drop phase: {phase!r}")
        return (
            decoded_tokens_before_step >= 0
            and (decoded_tokens_before_step + 1) % self.buffer == 0
            and resident_kv_tokens >= self.budget + self.buffer
        )

    def select_kept_token_positions(
        self, kv_by_layer: Mapping[str, KVView]
    ) -> Mapping[str, torch.Tensor]:
        """Return independent, algorithm-ordered positions per layer/KV head."""
        if not kv_by_layer:
            raise ValueError("R-KV selection requires non-empty KV views")
        layer_order = tuple(kv_by_layer)
        if layer_order != self._serving_layer_order:
            raise RuntimeError("R-KV serving layer order changed")

        retained = {}
        for layer_name, view in kv_by_layer.items():
            queries = self._serving_query_history.get(layer_name)
            if queries is None or queries.shape[0] < self.window_size:
                raise RuntimeError("R-KV does not have a full query window")
            keys = view.get_keys()
            if keys.ndim != 4 or keys.shape[0] != 1:
                raise ValueError("R-KV KVView keys must be [1, kv_heads, tokens, dim]")
            if keys.shape[2] < self.budget:
                raise ValueError("R-KV cannot compact fewer than budget tokens")
            query_window = queries[-self.window_size:].permute(1, 0, 2).unsqueeze(0)
            scores = self.score_kv(query_window, keys)
            kept_past = scores.topk(self.budget - self.window_size, dim=-1).indices
            recent = torch.arange(
                keys.shape[2] - self.window_size, keys.shape[2],
                device=keys.device, dtype=torch.long,
            ).view(1, 1, -1).expand(1, keys.shape[1], -1)
            retained[layer_name] = torch.cat([kept_past, recent], dim=-1)[0]

        self._serving_query_history.clear()
        return retained

    @classmethod
    def from_serving_config(cls, config: Mapping[str, Any]) -> "R1KV":
        """Build R-KV from the vLLM-port algorithm config."""
        if not isinstance(config, Mapping):
            raise ValueError("R-KV serving config must be a mapping")

        allowed = {
            "budget",
            "buffer",
            "window_size",
            "kernel_size",
            "mix_lambda",
            "retain_ratio",
            "retain_direction",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported R-KV serving config keys: {sorted(unknown)}"
            )

        values = {
            "budget": 128,
            "buffer": 128,
            "window_size": 8,
            "kernel_size": 7,
            "mix_lambda": 0.1,
            "retain_ratio": 0.1,
            "retain_direction": "last",
        }
        values.update(config)

        budget = values["budget"]
        buffer = values["buffer"]
        window_size = values["window_size"]
        kernel_size = values["kernel_size"]
        mix_lambda = values["mix_lambda"]
        retain_ratio = values["retain_ratio"]
        retain_direction = values["retain_direction"]

        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError("budget must be a positive integer")
        if not isinstance(buffer, int) or isinstance(buffer, bool) or buffer <= 0:
            raise ValueError("buffer must be a positive integer")
        if (
            not isinstance(window_size, int)
            or isinstance(window_size, bool)
            or window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer")
        if budget <= window_size:
            raise ValueError("budget must be greater than window_size")
        if buffer < window_size:
            raise ValueError("buffer must be >= window_size")
        if (
            not isinstance(kernel_size, int)
            or isinstance(kernel_size, bool)
            or kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be a positive odd integer")
        if not isinstance(mix_lambda, (int, float)) or isinstance(mix_lambda, bool):
            raise ValueError("mix_lambda must be numeric")
        if not 0.0 <= float(mix_lambda) <= 1.0:
            raise ValueError("mix_lambda must be in [0, 1]")
        if not isinstance(retain_ratio, (int, float)) or isinstance(
            retain_ratio, bool
        ):
            raise ValueError("retain_ratio must be numeric")
        if not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError("retain_ratio must be in (0, 1]")
        if retain_direction not in ("last", "first", "last_percent", "first_percent"):
            raise ValueError("Unsupported retain_direction")

        return cls(
            budget=budget,
            buffer=buffer,
            window_size=window_size,
            kernel_size=kernel_size,
            mix_lambda=float(mix_lambda),
            retain_ratio=float(retain_ratio),
            retain_direction=retain_direction,
        )

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            return key_states, value_states
        else:
            final_score, attn_weights = self._compute_scores(
                query_states, key_states
            )

            # shape: (bsz, num_kv_heads, budget - window_size)
            indices = final_score.topk(self.budget - self.window_size, dim=-1).indices

            #####################################################
            ###### Store evicted token indices start ############
            #####################################################
            # shape: (num_kv_heads, budget - window_size)
            if self.record_kept_token_indices:
                indices_cl = indices.clone().squeeze(0).to("cpu")

                similarity_cos_analysis = cal_similarity(
                    key_states,
                    retain_ratio=self.retain_ratio,
                    retain_direction=self.retain_direction,
                )

                attn_weights_sum_analysis = (
                    nn.functional.softmax(
                        attn_weights,
                        dim=-1,
                        dtype=torch.float32,
                    )
                    .mean(dim=-2)
                    .to(query_states.dtype)
                )

                attn_cache_analysis = F.max_pool1d(
                    attn_weights_sum_analysis,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )

                final_score_analysis = attn_cache_analysis * self.mix_lambda - similarity_cos_analysis * (
                    1 - self.mix_lambda
                )

                recent_window_indices = torch.arange(
                    kv_cache_len - self.window_size, kv_cache_len, device="cpu"
                ).expand(indices_cl.shape[0], -1)
                cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)

                #####################################################
                ### Store final scores, attention and similarity ####
                #####################################################

                # Gather the scores for the kept tokens
                attn_scores = attn_cache_analysis.clone().squeeze(0).to("cpu")
                sim_scores = similarity_cos_analysis.clone().squeeze(0).to("cpu")
                fin_scores = final_score_analysis.clone().squeeze(0).to("cpu")

                # print(f"cur_indices {cur_indices} attn_cache_analysis {attn_cache_analysis.shape} similarity_cos_analysis {similarity_cos_analysis.shape} final_score_analysis {final_score_analysis.shape}")

                # Gather the scores based on index
                kept_attn = torch.gather(attn_scores, dim=1, index=cur_indices)
                kept_sim = torch.gather(sim_scores, dim=1, index=cur_indices)
                kept_final = torch.gather(fin_scores, dim=1, index=cur_indices)

                #####################################################

                if self.evicted_token_num > 0:
                    prev_indices = self.kept_token_indices[-1]
                    mask = cur_indices < self.budget

                    for i in range(cur_indices.shape[0]):
                        positions = torch.where(mask[i])[0]

                        # For each position, get the value and use it as an index into prev_indices
                        for pos in positions:
                            val = cur_indices[i, pos].item()
                            cur_indices[i, pos] = prev_indices[i, val]

                    # For values >= self.budget, add the evicted token count
                    cur_indices[~mask] += self.evicted_token_num

                #####################################################
                ### Store final scores, attention and similarity ####
                #####################################################
                self.kept_attention_scores.append(kept_attn)
                self.kept_similarity_scores.append(kept_sim)
                self.kept_final_scores.append(kept_final)
                #####################################################

                self.kept_token_indices.append(cur_indices)
                self.evicted_token_num += kv_cache_len - self.budget
            ######################################################

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states
