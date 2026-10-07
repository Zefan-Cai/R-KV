from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cal_similarity, compute_attention_scores


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

        # Serving-only query history stays inside R-KV. The runtime forwards
        # request-local Q tensors but does not know R-KV's window semantics.
        self._serving_query_history = {}

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

    @classmethod
    def from_serving_config(cls, config):
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

        missing = {"budget", "buffer"} - set(config)
        if missing:
            raise ValueError(
                f"Missing required R-KV serving config keys: {sorted(missing)}"
            )

        values = {
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
        if retain_direction not in ("last", "first"):
            raise ValueError("retain_direction must be 'last' or 'first'")

        return cls(
            budget=budget,
            buffer=buffer,
            window_size=window_size,
            kernel_size=kernel_size,
            mix_lambda=float(mix_lambda),
            retain_ratio=float(retain_ratio),
            retain_direction=retain_direction,
        )

    def _compute_scores(
        self,
        key_states,
        query_states,
    ):
        """Compute the shared R-KV score used by all public APIs."""
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

    def observe_query(self, layer_name, query):
        """Record the request-local Q state R-KV needs for its next decision."""
        if query.ndim != 3 or query.shape[0] == 0:
            raise ValueError(
                "R-KV observation expects [tokens, q_heads, head_dim]"
            )

        # R-KV scores one decode-frontier query per observation step. Keep
        # bounded copies here; materialize the ordered window only when scoring.
        history = self._serving_query_history.setdefault(layer_name, [])
        history.append(query[-1:].clone())
        if len(history) > self.window_size:
            del history[:-self.window_size]

    def _serving_query_window(self, layer_name):
        history = self._serving_query_history.get(layer_name, [])
        if len(history) < self.window_size:
            raise RuntimeError(
                f"R-KV does not have a full query window for {layer_name!r}"
            )

        ordered = torch.cat(history, dim=0)
        return ordered.permute(1, 0, 2).unsqueeze(0).contiguous()

    def select_kept_positions(
        self,
        layer_key_states: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return one ordered retained-position set shared by all KV layers."""
        if not layer_key_states:
            raise ValueError("R-KV selection requires non-empty layer inputs")

        shared_scores = None
        kv_cache_len = None
        for layer_name, key_states in layer_key_states.items():
            query_states = self._serving_query_window(layer_name)
            if key_states.ndim != 4 or key_states.shape[0] != 1:
                raise ValueError("R-KV serving selection expects one batched request")
            if kv_cache_len is None:
                kv_cache_len = int(key_states.shape[-2])
            elif int(key_states.shape[-2]) != kv_cache_len:
                raise ValueError("R-KV selection requires one shared KV length")

            layer_score, _ = self._compute_scores(key_states, query_states)
            layer_score = layer_score.mean(dim=1)[0]
            shared_scores = (
                layer_score if shared_scores is None else shared_scores + layer_score
            )

        assert shared_scores is not None
        assert kv_cache_len is not None
        if not torch.isfinite(shared_scores).all():
            raise RuntimeError("R-KV computed non-finite scores; refusing to select")

        past_idx = shared_scores.topk(
            self.budget - self.window_size,
            dim=-1,
        ).indices
        window_idx = torch.arange(
            kv_cache_len - self.window_size,
            kv_cache_len,
            device=past_idx.device,
        )
        return torch.sort(torch.cat([past_idx, window_idx], dim=-1)).values

    def _crosses_buffer_boundary(
        self,
        *,
        num_decoded_tokens,
        num_new_tokens,
    ):
        prev_decoded_tokens = max(0, num_decoded_tokens - num_new_tokens)
        return (
            num_decoded_tokens > 0
            and num_decoded_tokens // self.buffer
            > prev_decoded_tokens // self.buffer
        )

    def should_observe_query(
        self,
        *,
        num_decoded_tokens,
        num_new_tokens,
        is_genuine_decode,
    ):
        """Return whether this decode step belongs to the next scoring window."""
        if not is_genuine_decode or num_new_tokens <= 0:
            return False

        prev_decoded_tokens = max(0, num_decoded_tokens - num_new_tokens)
        next_boundary = (prev_decoded_tokens // self.buffer + 1) * self.buffer
        observation_start = next_boundary - self.window_size + 1
        return num_decoded_tokens >= observation_start

    def should_compact(
        self,
        *,
        resident_len,
        num_decoded_tokens,
        num_new_tokens,
        is_genuine_decode,
    ):
        if not is_genuine_decode or num_new_tokens <= 0:
            return False
        if min(
            (len(history) for history in self._serving_query_history.values()),
            default=0,
        ) < self.window_size:
            return False
        if resident_len < self.budget + self.buffer:
            return False
        return self._crosses_buffer_boundary(
            num_decoded_tokens=num_decoded_tokens,
            num_new_tokens=num_new_tokens,
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
                key_states, query_states
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
