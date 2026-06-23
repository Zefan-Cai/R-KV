import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class KVCluster:
    def __init__(
        self,
        compress_method="RKV",
        max_capacity_prompts=128,
        attn_pooling="mean",
        window_size=8,
        kernel_size=7,
        pooling="avgpool",
        fix_obs_window=True,
        first_tokens=4,
        similarity_pooling="mean",
        sim_threshold=0.5,
        sim_normalization=True,
        mix_alpha=0.07,
        retain_num=None,
        retain_ratio=0.1,
        retain_direction="last",
        similarity_method="key",
        num_hidden_layers=None,
        layer_idx=None,
        record_kept_token_indices=False,
    ):
        # for all methods
        self.compress_method = compress_method
        self.max_capacity_prompts = max_capacity_prompts
        self.attn_pooling = attn_pooling

        # for RKV, snapkv, pyramidkv
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.fix_obs_window = fix_obs_window

        # for streamingllm
        self.first_tokens = first_tokens

        # for RKV
        self.similarity_pooling = similarity_pooling
        self.sim_threshold = sim_threshold
        self.sim_normalization = sim_normalization
        self.mix_alpha = mix_alpha
        self.retain_num = retain_num
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.similarity_method = similarity_method

        # for pyramidkv
        self.num_hidden_layers = num_hidden_layers
        self.layer_idx = layer_idx

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
    ):

        assert (
            self.max_capacity_prompts - self.window_size > 0
        ), "max_capacity_prompts must be greater than window_size"

        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.max_capacity_prompts:
            return key_states, value_states
        elif self.compress_method == "streamingllm":
            local_window_size = self.max_capacity_prompts - self.first_tokens
            # only select the first self.first_tokens tokens and the last local_window_size tokens
            key_states = torch.cat(
                [
                    key_states[:, :, : self.first_tokens],
                    key_states[:, :, -local_window_size:],
                ],
                dim=2,
            )
            value_states = torch.cat(
                [
                    value_states[:, :, : self.first_tokens],
                    value_states[:, :, -local_window_size:],
                ],
                dim=2,
            )
            return key_states, value_states
        else:
            if self.compress_method == "h2o":
                self.window_size = 1
                query_states = query_states[:, :, -1:, :]
            elif self.compress_method == "pyramidkv":
                min_num = (self.max_capacity_prompts - self.window_size) // 10
                max_num = (self.max_capacity_prompts - self.window_size) * 2 - min_num
                step = (max_num - min_num) // (self.num_hidden_layers - 1)
                self.max_capacity_prompts = max_num - self.layer_idx * step

            attn_weights = compute_attention_scores(
                query_states, key_states, pooling=self.attn_pooling
            )

            if not self.fix_obs_window:
                mask = torch.full(
                    (self.window_size, self.window_size),
                    torch.finfo(attn_weights.dtype).min,
                    device=attn_weights.device,
                )
                mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(attn_weights.device)
                attention_mask = mask[None, None, :, :]
                attn_weights[
                    :, :, -self.window_size :, -self.window_size :
                ] += attention_mask

                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)

                attn_weights_sum = attn_weights[
                    :, :, -self.window_size :, : -self.window_size
                ].mean(dim=-2)
            else:
                attn_weights_sum = (
                    nn.functional.softmax(
                        attn_weights[:, :, -self.window_size :, : -self.window_size],
                        dim=-1,
                        dtype=torch.float32,
                    )
                    .mean(dim=-2)
                    .to(query_states.dtype)
                )
            if self.compress_method != "h2o":  # h2o has no pooling
                if self.pooling == "avgpool":
                    attn_cache = F.avg_pool1d(
                        attn_weights_sum,
                        kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2,
                        stride=1,
                    )
                elif self.pooling == "maxpool":
                    attn_cache = F.max_pool1d(
                        attn_weights_sum,
                        kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2,
                        stride=1,
                    )
                else:
                    raise ValueError("Pooling method not supported")
            else:
                attn_cache = attn_weights_sum

            if self.compress_method == "RKV":
                if self.similarity_method == "key":
                    similarity_cos = cal_similarity(
                        key_states,
                        threshold=self.sim_threshold,
                        aggregation=self.similarity_pooling,
                        normalization=self.sim_normalization,
                        retain_num=self.retain_num,
                        retain_ratio=self.retain_ratio,
                        retain_direction=self.retain_direction,
                    )[:, :, : -self.window_size]
                elif self.similarity_method == "value":
                    similarity_cos = cal_similarity(
                        value_states,
                        threshold=self.sim_threshold,
                        aggregation=self.similarity_pooling,
                        normalization=self.sim_normalization,
                        retain_num=self.retain_num,
                        retain_ratio=self.retain_ratio,
                        retain_direction=self.retain_direction,
                    )[:, :, : -self.window_size]
                final_score = attn_cache * self.mix_alpha - similarity_cos * (
                    1 - self.mix_alpha
                )
            elif self.compress_method == "similarity":
                similarity_cos = cal_similarity(
                    key_states,
                    threshold=self.sim_threshold,
                    aggregation=self.similarity_pooling,
                    normalization=self.sim_normalization,
                    retain_num=self.retain_num,
                    retain_ratio=self.retain_ratio,
                    retain_direction=self.retain_direction,
                )[:, :, : -self.window_size]
                final_score = -similarity_cos
            elif (
                self.compress_method == "snapkv"
                or self.compress_method == "h2o"
                or self.compress_method == "pyramidkv"
            ):
                final_score = attn_cache
            else:
                raise ValueError("Compression method not supported")

            # shape: (bsz, num_kv_heads, max_capacity_prompts - window_size)
            indices = final_score.topk(
                self.max_capacity_prompts - self.window_size, dim=-1
            ).indices

            #####################################################
            ###### Store evicted token indices start ############
            #####################################################
            # shape: (num_kv_heads, max_capacity_prompts - window_size)
            if self.record_kept_token_indices:
                indices_cl = indices.clone().squeeze(0).to("cpu")
                recent_window_indices = (
                    torch.arange(
                        kv_cache_len - self.window_size, kv_cache_len, device="cpu"
                    ).expand(indices_cl.shape[0], -1)
                    + self.evicted_token_num
                )
                cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)
                if self.evicted_token_num > 0:
                    prev_indices = self.kept_token_indices[-1]
                    mask = cur_indices < self.max_capacity_prompts
                    cur_indices[mask] = prev_indices[mask]
                    cur_indices[~mask] += self.evicted_token_num

                self.kept_token_indices.append(cur_indices)
                self.evicted_token_num += kv_cache_len - self.max_capacity_prompts
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


def compute_attention_scores(query_states, key_states, pooling="mean"):
    batch_size, q_heads, seq_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    ratio = q_heads // kv_heads

    if ratio == 1:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
    else:
        # shape: [batch_size, kv_heads, ratio, seq_len, head_dim]
        query_states = query_states.view(batch_size, kv_heads, ratio, seq_len, head_dim)

        # shape: [batch_size, kv_heads, 1, seq_len, head_dim]
        key_states = key_states.unsqueeze(2)

        # average over ratio
        # shape: [batch_size, kv_heads, seq_len, seq_len]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 4)
        ) / math.sqrt(head_dim)

        if pooling == "mean":
            attn_weights = attn_weights.mean(dim=2)
        elif pooling == "max":
            attn_weights = attn_weights.max(dim=2).values
        else:
            raise ValueError("Pooling method not supported")

    return attn_weights


def cal_similarity(
    key_cache,
    threshold=0.5,
    aggregation="mean",
    normalization=False,
    retain_num=None,
    retain_ratio=None,
    retain_direction="last",
):
    _, _, seq_len, _ = key_cache.shape

    k_norm = key_cache / (key_cache.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_cache.device)
    similarity_cos.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)

    # shape: [num_heads, seq_len, seq_len]
    similarity_mask = similarity_cos > threshold

    if retain_ratio is not None and retain_num is not None:
        raise ValueError("retain_ratio and retain_num cannot be used together")
    if retain_ratio is None and retain_num is None:
        raise ValueError("retain_ratio or retain_num must be provided")
    if retain_num is not None:
        k = max(1, retain_num if retain_num is not None else 1)
    else:
        k = max(1, int(seq_len * retain_ratio))

    indices = torch.where(
        similarity_mask,
        torch.arange(seq_len, device=similarity_mask.device).view(1, 1, 1, seq_len),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    if retain_direction == "last":
        # find the last True index in each row
        similarity_retain = torch.max(indices, dim=-1)[0]
    elif retain_direction == "first":
        # find the first True index in each row
        similarity_retain = torch.min(indices, dim=-1)[0]
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]

    # zero the specified positions in similarity_cos
    similarity_cos.scatter_(-1, similarity_retain.unsqueeze(-1), 0)

    if aggregation == "mean":
        similarity_cos = similarity_cos.mean(dim=-2)
    elif aggregation == "max":
        similarity_cos = similarity_cos.max(dim=-2).values

    if normalization:
        similarity_cos = similarity_cos.softmax(dim=-1)

    return similarity_cos
