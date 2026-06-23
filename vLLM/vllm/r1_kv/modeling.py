import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_attention_scores(query_states, key_states, pooling="max"):
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    if query_group_size == 1:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
    else:
        query_states = query_states.view(
            batch_size, kv_heads, query_group_size, q_len, head_dim
        )
        key_states = key_states.unsqueeze(2)
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
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    _, _, seq_len, _ = key_states.shape

    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_states.device)
    similarity_cos.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)

    similarity_mask = similarity_cos > threshold
    k = max(1, int(seq_len * retain_ratio))
    indices = torch.where(
        similarity_mask,
        torch.arange(seq_len, device=similarity_mask.device).view(1, 1, 1, seq_len),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]
    else:
        raise ValueError("retain_direction not supported")

    similarity_cos.scatter_(-1, similarity_retain.unsqueeze(-1), 0)
    return similarity_cos.mean(dim=-2).softmax(dim=-1)


class R1KV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction

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
        indices = final_score.topk(self.budget - self.window_size, dim=-1).indices
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
