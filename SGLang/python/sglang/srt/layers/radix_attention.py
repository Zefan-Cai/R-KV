# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix attention."""

from torch import nn
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from typing import Tuple

def need_compress(l: int, req_indice: int, forward_batch: ForwardBatch) -> bool:
    if forward_batch.compress_algorithm is None:
        return False
    elif l <= forward_batch.compress_max_prompt:
        return False
    elif forward_batch.compress_algorithm == "RKV":
        if forward_batch.forward_mode.is_extend():
            return True
        elif forward_batch.req_to_token_pool.think_forbid[req_indice]:
            return False
        elif forward_batch.forward_mode.is_decode():
            if forward_batch.compress_divide_method == "newline":
                return forward_batch.req_to_token_pool.newline_compress[req_indice]
            elif forward_batch.compress_divide_method == "step_length":
                if forward_batch.compress_divide_length is None or forward_batch.compress_divide_length == 0:
                    raise ValueError(f"Invalid compress_divide_length: {forward_batch.compress_divide_length}")
                else:
                    return forward_batch.steps % forward_batch.compress_divide_length == 0
        else:
            return False
    else:
        raise ValueError(f"Invalid compress algorithm: {forward_batch.compress_algorithm}")

class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.k_scale = None
        self.v_scale = None

    def _get_metadata(self, idx: int, forward_batch: ForwardBatch) -> Tuple[int, int, int]:
        """
        Get forward_batch's necessary metadata for compression algorithms
        """
        if forward_batch.forward_mode.is_extend():
            req_indice = forward_batch.req_pool_indices[idx].item()
            seq_len = forward_batch.seq_lens[idx]
            prefix_len = forward_batch.extend_prefix_lens[idx].item()
            cur_len = seq_len - prefix_len
            return req_indice, seq_len, cur_len
        elif forward_batch.forward_mode.is_decode():
            req_indice = forward_batch.req_pool_indices[idx].item()
            seq_len = forward_batch.seq_lens[idx]
            prefix_len = 0
            cur_len = seq_len - prefix_len
            return req_indice, seq_len, cur_len

    def _get_kv_seq_len(self, idx: int, forward_batch: ForwardBatch) -> int:
        if hasattr(forward_batch, "kv_seq_lens") and forward_batch.kv_seq_lens is not None:
            return forward_batch.kv_seq_lens[idx].item()
        return forward_batch.seq_lens[idx].item()

    def _reshape_qkv_compress(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape q, k, v tensors to match KV compression cluster's input. 

        Input shape: (seq_len, num_heads, head_dim)

        Output shape: (batch_size=1, num_heads, seq_len, head_dim)
        """
        q_state = q.view(-1, self.tp_q_head_num, self.head_dim).transpose(0, 1).unsqueeze(0)[..., -forward_batch.compress_max_window : , :]
        k_state = k.transpose(0, 1).unsqueeze(0)
        v_state = v.transpose(0, 1).unsqueeze(0)
        return q_state, k_state, v_state
    
    def _reverse_kv_compress(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse k, v shapes to match Sglang's memory pool layout.

        Input shape: (batch_size=1, num_heads, seq_len, head_dim)

        Output shape: (seq_len, num_heads, head_dim)
        """
        k = k.transpose(1, 2).squeeze(0)
        v = v.transpose(1, 2).squeeze(0)
        return k, v

    def _apply_compress(self, forward_batch: ForwardBatch) -> None:
        """
        Apply compressed metadata to memory pool and batch objects after all layers finished forwarding
        """
        for i, req_indice in enumerate(forward_batch.req_pool_indices):
            req_indice, seq_len, _ = self._get_metadata(i, forward_batch)
            if not need_compress(seq_len, req_indice, forward_batch):
                continue
            if req_indice not in forward_batch.compress_out_locs:
                continue
            # extract old info
            old_kv_len = self._get_kv_seq_len(i, forward_batch)
            old_kv_loc = forward_batch.req_to_token_pool.req_to_token[req_indice][ : old_kv_len]
            # extract new info
            new_kv_loc = forward_batch.compress_out_locs[req_indice]
            new_seq_len = len(new_kv_loc)
            # delete old memory
            forward_batch.token_to_kv_pool.free(old_kv_loc)
            # write new memory location to the req_to_token_pool and ChunkCache
            forward_batch.tree_cache.compressed_req_len[req_indice] = new_seq_len
            forward_batch.req_to_token_pool.write(
                (req_indice, slice(0, new_seq_len)), 
                new_kv_loc,
            )

    def _get_new_kv_loc(self, forward_batch: ForwardBatch, req_indice: int, compressed_size: int) -> torch.Tensor:
        if self.layer_id == 0:
            # for layer 0, allocate KV space for current request
            new_kv_loc = forward_batch.token_to_kv_pool.alloc(compressed_size)
            if new_kv_loc is None:
                raise RuntimeError(
                    "Out of memory. "
                    "Please set a smaller number for `--max-running-requests`."
                )
            forward_batch.compress_out_locs[req_indice] = new_kv_loc
        else:
            # for the rest of layer, just get the newly created KV indices
            new_kv_loc = forward_batch.compress_out_locs[req_indice]
        return new_kv_loc

    def _cache_q(self, q: torch.Tensor, forward_batch: ForwardBatch) -> None:
        """
        cache q tensor for compression use
        """
        if forward_batch.forward_mode.is_extend():
            # For extend/prefill, Q contains all seq_len, need to seperate it one by one
            layer_id = self.layer_id
            start = end = 0
            for i in range(len(forward_batch.req_pool_indices)):
                req_indice, _, cur_len = self._get_metadata(i, forward_batch)
                end = start + cur_len - 1
                # we only need to retain the last 32 q tensor for current request
                forward_batch.req_to_token_pool.q_cache[layer_id][req_indice] = q[start : end + 1][-forward_batch.req_to_token_pool.max_window : , ...]
                start = end + 1
            assert sum(forward_batch.seq_lens) - sum(forward_batch.extend_prefix_lens) == q.shape[0]
        elif forward_batch.forward_mode.is_decode():
            layer_id = self.layer_id
            # For decode, every indice has only one q tensor
            for i in range(len(forward_batch.req_pool_indices)):
                req_indice = forward_batch.req_pool_indices[i].item()
                old_tensor = forward_batch.req_to_token_pool.q_cache[layer_id][req_indice]
                forward_batch.req_to_token_pool.q_cache[layer_id][req_indice] = torch.cat((old_tensor, q[i].unsqueeze(0)), dim=0)[-forward_batch.req_to_token_pool.max_window : , ...]
                del old_tensor
            assert q.shape[0] == len(forward_batch.req_pool_indices)

    def compress(self, k: torch.Tensor, v: torch.Tensor, forward_batch: ForwardBatch) -> None:
        """
        Compress algorithm entry
        """
        if forward_batch.forward_mode.is_extend():
            # Record start and end position for current request in a batch
            start = end = 0
            for i in range(len(forward_batch.req_pool_indices)):
                # compress for every request separately
                req_indice, seq_len, _ = self._get_metadata(i, forward_batch)
                # Update end positions
                end = start + seq_len - 1
                if need_compress(seq_len, req_indice, forward_batch):
                    q_state = forward_batch.req_to_token_pool.q_cache[self.layer_id][req_indice]
                    q_state, k_state, v_state = self._reshape_qkv_compress(q_state, k[start : end + 1, ...], v[start : end + 1, ...], forward_batch)
                    key_states_compress, value_states_compress = forward_batch.compress_cluster.update_kv(k_state, q_state, v_state, None)
                    key_states_compress, value_states_compress = self._reverse_kv_compress(key_states_compress, value_states_compress)
                    compressed_size = key_states_compress.shape[0]
                    new_kv_loc = self._get_new_kv_loc(forward_batch, req_indice, compressed_size)
                    # For all layers, set compressed kv buffer
                    forward_batch.token_to_kv_pool.set_kv_buffer(self, new_kv_loc, key_states_compress, value_states_compress)
                # Update start positions
                start = end + 1
        elif forward_batch.forward_mode.is_decode():
            for i in range(len(forward_batch.req_pool_indices)):
                # compress for every request separately
                req_indice, seq_len, _ = self._get_metadata(i, forward_batch)
                if need_compress(seq_len, req_indice, forward_batch):
                    q_state = forward_batch.req_to_token_pool.q_cache[self.layer_id][req_indice]
                    kv_seq_len = self._get_kv_seq_len(i, forward_batch)
                    kv_locs = forward_batch.req_to_token_pool.req_to_token[req_indice][ : kv_seq_len]
                    k_state = forward_batch.token_to_kv_pool.k_buffer[self.layer_id][kv_locs]
                    v_state = forward_batch.token_to_kv_pool.v_buffer[self.layer_id][kv_locs]
                    q_state, k_state, v_state = self._reshape_qkv_compress(q_state, k_state, v_state, forward_batch)
                    key_states_compress, value_states_compress = forward_batch.compress_cluster.update_kv(k_state, q_state, v_state, None)
                    key_states_compress, value_states_compress = self._reverse_kv_compress(key_states_compress, value_states_compress)
                    compressed_size = key_states_compress.shape[0]
                    new_kv_loc = self._get_new_kv_loc(forward_batch, req_indice, compressed_size)
                    # For all layers, set compressed kv buffer
                    forward_batch.token_to_kv_pool.set_kv_buffer(self, new_kv_loc, key_states_compress, value_states_compress)
    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        # TODO: for this project, I set chunked_prefill_size as 98304, and assume all prompt token length will be smaller than this value.
        # In the future, we will support any length of prompts!
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        if forward_batch.compress_algorithm is not None:
            # cache q tensor
            self._cache_q(q, forward_batch)
            # call compression algorithm for extend/prefill
            if forward_batch.forward_mode.is_extend():
                self.compress(k, v, forward_batch)
                assert sum(forward_batch.seq_lens) - sum(forward_batch.extend_prefix_lens) == q.shape[0]
            elif forward_batch.forward_mode.is_decode():
                # save the latest k and v in advance
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not self.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_kv_buffer(self, cache_loc, k, v)
                self.compress(k, v, forward_batch)
                assert q.shape[0] == len(forward_batch.req_pool_indices)
                    
        # Call backend to make self attention
        o = forward_batch.attn_backend.forward(
            q, k, v, self, forward_batch, save_kv_cache
        )
        # If current forward batch meets compression condition, update memory locations and free old memory after finishing the last forward
        if forward_batch.compress_algorithm is not None and self.layer_id == forward_batch.token_to_kv_pool.layer_num - 1:
            self._apply_compress(forward_batch)
        # Return output hidden layer
        return o
