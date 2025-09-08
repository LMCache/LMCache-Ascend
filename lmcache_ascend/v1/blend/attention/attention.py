# SPDX-License-Identifier: Apache-2.0
# Third Party
from vllm.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
# from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
import torch
from torch import nn
from typing import Optional
import torch.distributed as dist
import pathlib

# First Party
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendMetadata

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim)

# ref from transformers.models.llama.modeling_llama import eager_attention_forward
def eager_attention_causal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    q_positions: Optional[torch.Tensor],
    scaling: float,
    **kwargs
):
    num_key_value_groups = query.shape[-2] // key.shape[-2]
    query_states = query.transpose(0, 1) # (heads, slen, dim)
    key_states = repeat_kv(key, num_key_value_groups).transpose(0, 1) # (heads, slen, dim)
    value_states = repeat_kv(value, num_key_value_groups).transpose(0, 1)

    attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) * scaling
    if q_positions is None:
        q_positions = torch.arange(query.shape[0]).to(attn_weights.device)
    if q_positions is not None:
        causal_mask = (q_positions.unsqueeze(1) < torch.arange(key.shape[0], device=q_positions.device, dtype=q_positions.dtype).unsqueeze(0))
        causal_mask = causal_mask.to(dtype=attn_weights.dtype) * torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights + causal_mask

    # dim: (heads, q, k) cf causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    del attn_weights, value_states, query_states, key_states
    if q_positions is not None:
        del causal_mask
    torch.npu.empty_cache()
    attn_output = attn_output.transpose(0, 1).contiguous()

    out.copy_(attn_output)
    del attn_output
    torch.npu.empty_cache()

class LMCFlashAttnBackend(AttentionInterface):
    """
    FlashAttention backend for LMCache.
    This backend uses the FlashAttention implementation
    for efficient attention computation.
    """

    def __init__(
        self,
        vllm_attn: Attention,
    ):
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl: FlashAttentionImpl = vllm_attn.impl

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: LMCFlashAttnMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # num_actual_tokens = query.shape[0]

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        output = flash_attn_varlen_func(
            q=query.contiguous(),  # contiguous
            k=key.contiguous(),  # contiguous
            v=value.contiguous(),  # contiguous
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            # seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.vllm_attn_impl.scale,
            causal=False, # https://github.com/LMCache/LMCache/issues/1152
            alibi_slopes=self.vllm_attn_impl.alibi_slopes,
            window_size=self.vllm_attn_impl.sliding_window,
            block_table=None,
            # softcap=self.vllm_attn_impl.logits_soft_cap,
            # scheduler_metadata=scheduler_metadata,
            # fa_version=self.vllm_attn_impl.vllm_flash_attn_version,
            q_descale=self.vllm_attn._q_scale.expand(descale_shape),
            k_descale=self.vllm_attn._k_scale.expand(descale_shape),
            v_descale=self.vllm_attn._v_scale.expand(descale_shape),
        )

        return output

class LMCAttnBackend(AttentionInterface):
    """
    FlashAttention backend for LMCache.
    This backend uses the FlashAttention implementation
    for efficient attention computation.
    """

    def __init__(
        self,
        vllm_attn: Attention,
    ):
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl: FlashAttentionImpl = vllm_attn.impl

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: LMCFlashAttnMetadata,
        **kwargs,
    ) -> torch.Tensor:
        blend_metadata: 'LMCBlendMetadata' = kwargs['blend_metadata']
        eager_attention_causal(
            query,
            key,
            value,
            out=output,
            q_positions = blend_metadata.imp_indices,
            scaling = self.vllm_attn_impl.scale,
        )

        return output