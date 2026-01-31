# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendMetadata
from torch import nn

# from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
from vllm.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
import torch
import torch_npu
import transformers.integrations.npu_flash_attention as nfa

if not hasattr(nfa, "npu_fusion_attention"):
    nfa.npu_fusion_attention = torch_npu.npu_fusion_attention
# Third Party
from torch_npu import npu_fused_infer_attention_score
from transformers.integrations.npu_flash_attention import (
    npu_flash_attn_varlen_func as flash_attn_varlen_func,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(
        slen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim)


# ref from transformers.models.llama.modeling_llama import eager_attention_forward
def eager_attention_causal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    q_positions: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    num_key_value_groups = query.shape[-2] // key.shape[-2]
    query_states = query.transpose(0, 1)  # (heads, slen, dim)
    key_states = repeat_kv(key, num_key_value_groups).transpose(
        0, 1
    )  # (heads, slen, dim)
    value_states = repeat_kv(value, num_key_value_groups).transpose(0, 1)

    attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) * scaling
    if q_positions is None:
        q_positions = torch.arange(query.shape[0]).to(attn_weights.device)
    if q_positions is not None:
        causal_mask = q_positions.unsqueeze(1) < torch.arange(
            key.shape[0], device=q_positions.device, dtype=q_positions.dtype
        ).unsqueeze(0)
        causal_mask = (
            causal_mask.to(dtype=attn_weights.dtype)
            * torch.finfo(attn_weights.dtype).min
        )
        attn_weights = attn_weights + causal_mask

    # dim: (heads, q, k) cf causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )

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
            causal=False,  # https://github.com/LMCache/LMCache/issues/1152
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

    def init_attn_metadata(
        self,
        input_ids: torch.tensor,
        **kwargs,
    ) -> LMCFlashAttnMetadata:
        seq_len = input_ids.shape[0]
        device = input_ids.device
        return LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor([seq_len], device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_query_len=seq_len,
            max_seq_len=seq_len,
        )


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
        blend_metadata: "LMCBlendMetadata" = kwargs["blend_metadata"]
        eager_attention_causal(
            query,
            key,
            value,
            out=output,
            q_positions=blend_metadata.imp_indices,
            scaling=self.vllm_attn_impl.scale,
        )

        return output

    def init_attn_metadata(
        self,
        input_ids: torch.tensor,
        **kwargs,
    ) -> LMCFlashAttnMetadata:
        pass


class ZLMCFlashAttnBackend(AttentionInterface):
    """
    FlashAttention backend for LMCache on Ascend NPU.
    Wrapper for torch_npu.npu_fused_infer_attention_score.
    """

    def __init__(
        self,
        vllm_attn: "Attention",
    ):
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl = vllm_attn.impl

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCFlashAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        """
        Executes NPU Fused Attention.
        Handles both [Batch, Seq, Heads, Dim] and [TotalSeq, Heads, Dim] inputs.
        """
        # 1. Extract LMCache specific metadata
        blend_metadata = kwargs.get("blend_metadata")

        # 2. Handle Input Shapes and Contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # 3. Robust Dimension Extraction
        if query.dim() == 3:
            # Input is (TotalSeqLen, NumHeads, HeadDim)
            # We treat this as Batch=1, Seq=TotalSeqLen
            q_seq_len = query.shape[0]
            num_heads = query.shape[1]
            # head_dim = query.shape[2]

            kv_seq_len = key.shape[0]
            num_kv_heads = key.shape[1]

            # Unsqueeze to create pseudo-batch dimension for NPU (BSND requirements)
            query_input = query.unsqueeze(0)
            key_input = key.unsqueeze(0)
            value_input = value.unsqueeze(0)
        else:
            raise NotImplementedError

        # 4. Determine Positions
        q_positions = None

        if blend_metadata is not None and hasattr(blend_metadata, "imp_indices"):
            q_positions = blend_metadata.imp_indices
        elif "q_positions" in kwargs:
            q_positions = kwargs["q_positions"]

        if q_positions is None:
            # Note: For flattened 3D input, q_seq_len is the total token count
            q_positions = torch.arange(q_seq_len, device=query.device, dtype=torch.long)

        k_positions = torch.arange(kv_seq_len, device=query.device, dtype=torch.long)

        # 5. Construct Custom Boolean Mask for NPU
        if q_positions.dim() == 1:
            mask_condition = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)
            # Reshape to (1, 1, Q, K) to broadcast over Batch and Heads
            atten_mask = mask_condition.unsqueeze(0).unsqueeze(0)
        else:
            # If q_positions is [Batch, Seq]
            raise NotImplementedError

        atten_mask = atten_mask.to(torch.bool)

        result_tuple = npu_fused_infer_attention_score(
            query=query_input,  # Use the prepared input
            key=key_input,
            value=value_input,
            atten_mask=atten_mask,
            actual_seq_lengths=None,
            actual_seq_lengths_kv=None,
            num_heads=num_heads,  # Now correctly extracted
            num_key_value_heads=num_kv_heads,  # Now correctly extracted
            scale=self.vllm_attn_impl.scale,
            input_layout="BSND",
            sparse_mode=0,
            softmax_lse_flag=False,
        )

        attention_out = result_tuple[0]

        # 7. Copy to output
        if output is not None:
            # Ensure shape matches output
            if output.shape != attention_out.shape:
                # Flatten back if input was 3D but we processed as 4D
                attention_out = attention_out.reshape(output.shape)
            output.copy_(attention_out)
            return output

        return attention_out

    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> "LMCFlashAttnMetadata":
        """
        Initialize attention metadata.
        """
        raise NotImplementedError
