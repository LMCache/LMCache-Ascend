# SPDX-License-Identifier: Apache-2.0
# Third Party
from torch import nn
import torch

# First Party
from lmcache_ascend.v1.blend.attention.attention import LMCAttnBackend
from lmcache_ascend.v1.blend.attention.attention import LMCFlashAttnMetadata
from lmcache_ascend.v1.blend.models.models import LMCModel
from lmcache_ascend.v1.blend.positional_encoding import get_fused_rope

def qk_post_processing(q, k, attn_layer, positions):
    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // attn_layer.head_dim, attn_layer.head_dim)
    q_by_head = attn_layer.q_norm(q_by_head)
    q = q_by_head.view(q.shape)
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // attn_layer.head_dim, attn_layer.head_dim)
    k_by_head = attn_layer.k_norm(k_by_head)
    k = k_by_head.view(k.shape)
    q, k = attn_layer.rotary_emb(positions, q, k)
    return q, k

class LMCQwen3Model(LMCModel):

    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L353 https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L268
    def compute_layer(
        self,
        input_ids: torch.Tensor,
    ):
        hidden_states = self.vllm_model.get_input_embeddings(input_ids.cuda())
        residual = None

        # TODO (Jiayi): reduce the number of calls
        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, input_ids.shape[0]], dtype=torch.int32, device=hidden_states.device
            ),
            seq_lens=torch.tensor([input_ids.shape[0]], device=hidden_states.device),
            cu_seqlens_k=torch.tensor(
                [0, input_ids.shape[0]], dtype=torch.int32, device=hidden_states.device
            ),
            max_query_len=input_ids.shape[0],
            max_seq_len=input_ids.shape[0],
        )

        no_more_queries = False

        for idx, layer in enumerate(
            self.vllm_model.model.layers[
                self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
            ]
        ):
            if no_more_queries:
                yield
                continue
            # TODO(Jiayi) The last layer doesn't have to be computed
            # hidden_states, residual = layer(positions, hidden_states, residual)

            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            # hidden_states = self.self_attn(positions=positions,
            #                            hidden_states=hidden_states)

            # # According to HF transformers
            # residual = hidden_states
            # hidden_states = layer.input_layernorm(hidden_states)
            # #

            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )

            num_heads = self.vllm_attn_layers[idx].num_heads
            num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
            head_size = self.vllm_attn_layers[idx].head_size

            q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
                q, k, v, residual, idx, attn_output, attn_metadata, qk_post_processing=qk_post_processing
            )
            if q.numel() == 0:
                no_more_queries = True
                yield
                continue

            q = q.view(-1, num_heads, head_size)
            k = k.view(-1, num_kv_heads, head_size)
            v = v.view(-1, num_kv_heads, head_size)
            attn_output = attn_output.view(-1, num_heads, head_size)

            attn_output = self.lmc_attn_layers[idx].forward_contiguous(q, k, v, attn_output, attn_metadata, blend_metadata=self.blender.metadata, layer_id=idx)

            attn_output = attn_output.view(-1, num_heads * head_size)
            k = k.view(-1, num_kv_heads * head_size)
            v = v.view(-1, num_kv_heads * head_size)

            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # # According to hf transformers
            # hidden_states = residual + hidden_states
            # residual = hidden_states
            # hidden_states = layer.post_attention_layernorm(hidden_states)
            # hidden_states = layer.mlp(hidden_states)
            # hidden_states = residual + hidden_states
            # #

            # Fully Connected
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

            yield
