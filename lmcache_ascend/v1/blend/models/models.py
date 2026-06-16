# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
from torch import nn

# First Party
from lmcache_ascend.v1.blend.attention.attention import ZLMCFlashAttnBackend
from lmcache_ascend.v1.blend.positional_encoding import (
    get_fused_rope,
    get_fused_rope_from_rotary_emb,
)

LIVE_ROPE = True


class LMCModel(nn.Module):
    def __init__(
        self,
        vllm_model,
        blender,
    ):
        super().__init__()
        self.vllm_model = vllm_model

        self.num_layers = len(vllm_model.model.layers)

        self.vllm_attn_layers = []
        self.lmc_attn_layers: list[ZLMCFlashAttnBackend] = []
        for i in range(self.num_layers):
            vllm_attn = vllm_model.model.layers[i].self_attn.attn
            self.vllm_attn_layers.append(vllm_attn)
            self.lmc_attn_layers.append(ZLMCFlashAttnBackend(vllm_attn))

        # NOTE(Jiayi): better not to pass the blender in init
        # if we want to make this LMCModel more general.
        self.blender = blender

        rotary_emb = vllm_model.model.layers[0].self_attn.rotary_emb
        if LIVE_ROPE:
            self.fused_rotary_emb = get_fused_rope_from_rotary_emb(
                rotary_emb,
            )
        else:
            # Legacy path kept for side-by-side analysis with the live-rope variant.
            head_dim = rotary_emb.head_size
            max_position_embeddings = rotary_emb.max_position_embeddings
            rope_scaling = None
            base = rotary_emb.base
            is_neox_style = rotary_emb.is_neox_style
            dtype = rotary_emb.dtype
            self.fused_rotary_emb = get_fused_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=max_position_embeddings,
                base=base,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                dtype=dtype,
            )

    def embed_input_ids(self, input_ids):
        if hasattr(self.vllm_model, "embed_input_ids"):
            return self.vllm_model.embed_input_ids(input_ids)

        get_input_embeddings = getattr(self.vllm_model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embedding_layer = get_input_embeddings()
            if callable(embedding_layer):
                return embedding_layer(input_ids)

        model = getattr(self.vllm_model, "model", None)
        if model is not None and hasattr(model, "embed_input_ids"):
            return model.embed_input_ids(input_ids)
        if model is not None and hasattr(model, "embed_tokens"):
            return model.embed_tokens(input_ids)

        raise AttributeError(
            f"{type(self.vllm_model).__name__} does not expose a supported "
            "input embedding API for LMCache-Ascend blending."
        )
