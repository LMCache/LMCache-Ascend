# SPDX-License-Identifier: Apache-2.0
"""Hole-specific model adapters and attention synchronization."""

# Third Party
import torch

# First Party
from lmcache_ascend.v1.blend.attention.attention import ZLMCFlashAttnBackend
from lmcache_ascend.v1.blend.models.llama import LMCLlamaModel
from lmcache_ascend.v1.blend.models.qwen3 import LMCQwen3Model


class HoleSynchronizedAttentionBackend(ZLMCFlashAttnBackend):
    """Record when attention is done reading a hole connector buffer."""

    def __init__(self, vllm_attn, gpu_connector, layer_id: int):
        super().__init__(vllm_attn)
        self.gpu_connector = gpu_connector
        self.layer_id = layer_id

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata,
        **kwargs,
    ) -> torch.Tensor:
        result = super().forward_contiguous(
            query,
            key,
            value,
            output,
            attn_metadata,
            **kwargs,
        )
        self.gpu_connector.record_scatter_done(self.layer_id)
        return result


class HoleModelMixin:
    """Install hole-aware attention backends without changing shared models."""

    def __init__(self, vllm_model, blender):
        super().__init__(vllm_model, blender)
        self.lmc_attn_layers = [
            HoleSynchronizedAttentionBackend(attn, blender.gpu_connector, layer_id)
            for layer_id, attn in enumerate(self.vllm_attn_layers)
        ]


class LMCQwen3HoleModel(HoleModelMixin, LMCQwen3Model):
    pass


class LMCLlamaHoleModel(HoleModelMixin, LMCLlamaModel):
    pass


def infer_hole_model_from_vllm(vllm_model, blender):
    model_name = type(vllm_model).__name__
    if model_name == "LlamaForCausalLM":
        return LMCLlamaHoleModel(vllm_model, blender)
    if "Qwen3ForCausalLM" in model_name:
        return LMCQwen3HoleModel(vllm_model, blender)
    raise NotImplementedError(
        f"Model type {model_name} is not supported in hole-mode LMCache."
    )
