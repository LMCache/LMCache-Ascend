# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import time
from typing import Optional, Union

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.trace_utils import (
    emit_layer_timer,
    mask_to_string,
    tensor_to_list,
    trace_flow,
)
import torch

# First Party
from lmcache_ascend.v1.blend.models.utils import infer_model_from_vllm

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        self.layerwise_model = infer_model_from_vllm(vllm_model, self)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # TODO (Jiayi): make this less hard-coded
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )
        trace_value = str(os.environ.get("LMCACHE_TRACE_BLEND", "0")).strip().lower()
        self.trace_blend = trace_value not in {"", "0", "false", "no", "off"}
        self._trace_tokens_cpu: Optional[torch.Tensor] = None
        self._trace_req_id: Optional[str] = None
        self._active_timer_req_id: Optional[str] = None
        self._active_timer_path: str = "nohole"
        self._active_timer_load_mode: Optional[str] = None
        self._layer_topk_ms: dict[int, float] = {}

    def _emit_timer(self, bucket: str, layer_id: int, duration_ms: float) -> None:
        emit_layer_timer(
            bucket,
            req_id=self._active_timer_req_id,
            layer_id=layer_id,
            duration_ms=duration_ms,
            path=self._active_timer_path,
            load_mode=self._active_timer_load_mode,
        )

    def get_last_topk_ms(self, layer_id: int) -> float:
        return float(self._layer_topk_ms.get(layer_id, 0.0))

    def emit_blend_timer(self, layer_id: int, duration_ms: float) -> None:
        blend_ms = max(float(duration_ms) - self.get_last_topk_ms(layer_id), 0.0)
        self._emit_timer("blend", layer_id, blend_ms)

    def _log_recomputed_tokens(
        self,
        layer_id: int,
        absolute_positions: torch.Tensor,
        eligible_count: int,
    ) -> None:
        if not self.trace_blend:
            return
        if self._trace_tokens_cpu is None:
            return
        positions_cpu = absolute_positions.detach().to(device="cpu", dtype=torch.long)
        token_ids = [
            int(self._trace_tokens_cpu[pos].item())
            for pos in positions_cpu.tolist()
            if 0 <= int(pos) < int(self._trace_tokens_cpu.shape[0])
        ]
        logger.info(
            "Blend recompute layer=%d req_id=%s eligible_tokens=%d "
            "recomputed_positions=%s recomputed_token_ids=%s",
            layer_id,
            self._trace_req_id if self._trace_req_id is not None else "unknown",
            eligible_count,
            positions_cpu.tolist(),
            token_ids,
        )
        trace_flow(
            "blender.nohole",
            "recompute_positions",
            layer_id=layer_id,
            req_id=self._trace_req_id,
            eligible_tokens=eligible_count,
            recomputed_positions=positions_cpu.tolist(),
            recomputed_token_ids=token_ids,
        )

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        old_k, old_v = self.gpu_connector.get_kv(layer_id)
        self._layer_topk_ms[layer_id] = 0.0

        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # perform positional encoding
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        if "qk_post_processing" in kwargs:
            q, k = kwargs["qk_post_processing"](
                q, k, attn_layer, self.metadata.positions
            )
        else:
            q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        if (
            layer_id in self.common_metadata.check_layers
            and self.common_metadata.recomp_ratios[0] > 0
        ):
            topk_start = time.perf_counter()
            assert k[num_falses:].shape[0] == old_k.shape[0], (
                "Mismatch between number of tokens in k "
                "(after skipping falses) and old_k"
            )

            diff_k = torch.sum(
                (k[num_falses:].to(torch.float32) - old_k.to(torch.float32)) ** 2,
                dim=[1],
            )

            total_len = diff_k.shape[0]

            # TODO(Jiayi): remove `[0]` hardcode
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
            topk_num = max(topk_num, 1)

            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)
            absolute_top_indices = top_indices + num_falses
            topk_ms = (time.perf_counter() - topk_start) * 1000.0
            self._layer_topk_ms[layer_id] = topk_ms
            self._emit_timer("topk_l1", layer_id, topk_ms)
            self._log_recomputed_tokens(layer_id, absolute_top_indices, total_len)
            trace_flow(
                "blender.nohole",
                "process_qkv",
                layer_id=layer_id,
                req_id=kwargs.get("req_id", self._trace_req_id),
                num_falses=num_falses,
                eligible_tokens=total_len,
                topk_num=topk_num,
                top_indices=top_indices.tolist(),
                absolute_top_indices=absolute_top_indices.tolist(),
            )

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")
            logger.debug(f"Picking indices: {top_indices}")
            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.update_from_top_indices(top_indices)
            attn_metadata.max_query_len = topk_num
            attn_metadata.query_start_loc = torch.tensor(
                [0, topk_num], dtype=torch.int32, device=q.device
            )

        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        layerwise_model_executor = self.layerwise_model.compute_layer(
            tokens,
            mask,
            req_id=kwargs.get("req_id"),
        )
        layerwise_retriever = self.cache_engine.retrieve_layer(
            tokens,
            mask,
            gpu_connector_override=self.gpu_connector,
            **kwargs,
        )

        next(layerwise_retriever)
        yield

        for i in range(self.num_layers):
            wait_start = time.perf_counter()
            next(layerwise_retriever)
            self._emit_timer("wait_reuse", i, (time.perf_counter() - wait_start) * 1000.0)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)

        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).npu()
        trace_flow(
            "blender.nohole",
            "blend_start",
            req_id=kwargs.get("req_id"),
            token_count=len(tokens),
            mask=mask_to_string(mask),
            token_ids=tensor_to_list(tokens, dtype=torch.long),
        )
        if self.trace_blend:
            self._trace_tokens_cpu = tokens.detach().to(device="cpu", dtype=torch.long)
            req_id = kwargs.get("req_id")
            self._trace_req_id = None if req_id is None else str(req_id)
        req_id = kwargs.get("req_id")
        self._active_timer_req_id = None if req_id is None else str(req_id)
        self._active_timer_path = str(kwargs.get("timer_path", "nohole"))
        timer_load_mode = kwargs.get("timer_load_mode")
        self._active_timer_load_mode = (
            None if timer_load_mode is None else str(timer_load_mode)
        )
        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        try:
            for i in range(self.num_layers + 2):
                next(layerwise_blender)
        finally:
            self._trace_tokens_cpu = None
            self._trace_req_id = None
            self._active_timer_req_id = None
            self._active_timer_path = "nohole"
            self._active_timer_load_mode = None
            self._layer_topk_ms.clear()
