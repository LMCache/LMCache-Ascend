# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Optional

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import LayoutHints, need_gpu_interm_buffer
from lmcache.v1.metadata import LMCacheMetadata
import torch

# First Party
from lmcache_ascend import _build_info

if _build_info.__framework_name__ == "pytorch":
    # First Party
    from lmcache_ascend.v1.npu_connector.npu_connectors import (
        VLLMBufferLayerwiseNPUConnector,
        VLLMPagedMemLayerwiseNPUConnector,
        VLLMPagedMemNPUConnectorV2,
    )
elif _build_info.__framework_name__ == "mindspore":
    # First Party
    from lmcache_ascend.mindspore.v1.npu_connector import (
        VLLMBufferLayerwiseNPUConnector,
        VLLMPagedMemLayerwiseNPUConnector,
        VLLMPagedMemNPUConnectorV2,
    )

logger = init_logger(__name__)


def _attach_lmcache_slot_mapping_staging(
    connector: GPUConnectorInterface,
    vllm_config: Optional[Any],
    device: torch.device,
) -> None:
    """Pre-allocate NPU + pinned-CPU buffers for ``slot_mapping`` staging.

    ``slot_mapping`` is indexed per **request** and can be as long as
    ``model_config.max_model_len``, which may exceed
    ``scheduler_config.max_num_batched_tokens``. Capacity is the max of
    those bounds when both are present.

    For a 256k tokens with long, we allocate about 256k*8 bytes,
        i.e. about 2MB. 1M tokens with int64 will take about 8MB.
    """
    if vllm_config is None or device.type != "npu":
        return
    bounds: list[int] = []
    model_cfg = getattr(vllm_config, "model_config", None)
    if model_cfg is not None:
        mml = getattr(model_cfg, "max_model_len", None)
        if mml is not None and mml > 0:
            bounds.append(int(mml))
    sched = getattr(vllm_config, "scheduler_config", None)
    if sched is not None:
        mnb = getattr(sched, "max_num_batched_tokens", None)
        if mnb is not None and mnb > 0:
            bounds.append(int(mnb))
    if not bounds:
        return
    max_tokens = max(bounds)
    connector._lmcache_max_slot_mapping_tokens = int(max_tokens)
    connector._lmcache_slot_mapping_npu_buf = torch.empty(
        int(max_tokens), dtype=torch.long, device=device
    )
    connector._lmcache_slot_mapping_cpu_pinned = torch.empty(
        int(max_tokens), dtype=torch.long, device="cpu", pin_memory=True
    )


def CreateNPUConnector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    engine: EngineType,
    layout_hints: Optional[LayoutHints] = None,
    vllm_config: Optional[Any] = None,
) -> GPUConnectorInterface:
    """Factory function to create NPU connectors on Ascend.

    Replaces upstream CreateGPUConnector to return Ascend NPU-specific
    connector implementations.
    """
    use_gpu = need_gpu_interm_buffer(config)

    num_gpus = torch.npu.device_count()
    local_rank = metadata.worker_id % num_gpus
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    if engine == EngineType.VLLM:
        if metadata.use_mla and config.use_layerwise and config.enable_blending:
            raise ValueError(
                "We haven't supported MLA with Cacheblend yet. Please disable blending."
            )

        if config.use_layerwise:
            if config.enable_blending:
                conn = VLLMBufferLayerwiseNPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            else:
                conn = VLLMPagedMemLayerwiseNPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            _attach_lmcache_slot_mapping_staging(conn, vllm_config, device)
            return conn

        if config.use_gpu_connector_v3:
            raise NotImplementedError(
                "GPU Connector v3 is not supported yet. Please contact LMCache-Ascend."
            )
        else:
            conn = VLLMPagedMemNPUConnectorV2.from_metadata(
                metadata, use_gpu, device, layout_hints=layout_hints
            )
            _attach_lmcache_slot_mapping_staging(conn, vllm_config, device)
            return conn
    elif engine == EngineType.SGLANG:
        # First Party
        from lmcache_ascend.v1.npu_connector.npu_connectors import (
            SGLangLayerwiseNPUConnector,
            SGLangNPUConnector,
        )

        num_layer, _, chunk_size, num_kv_head, head_dim = metadata.kv_shape
        hidden_dim_size = num_kv_head * head_dim
        kv_dtype = metadata.kv_dtype

        if config.use_layerwise:
            conn = SGLangLayerwiseNPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        else:
            conn = SGLangNPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        _attach_lmcache_slot_mapping_staging(conn, vllm_config, device)
        return conn
    else:
        raise RuntimeError(f"Unsupported engine type for Ascend: {engine}")
