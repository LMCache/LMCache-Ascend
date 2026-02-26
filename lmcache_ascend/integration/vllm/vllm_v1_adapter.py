# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from typing import Optional

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import ENGINE_NAME, mla_enabled
from lmcache.integration.vllm.vllm_v1_adapter import (
    _calculate_draft_layers,
    need_gpu_interm_buffer,
)
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import GPUConnectorInterface
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
try:
    # Third Party
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
except ImportError:
    # Third Party
    from vllm.utils import get_kv_cache_torch_dtype

import torch

# First Party
from lmcache_ascend import _build_info

if _build_info.__framework_name__ == "pytorch":
    # First Party
    from lmcache_ascend.v1.npu_connector import (
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


# We need to patch this function due to connector modification
def init_lmcache_engine(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
    role: str,
) -> LMCacheEngine:
    """Initialize the LMCache engine by the given model config and parallel
    config. This function will check the environment variable
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param lmcache_config: The LMCache configuration.
    :type lmcache_config: LMCacheEngineConfig
    :param vllm_config: The vLLM configuration.
    :type vllm_config: VllmConfig

    :return: The initialized LMCache engine
    :rtype: LMCacheEngine
    """

    curr_engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if curr_engine:
        return curr_engine

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    assert isinstance(lmcache_config, LMCacheEngineConfig), (
        "LMCache v1 configuration is should be passed."
    )

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

    use_mla = mla_enabled(model_config)
    if use_mla and (
        lmcache_config.remote_serde != "naive"
        and lmcache_config.remote_serde is not None
    ):
        raise ValueError("MLA only works with naive serde mode..")

    # MLA requires save_unfull_chunk=True for correct KV cache storage and retrieval.
    # Without this, partial chunks would be discarded, causing incomplete cache
    # and incorrect results in MLA mode.
    if use_mla and not lmcache_config.save_unfull_chunk:
        logger.warning(
            "MLA (Multi-Level Attention) requires save_unfull_chunk=True "
            "for correct KV cache storage. Automatically setting "
            "save_unfull_chunk=True."
        )
        lmcache_config.save_unfull_chunk = True
    elif use_mla:
        logger.info(
            "MLA mode enabled with save_unfull_chunk=True - all KV cache "
            "including partial chunks will be stored"
        )

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    num_draft_layers = _calculate_draft_layers(vllm_config, model_config)
    num_layer += num_draft_layers
    chunk_size = lmcache_config.chunk_size
    # this is per gpu
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
    logger.info(
        f"num_layer: {num_layer}, chunk_size: {chunk_size}, "
        f"num_kv_head (per gpu): {num_kv_head}, head_size: {head_size}, "
        f"hidden_dim (D) for KV (per gpu): {num_kv_head * head_size}, "
        f"use mla: {use_mla}, kv shape: {kv_shape}, num_draft_layers:{num_draft_layers}"
    )

    # Change current device.
    num_gpus = torch.npu.device_count()
    local_rank = parallel_config.rank % num_gpus
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    metadata = LMCacheEngineMetadata(
        model_config.model,
        parallel_config.world_size,
        parallel_config.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
        role,
        served_model_name=model_config.served_model_name,
        chunk_size=lmcache_config.chunk_size,
    )

    use_gpu = need_gpu_interm_buffer(lmcache_config)
    vllm_gpu_connector: Optional[GPUConnectorInterface]

    if use_mla and lmcache_config.use_layerwise and lmcache_config.enable_blending:
        raise ValueError(
            "We haven't supported MLA with Cacheblend yet. Please disable blending."
        )

    if role == "scheduler":
        vllm_gpu_connector = None
        # Create a dummy tpg object with broadcast and broadcast_object methods
        tpg = SimpleNamespace()
        tpg.broadcast = lambda tensor, src: tensor
        tpg.broadcast_object = lambda obj, src: obj
    elif lmcache_config.use_layerwise:
        if lmcache_config.enable_blending:
            # Use layerwise connector for blending
            vllm_gpu_connector = VLLMBufferLayerwiseNPUConnector.from_metadata(
                metadata, use_gpu, device
            )
        else:
            vllm_gpu_connector = VLLMPagedMemLayerwiseNPUConnector.from_metadata(
                metadata, use_gpu, device
            )
        tpg = get_tp_group()
    else:
        # TODO (gingfung): gpu_connector_v3
        if lmcache_config.use_gpu_connector_v3:
            raise NotImplementedError(
                "GPU Connector v3 is not supported yet. Please contact LMCache-Ascend."
            )
        else:
            vllm_gpu_connector = VLLMPagedMemNPUConnectorV2.from_metadata(
                metadata, use_gpu, device
            )
        tpg = get_tp_group()

    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        lmcache_config,
        metadata,
        vllm_gpu_connector,
        tpg.broadcast,
        tpg.broadcast_object,
    )

    if role == "scheduler" and lmcache_config.enable_scheduler_bypass_lookup:
        assert engine.save_only_first_rank or lmcache_config.get_extra_config_value(
            "remote_enable_mla_worker_id_as0", metadata.use_mla
        ), (
            "enable_scheduler_bypass_lookup is only supported with "
            "save_only_first_rank or remote_enable_mla_worker_id_as0"
        )
    return engine
