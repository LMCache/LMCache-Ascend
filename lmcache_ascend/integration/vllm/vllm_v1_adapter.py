# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

# Third Party
# TODO: Currently we patch all the cuda calls due to effort to port all torch.cuda
# will disabled torch.jit
from torch_npu.contrib import transfer_to_npu

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import ENGINE_NAME, lmcache_get_config
from lmcache.integration.vllm.vllm_adapter import (
    VLLM_CACHE_CONFIG,
    VLLM_MODEL_CONFIG,
    VLLM_PARALLEL_CONFIG,
    VLLM_SCHEDULER_CONFIG,
    need_gpu_interm_buffer,
)
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import (
    VLLMBufferLayerwiseGPUConnector,
)
from lmcache_ascend.v1.npu_connector import (
    VLLMPagedMemNPUConnectorV2,
    VLLMPagedMemLayerwiseNPUConnector,
)
from lmcache.logging import init_logger

# Third Party
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from vllm.utils import get_kv_cache_torch_dtype

logger = init_logger(__name__)


# We need to patch this function due to connector modification
def init_lmcache_engine(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    scheduler_config: SchedulerConfig,
) -> Optional[LMCacheEngine]:
    """Initialize the LMCache engine by the given model config and parallel
    config. This function will check the environment variable
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param model_config: The model configuration in vLLM.
    :type model_config: ModelConfig
    :param parallel_config: The parallel configuration in vLLM.
    :type parallel_config: ParallelConfig
    :param cache_config: The KV cache configuration in vLLM.
    :type cache_config: CacheConfig
    :param scheduler_config: The scheduler configuration in vLLM.
    :type scheduler_config: SchedulerConfig

    :return: The initialized LMCache engine or None (if the environment variable
        `LMCACHE_CONFIG_FILE` is not set).
    :rtype: Optional[LMCacheEngine]
    """
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return None

    global VLLM_CACHE_CONFIG
    global VLLM_PARALLEL_CONFIG
    global VLLM_MODEL_CONFIG
    global VLLM_SCHEDULER_CONFIG
    VLLM_CACHE_CONFIG = cache_config
    VLLM_PARALLEL_CONFIG = parallel_config
    VLLM_MODEL_CONFIG = model_config
    VLLM_SCHEDULER_CONFIG = scheduler_config

    config = lmcache_get_config()

    assert isinstance(config, LMCacheEngineConfig), (
        "LMCache v1 configuration is should be passed."
    )

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

    use_mla = False
    if (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    ):
        use_mla = True

    if use_mla and (config.remote_serde != "naive" and config.remote_serde is not None):
        raise ValueError("MLA only works with naive serde mode..")

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    chunk_size = config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
    logger.info(f"use mla: {use_mla}, kv shape: {kv_shape}")

    # Change current device.
    torch.cuda.device(parallel_config.rank)
    device = torch.device(f"cuda:{parallel_config.rank}")
    metadata = LMCacheEngineMetadata(
        model_config.model,
        parallel_config.world_size,
        parallel_config.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
    )

    use_gpu = need_gpu_interm_buffer(config)
    vllm_gpu_connector: Union[
        VLLMPagedMemNPUConnectorV2,
        VLLMPagedMemLayerwiseNPUConnector,
    ]

    if use_mla and config.use_layerwise:
        raise ValueError("layerwise MLA connector is not supported yet")

    # When use_mla is True, num_kv_head is 1
    hidden_dim_size = num_kv_head * head_size
    if config.use_layerwise:
        if config.enable_blending:
            # Use layerwise connector for blending
            raise NotImplementedError("Blending is not yet supported for Ascend.")
        else:
            vllm_gpu_connector = VLLMPagedMemLayerwiseNPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
    else:
        vllm_gpu_connector = VLLMPagedMemNPUConnectorV2(
            hidden_dim_size,
            num_layer,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=kv_dtype,
            device=device,
            use_mla=use_mla,
        )
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME, config, metadata, vllm_gpu_connector
    )

    return engine
