# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from typing import Optional

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import ENGINE_NAME, mla_enabled
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    _calculate_draft_layers,
    need_gpu_interm_buffer,
)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import GPUConnectorInterface
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
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


# Patching wait_for_save to remove the PD disagg_spec skip_leading_tokens
# override. The upstream code does:
#   if self.kv_role == "kv_producer" and request.disagg_spec:
#       skip_leading_tokens = min(skip_leading_tokens,
#                                 request.disagg_spec.num_transferred_tokens)
# save_spec.skip_leading_tokens is already aligned with the number of tokens
# that have been saved, in chunk prefills and delay pull mode, this can cause
# redundant full re-saves when there is an existing cache hit.
# In push mode, this is not a problem, because the skip leading tokens
# already aligns with the number of tokens that have been saved.
@_lmcache_nvtx_annotate
def wait_for_save(self):
    """Blocking until the KV cache is saved to the connector buffer."""

    connector_metadata = self._parent._get_connector_metadata()
    assert isinstance(connector_metadata, LMCacheConnectorMetadata)

    if self.kv_role == "kv_consumer":
        return

    if self.use_layerwise:
        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)

        for request in connector_metadata.requests:
            self.lmcache_engine.lookup_unpin(request.req_id)
        return

    assert len(self.kv_caches) > 0
    kvcaches = list(self.kv_caches.values())

    assert self.lmcache_engine is not None

    for request in connector_metadata.requests:
        self.lmcache_engine.lookup_unpin(request.req_id)

        save_spec = request.save_spec
        if (
            save_spec is None or not save_spec.can_save
        ) and self.kv_role != "kv_producer":
            continue

        token_ids = request.token_ids

        slot_mapping = request.slot_mapping
        assert isinstance(slot_mapping, torch.Tensor)
        assert len(slot_mapping) == len(token_ids)

        slot_mapping = slot_mapping.to(self.device)

        skip_leading_tokens = save_spec.skip_leading_tokens

        if skip_leading_tokens == len(token_ids):
            continue
        skip_leading_tokens = (
            skip_leading_tokens // self._lmcache_chunk_size * self._lmcache_chunk_size
        )

        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip_leading_tokens] = False

        logger.info(
            "Storing KV cache for %d out of %d tokens "
            "(skip_leading_tokens=%d) for request %s",
            len(token_ids) - skip_leading_tokens,
            len(token_ids),
            skip_leading_tokens,
            request.req_id,
        )

        is_last_prefill = request.is_last_prefill
        if is_last_prefill:
            if request.disagg_spec:
                request.disagg_spec.is_last_prefill = True
        else:
            if not self.enable_blending:
                token_len = len(token_ids)
                aligned_token_len = (
                    token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                )
                token_ids = token_ids[:aligned_token_len]
                store_mask = store_mask[:aligned_token_len]
                slot_mapping = slot_mapping[:aligned_token_len]

        self.lmcache_engine.store(
            token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            transfer_spec=request.disagg_spec,
            request_configs=request.request_configs,
            req_id=request.req_id,
        )

        if get_pp_group().is_last_rank:
            save_spec.skip_leading_tokens = len(token_ids)
            if request.disagg_spec:
                request.disagg_spec.num_transferred_tokens = len(token_ids)
