# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from typing import Optional

# Third Party
from lmcache.integration.vllm.utils import ENGINE_NAME, mla_enabled
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
)
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import need_gpu_interm_buffer
from lmcache.v1.metadata import LMCacheMetadata
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group

try:
    # Third Party
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
except ImportError:
    # Third Party
    from vllm.utils import get_kv_cache_torch_dtype

# Third Party
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


def ascend_create_gpu_connector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    engine: EngineType,
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
                "We haven't supported MLA with Cacheblend yet. "
                "Please disable blending."
            )

        if config.use_layerwise:
            if config.enable_blending:
                return VLLMBufferLayerwiseNPUConnector.from_metadata(
                    metadata, use_gpu, device
                )
            else:
                return VLLMPagedMemLayerwiseNPUConnector.from_metadata(
                    metadata, use_gpu, device
                )

        if config.use_gpu_connector_v3:
            raise NotImplementedError(
                "GPU Connector v3 is not supported yet. "
                "Please contact LMCache-Ascend."
            )
        else:
            return VLLMPagedMemNPUConnectorV2.from_metadata(
                metadata, use_gpu, device
            )
    elif engine == EngineType.SGLANG:
        # First Party
        from lmcache_ascend.v1.npu_connector import (
            SGLangLayerwiseNPUConnector,
            SGLangNPUConnector,
        )

        num_layer, _, chunk_size, num_kv_head, head_dim = metadata.kv_shape
        hidden_dim_size = num_kv_head * head_dim
        kv_dtype = metadata.kv_dtype

        if config.use_layerwise:
            return SGLangLayerwiseNPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        else:
            return SGLangNPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
    else:
        raise RuntimeError(f"Unsupported engine type for Ascend: {engine}")


def ascend_create_lmcache_engine(self, role: str) -> LMCacheEngine:
    """Patched version of LMCacheManager._create_lmcache_engine for Ascend.

    Sets NPU device and uses Ascend-specific GPU connector factory.
    """
    # Third Party
    from lmcache.integration.vllm.utils import ENGINE_NAME, mla_enabled

    if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
        return curr_engine

    assert self._vllm_config is not None, "vllm_config required for vLLM mode"

    model_config = self._vllm_config.model_config
    parallel_config = self._vllm_config.parallel_config
    cache_config = self._vllm_config.cache_config

    kv_dtype = get_kv_cache_torch_dtype(
        cache_config.cache_dtype, model_config.dtype
    )

    use_mla = mla_enabled(model_config)
    self._validate_mla_config(use_mla)

    # Construct kv shape
    num_layer = model_config.get_num_layers(parallel_config)
    num_draft_layers = self._calculate_draft_layers()
    num_layer += num_draft_layers
    chunk_size = self._config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)

    logger.info(
        "num_layer: %d, chunk_size: %d, num_kv_head (per gpu): %d, "
        "head_size: %d, hidden_dim (D) for KV (per gpu): %d, "
        "use mla: %s, kv shape: %s, num_draft_layers: %d",
        num_layer,
        chunk_size,
        num_kv_head,
        head_size,
        num_kv_head * head_size,
        use_mla,
        kv_shape,
        num_draft_layers,
    )

    # Extract engine_id and kv_connector_extra_config from vllm_config
    engine_id = None
    kv_connector_extra_config = None
    if hasattr(self._vllm_config, "kv_transfer_config"):
        kv_transfer_config = self._vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            engine_id = getattr(kv_transfer_config, "engine_id", None)
            kv_connector_extra_config = getattr(
                kv_transfer_config, "kv_connector_extra_config", None
            )

    # Calculate local rank / world size for NPU
    num_gpus = torch.npu.device_count()
    local_rank = parallel_config.rank % num_gpus
    local_world_size = min(parallel_config.world_size, num_gpus)

    metadata = LMCacheMetadata(
        model_name=model_config.model,
        world_size=parallel_config.world_size,
        local_world_size=local_world_size,
        worker_id=parallel_config.rank,
        local_worker_id=local_rank,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
        role=role,
        served_model_name=model_config.served_model_name,
        chunk_size=self._config.chunk_size,
        engine_id=engine_id,
        kv_connector_extra_config=kv_connector_extra_config,
    )

    # Change NPU device
    torch.npu.set_device(local_rank)

    # Get tensor parallel group
    if role == "scheduler":
        tpg = SimpleNamespace()
        tpg.broadcast = lambda tensor, src: tensor
        tpg.broadcast_object = lambda obj, src: obj
        vllm_gpu_connector = None
    else:
        tpg = get_tp_group()
        vllm_gpu_connector = ascend_create_gpu_connector(
            self._config, metadata, EngineType.VLLM
        )

    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        self._config,
        metadata,
        vllm_gpu_connector,
        tpg.broadcast,
        tpg.broadcast_object,
    )

    if role == "scheduler" and self._config.enable_scheduler_bypass_lookup:
        assert engine.save_only_first_rank or self._config.get_extra_config_value(
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
        for request in connector_metadata.requests:
            layerwise_storer = self._layerwise_save_storers.pop(
                request.req_id, None
            )
            if layerwise_storer is not None:
                next(layerwise_storer)
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
