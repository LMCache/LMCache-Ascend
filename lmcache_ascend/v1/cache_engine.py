# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import (
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
import time

# Third Party
import torch

from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    convert_tokens_to_list,
)
from lmcache.v1.gpu_connector import (
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryObj,
)

logger = init_logger(__name__)

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


@_lmcache_nvtx_annotate
@torch.inference_mode()
def store_layer(
    self,
    tokens: Union[torch.Tensor, list[int]],
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Generator[None, None, None]:
    """
    Store the KV cache in a layerwise manner.

    :param torch.Tensor tokens: The tokens of the corresponding KV caches.

    :param Optional[torch.Tensor] mask: The mask for the tokens. Should
        have the same length as tokens. And the mask should ALWAYS be like
        FFFFFTTTTTTT, where True means the tokens needs to be matched.

    :param **kwargs: The additional arguments for the storage backend which
        will be passed into the gpu_connector.

    return: A generator that yields None. In the first iteration, the
        generator allocates the memory objects for all layers and moves
        the KV cache of the first layer from GPU to CPU. In the next
        iterations, it moves the KV cache of layer i from GPU to the memory
        objects (on CPU) and puts the memory objects of layer i-1 to the
        storage backends. In the last iteration, it puts the memory objects
        of the last layer to the storage backends.
    """
    assert self.storage_manager is not None
    assert self.gpu_connector is not None, (
        "gpu_connector is required for store_layer operation"
    )

    if mask is not None:
        num_to_store_tokens = torch.sum(mask).item()
    else:
        num_to_store_tokens = len(tokens)

    # KVCache Check logging
    self._log_kvcache_for_check(
        operation="Layerwise store",
        kwargs=kwargs,
        token_count=num_to_store_tokens,
        require_req_id=True,
    )

    monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

    # Check if freeze mode is enabled
    if self.is_frozen():
        logger.debug(
            "Freeze mode enabled, skipping store_layer for %d tokens",
            num_to_store_tokens,
        )
        # Still need to yield to avoid StopIteration
        for layer_id in range(self.num_layers):
            yield
        return

    starts = []
    ends = []
    keys = []
    memory_objs = []
    tot_token_num = 0
    kv_dtype = self.metadata.kv_dtype
    request_configs = kwargs.get("request_configs")
    if request_configs is not None and len(request_configs) != 0:
        assert isinstance(request_configs, dict)

    prev_key = 0
    for start, end, key in self.token_database.process_tokens(
        tokens=tokens, mask=mask, request_configs=request_configs
    ):
        assert isinstance(key, CacheEngineKey)

        keys_multi_layer = key.split_layers(self.num_layers)
        # Only check the first layer
        if self.storage_manager.contains(keys_multi_layer[0]):
            continue

        # Allocate the memory object
        num_tokens = end - start
        kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

        memory_objs_multi_layer = self.storage_manager.batched_allocate(
            kv_shape_single_layer,
            kv_dtype,
            batch_size=self.num_layers,
            fmt=self.fmt,
            busy_loop=self.force_store_wait,
        )

        if memory_objs_multi_layer is None:
            logger.warning(
                "Local cpu memory under pressure so choosing to not store the KV cache."
            )
            break

        starts.append(start)
        ends.append(end)
        keys.append(keys_multi_layer)
        memory_objs.append(memory_objs_multi_layer)
        tot_token_num += num_tokens

        # Create KV event
        if self.kv_events_enabled and tokens is not None:
            stored_event = CacheStoreEvent(
                block_hashes=[key.chunk_hash],
                parent_block_hash=None if start == 0 else prev_key,
                token_ids=[],
                block_size=num_tokens,
                lora_id=None,
                medium="cpu",
            )
            if tokens is not None:
                stored_event.token_ids = convert_tokens_to_list(
                    tokens,
                    start,
                    end,
                )
                if isinstance(tokens, torch.Tensor):
                    stored_event.medium = tokens.device
            logger.debug(
                f"Added kv cache event '{stored_event}' to kv cache events queue"
            )
            self.kv_events.append(stored_event)
            prev_key = key.chunk_hash

    if keys:
        # Transpose the keys and memory objects into layer major format
        memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
        keys = [list(row) for row in zip(*keys, strict=False)]

        # Calculate total KV size for logging
        tot_kv_size = sum(
            mo.get_size() for layer_objs in memory_objs for mo in layer_objs
        )

        assert isinstance(
            self.gpu_connector,
            (
                VLLMPagedMemLayerwiseGPUConnector,
                VLLMBufferLayerwiseGPUConnector,
                SGLangLayerwiseGPUConnector,
            ),
        )

        t_start = time.perf_counter()
        mem_obj_generator = self.gpu_connector.batched_from_gpu(
            memory_objs, starts, ends, **kwargs
        )

        next(mem_obj_generator)

        for layer_id in range(self.num_layers):
            yield
            next(mem_obj_generator)
            self.storage_manager.batched_put(keys[layer_id], memory_objs[layer_id])

        # NOTE(niming): Temporary INFO logging for sglang compatibility.
        # TODO: Remove this patch in the next release.
        tot_time = time.perf_counter() - t_start
        logger.info(
            "Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
            tot_token_num,
            len(tokens),
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
        )
    else:
        # If no cache are found, we still need to yield to avoid
        # `StopIteration`
        for layer_id in range(self.num_layers):
            yield

    self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
    # logger.debug(f"Stored {tot_token_num} out of total {len(tokens)} tokens")
    yield
