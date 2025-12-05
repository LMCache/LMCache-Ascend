# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import List, Optional, Union
import asyncio
import time
from copy import deepcopy

# Third Party
import torch
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
@torch.inference_mode()
def LMCacheEngine_store(
    self,
    tokens: np.ndarray,
    mask: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """Store the tokens and mask into the cache engine.

    :param torch.Tensor tokens: The tokens of the corresponding KV caches.

    :param Optional[torch.Tensor] mask: The mask for the tokens. Should
        have the same length as tokens. And the mask should ALWAYS be like
        FFFFFTTTTTTT, where True means the tokens needs to be matched,
        and the Falses will ALWAYS be at the PREFIX of the tensor.

    :param **kwargs: The additional arguments for the storage backend which
        will be passed into the gpu_connector.
        Should include KV cache specific information (e.g., paged KV buffer
        and the page tables).

    :raises: ValueError if the number of Falses in the mask is not a
        multiple of the chunk size.
    """

    if mask is not None:
        num_stored_tokens = int(np.sum(mask))
    else:
        num_stored_tokens = tokens.size
    monitor_req_id = self.stats_monitor.on_store_request(num_stored_tokens)

    starts = []
    ends = []
    keys = []
    memory_objs = []

    offload_time = 0.0
    put_time = 0.0
    tot_kv_size = 0
    t = time.perf_counter()

    for start, end, key in self.token_database.process_tokens(tokens, mask):
        assert isinstance(key, CacheEngineKey)
        if self.storage_manager.contains(key):
            continue
        # Allocate the memory object
        num_tokens = end - start
        kv_shape = self.gpu_connector.get_shape(num_tokens)
        kv_dtype = self.metadata.kv_dtype
        memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype)
        if memory_obj is None:
            logger.warning(
                "Failed to allocate memory for the KV cache.\n"     
                "The KV cache will not be stored."
            )
            break

        starts.append(start)
        ends.append(end)
        keys.append(key)
        memory_objs.append(memory_obj)
        tot_kv_size += memory_obj.get_size()    
    self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
    offload_time += time.perf_counter() - t

    t = time.perf_counter()
    self.storage_manager.batched_put(keys, memory_objs)
    put_time += time.perf_counter() - t

    tot_time = offload_time + put_time

    if self.lookup_server is not None:
        self.lookup_server.batched_insert(keys)

    logger.debug(
        "Store %d tokens takes: %.4f ms, throughput: %.4f GB/s; "
        "offload_time: %.4f ms, put_time: %.4f ms",
        num_stored_tokens,
        tot_time * 1000,
        tot_kv_size / tot_time / 1024**3,
        offload_time * 1000,
        put_time * 1000,
    )

    self.stats_monitor.on_store_finished(monitor_req_id)

    logger.debug(f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")

@_lmcache_nvtx_annotate
@torch.inference_mode()
def LMCacheEngine_retrieve(
    self,
    tokens: np.ndarray,
    mask: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Retrieve the KV caches from the cache engine. And put the retrieved
    KV cache to the serving engine via the GPU connector.

    :param torch.Tensor tokens: The tokens of the corresponding KV caches.

    :param Optional[torch.Tensor] mask: The mask for the tokens. Should
        have the same length as tokens. And the mask should ALWAYS be like
        FFFFFTTTTTTT, where True means the tokens needs to be matched,
        and the Falses will ALWAYS be at the PREFIX of the tensor.

    :param **kwargs: The additional arguments for the storage backend which
        will be passed into the gpu_connector.
        Should include KV cache specific information (e.g., paged KV buffer
        and the page tables).

    :return: the boolean mask indicating which tokens are retrieved. The
        length of the mask should be the same as the tokens. On CPU.

    :raises: ValueError if the number of Falses in the mask is not a
        multiple of the chunk size.
    """
    if mask is not None:
        num_required_tokens = int(np.sum(mask))
    else:
        num_required_tokens = tokens.size

    # kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
    # with open('/home/junyuan/zhouhe/orig_kvcaches.txt', 'w') as f:
    #     print(kvcaches, file = f)

    monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

    ret_mask = np.zeros(tokens.shape, dtype=np.bool_)

    for start, end, key in self.token_database.process_tokens(tokens, mask):
        assert isinstance(key, CacheEngineKey)

        # Get the memory object from the storage backend
        memory_obj = self.storage_manager.get(key)

        if memory_obj is None:
            if self.enable_p2p:
                future_memory_obj = asyncio.run_coroutine_threadsafe(
                    self.distributed_server.issue_get(key),
                    self.distributed_loop,
                )
                memory_obj = future_memory_obj.result()
            if memory_obj is None:
                break

        ret_mask[start:end] = True

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        # t1 = time.perf_counter()
        self.gpu_connector.to_gpu(memory_obj, start, end, **kwargs)
        # t2 = time.perf_counter()
        # print("To gpu: ", t2-t1)
        memory_obj.ref_count_down()

        # NOTE (ApostaC): This is only for the current implementation:
        # When the object is retrieved back to vLLM, the storage backend
        # will immediately remove the object from itself
        if self.remove_after_retrieve:
            self.storage_manager.remove(key)
        else:
            self.storage_manager.batched_unpin([key])

    # kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
    # with open('/home/junyuan/zhouhe/upd_kvcaches.txt', 'w') as f:
    #     print(kvcaches, file = f)

    retrieved_tokens = np.sum(ret_mask)
    self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
    logger.debug(
        f"Retrieved {retrieved_tokens} "
        f"out of {num_required_tokens} "
        f"out of total {len(tokens)} tokens"
    )
    return ret_mask

