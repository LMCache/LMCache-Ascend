# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Sequence
import asyncio
import threading

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey, start_loop_in_thread_with_exceptions
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend import CreateStorageBackends, is_cuda_worker
from lmcache.v1.storage_backend.storage_manager import AsyncSingleSerializer, AsyncSerializer
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
import torch

# First Party
from lmcache_ascend.v1.npu_connector import is_310p

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )


# Helper function to allocate and copy memory objects between D and H
def allocate_and_copy_objects_310p(
    allocator_backend: AllocatorBackendInterface,
    keys: Sequence[CacheEngineKey],
    src_memory_objs: list[MemoryObj],
    stream: torch.cuda.Stream,
) -> tuple[Sequence[CacheEngineKey], list[MemoryObj]]:
    """
    Allocate the memory objects and copy the data from src_memory_objs to
    the newly allocated memory objects

    Args:
        allocator_backend: the allocator backend to allocate the new memory
          objects
        keys: the cache engine keys corresponding to the memory objects
        src_memory_objs: the memory objects to copy from
        stream: the cuda stream to run the copy in

    Returns:
        - list of cache engine keys that corresponds to the memory objects
          that has been successfully allocated
        - list of the memory objects that has been successfully allocated
    """
    allocated_objects = []
    for key, src_memory_obj in zip(keys, src_memory_objs, strict=False):
        if allocator_backend.contains(key):
            continue
        memory_obj = allocator_backend.allocate(
            shape=src_memory_obj.get_shape(),
            dtype=src_memory_obj.get_dtype(),
            fmt=src_memory_obj.meta.fmt,
            eviction=True,
            busy_loop=False,
        )

        if memory_obj is None or memory_obj.tensor is None:
            break

        if is_310p():
            memory_obj.tensor.copy_(src_memory_obj.tensor, non_blocking=True)
        else:
            with torch.cuda.stream(stream):
                memory_obj.tensor.copy_(src_memory_obj.tensor, non_blocking=True)
        allocated_objects.append(memory_obj)

    if is_310p():
        torch.cuda.synchronize()
    else:
        stream.synchronize()

    return keys[: len(allocated_objects)], allocated_objects


def StorageManager__init__(
    self,
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    event_manager: EventManager,
    lmcache_worker: Optional["LMCacheWorker"] = None,
    async_lookup_server: Optional["LMCacheAsyncLookupServer"] = None,
):
    self.config = config
    self.metadata = metadata
    self.loop = asyncio.new_event_loop()

    self.thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(self.loop,),
        name="storage-manger-event-loop",
    )
    self.thread.start()

    # For scheduler role, always use CPU device
    if is_cuda_worker(metadata):
        dst_device = "cuda"
    else:
        dst_device = "cpu"
    self.storage_backends: OrderedDict[str, StorageBackendInterface] = (
        CreateStorageBackends(
            config,
            metadata,
            self.loop,
            dst_device,
            lmcache_worker,
        )
    )

    # the backend used for actual storage
    self.non_allocator_backends = self.get_non_allocator_backends()

    self.enable_pd = config.enable_pd

    self.allocator_backend = None
    if metadata.role != "scheduler":
        self.allocator_backend = self._get_allocator_backend(config)
    if config.local_cpu:
        self.local_cpu_backend = self.storage_backends["LocalCPUBackend"]

    self.manager_lock = threading.Lock()

    self.lmcache_worker = lmcache_worker
    self.instance_id = config.lmcache_instance_id
    self.worker_id = metadata.worker_id

    self.event_manager = event_manager

    self.async_lookup_server: Optional["LMCacheAsyncLookupServer"] = (
        async_lookup_server
    )
    self.async_serializer: Optional[AsyncSerializer] = None

    # The cuda stream for internal copies during put
    if is_cuda_worker(metadata) and not is_310p():
        self.internal_copy_stream = torch.cuda.Stream()
    else:
        self.internal_copy_stream = None

    # freeze mode: only use local_cpu backend for retrieval
    self._freeze = False
    self._freeze_lock = threading.RLock()

    if not self.enable_pd and self.config.enable_async_loading:
        assert self.allocator_backend is not None
        self.async_serializer = AsyncSingleSerializer(self.loop)

    self._setup_metrics()
