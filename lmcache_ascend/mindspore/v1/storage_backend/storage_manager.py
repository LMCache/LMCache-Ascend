# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Generator,
    List,
    Optional,
    Sequence,
)
import asyncio
import functools
import threading

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    _lmcache_nvtx_annotate,
    start_loop_in_thread_with_exceptions,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import (
    StorageManager,
)

from lmcache_ascend.v1.npu_connector import is_310p

if TYPE_CHECKING:
    # First Party
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

    if torch.cuda.is_available():
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

    self.enable_pd = config.enable_pd

    self.allocator_backend = self._get_allocator_backend(config)
    if config.local_cpu:
        self.local_cpu_backend = self.storage_backends["LocalCPUBackend"]

    self.manager_lock = threading.Lock()

    self.lmcache_worker = lmcache_worker
    self.instance_id = config.lmcache_instance_id
    self.worker_id = metadata.worker_id

    self.event_manager = event_manager

    self.async_lookup_server: Optional["LMCacheAsyncLookupServer"] = None

    # The cuda stream for internal copies during put
    if torch.cuda.is_available():
        if is_310p():
            self.internal_copy_stream = None
        else:
            self.internal_copy_stream = torch.cuda.Stream()
    else:
        self.internal_copy_stream = None
