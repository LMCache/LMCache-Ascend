# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import functools
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Generator, List, Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, start_loop_in_thread_with_exceptions
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventType
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )

# Third Party
from lmcache.v1.storage_backend.storage_manager import AsyncSerializer

logger = init_logger(__name__)


# Fix from https://github.com/LMCache/LMCache/pull/1795
# TODO (gingfung): remove when in v0.3.9
def post_init_fix(self, **kwargs) -> None:
    if "async_lookup_server" in kwargs:
        assert not self.config.save_unfull_chunk, (
            "save_unfull_chunk should be automatically set to False "
            "when using async loading."
        )
        self.async_lookup_server = kwargs.pop("async_lookup_server")
    self.async_serializer = AsyncSerializer(self.allocator_backend, self.loop)


def __init__(
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
    self.async_serializer: Optional[AsyncSerializer] = None

    # The cuda stream for internal copies during put
    if torch.cuda.is_available():
        self.internal_copy_stream = torch.cuda.Stream()
    else:
        self.internal_copy_stream = None
    self.async_serializer = AsyncSerializer(self.allocator_backend, self.loop)


def layerwise_batched_get(
    self,
    keys: List[List[CacheEngineKey]],
    location: Optional[str] = None,
) -> Generator[Future, None, None]:
    """
    Non-blocking function to get the memory objects into the storages
    in a layerwise manner.
    Do not store if the same object is being stored (handled here by
    storage manager) or has been stored (handled by storage backend).

    :param List[List[CacheEngineKey]] keys: The keys to get. The first
        dimension corresponds to the number of layers, and the second
        dimension corresponds to the number of chunks.

    :return: A generator that yields a future for each layer.
    """
    if location is None:
        location = "LocalCPUBackend"

    for keys_multi_chunk in keys:
        # Retrieve all chunks for one layer
        backend = self.storage_backends[location]
        # TODO(Jiayi): need to make async loading and layerwise compatible
        assert self.async_serializer is not None, (
            "Async serializer must be initialized via post_init before using "
            "layerwise_batched_get."
        )
        coro = self.async_serializer.run(
            backend.batched_get_non_blocking("fake_lookup_id", keys_multi_chunk),
            len(keys_multi_chunk),
        )
        task = asyncio.run_coroutine_threadsafe(coro, self.loop)
        yield task


async def async_lookup_and_prefetch(
    self,
    lookup_id: str,
    keys: list[CacheEngineKey],
    cum_chunk_lengths: list[int],
    search_range: Optional[list[str]] = None,
    pin: bool = False,
) -> None:
    """
    Perform asynchronous lookup and prefetching across all storage backends.

    :param str lookup_id: The unique id (e.g., request id) for the request.
    :param list[CacheEngineKey] keys: The keys to lookup and prefetch.
    :param list[int] cum_chunk_lengths: The cumulative lengths of the chunks.
    :param Optional[list[str]] search_range: The range of storage backends
    to search in. Should be a subset of ["LocalCPUBackend",
    "LocalDiskBackend"] for now. If None, search in all backends.
    :param bool pin: Whether to pin the keys.
    """

    # NOTE(Jiayi): Currently, the retrieval pattern is always
    # prefix-based. That is, we retrieve 0-t1 tokens from backend 1
    # and retrieve t1-t2 tokens from backend 2, etc. The assumption
    # here is that the suffix chunks are more likely to be evicted
    # than the prefix chunks.
    # TODO(Jiayi): We need to change/optimize this for non-prefix
    # based retrieval patterns or cases where middle chunks are missing.

    # NOTE(Jiayi): We can tolerate the last tier to have fewer loaded
    # chunks than its lookup result indicated. This is especially helpful
    # for P2PBackend.

    num_total_chunks = len(keys)
    num_total_hit_chunks = 0
    num_last_tier_hit_chunks = 0
    cum_chunk_lengths_total = cum_chunk_lengths[:]
    loading_tasks = []
    for backend_name, backend in self.storage_backends.items():
        if search_range and backend_name not in search_range:
            continue
        num_hit_chunks = await backend.batched_async_contains(lookup_id, keys, pin)

        if num_hit_chunks == 0:
            continue

        num_last_tier_hit_chunks = num_hit_chunks

        num_total_hit_chunks += num_hit_chunks

        assert self.async_serializer is not None, (
            "Async serializer must be initialized via post_init before using "
            "async_lookup_and_prefetch."
        )
        get_coro = self.async_serializer.run(
            backend.batched_get_non_blocking(
                lookup_id,
                keys[:num_hit_chunks],
                {"cum_chunk_lengths": cum_chunk_lengths[: num_hit_chunks + 1]},
            ),
            num_hit_chunks,
        )
        loading_task = asyncio.create_task(get_coro)

        loading_task.add_done_callback(
            functools.partial(
                self.prefetch_single_done_callback,
                keys=keys,
                backend_name=backend_name,
            )
        )

        loading_tasks.append(loading_task)

        cum_chunk_lengths = cum_chunk_lengths[num_hit_chunks:]

        if num_total_hit_chunks == num_total_chunks:
            break
        keys = keys[num_hit_chunks:]

    # If no chunks were hit across all backends, respond immediately and return.
    if num_total_hit_chunks == 0:
        if self.async_lookup_server is not None:
            self.async_lookup_server.send_response_to_scheduler(lookup_id, 0)
        return

    all_done = asyncio.gather(*loading_tasks)
    # Register the event before adding the callback to avoid race conditions
    self.event_manager.add_event(
        EventType.LOADING,
        lookup_id,
        all_done,
    )

    all_done.add_done_callback(
        lambda future: self.prefetch_all_done_callback(
            future,
            lookup_id,
            cum_chunk_lengths_total[num_total_hit_chunks - num_last_tier_hit_chunks :],
        )
    )
