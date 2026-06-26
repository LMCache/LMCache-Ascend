# SPDX-License-Identifier: Apache-2.0
"""Ascend overrides for ``lmcache.v1.storage_backend`` resilience.

These rebind:

* ``StorageManager.get`` / ``StorageManager.batched_get`` -- delay-pull proxy
  write-back guard (see "Why the proxy guard is needed" below), and
* ``LocalCPUBackend.touch_cache`` / ``LocalDiskBackend.touch_cache`` -- make the
  LRU bookkeeping strictly best-effort so a single evicted key can never abort a
  lookup (see "Why touch_cache must not raise" below),
* ``StorageManager.prefetch_all_done_callback`` -- mirror prefetched tiers into
  the local hot cache when enabled (see ``patched_prefetch_all_done_callback``).

so the fixes live in the Ascend overlay instead of mutating the upstream
LMCache tree.

Why the proxy guard is needed
-----------------------------
With ``p2p_delay_pull`` the (Ascend) P2P backend returns ``ProxyMemoryObj``
placeholders that carry *no data* at retrieve time -- the real KV is pulled
later straight into transient device buffers and the proxy is then consumed.

The upstream write-back path eagerly mirrors every non-local hit into
``LocalCPUBackend`` for faster reuse. Mirroring a data-less proxy poisons the
hot cache with a stale entry that:

  (a) re-issues one-sided reads against the sender's already-released buffers on
      the next "local hit", and
  (b) blocks the legitimate save path from ever storing the real KV for that key
      (``submit_put_task`` no-ops when the key is already present).

Skipping proxies (``is_proxy``) keeps the local write-back correct while leaving
every non-proxy path identical to upstream.

Why touch_cache must not raise
------------------------------
``CacheEngine.lookup`` pins hits via ``batched_contains(pin=True)`` (which
appends each hit key to the backend's ``keys_in_request``) and then, in a
``finally`` block, calls ``StorageManager.touch_cache`` to refresh the eviction
policy ordering. Upstream ``LocalCPUBackend.touch_cache`` / ``LocalDiskBackend
.touch_cache`` do::

    for key in reversed(self.keys_in_request):
        self.cache_policy.update_on_hit(key, self.<dict>)   # may KeyError(key)
    self.keys_in_request = []                                # only on success

For LRU/MRU ``update_on_hit`` runs ``cache_dict.move_to_end(key)`` and for LFU
``self.key_to_freq[key]`` -- both raise ``KeyError(key)`` if the pinned key was
removed (concurrent overwrite/eviction) between ``contains`` and ``touch_cache``.

Because the raise happens in ``lookup``'s ``finally``, it *discards the already
-computed local hit count* and propagates out of ``lookup``. The lookup RPC
handler then sends no reply, so the scheduler times out (and recreates sockets)
and the local cache hit is lost. And since ``keys_in_request`` is never cleared
on the raising path, the stale key poisons every later ``touch_cache``, turning
a transient race into a permanent, every-request failure across all ranks.

The overrides below update each key independently and *always* clear
``keys_in_request`` (``finally``), so eviction-policy bookkeeping degrades to a
no-op for missing keys instead of aborting the lookup.
"""

# Standard
from typing import List, Optional, Sequence, cast
import contextvars

# Third Party
import torch
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.event_manager import EventStatus, EventType
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

_current_pd_lookup_id: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("current_pd_lookup_id", default=None)
)
_current_pd_retrieve_id: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("current_pd_retrieve_id", default=None)
)


def set_current_pd_lookup_id(request_id: str) -> contextvars.Token:
    return _current_pd_lookup_id.set(request_id)


def reset_current_pd_lookup_id(token: contextvars.Token) -> None:
    _current_pd_lookup_id.reset(token)


def set_current_pd_retrieve_id(request_id: str) -> contextvars.Token:
    return _current_pd_retrieve_id.set(request_id)


def reset_current_pd_retrieve_id(token: contextvars.Token) -> None:
    _current_pd_retrieve_id.reset(token)


def allocate_and_copy_objects(
    allocator_backend: AllocatorBackendInterface,
    keys: Sequence[CacheEngineKey],
    src_memory_objs: list[MemoryObj],
    stream: torch.cuda.Stream,
) -> tuple[Sequence[CacheEngineKey], list[MemoryObj]]:
    """Allocate/copy objects while preserving key-object alignment.

    Upstream LMCache returns ``keys[:len(allocated_objects)]`` after skipping
    keys that already exist in the target allocator. That can pair newly copied
    suffix objects with already-skipped prefix keys. Keep the allocated keys
    alongside the objects so StorageManager fan-out submits aligned pairs.
    """
    allocated_keys = []
    allocated_objects = []
    for key, src_memory_obj in zip(keys, src_memory_objs, strict=False):
        if allocator_backend.contains(key):
            continue
        memory_obj = allocator_backend.allocate(
            src_memory_obj.get_shape(),
            src_memory_obj.get_dtype(),
            fmt=src_memory_obj.meta.fmt,
            eviction=True,
            busy_loop=False,
        )

        if memory_obj is None:
            break

        if memory_obj.tensor is None:
            logger.warning(
                "Allocated MemoryObj has None tensor, this is unexpected. "
                "Releasing the memory object."
            )
            memory_obj.ref_count_down()
            break

        with torch.cuda.stream(stream):
            memory_obj.tensor.copy_(src_memory_obj.tensor, non_blocking=True)
        allocated_keys.append(key)
        allocated_objects.append(memory_obj)

    if stream is not None:
        stream.synchronize()
    return allocated_keys, allocated_objects


def get(
    self,
    key: CacheEngineKey,
    location: Optional[str] = None,
) -> Optional[MemoryObj]:
    """Blocking get with a delay-pull proxy guard on local write-back."""
    # Search all backends for blocking get
    for backend_name, backend in self.get_active_storage_backends(location):
        # TODO(Jiayi): need to make sure all memory_objs returned
        # are allocated by the allocator backend.
        memory_obj = backend.get_blocking(key)
        if memory_obj:
            # Skip deferred-fetch proxies (e.g. P2P delay-pull): they hold no
            # data here, so caching them poisons the hot cache. See module note.
            if (
                backend_name not in ["LocalCPUBackend", "PDBackend", "MaruBackend"]
                and "LocalCPUBackend" in self.storage_backends
                and not getattr(memory_obj, "is_proxy", False)
            ):
                local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                assert isinstance(local_cpu_backend, LocalCPUBackend)
                local_cpu_backend.submit_put_task(key, memory_obj)
            return memory_obj

    return None


def batched_get(
    self,
    keys: List[CacheEngineKey],
    location: Optional[str] = None,
) -> List[Optional[MemoryObj]]:
    """Blocking batched get with a delay-pull proxy guard on local write-back."""
    # TODO (ApostaC): remove the nested optional here
    for backend_name, storage_backend in self.get_active_storage_backends(location):
        request_id = _current_pd_retrieve_id.get()
        if (
            backend_name == "PDBackend"
            and request_id is not None
            and hasattr(storage_backend, "batched_get_blocking_for_request")
        ):
            memory_objs = storage_backend.batched_get_blocking_for_request(
                keys,
                request_id,
            )
        else:
            memory_objs = storage_backend.batched_get_blocking(keys)
            
        if memory_objs and any(m is not None for m in memory_objs):
            # Align with single-key `get()` logic:
            # auto-write remote data to local CPU cache, but skip deferred-fetch
            # proxies (e.g. P2P delay-pull ProxyMemoryObj) -- see module note.
            if (
                backend_name not in ["LocalCPUBackend", "PDBackend", "MaruBackend"]
                and "LocalCPUBackend" in self.storage_backends
                and None not in memory_objs
                and not any(getattr(m, "is_proxy", False) for m in memory_objs)
            ):
                logger.debug(
                    "Storing %s objects from %s to LocalCPUBackend",
                    len(keys),
                    backend_name,
                )
                local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                assert isinstance(local_cpu_backend, LocalCPUBackend)
                # Type cast: Safe (we verified no Nones above)
                # `batched_submit_put_task` expects list[MemoryObj]
                memory_objs_no_none = cast(List[MemoryObj], memory_objs)
                local_cpu_backend.batched_submit_put_task(keys, memory_objs_no_none)
            return memory_objs
    return [None] * len(keys)


def batched_contains(
    self,
    keys: List[CacheEngineKey],
    search_range: Optional[List[str]] = None,
    pin: bool = False,
) -> tuple[int, dict]:
    """Prefix lookup with PD receiver request-lease registration."""
    total_keys = len(keys)
    total_hit_chunks = 0
    block_mapping = {}
    for backend_name, backend in self.get_active_storage_backends(
        search_range=search_range
    ):
        request_id = _current_pd_lookup_id.get()
        if (
            backend_name == "PDBackend"
            and pin
            and request_id is not None
            and hasattr(backend, "batched_contains_and_lease")
        ):
            hit_chunks = backend.batched_contains_and_lease(keys, request_id)
        else:
            # Preserve upstream semantics: PDBackend is not pinned by the
            # generic pin path. Ascend PD leases are handled above.
            pin_in_backend = pin if backend_name != "PDBackend" else False
            hit_chunks = backend.batched_contains(keys, pin_in_backend)

        if hit_chunks == 0:
            continue
        block_mapping[backend_name] = keys[:hit_chunks]
        total_hit_chunks += hit_chunks
        if total_hit_chunks == total_keys:
            break
        keys = keys[hit_chunks:]

    return total_hit_chunks, block_mapping


def _best_effort_touch_cache(self, cache_dict) -> None:
    """Refresh eviction-policy order for ``keys_in_request`` without raising.

    Mirrors upstream ``touch_cache`` but (a) guards every ``update_on_hit`` so a
    key removed since it was pinned (``KeyError``) is skipped instead of
    aborting the enclosing ``CacheEngine.lookup``, and (b) always clears
    ``keys_in_request`` so a missing key cannot poison subsequent lookups. See
    the module docstring ("Why touch_cache must not raise").

    Shared by the CPU and disk backend overrides; ``cache_dict`` is the backend's
    backing store (``hot_cache`` / ``dict``).
    """
    try:
        for key in reversed(self.keys_in_request):
            try:
                self.cache_policy.update_on_hit(key, cache_dict)
            except Exception as e:
                # Best-effort LRU/LFU bookkeeping: a key that was pinned during
                # lookup may have been overwritten/evicted before touch_cache.
                # Skipping it keeps lookup returning its local hit count.
                logger.debug(
                    "touch_cache: skipping eviction-policy update for missing "
                    "key (%s): %s",
                    type(e).__name__,
                    e,
                )
    finally:
        self.keys_in_request = []


def local_cpu_touch_cache(self) -> None:
    """Best-effort ``LocalCPUBackend.touch_cache`` (never raises). See module doc."""
    with self.cpu_lock:
        _best_effort_touch_cache(self, self.hot_cache)


def local_disk_touch_cache(self) -> None:
    """Best-effort ``LocalDiskBackend.touch_cache`` (never raises). See module doc."""
    with self.disk_lock:
        _best_effort_touch_cache(self, self.dict)


def patched_prefetch_all_done_callback(
    self,
    task,
    lookup_id,
    cum_chunk_lengths_total,
    tier_expected_chunks,
):
    assert self.async_lookup_server is not None
    self.event_manager.update_event_status(
        EventType.LOADING, lookup_id, status=EventStatus.DONE
    )

    res = task.result()

    total_retrieved_chunks = 0
    for tier_idx, tier_result in enumerate(res):
        actual_chunks = len(tier_result)
        total_retrieved_chunks += actual_chunks
        if actual_chunks < tier_expected_chunks[tier_idx]:
            for subsequent_tier in res[tier_idx + 1 :]:
                for mem_obj in subsequent_tier:
                    mem_obj.ref_count_down()
            break

    # inject hotcache start ---------------------
    if (
        self.local_cpu_backend is not None
        and self.local_cpu_backend.use_hot
        and total_retrieved_chunks > 0
    ):
        chunk_count = 0
        for tier_idx, tier_result in enumerate(res):
            tier_keys = []
            tier_objs = []
            for key, mem_obj in tier_result:
                if chunk_count >= total_retrieved_chunks:
                    break
                tier_keys.append(key)
                tier_objs.append(mem_obj)
                chunk_count += 1
            if tier_keys:
                self.local_cpu_backend.batched_submit_put_task(tier_keys, tier_objs)
            if chunk_count >= total_retrieved_chunks:
                break
    # inject hotcache end ---------------------

    retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]
    logger.info(
        f"Responding to scheduler for lookup id {lookup_id}"
        f" with retrieved length {retrieved_length}"
    )
    self.async_lookup_server.send_response_to_scheduler(lookup_id, retrieved_length)
