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
from typing import List, Optional, cast

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.event_manager import EventStatus, EventType
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


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
        memory_objs = storage_backend.batched_get_blocking(keys)
        if memory_objs:
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
