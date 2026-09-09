# SPDX-License-Identifier: Apache-2.0
"""Ascend overrides for ``LocalDiskBackend`` multi-group disk save/load."""

# Standard
from copy import deepcopy
from typing import Any, Callable, Optional
import asyncio
import os

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# First Party
from lmcache_ascend.v1.memory_management import is_multi_group_memory_obj

_orig_async_save_bytes_to_disk = None
_orig_submit_put_task = None
_orig_get_blocking = None
logger = init_logger(__name__)


def _is_state_meta(meta) -> bool:
    # Paged allocators reuse metadata objects: an old dynamic field alone is
    # insufficient to identify a state allocation.
    return (
        meta.fmt == MemoryFormat.BINARY
        and getattr(meta, "state_checkpoint", None) is not None
    )


def _release_failed_state_put(self, key, size) -> None:
    try:
        with self.disk_lock:
            self.current_cache_size -= size
            self.cache_policy.update_on_force_evict(key)
    finally:
        self.disk_worker.remove_put_task(key)


def local_disk_submit_put_task(self, key, memory_obj, on_complete_callback=None):
    """Reserve a state write once; rejection leaves no pending key or owned ref."""
    if not _is_state_meta(memory_obj.meta):
        assert _orig_submit_put_task is not None
        return _orig_submit_put_task(self, key, memory_obj, on_complete_callback)
    size = memory_obj.get_size()
    # Serialize duplicate admission with reservation. Ordinary keys use a
    # separate namespace and retain their original submission path.
    with self.disk_lock:
        if key in self.dict or self.exists_in_put_tasks(key):
            return None
        if size > self.max_cache_size:
            return None
        while self.current_cache_size + size > self.max_cache_size:
            candidates = self.cache_policy.get_evict_candidates(
                self.dict, num_candidates=1
            )
            if not candidates:
                return None
            for candidate in candidates:
                evicted_size = self.dict[candidate].size
                try:
                    self.batched_remove([candidate], force=False)
                finally:
                    # remove drops the index before unlinking the file. An
                    # unlink error must not leave that removed entry charged.
                    if candidate not in self.dict:
                        self.current_cache_size -= evicted_size
        self.cache_policy.update_on_put(key)
        self.disk_worker.insert_put_task(key)
        self.current_cache_size += size
    memory_obj.ref_count_up()
    started = False

    def write():
        nonlocal started
        started = True
        self.async_save_bytes_to_disk(key, memory_obj, on_complete_callback)

    async def submit():
        try:
            await self.disk_worker.submit_task("put", write)
        except Exception:
            if not started:
                try:
                    _release_failed_state_put(self, key, size)
                finally:
                    memory_obj.ref_count_down()
            logger.exception("State disk write failed: key=%s", key)
            raise

    coro = submit()
    try:
        asyncio.run_coroutine_threadsafe(coro, self.loop)
    except BaseException:
        coro.close()
        try:
            _release_failed_state_put(self, key, size)
        finally:
            memory_obj.ref_count_down()
        raise


def _save_state_bytes(self, key, memory_obj, on_complete_callback) -> None:
    """Publish complete byte/layout metadata under the disk index lock."""
    size = memory_obj.get_size()
    published = False
    path = None
    try:
        meta = memory_obj.meta
        path = self._key_to_path(key)
        disk_meta = DiskCacheMetadata(
            path, size, meta.shape, meta.dtype, meta.cached_positions, meta.fmt, 0
        )
        disk_meta.shapes = list(meta.shapes)
        disk_meta.dtypes = list(meta.dtypes)
        disk_meta.state_checkpoint = deepcopy(meta.state_checkpoint)
        buffer = memory_obj.byte_array
        self.write_file(buffer, path)
        if os.path.getsize(path) != len(buffer):
            raise OSError("Incomplete state checkpoint disk write")
        with self.disk_lock:
            self.dict[key] = disk_meta
            published = True
            self.usage += size
        # Notification/statistics failures cannot invalidate a completed entry.
        try:
            self.stats_monitor.update_local_storage_usage(self.usage)
            if self.batched_msg_sender is not None:
                self.batched_msg_sender.add_kv_op(
                    op_type=OpType.ADMIT, key=key.chunk_hash
                )
        except Exception:
            logger.exception("State disk publication bookkeeping failed: key=%s", key)
    finally:
        try:
            if published:
                self.disk_worker.remove_put_task(key)
            else:
                if path is not None:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logger.exception(
                            "Failed to remove incomplete state file: %s", path
                        )
                _release_failed_state_put(self, key, size)
        finally:
            memory_obj.ref_count_down()
    if on_complete_callback is not None:
        try:
            on_complete_callback(key)
        except Exception:
            logger.exception("State disk completion callback failed: key=%s", key)


def local_disk_get_blocking(self, key) -> Optional[MemoryObj]:
    """Hold a temporary disk pin until get has produced an owned CPU object."""
    if not self.pin(key):
        return None
    try:
        assert _orig_get_blocking is not None
        return _orig_get_blocking(self, key)
    finally:
        self.unpin(key)


def _read_bytes_exact(self, buffer, path) -> None:
    """Use existing byte I/O conventions, rejecting missing or short payloads."""
    if self.use_odirect and len(buffer) % self.os_disk_bs == 0:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        file = os.fdopen(fd, "rb", buffering=0)
    else:
        file = open(path, "rb")
    with file:
        if os.fstat(file.fileno()).st_size != len(buffer):
            raise OSError("Incompatible disk cache byte length")
        if file.readinto(buffer) != len(buffer):
            raise OSError("Incomplete disk cache read")


def _allocate_from_disk_meta(
    local_cpu_backend: LocalCPUBackend,
    disk_meta,
    fmt: MemoryFormat,
    *,
    busy_loop: bool = True,
) -> Optional[MemoryObj]:
    shapes = getattr(disk_meta, "shapes", None)
    dtypes = getattr(disk_meta, "dtypes", None)
    if shapes and dtypes:
        return local_cpu_backend.allocate(
            shapes,
            dtypes,
            fmt,
            busy_loop=busy_loop,
        )
    return local_cpu_backend.allocate(
        disk_meta.shape,
        disk_meta.dtype,
        fmt,
        busy_loop=busy_loop,
    )


def local_disk_async_save_bytes_to_disk(
    self,
    key: CacheEngineKey,
    memory_obj: MemoryObj,
    on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
) -> None:
    """Publish state metadata before visibility; retain ordinary multi-group I/O."""
    if _is_state_meta(memory_obj.meta):
        return _save_state_bytes(self, key, memory_obj, on_complete_callback)
    assert _orig_async_save_bytes_to_disk is not None
    _orig_async_save_bytes_to_disk(
        self,
        key,
        memory_obj,
        on_complete_callback=on_complete_callback,
    )
    if is_multi_group_memory_obj(memory_obj):
        with self.disk_lock:
            disk_meta = self.dict.get(key)
            if disk_meta is not None:
                disk_meta.shapes = list(memory_obj.meta.shapes)
                disk_meta.dtypes = list(memory_obj.meta.dtypes)


def local_disk_load_bytes_from_disk(
    self,
    key: CacheEngineKey,
    path: str,
    dtype,
    shape,
    fmt: MemoryFormat,
) -> Optional[MemoryObj]:
    """Load bytes from disk, restoring multi-group allocation when present."""
    with self.disk_lock:
        disk_meta = self.dict[key]
    state = _is_state_meta(disk_meta)
    if ("state.domain", "checkpoint") in (key.tags or ()) and not state:
        return None
    memory_obj = _allocate_from_disk_meta(
        self.local_cpu_backend, disk_meta, fmt, busy_loop=False
    )
    if memory_obj is None:
        return None
    try:
        buffer = memory_obj.byte_array
        # Attention participates in the same hybrid restore. The upstream reader
        # ignores missing files and short reads, which can fabricate a token hit.
        _read_bytes_exact(self, buffer, path)
        if state:
            memory_obj.meta.state_checkpoint = deepcopy(disk_meta.state_checkpoint)
        memory_obj.metadata.cached_positions = disk_meta.cached_positions
        return memory_obj
    except BaseException:
        memory_obj.ref_count_down()
        raise


async def local_disk_batched_get_non_blocking(
    self,
    lookup_id: str,
    keys: list[CacheEngineKey],
    transfer_spec: Any = None,
) -> list[MemoryObj]:
    """Prefetch from disk with multi-group allocation when metadata exists."""
    mem_objs: list[MemoryObj] = []
    paths: list[str] = []

    for key in keys:
        self.disk_lock.acquire()
        assert key in self.dict, f"Key {key} not found in disk cache after pinning"

        disk_meta = self.dict[key]
        path = disk_meta.path
        fmt = disk_meta.fmt

        assert disk_meta.dtype is not None
        assert disk_meta.shape is not None

        memory_obj = _allocate_from_disk_meta(
            self.local_cpu_backend,
            disk_meta,
            fmt,
            busy_loop=False,
        )

        if memory_obj is None:
            self.disk_lock.release()
            return mem_objs

        self.dict[key].pin()
        self.cache_policy.update_on_hit(key, self.dict)
        self.disk_lock.release()

        memory_obj.pin()
        mem_objs.append(memory_obj)
        paths.append(path)

    return await self.disk_worker.submit_task(
        "prefetch",
        self.batched_async_load_bytes_from_disk,
        paths=paths,
        keys=keys,
        memory_objs=mem_objs,
    )
