# SPDX-License-Identifier: Apache-2.0
"""Ascend overrides for ``LocalDiskBackend`` multi-group disk save/load."""

# Standard
from typing import Any, Callable, Optional

# Third Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# First Party
from lmcache_ascend.v1.memory_management import is_multi_group_memory_obj

_orig_async_save_bytes_to_disk = None


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
    """Persist per-group shapes/dtypes after upstream disk write."""
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
    memory_obj = _allocate_from_disk_meta(self.local_cpu_backend, disk_meta, fmt)
    assert memory_obj is not None, "Memory allocation failed during disk load."

    buffer = memory_obj.byte_array
    self.read_file(key, buffer, path)

    cached_positions = disk_meta.cached_positions
    memory_obj.metadata.cached_positions = cached_positions

    return memory_obj


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
