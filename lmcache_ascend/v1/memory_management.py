# SPDX-License-Identifier: Apache-2.0
"""Ascend helpers for multi-group ``MemoryObj`` metadata."""

# Third Party
from lmcache.v1.memory_management import MemoryObj


def is_multi_group_memory_obj(memory_obj: MemoryObj) -> bool:
    """Return True when ``memory_obj`` spans more than one KV layer group."""
    return len(getattr(memory_obj, "group_prefix_sum", (0,))) > 2


def sync_group_prefix_sum(memory_obj: MemoryObj) -> None:
    """Rebuild ``group_prefix_sum`` from ``meta.shapes`` / ``meta.dtypes``.

    Upstream ``PagedTensorMemoryAllocator.allocate`` updates ``meta.shapes``
    when the request differs from the pool page layout but does not refresh
    ``group_prefix_sum`` (computed only in ``TensorMemoryObj.__init__``).
    Installed via ``_patch_paged_allocator_sync_group_prefix`` so every
    freelist allocate path refreshes prefixes.
    """
    meta = memory_obj.meta
    shapes = meta.shapes
    dtypes = meta.dtypes
    if shapes is None or dtypes is None:
        return
    prefix = [0]
    nbytes = 0
    for shape, dtype in zip(shapes, dtypes, strict=True):
        nbytes += int(shape.numel()) * dtype.itemsize
        prefix.append(nbytes)
    memory_obj.group_prefix_sum = prefix


def patch_mixed_allocator_binary():
    """Route independent state objects through the existing managed host pool."""
    # Third Party
    from lmcache.v1.memory_management import (
        MemoryFormat,
        MixedMemoryAllocator,
        PagedTensorMemoryAllocator,
        get_size_bytes,
    )

    original_allocate = MixedMemoryAllocator.allocate
    original_batched_allocate = MixedMemoryAllocator.batched_allocate
    original_free = MixedMemoryAllocator.free
    original_batched_free = MixedMemoryAllocator.batched_free

    def check_page(self, shapes, dtypes):
        if isinstance(self.pin_allocator, PagedTensorMemoryAllocator):
            shapes, dtypes = self.pin_allocator._adapt_shapes_and_dtypes(shapes, dtypes)
            if get_size_bytes(shapes, dtypes) > self.pin_allocator.align_bytes:
                raise ValueError("State BINARY payload exceeds managed pool page size")

    def allocate(self, shapes, dtypes, fmt=MemoryFormat.KV_2LTD, allocator_type=None):
        if fmt != MemoryFormat.BINARY:
            return original_allocate(self, shapes, dtypes, fmt, allocator_type)
        with self.host_mem_lock:
            check_page(self, shapes, dtypes)
            return self.pin_allocator.allocate(shapes, dtypes, fmt, str(self))

    def batched_allocate(
        self, shapes, dtypes, batch_size, fmt=MemoryFormat.KV_2LTD, allocator_type=None
    ):
        if fmt != MemoryFormat.BINARY:
            return original_batched_allocate(
                self, shapes, dtypes, batch_size, fmt, allocator_type
            )
        with self.host_mem_lock:
            check_page(self, shapes, dtypes)
            return self.pin_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
            )

    def free(self, memory_obj, allocator_type=None):
        if memory_obj.meta.fmt != MemoryFormat.BINARY:
            return original_free(self, memory_obj, allocator_type)
        with self.host_mem_lock:
            return self.pin_allocator.free(memory_obj)

    def batched_free(self, memory_objs, allocator_type=None, update_stats=True):
        if not memory_objs or memory_objs[0].meta.fmt != MemoryFormat.BINARY:
            return original_batched_free(
                self, memory_objs, allocator_type, update_stats
            )
        with self.host_mem_lock:
            return self.pin_allocator.batched_free(
                memory_objs, update_stats=update_stats
            )

    MixedMemoryAllocator.allocate = allocate
    MixedMemoryAllocator.batched_allocate = batched_allocate
    MixedMemoryAllocator.free = free
    MixedMemoryAllocator.batched_free = batched_free
