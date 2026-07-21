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
