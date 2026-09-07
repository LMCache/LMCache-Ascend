# SPDX-License-Identifier: Apache-2.0
"""Independent checkpoint allocation using LMCache's managed tensor pools."""

# Standard
from dataclasses import dataclass, field

# Third Party
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
import torch

# First Party
from lmcache_ascend.v1.memory_management import sync_group_prefix_sum
from lmcache_ascend.v1.state_layout import StateGroupLayout


@dataclass
class StateCheckpointBuffer:
    """Own one allocation reference; borrowed plane views require this owner alive.

    Use close() or a context manager to return the allocation to its pool. Closing
    does not make previously borrowed views safe to use; callers must stop using them.
    """

    layout: StateGroupLayout
    memory_obj: MemoryObj
    _planes: tuple[torch.Tensor, ...] = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def planes(self) -> tuple[torch.Tensor, ...]:
        """Borrow typed plane views while the allocation is owned and valid."""
        if self._released or not self.memory_obj.is_valid():
            raise RuntimeError("State checkpoint buffer has been released")
        return self._planes

    def close(self) -> None:
        """Release exactly the allocation reference owned by this buffer."""
        if not self._released:
            self._released = True
            self.memory_obj.ref_count_down()

    def __enter__(self) -> "StateCheckpointBuffer":
        _ = self.planes
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def allocate_state_checkpoint(
    layout: StateGroupLayout, allocator: MemoryAllocatorInterface
) -> StateCheckpointBuffer:
    """Allocate one payload from a tensor-backed pool accepting BINARY objects.

    Composite typed segments and explicit uint8 padding preserve payload offsets.
    This entry does not route through token-shaped StorageManager allocation.
    Raises MemoryError on pool exhaustion and releases allocations on view failure.
    """
    shapes, dtypes, plane_indices = [], [], []
    end = 0
    for plane in layout.planes:
        if plane.offset < end or plane.offset % layout.alignment:
            raise ValueError("Invalid state payload offsets/alignment")
        if plane.offset > end:
            shapes.append(torch.Size([plane.offset - end]))
            dtypes.append(torch.uint8)
        plane_indices.append(len(shapes))
        shapes.append(torch.Size(plane.shape))
        dtypes.append(plane.dtype)
        end = plane.offset + plane.nbytes
    if end != layout.nbytes:
        raise ValueError("State payload size does not match its planes")
    obj = allocator.allocate(shapes, dtypes, fmt=MemoryFormat.BINARY)
    if obj is None:
        raise MemoryError(
            f"Cannot allocate {layout.nbytes} bytes for a state checkpoint"
        )
    try:
        # Also supports reused composite metadata from the existing paged pool.
        sync_group_prefix_sum(obj)
        raw = obj.raw_tensor
        if raw is None or raw.numel() * raw.element_size() < layout.nbytes:
            raise ValueError(
                "Allocator returned an undersized/non-tensor state payload"
            )
        views = []
        for index, plane in zip(plane_indices, layout.planes, strict=True):
            view = obj.get_tensor(index)
            if (
                view is None
                or tuple(view.shape) != plane.shape
                or view.dtype != plane.dtype
                or view.data_ptr() != raw.data_ptr() + plane.offset
                or view.data_ptr() % layout.alignment
            ):
                raise ValueError(f"Invalid allocated view for state plane {plane.name}")
            views.append(view)
        return StateCheckpointBuffer(layout, obj, tuple(views))
    except Exception:
        obj.ref_count_down()
        raise
