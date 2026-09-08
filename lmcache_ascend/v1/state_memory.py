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


def state_checkpoint_metadata(layout: StateGroupLayout) -> dict:
    """Portable CPU/disk metadata; persist this field before publishing a disk key.

    Disk readers must restore ``meta.state_checkpoint`` before adoption, alongside
    plural shapes/dtypes (including padding). No process-local layout is required
    in storage metadata. The caller still supplies its expected runtime layout.
    """

    def portable(value):
        return [portable(item) for item in value] if isinstance(value, tuple) else value

    return {"group_index": layout.group_index, "layout": portable(layout.signature)}


def _state_segments(layout: StateGroupLayout):
    shapes, dtypes, plane_indices = [], [], []
    if layout.alignment <= 0 or not layout.planes or not layout.layer_names:
        raise ValueError("Invalid state payload layout")
    end = 0
    for plane in layout.planes:
        if (
            not plane.shape
            or plane.shape[0] != len(layout.layer_names)
            or any(size <= 0 for size in plane.shape)
            or plane.offset < end
            or plane.offset % layout.alignment
            or plane.offset % plane.dtype.itemsize
        ):
            raise ValueError("Invalid state payload offsets/alignment or shape")
        if plane.offset > end:
            shapes.append(torch.Size([plane.offset - end]))
            dtypes.append(torch.uint8)
        plane_indices.append(len(shapes))
        shapes.append(torch.Size(plane.shape))
        dtypes.append(plane.dtype)
        end = plane.offset + plane.nbytes
    if end != layout.nbytes:
        raise ValueError("State payload size does not match its planes")
    return shapes, dtypes, plane_indices


def adopt_state_checkpoint(
    layout: StateGroupLayout, owned_memory_obj: MemoryObj
) -> StateCheckpointBuffer:
    """Consume exactly one owned reference, releasing it on validation failure.

    Backend get returns that reference. Never pass a borrowed backend object.
    Success transfers it to the buffer with no retain, allocation or payload copy.
    """
    obj = owned_memory_obj
    try:
        shapes, dtypes, plane_indices = _state_segments(layout)
        if (
            not obj.is_valid()
            or obj.meta.fmt != MemoryFormat.BINARY
            or getattr(obj.meta, "state_checkpoint", None)
            != state_checkpoint_metadata(layout)
            or obj.meta.shapes != shapes
            or obj.meta.dtypes != dtypes
        ):
            raise ValueError("Incompatible state checkpoint metadata")
        sync_group_prefix_sum(obj)
        raw = obj.raw_tensor
        if (
            raw is None
            or raw.device.type != "cpu"
            or not raw.is_contiguous()
            or raw.numel() * raw.element_size() < layout.nbytes
            or obj.get_size() != layout.nbytes
        ):
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
                or not view.is_contiguous()
                or view.data_ptr() != raw.data_ptr() + plane.offset
                or view.data_ptr() % layout.alignment
            ):
                raise ValueError(f"Invalid allocated view for state plane {plane.name}")
            views.append(view)
        return StateCheckpointBuffer(layout, obj, tuple(views))
    except Exception:
        obj.ref_count_down()
        raise


def allocate_state_checkpoint(
    layout: StateGroupLayout,
    allocator: MemoryAllocatorInterface,
    *,
    busy_loop: bool | None = None,
) -> StateCheckpointBuffer:
    """Allocate a managed BINARY payload, including explicit padding segments."""
    shapes, dtypes, _ = _state_segments(layout)
    kwargs = {} if busy_loop is None else {"busy_loop": busy_loop}
    obj = allocator.allocate(shapes, dtypes, fmt=MemoryFormat.BINARY, **kwargs)
    if obj is None:
        raise MemoryError(
            f"Cannot allocate {layout.nbytes} bytes for a state checkpoint"
        )
    obj.meta.state_checkpoint = state_checkpoint_metadata(layout)
    return adopt_state_checkpoint(layout, obj)
