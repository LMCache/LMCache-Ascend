# SPDX-License-Identifier: Apache-2.0
"""Immutable plane-major checkpoint layouts, independent of request progress."""

# Standard
from dataclasses import dataclass
from math import lcm, prod
from typing import Sequence

# Third Party
import torch


@dataclass(frozen=True)
class StatePlaneLayout:
    """One typed plane across layers; source strides are in tensor elements."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    source_strides: tuple[tuple[int, ...], ...]

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize

    @property
    def block_stride_bytes(self) -> tuple[int, ...]:
        return tuple(stride[0] * self.dtype.itemsize for stride in self.source_strides)


@dataclass(frozen=True)
class StateGroupLayout:
    """One scheduler group's payload geometry, without addresses or availability."""

    group_index: int
    layer_names: tuple[str, ...]
    planes: tuple[StatePlaneLayout, ...]
    alignment: int
    nbytes: int
    version: int = 1

    @property
    def signature(self) -> tuple:
        """Payload compatibility; source strides and addresses are excluded."""
        return (
            self.version,
            self.layer_names,
            tuple((p.name, p.shape, str(p.dtype), p.offset) for p in self.planes),
            self.alignment,
            self.nbytes,
        )


def build_state_group_layout(
    group_index: int,
    layer_names: Sequence[str],
    tensors: Sequence[Sequence[torch.Tensor]],
) -> StateGroupLayout:
    """Pack validated GDN tensors by plane using natural dtype alignment.

    Each block's elements must be contiguous; gaps between runtime blocks are
    allowed. Layers must agree on each plane's shape/dtype. No tensors are copied.
    Raises ValueError for reordered elements or overlapping blocks.
    """
    if not layer_names or len(layer_names) != len(tensors):
        raise ValueError("State layout requires one tensor entry per layer")
    if any(len(entry) != 2 for entry in tensors):
        raise ValueError("State layout requires conv and SSM planes")
    alignment = lcm(*(tensor.element_size() for tensor in tensors[0]))
    offset = 0
    planes = []
    for index, name in enumerate(("conv", "ssm")):
        first = tensors[0][index]
        shape = tuple(first.shape[1:])
        if not shape or any(size <= 0 for size in shape):
            raise ValueError(f"State plane {name}: empty state shape")
        strides = []
        for layer_name, entry in zip(layer_names, tensors, strict=True):
            tensor = entry[index]
            if tuple(tensor.shape[1:]) != shape or tensor.dtype != first.dtype:
                raise ValueError(
                    f"State layer {layer_name}, plane {name}: layout mismatch"
                )
            if (
                tensor.shape[0] <= 0
                or not tensor[0].is_contiguous()
                or tensor.stride(0) < prod(shape)
            ):
                raise ValueError(
                    f"State layer {layer_name}, plane {name}: unsupported strides; "
                    "require contiguous elements and non-overlapping blocks"
                )
            strides.append(tuple(tensor.stride()))
        offset = (offset + alignment - 1) // alignment * alignment
        plane = StatePlaneLayout(
            name, (len(layer_names), *shape), first.dtype, offset, tuple(strides)
        )
        planes.append(plane)
        offset += plane.nbytes
    return StateGroupLayout(
        group_index, tuple(layer_names), tuple(planes), alignment, offset
    )
