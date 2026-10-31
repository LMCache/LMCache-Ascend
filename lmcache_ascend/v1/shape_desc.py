# SPDX-License-Identifier: Apache-2.0
"""Ascend helpers for the plugin PageBufferShapeDesc plane extras.

CUDA's compiled struct is unchanged. After bind_native, factory/grouping
construct ``device_ops.PageBufferShapeDesc()`` (``c_ops``). NPU kernels
take that class by value.

The fmt 16/17 per-plane byte geometry is attached by upstream's
``make_page_buffer_shape_desc`` itself (it owns the format knowledge);
the plugin-side factory wrap was retired with that upstream change. The
``attach_tuple_*`` helpers below remain for callers that build a desc
by hand (e.g. tests).
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import Any

_MAX_PLANES = 4


def attach_tuple_planes(
    desc: Any,
    plane_slot_bytes: Sequence[int],
) -> None:
    """Record per-token plane widths (bytes) on ``desc``.

    ``plane_slot_bytes[i]`` is ``numel * itemsize // (nb * bs)`` for plane
    ``i``. Caps at ``_MAX_PLANES``; the packed kernel only accepts NP==2.
    """
    slots = tuple(int(b) for b in plane_slot_bytes)
    if len(slots) > _MAX_PLANES:
        raise ValueError(
            f"attach_tuple_planes: {len(slots)} planes exceeds max {_MAX_PLANES}"
        )
    desc.num_planes = len(slots)
    desc.plane_slot_bytes = slots


def attach_tuple_block_strides(
    desc: Any,
    plane_block_stride_bytes: Sequence[int],
) -> None:
    """Record per-block dim-0 byte strides on ``desc``.

    ``plane_block_stride_bytes[i]`` is ``stride(0) * itemsize`` for plane
    ``i``. Packed native uses plane 0/1; 0 means fall back to the shared
    ``block_stride_elems``.
    """
    strides = tuple(int(b) for b in plane_block_stride_bytes)
    if len(strides) > _MAX_PLANES:
        raise ValueError(
            "attach_tuple_block_strides: "
            f"{len(strides)} planes exceeds max {_MAX_PLANES}"
        )
    desc.plane_block_stride_bytes = strides
