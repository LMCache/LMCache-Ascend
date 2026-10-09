# SPDX-License-Identifier: Apache-2.0
"""Ascend helpers for the plugin PageBufferShapeDesc plane extras.

CUDA's compiled struct is unchanged. After bind_native, factory/grouping
construct ``device_ops.PageBufferShapeDesc()`` (``c_ops``). NPU kernels
take that class by value. Also hosts the wrap that attaches per-plane
byte geometry to upstream's desc factory (:func:`install_plane_geometry_fill`).
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence

_MAX_PLANES = 4


def attach_tuple_planes(
    desc: object,
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
    desc: object,
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


def install_plane_geometry_fill() -> None:
    """Wrap upstream ``make_page_buffer_shape_desc`` with plane geometry.

    Upstream's factory fills only the scalar fields; its compiled struct
    carries no plane arrays (the pybind class is ``dynamic_attr``), so the
    per-plane byte geometry both Ascend formats require is attached here,
    at the same construction point:

    - fmt 17 (``NL_X_NP_X_NB_BS_ONE_HS``): MLA/DSA planes
      ``[NB, BS, 1, HS]``;
    - fmt 16 (``NL_X_TWO_X_NB_BS_NH_HS``): per-layer ``(K, V)`` planes
      ``[NB, BS, NH, HS]``.

    Both share the per-layer plane-tuple structure: the per-token row is
    the product of the trailing two dims (``shape[2] * shape[3] *
    element_size``; ``shape[2] == 1`` for fmt 17, reducing to its plane
    width) and the per-block stride is ``stride(0) * element_size``.

    Upstream call sites (``kv_layer_groups`` and the MP transfer-context
    gather/scatter utilities) import the function lazily inside the call,
    so rebinding the module attribute takes effect regardless of import
    order.
    """
    # Third Party
    import lmcache.lmcache_native as lmcache_native
    import lmcache.v1.gpu_connector.utils as gpu_connector_utils

    original = gpu_connector_utils.make_page_buffer_shape_desc
    plane_tuple_fmts = (
        lmcache_native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS,
        lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS,
    )

    def make_page_buffer_shape_desc(
        kv_caches,
        engine_kv_format,
        layer_idx: int,
        num_layers_in_group: int,
        num_blocks: int,
        block_size: int,
        block_stride_elems=None,
    ):
        desc = original(
            kv_caches,
            engine_kv_format,
            layer_idx,
            num_layers_in_group,
            num_blocks,
            block_size,
            block_stride_elems,
        )
        if engine_kv_format in plane_tuple_fmts:
            planes = kv_caches[layer_idx]
            desc.num_planes = len(planes)
            desc.plane_slot_bytes = tuple(
                int(t.shape[2]) * int(t.shape[3]) * int(t.element_size())
                for t in planes
            )
            desc.plane_block_stride_bytes = tuple(
                int(t.stride(0)) * int(t.element_size()) for t in planes
            )
        return desc

    gpu_connector_utils.make_page_buffer_shape_desc = make_page_buffer_shape_desc
