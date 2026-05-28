# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU bodies installed onto upstream ``KVLayerGroupsManager``
by :func:`lmcache_ascend.v1.kv_layer_groups.build_kv_layer_groups` (via
npu_connectors early init; ``_patch_kv_layer_group`` removed in __init__.py)."""

# Standard
from collections import defaultdict
from typing import Any, Optional, Sequence, Union

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
import torch

# First Party
from lmcache_ascend.v1.kv_format import (
    KVCacheFormat,
    _get_primary_blob_view,
    _is_multi_plane_tuple,
    _is_shared_storage_blob,
    _plane_block_size,
)
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

_LayerKV = Union[torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor]]


# Multi-plane layers (DSA/DSA-C8) have planes with different block sizes;
# we need per-plane byte widths to pack them into LMCache's flat chunk layout.
def _multi_plane_plane_bytes(kv_cache: Sequence[torch.Tensor]) -> list[int]:
    """Per-plane hidden bytes per paged slot for a multi-plane layer tuple."""
    out: list[int] = []
    for t in kv_cache:
        nb = int(t.shape[0])
        bs = _plane_block_size(t)
        slots = nb * bs
        if slots == 0:
            out.append(0)
            continue
        out.append(int(t.numel() * t.element_size()) // slots)
    return out


# Total chunk footprint per layer: sum of (hd * num_tokens) per plane.
# LMCache chunk sizes are multiples of 256, so each plane stride is
# already 32B-aligned; we assert rather than silently padding.
def _multi_plane_layer_block_bytes(
    plane_bytes: Sequence[int], num_tokens: int
) -> int:
    """Bytes per layer: sum of (hd * num_tokens) per plane block."""
    total = 0
    for hd in plane_bytes:
        plane_stride = hd * num_tokens
        assert plane_stride % 32 == 0, (
            f"Plane stride {plane_stride} (hd={hd} * num_tokens={num_tokens}) "
            f"is not 32B-aligned; chunk_size must be a multiple of 32"
        )
        total += plane_stride
    return total


# The LMCache chunk physical layout per layer is [plane0|plane1|...|planeP-1] where
# each plane block is (hd_p * num_tokens) bytes stored contiguously.
# A "row" is the per-token amortised width of the ENTIRE layer block (all planes),
# i.e. sum(hd_p).  Needed because upstream allocates chunks as
# [kv_size, nl, num_tokens, hidden] and `hidden = row_bytes`.
def _multi_plane_lmc_row_bytes(plane_bytes: Sequence[int], num_tokens: int) -> int:
    """LMCache chunk last dim (bytes): ceil(layer_block / num_tokens)."""
    if num_tokens <= 0:
        # Empty g_end chunk: kernel is not invoked; keep last dim positive for allocation.
        return max(1, sum(int(b) for b in plane_bytes))
    block = _multi_plane_layer_block_bytes(plane_bytes, num_tokens)
    return (block + num_tokens - 1) // num_tokens


# Upstream LMCache expects a single [num_blocks, block_size, hidden] shape per
# group; this collapses a multi-tensor tuple into that canonical 3-D form.
def _get_tuple_storage_shape(
    kv_cache: tuple[torch.Tensor, ...], *, is_310p: bool = False
) -> torch.Size:
    """Return the flattened LMCache storage shape for tuple-based KV caches.

    For MLA / DSA / DSA-C8, LMCache stores multiple KV tensors as a
    single contiguous hidden dimension, so we derive the flattened
    hidden size from the whole tuple instead of only looking at the
    first tensor. ``is_310p`` accounts for the head-packing layout
    where ``block_size`` lives at ``shape[-2]`` instead of ``shape[1]``.
    """
    first = kv_cache[0]
    num_blocks = int(first.shape[0])
    block_size = int(first.shape[-2]) if is_310p else int(first.shape[1])

    if (
        len(kv_cache) == 2
        and kv_cache[0].shape == kv_cache[1].shape
        and kv_cache[0].dtype == kv_cache[1].dtype
    ):
        # SEPARATE: K and V agree → single per-K shape.
        per_k = int(first.numel()) // (num_blocks * block_size)
        return torch.Size([num_blocks, block_size, per_k])

    total_hidden = 0
    for t in kv_cache:
        t_nb = int(t.shape[0])
        t_bs = int(t.shape[-2]) if is_310p else int(t.shape[1])
        if t_nb * t_bs == 0:
            continue
        total_hidden += int(t.numel()) // (t_nb * t_bs)
    return torch.Size([num_blocks, block_size, total_hidden])


# Shared-storage blobs expose multiple views over one allocation; only the
# primary view carries the true paging geometry, so we derive shape from it.
def _get_blob_storage_shape(
    kv_cache: Sequence[torch.Tensor],
    vllm_block_size: int,
    *,
    is_310p: bool = False,
) -> torch.Size:
    """Storage shape for a shared blob using the primary view's tensor shape."""
    del vllm_block_size
    primary = _get_primary_blob_view(kv_cache)
    num_blocks = int(primary.shape[0])
    block_size = int(primary.shape[-2]) if is_310p else int(primary.shape[1])
    hidden_per_token = int(primary.numel()) // (num_blocks * block_size)
    return torch.Size([num_blocks, block_size, hidden_per_token])


# Layers sharing a grouping key can ride one kernel launch; this extracts the
# 5-tuple identity (kv_size, hidden, block_size, dtype_key, num_tensors).
def _get_kv_cache_group_key_and_info(
    kv_cache: _LayerKV,
    *,
    is_310p: bool = False,
    vllm_block_size: Optional[int] = None,
) -> tuple[int, int, int, Any, int]:
    """Build a stable grouping key plus the LMCache storage shape/dtype.

    The fourth tuple element is ``torch.dtype`` when all tuple tensors share
    one dtype; otherwise it is the ``tuple`` of per-tensor dtypes (used for
    DSA-C8 mixed dtypes). ``PageBufferShapeDesc.element_size`` is set from the
    maximum tensor itemsize in :func:`build_kv_layer_groups`.
    """
    # Accept ``list`` too: upstream ``initialize_kvcaches_ptr`` relists every
    # per-layer tuple before grouping runs (see ``KVCacheFormat.detect``).
    if isinstance(kv_cache, (tuple, list)):
        if _is_multi_plane_tuple(kv_cache):
            # Grouping key: sum of per-slot plane widths (chunk-independent).
            plane_bytes = _multi_plane_plane_bytes(kv_cache)
            total_plane_bytes = sum(plane_bytes)
            primary_bs = _plane_block_size(kv_cache[0])
            return (1, total_plane_bytes, primary_bs, torch.uint8, len(kv_cache))

        if _is_shared_storage_blob(kv_cache):
            if vllm_block_size is None or int(vllm_block_size) <= 0:
                raise ValueError(
                    "Shared-storage KV blob layers require "
                    "layout_hints['vllm_block_size'] from the vLLM adapter."
                )
            primary = _get_primary_blob_view(kv_cache)
            _, bs, hidden_per_token = (
                int(x)
                for x in _get_blob_storage_shape(
                    kv_cache, int(vllm_block_size), is_310p=is_310p
                )
            )
            # kv_size=2 matches SEPARATE_KV LMCache chunk layout; num_tensors=0 marks blob.
            return (2, hidden_per_token, bs, primary.dtype, 0)

        dtypes = tuple(tensor.dtype for tensor in kv_cache)
        dtype_key: Any = dtypes[0] if len(set(dtypes)) == 1 else dtypes
        # NOTE(gingfung): Ascend tuple formats give per-tensor shape
        # (num_blocks, block_size, heads, headdim) on 910B; on 310P the
        # block_size moves to dim ``-2``. MLA / DSA / DSA-C8 may have
        # mismatched inner dims across tuple members.
        _, bs, hidden = (
            int(x) for x in _get_tuple_storage_shape(kv_cache, is_310p=is_310p)
        )
        is_separate = (
            len(kv_cache) == 2
            and kv_cache[0].shape == kv_cache[1].shape
            and kv_cache[0].dtype == kv_cache[1].dtype
        )
        kv_size = 2 if is_separate else 1
        return (kv_size, hidden, bs, dtype_key, len(kv_cache))

    if isinstance(kv_cache, torch.Tensor):
        # Flattened multi-spec / MLA page buffer: [num_blocks, block_size, hidden]
        # Only used when bundling is disabled (LMCACHE_ASCEND_BUNDLE_MULTI_SPEC=0).
        if kv_cache.ndim == 3:
            num_blocks = int(kv_cache.shape[0])
            bs = int(kv_cache.shape[1])
            hidden = int(kv_cache.shape[2])
            return (1, hidden, bs, kv_cache.dtype, 1)

        # MERGED_KV single tensor: 910B [2, NB, BS, NH, HS] / 310P
        # [2, NB, NH*HS//16, BS, 16]; second-form [NB, 2, BS, NH, HS]
        # also accepted (flash-infer).
        rep = kv_cache
        dtype = rep.dtype
        if is_310p:
            kv_size = int(rep.shape[0])
            num_blocks = int(rep.shape[1])
            bs = int(rep.shape[3])
        elif rep.ndim >= 5 and int(rep.shape[0]) == 2:
            kv_size, num_blocks, bs = 2, int(rep.shape[1]), int(rep.shape[2])
        elif rep.ndim >= 5 and int(rep.shape[1]) == 2:
            kv_size, num_blocks, bs = 2, int(rep.shape[0]), int(rep.shape[2])
        else:
            kv_size = max(1, int(rep.shape[0]))
            num_blocks = int(rep.shape[1]) if rep.ndim > 1 else 0
            bs = int(rep.shape[2]) if rep.ndim > 2 else 0
        denom = kv_size * num_blocks * bs
        hidden = int(rep.numel()) // denom if denom > 0 else 0
        return (kv_size, hidden, bs, dtype, 1)

    raise RuntimeError(f"Unknown KVCache type: {type(kv_cache)}")


def build_kv_layer_groups(
    self,
    kv_caches: Sequence[_LayerKV],
    *,
    kv_format: KVCacheFormat,
    num_blocks: int,
    is_310p: bool = False,
    layout_hints: Optional[dict] = None,
    lmcache_logical_chunk_size: int = 256,
) -> None:
    """Build KV layer groups by analyzing each layer's shape and dtype.

    Layers with the same ``(kv_size, hidden, block_size, dtype_key,
    num_tensors)`` are grouped together.  The grouping exists because the
    transfer kernel is parameterised by a single ``PageBufferShapeDesc``
    (introduced upstream in 167d6aeb — "[Refactor] single source of truth
    for Layer Group Metadata (#3078)"): all layers in one launch must
    share the same hidden width, block size, element size, and KV layout.
    Layers fall into *different* groups when
    they have different geometry — e.g. DeepSeek V4 attention layers vs.
    MLA/indexer layers (different hidden dims and block sizes), or int8
    quantised layers vs. fp16 layers (different dtypes/element sizes).
    Each group gets its own kernel launch with its own shape descriptor.

    ``dtype_key`` is either a single ``torch.dtype`` or a tuple of
    per-tensor dtypes for DSA+C8 mixed-type tuples.

    This override exists because upstream's grouping relies on
    ``GPUKVFormat``-keyed accessors (``get_num_heads``, ``get_block_size``,
    etc. in ``gpu_connector/utils.py``) that only understand standard CUDA
    tensor layouts.  Ascend's tuple-based, multi-plane, mixed-dtype, and
    shared-blob KV caches cannot be expressed through that enum, so we
    replace the format-discovery mechanism with direct tensor introspection
    via our own ``KVCacheFormat``.

    Body of ``KVLayerGroupsManager.__init__`` on the Ascend dispatch path
    (see npu_connectors early init).
    """
    self.kv_layer_groups = []
    self._vllm_block_size = (layout_hints or {}).get("vllm_block_size")
    self._kv_format = kv_format
    self.inference_engine_logical_block_size_: int | None = (
        layout_hints.get("inference_engine_logical_block_size")
        if layout_hints
        else None
    )

    if len(kv_caches) == 0:
        logger.debug("No KV caches available, skipping KV layer groups building")
        return

    # Group layers by key in a single loop
    groups_dict: dict[tuple, list[int]] = defaultdict(list)
    vllm_bs = self._vllm_block_size
    for idx, layer in enumerate(kv_caches):
        key = _get_kv_cache_group_key_and_info(
            layer, is_310p=is_310p, vllm_block_size=vllm_bs
        )
        groups_dict[key].append(idx)

    # Sort groups by the first layer index to maintain order
    def _get_first_layer_index(key):
        return groups_dict[key][0]

    sorted_keys = sorted(groups_dict.keys(), key=_get_first_layer_index)

    kv_layer_groups: list[KVLayerGroupInfo] = []
    for group_idx, key in enumerate(sorted_keys):
        kv_size, hidden, bs, dtype_key, _ = key
        indices = groups_dict[key]
        rep = kv_caches[indices[0]]

        shape_desc = lmc_ops.PageBufferShapeDesc()
        shape_desc.kv_size = kv_size
        shape_desc.nl = len(indices)
        shape_desc.nb = num_blocks
        shape_desc.bs = bs
        # ``nh=1, hs=hidden`` keeps upstream's
        # ``hidden_dim_size = nh * hs`` correct without overriding.
        shape_desc.nh = 1
        shape_desc.hs = hidden
        if isinstance(rep, (tuple, list)) and _is_shared_storage_blob(rep):
            shape_desc.element_size = int(
                _get_primary_blob_view(rep).element_size()
            )
        elif isinstance(rep, (tuple, list)):
            shape_desc.element_size = max(int(t.element_size()) for t in rep)
        else:
            shape_desc.element_size = rep.element_size
        stride_hint = (layout_hints or {}).get("block_stride_elems", 0) or 0
        if not stride_hint and isinstance(rep, (tuple, list)) and rep:
            first_t = rep[0]
            if first_t.dim() >= 1 and int(first_t.shape[0]) > 0:
                tight = int(first_t.numel()) // int(first_t.shape[0])
                s0 = int(first_t.stride(0))
                if s0 > tight:
                    stride_hint = s0
        shape_desc.block_stride_elems = int(stride_hint) if stride_hint else 0

        compress_ratio, physical_chunk_size = (
            KVLayerGroupsManager._derive_compression_metadata(
                group_idx=group_idx,
                bs=bs,
                ie_logical_block_size=self.inference_engine_logical_block_size_,
                lmcache_logical_chunk_size=lmcache_logical_chunk_size,
            )
        )
        multi_plane_hidden_bytes: tuple[int, ...] | None = None
        if isinstance(rep, (tuple, list)) and _is_multi_plane_tuple(rep):
            rep_dtype = torch.uint8
            plane_bytes = _multi_plane_plane_bytes(rep)
            shape_desc.hs = _multi_plane_lmc_row_bytes(
                plane_bytes, lmcache_logical_chunk_size
            )
            multi_plane_hidden_bytes = tuple(plane_bytes)
        elif isinstance(rep, (tuple, list)) and _is_shared_storage_blob(rep):
            rep_dtype = _get_primary_blob_view(rep).dtype
        elif isinstance(rep, (tuple, list)):
            rep_dtype = rep[0].dtype
        else:
            rep_dtype = rep.dtype
        group_info = KVLayerGroupInfo(
            layer_indices=indices,
            shape_desc=shape_desc,
            dtype=rep_dtype,
            compress_ratio=compress_ratio,
            physical_chunk_size=physical_chunk_size,
        )
        if multi_plane_hidden_bytes is not None:
            group_info.multi_plane_hidden_bytes = multi_plane_hidden_bytes
        kv_layer_groups.append(group_info)

    self.kv_layer_groups = kv_layer_groups
    self.inference_engine_logical_block_size_ = (
        self.inference_engine_logical_block_size_
        or self.kv_layer_groups[0].shape_desc.bs
    )
    logger.info("KV layer groups: %s", kv_layer_groups)
