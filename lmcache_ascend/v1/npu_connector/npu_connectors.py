# SPDX-License-Identifier: Apache-2.0
# Standard
import copy
import dataclasses
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

# Third Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.gpu_connector.gpu_connectors import (
    SGLangGPUConnector,
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
import torch

# First Party
from lmcache_ascend.v1.kv_format import (
    KVCacheFormat,
    _get_primary_blob_view,
    _is_shared_storage_blob,
)
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.transfer_context import AscendBaseTransferContext
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

_IS_310P = None


# MLA/DSA tuple formats use the v2 multi-layer kernel (interleaved LMCache rows).
# DSA-C8 uses ``multi_layer_kv_transfer_multi_plane`` (plane-block layout; scale bulk copy).
def _kv_tuple_formats_use_multi_layer_transfer(fmt: KVCacheFormat) -> bool:
    """Formats handled by ``multi_layer_kv_transfer_kernel_v2``, not single-layer."""
    return fmt in (KVCacheFormat.MLA_KV, KVCacheFormat.DSA_KV)


def _uses_multi_plane_kv_transfer(group_params: dict[str, Any]) -> bool:
    """True when group params describe a multi-plane (plane-block LMCache) transfer."""
    if group_params.get("kv_format") == KVCacheFormat.MULTI_PLANE_KV.value:
        return True
    if group_params.get("kv_format") == KVCacheFormat.DSA_C8_KV.value:
        return int(group_params.get("num_planes", 0)) > 0
    return int(group_params.get("num_planes", 0)) > 0


def _build_dsa_c8_multi_plane_group_params(
    plane_bytes: Sequence[int],
    *,
    block_size: int,
    page_buffer_size: int,
    num_tokens: int,
    per_plane_block_sizes: Optional[Sequence[int]] = None,
    per_plane_page_buffer_sizes: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Build multi-plane kernel params for DSA-C8 (plane-block uint8 LMCache chunk)."""
    from lmcache_ascend.v1.kv_layer_groups import _multi_plane_lmc_row_bytes

    pb = list(plane_bytes)
    if len(pb) != 4:
        raise ValueError(f"DSA-C8 expects 4 plane byte widths, got {len(pb)}")
    n = 4
    chunk_tokens = int(num_tokens)
    row_bytes = _multi_plane_lmc_row_bytes(pb, chunk_tokens)
    kb, vb, db, sb = pb
    bs_list = (
        [int(x) for x in per_plane_block_sizes]
        if per_plane_block_sizes is not None
        else [int(block_size)] * n
    )
    pbs_list = (
        [int(x) for x in per_plane_page_buffer_sizes]
        if per_plane_page_buffer_sizes is not None
        else [int(page_buffer_size)] * n
    )
    return {
        "kv_format": KVCacheFormat.DSA_C8_KV.value,
        "use_mla": True,
        "num_planes": n,
        "per_plane_hidden_dim_bytes": pb,
        "per_plane_block_sizes": bs_list,
        "per_plane_page_buffer_sizes": pbs_list,
        "scheduler_groups_per_plane": [0, 0, 0, 0],
        "dsa_c8_plane_bytes": tuple(pb),
        "block_size": int(block_size),
        "page_buffer_size": int(page_buffer_size),
        "k_extra": row_bytes,
        "v_extra": vb,
        "d_extra": db,
        "s_extra": sb,
    }


# KV caches may be plain tensors or tuples depending on the format; callers that
# only need device/dtype/shape from a representative tensor should not branch everywhere.
def _first_layer_tensor(
    kv_caches: List[Union[torch.Tensor, tuple, list]],
) -> torch.Tensor:
    first = kv_caches[0]
    if isinstance(first, (tuple, list)):
        return first[0]
    return first


# Stage-2 NPU grouping after upstream KVLayerGroupsManager (or build_kv_layer_groups): shape-only
# buckets can mix flat layers that map to different vLLM scheduler groups (DSv4: ~3 shape groups
# -> 6 NPU groups). Split so each group has one scheduler_slot_group for store/load.
# multi_layer_kv_transfer applies one slot index per token to every pointer in group_ptrs; without
# split, _scheduler_slot_group_for_npu_group raises when layer_indices span multiple sched groups.
# _invoke_multi_plane_kv_transfer slot concat does not replace this: perPlaneSlotOffsets are keyed
# by planeIdx within one logical layer (bundled tuple), not by flat layer index across SEPARATE_KV
# entries; num_layers>1 with num_planes=1 would reuse one slot stream for all layerIdx values.
# TODO(marco): per-layer slot offsets in multi_layer kernel + MemoryObj/get_shapes plane-packed vs
# nl-row layout if we want to fold some shape-merged groups without re-bundling flat tensors (§2.6.2).
def _split_kv_layer_groups_by_scheduler_slot(
    manager: Any,
    sched_map: Sequence[int],
) -> None:
    """Subdivide upstream groups when flattened layers share a shape but not a slot group."""
    from lmcache.v1.kv_layer_groups import KVLayerGroupInfo

    split_groups: list[KVLayerGroupInfo] = []
    for group in manager.kv_layer_groups:
        buckets: dict[int, list[int]] = defaultdict(list)
        for layer_idx in group.layer_indices:
            buckets[int(sched_map[layer_idx])].append(layer_idx)
        if len(buckets) == 1:
            split_groups.append(group)
            continue
        for sched_g in sorted(buckets):
            indices = buckets[sched_g]
            new_sd = copy.copy(group.shape_desc)
            new_sd.nl = len(indices)
            split_groups.append(
                dataclasses.replace(
                    group,
                    layer_indices=indices,
                    shape_desc=new_sd,
                )
            )
            logger.debug(
                "Split KV layer group by scheduler_slot_group=%d -> layers=%s",
                sched_g, indices,
            )
    manager.kv_layer_groups = split_groups


# Each scheduler group may address fewer slots than tokens when its block size
# is smaller than the logical block size; this translates the global token window
# into the correct slot_mapping slice for one group's kernel call.
def multi_plane_slot_slice_bounds(
    token_start: int,
    token_end: int,
    sched_g: int,
    compress_ratios: Sequence[int],
    sm_len: int,
) -> tuple[int, int]:
    """Map token range ``[token_start, token_end)`` to slot slice bounds for one group."""
    ratio = int(compress_ratios[sched_g]) if sched_g < len(compress_ratios) else 1
    if ratio <= 1:
        s0, s1 = int(token_start), min(int(token_end), sm_len)
    else:
        s0 = int(token_start) // ratio
        s1 = min((int(token_end) + ratio - 1) // ratio, sm_len)
    return s0, max(s0, s1)


# Mixed-format caches can contain non-tensor entries (e.g. Mamba state lists);
# we must gate kernel dispatch to avoid passing incompatible objects to the C++ layer.
def _is_kernel_compatible_entry(
    entry: Union[torch.Tensor, Tuple[torch.Tensor, ...], list],
) -> bool:
    """True if the entry is a standard KV cache format handled by transfer kernels."""
    if isinstance(entry, torch.Tensor):
        return entry.ndim >= 3
    if isinstance(entry, (tuple, list)):
        if len(entry) < 1:
            return False
        return all(
            isinstance(t, torch.Tensor) and t.ndim >= 3 for t in entry
        )
    return False


# The v2 kernel reads a flat int64 pointer array; each format packs a different
# number of planes per layer, so we need format-aware pointer extraction here.
def _pointers_for_entry(
    entry: Union[torch.Tensor, Tuple[torch.Tensor, ...], list],
    entry_format: KVCacheFormat,
    *,
    multi_plane: bool = False,
) -> list[int]:
    """Return device pointers for one layer entry in kernel-expected order."""
    if entry_format == KVCacheFormat.MULTI_PLANE_KV:
        return [t.data_ptr() for t in entry]
    if entry_format == KVCacheFormat.DSA_KV:
        k_cache, v_cache, dsa_k_cache = entry
        return [
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            dsa_k_cache.data_ptr(),
        ]
    if entry_format == KVCacheFormat.DSA_C8_KV:
        k_cache, v_cache, dsa_k_cache, dsa_k_scale = entry
        return [
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            dsa_k_cache.data_ptr(),
            dsa_k_scale.data_ptr(),
        ]
    if entry_format == KVCacheFormat.MLA_KV:
        k_cache, v_cache = entry
        return [k_cache.data_ptr(), v_cache.data_ptr()]
    if entry_format == KVCacheFormat.SEPARATE_KV:
        if isinstance(entry, torch.Tensor):
            return [entry.data_ptr()]
        if isinstance(entry, (tuple, list)) and _is_shared_storage_blob(entry):
            blob_ptr = entry[0].data_ptr()
            if multi_plane:
                return [_get_primary_blob_view(entry).data_ptr()]
            return [blob_ptr, blob_ptr]
        k_cache, v_cache = entry
        return [k_cache.data_ptr(), v_cache.data_ptr()]
    if isinstance(entry, torch.Tensor):
        return [entry.data_ptr()]
    raise ValueError(f"Unsupported KV cache entry type: {type(entry)}")


# Each KV layer group needs its own block_size, page_buffer_size, and hidden-dim
# byte counts passed to the transfer kernel; this centralises that per-format logic
# so _initialize_group_pointers_and_params doesn't duplicate it for every format.
def _derive_group_params(
    entry: Union[torch.Tensor, Tuple[torch.Tensor, ...], list],
    entry_format: KVCacheFormat,
    shape_desc: Any,
    *,
    logical_page_slots: Optional[int] = None,
    layout_hints: Optional[dict] = None,
    num_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Kernel parameters for one KV layer group."""
    nb = int(shape_desc.nb)
    bs = int(shape_desc.bs)
    params: dict[str, Any] = {
        "kv_format": entry_format.value,
        "use_mla": entry_format
        in (KVCacheFormat.MLA_KV, KVCacheFormat.DSA_KV, KVCacheFormat.DSA_C8_KV),
        "block_size": bs,
        "page_buffer_size": nb * bs,
        "k_extra": 0,
        "v_extra": 0,
        "d_extra": 0,
        "s_extra": 0,
    }

    if entry_format == KVCacheFormat.MULTI_PLANE_KV:
        # Plane-tail layout: dense hd bytes per token, hd*T per plane block.
        from lmcache_ascend.v1.kv_layer_groups import _multi_plane_lmc_row_bytes

        planes = list(entry)
        num_planes = len(planes)
        plane_block_sizes: list[int] = []
        plane_page_buffer_sizes: list[int] = []
        plane_hidden_bytes: list[int] = []
        sched_groups = (layout_hints or {}).get("layer_to_scheduler_groups", {})
        layer_name = (layout_hints or {}).get("_current_layer_name")
        sched_per_plane: list[int] = []
        if layer_name and layer_name in sched_groups:
            sched_per_plane = list(sched_groups[layer_name])
        elif layout_hints and "scheduler_groups_per_plane" in layout_hints:
            sched_per_plane = list(layout_hints["scheduler_groups_per_plane"])
        for t in planes:
            bs = int(t.shape[1])
            slots = int(t.shape[0]) * bs
            plane_block_sizes.append(bs)
            plane_page_buffer_sizes.append(slots)
            plane_hidden_bytes.append(
                int(t.numel() * t.element_size()) // slots if slots else 0
            )
        params["num_planes"] = num_planes
        params["per_plane_block_sizes"] = plane_block_sizes
        params["per_plane_page_buffer_sizes"] = plane_page_buffer_sizes
        params["per_plane_hidden_dim_bytes"] = plane_hidden_bytes
        params["scheduler_groups_per_plane"] = sched_per_plane
        params["block_size"] = plane_block_sizes[0]
        params["page_buffer_size"] = plane_page_buffer_sizes[0]
        chunk_tokens = int(num_tokens) if num_tokens is not None else 256
        params["k_extra"] = _multi_plane_lmc_row_bytes(
            plane_hidden_bytes, chunk_tokens
        )
    elif entry_format == KVCacheFormat.DSA_KV:
        k_cache, v_cache, dsa_k_cache = entry
        params["block_size"] = int(k_cache.shape[1])
        slots = logical_page_slots or (
            int(k_cache.shape[0]) * int(k_cache.shape[1])
        )
        params["page_buffer_size"] = slots
        params["k_extra"] = int(k_cache.shape[-1])
        params["v_extra"] = int(v_cache.shape[-1])
        params["d_extra"] = int(dsa_k_cache.shape[-1])
    elif entry_format == KVCacheFormat.DSA_C8_KV:
        k_cache, v_cache, dsa_k_cache, dsa_k_scale = entry
        planes = [k_cache, v_cache, dsa_k_cache, dsa_k_scale]
        block_size = int(k_cache.shape[1])
        slots = logical_page_slots or (
            int(k_cache.shape[0]) * int(k_cache.shape[1])
        )
        plane_block_sizes: list[int] = []
        plane_page_buffer_sizes: list[int] = []
        plane_hidden_bytes: list[int] = []
        for t in planes:
            bs_p = int(t.shape[1])
            slots_p = int(t.shape[0]) * bs_p
            plane_block_sizes.append(bs_p)
            plane_page_buffer_sizes.append(slots_p)
            plane_hidden_bytes.append(
                int(t.numel() * t.element_size()) // slots_p if slots_p else 0
            )
        chunk_tokens = int(num_tokens) if num_tokens is not None else 256
        params.update(
            _build_dsa_c8_multi_plane_group_params(
                plane_hidden_bytes,
                block_size=block_size,
                page_buffer_size=slots,
                num_tokens=chunk_tokens,
                per_plane_block_sizes=plane_block_sizes,
                per_plane_page_buffer_sizes=plane_page_buffer_sizes,
            )
        )
    elif entry_format == KVCacheFormat.MLA_KV:
        k_cache, v_cache = entry
        params["block_size"] = int(k_cache.shape[1])
        params["page_buffer_size"] = int(k_cache.shape[0]) * int(k_cache.shape[1])
        params["k_extra"] = int(k_cache.shape[-1])
        params["v_extra"] = int(v_cache.shape[-1])
    elif entry_format == KVCacheFormat.SEPARATE_KV and _is_shared_storage_blob(entry):
        primary = _get_primary_blob_view(entry)
        num_blocks = int(primary.shape[0])
        block_size = int(primary.shape[1])
        if block_size <= 0 or num_blocks <= 0:
            raise ValueError(
                "Shared-storage KV blob requires a positive block_size from "
                "the primary view tensor shape."
            )
        params["block_size"] = block_size
        params["page_buffer_size"] = num_blocks * block_size
        hidden_bytes = int(primary.numel() * primary.element_size()) // (
            num_blocks * block_size
        )
        params["k_extra"] = hidden_bytes
        params["v_extra"] = hidden_bytes
    elif entry_format == KVCacheFormat.SEPARATE_KV:
        if isinstance(entry, torch.Tensor):
            k_cache = entry
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            k_cache, v_cache = entry
        elif isinstance(entry, (tuple, list)) and len(entry) > 0:
            k_cache = entry[0]
        else:
            return params
        if is_310p():
            params["block_size"] = int(k_cache.shape[-2])
            params["page_buffer_size"] = int(k_cache.shape[0]) * params["block_size"]
        else:
            params["block_size"] = int(k_cache.shape[1])
            params["page_buffer_size"] = (
                int(k_cache.shape[0]) * int(k_cache.shape[1])
            )
        hidden_bytes = int(k_cache.shape[-1]) * k_cache.element_size()
        params["k_extra"] = hidden_bytes
        if (
            isinstance(entry, (tuple, list))
            and len(entry) >= 2
            and not _is_shared_storage_blob(entry)
        ):
            v_cache = entry[1]
            params["v_extra"] = int(v_cache.shape[-1]) * v_cache.element_size()
        else:
            params["v_extra"] = hidden_bytes
        if hidden_bytes > 0 and hidden_bytes < 16:
            # Per-token MTE copy is under 32 bytes; route to processScalePlane.
            params["s_extra"] = hidden_bytes
    elif entry_format == KVCacheFormat.MERGED_KV:
        tensor = entry
        if is_310p():
            params["block_size"] = int(tensor.shape[-2])
            params["page_buffer_size"] = int(tensor.shape[1]) * params["block_size"]
        else:
            params["page_buffer_size"] = int(tensor.shape[1]) * int(tensor.shape[2])
            params["block_size"] = int(tensor.shape[2])

    if (
        entry_format == KVCacheFormat.SEPARATE_KV
        and int(getattr(shape_desc, "kv_size", 2)) == 1
        and not (
            isinstance(entry, (tuple, list))
            and len(entry) >= 2
            and not _is_shared_storage_blob(entry)
            and int(params["v_extra"]) != int(params["k_extra"])
        )
    ):
        params["num_planes"] = 1
        params["per_plane_hidden_dim_bytes"] = [int(params["k_extra"])]
        params["per_plane_block_sizes"] = [int(params["block_size"])]
        params["per_plane_page_buffer_sizes"] = [int(params["page_buffer_size"])]

    stride_elems = int(getattr(shape_desc, "block_stride_elems", 0) or 0)
    if stride_elems > 0:
        params["page_buffer_size"] = nb * bs

    return params


def _init_mla_dsa_connector_dims(
    connector: Any,
    entry: Union[torch.Tensor, tuple, list],
    kv_format: KVCacheFormat,
    *,
    logical_page_slots: Optional[int] = None,
) -> Any:
    """Derive MLA/DSA/DSA-C8 connector fields from the first layer via ``_derive_group_params``."""
    k0 = entry[0] if isinstance(entry, (tuple, list)) else entry
    sd = SimpleNamespace(
        nb=int(k0.shape[0]), bs=int(k0.shape[1]), block_stride_elems=0
    )
    params = _derive_group_params(
        entry, kv_format, sd, logical_page_slots=logical_page_slots
    )
    connector.block_size = int(params["block_size"])
    connector.page_buffer_size = int(params["page_buffer_size"])
    k_cache, v_cache = entry[0], entry[1]
    connector.kv_lora_rank = int(k_cache.shape[-1])
    connector.qk_rope_head_dim = int(v_cache.shape[-1])
    connector.dsa_head_dim = (
        int(entry[2].shape[-1]) if len(entry) >= 3 else 0
    )
    connector.dsa_c8_plane_bytes = (
        tuple(params["dsa_c8_plane_bytes"])
        if kv_format == KVCacheFormat.DSA_C8_KV
        else (0, 0, 0, 0)
    )
    return k0.shape


# The layerwise connector transfers one layer at a time and needs a device-resident
# pointer tensor for that single layer; this builds it without the full flat table.
def _layer_paged_kv_ptrs_tensor(
    layer_cache: Union[torch.Tensor, tuple, list], kv_format: KVCacheFormat
) -> torch.Tensor:
    """Build a 1-D int64 pointer tensor on the same device as the paged KV tensors."""
    if kv_format in (
        KVCacheFormat.MLA_KV,
        KVCacheFormat.DSA_KV,
        KVCacheFormat.DSA_C8_KV,
    ):
        if not isinstance(layer_cache, (tuple, list)):
            raise ValueError(
                f"_layer_paged_kv_ptrs_tensor: {kv_format.name} expects a tuple entry"
            )
        ptrs = _pointers_for_entry(layer_cache, kv_format)
        dev = layer_cache[0].device
        return torch.tensor(ptrs, dtype=torch.int64, device=dev)
    if kv_format == KVCacheFormat.SEPARATE_KV and _is_shared_storage_blob(layer_cache):
        blob_ptr = layer_cache[0].untyped_storage().data_ptr()
        dev = layer_cache[0].device
        return torch.tensor(
            [blob_ptr, blob_ptr],
            dtype=torch.int64,
            device=dev,
        )
    raise ValueError(f"_layer_paged_kv_ptrs_tensor: unexpected format {kv_format}")


def is_310p():
    global _IS_310P
    if _IS_310P is None:
        # First Party
        from lmcache_ascend import _build_info

        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P


class _V2KVTransferMixin:
    """Shared v2 kernel plane extras and staging hidden dim for paged NPU connectors."""

    kv_format: KVCacheFormat
    kv_lora_rank: int
    qk_rope_head_dim: int
    dsa_head_dim: int
    dsa_c8_plane_bytes: tuple[int, int, int, int]

    @property
    def v2_plane_extras(self) -> tuple[int, int, int, int]:
        if self.kv_format == KVCacheFormat.DSA_C8_KV:
            return self.dsa_c8_plane_bytes
        return (self.kv_lora_rank, self.qk_rope_head_dim, self.dsa_head_dim, 0)

    @property
    def v2_staging_hidden_dim(self) -> int:
        if self.kv_format == KVCacheFormat.DSA_C8_KV:
            # Plane-block row width; get_shape recomputes with num_tokens for alignment.
            return sum(self.dsa_c8_plane_bytes)
        if self.kv_format == KVCacheFormat.DSA_KV:
            return self.kv_lora_rank + self.qk_rope_head_dim + self.dsa_head_dim
        return self.kv_lora_rank + self.qk_rope_head_dim

    def _invoke_multi_plane_kv_transfer(
        self,
        *,
        mem_tensor: torch.Tensor,
        group_ptrs: torch.Tensor,
        group_params: dict[str, Any],
        slot_mappings_by_group: Union[tuple[torch.Tensor, ...], list[torch.Tensor]],
        compress_ratios: tuple[int, ...],
        g_start: int,
        g_end: int,
        is_store: bool,
        npu_group_idx: int,
    ) -> None:
        """One logical kernel op per bundled multi-spec layer (per-plane transfers)."""
        if (
            mem_tensor.dtype != torch.uint8
            and int(group_params.get("num_planes", 0)) > 0
        ):
            mem_tensor = mem_tensor.view(torch.uint8)
        num_planes = int(group_params["num_planes"])
        sched_groups = group_params.get("scheduler_groups_per_plane") or []
        if len(sched_groups) != num_planes:
            sched_groups = [int(group_params.get("scheduler_slot_group", 0))] * num_planes

        sm_parts: list[torch.Tensor] = []
        for sched_g in sched_groups:
            if sched_g >= len(slot_mappings_by_group):
                raise IndexError(
                    f"MULTI_PLANE: scheduler group {sched_g} out of range "
                    f"(num={len(slot_mappings_by_group)}) for NPU group {npu_group_idx}"
                )
            sm = slot_mappings_by_group[sched_g]
            s0, s1 = multi_plane_slot_slice_bounds(
                g_start, g_end, sched_g, compress_ratios, int(sm.shape[0])
            )
            sm_parts.append(sm[s0:s1])

        slot_concat = torch.cat(sm_parts, dim=0)
        offsets = torch.zeros(
            num_planes + 1, dtype=torch.int32, device=slot_concat.device
        )
        cursor = 0
        for i, part in enumerate(sm_parts):
            offsets[i] = cursor
            cursor += int(part.shape[0])
        offsets[num_planes] = cursor

        pbs = torch.tensor(
            group_params["per_plane_page_buffer_sizes"],
            dtype=torch.int32,
            device=self.kvcaches_device,
        )
        bss = torch.tensor(
            group_params["per_plane_block_sizes"],
            dtype=torch.int32,
            device=self.kvcaches_device,
        )
        plane_hidden_bytes = group_params["per_plane_hidden_dim_bytes"]
        hds = torch.tensor(
            plane_hidden_bytes,
            dtype=torch.int32,
            device=self.kvcaches_device,
        )
        max_hidden_dim_bytes = max(plane_hidden_bytes)

        lmc_ops.multi_layer_kv_transfer_multi_plane(
            mem_tensor,
            group_ptrs,
            slot_concat,
            offsets,
            pbs,
            bss,
            hds,
            max_hidden_dim_bytes,
            self.kvcaches_device,
            is_store,
            num_planes,
        )


class VLLMBufferLayerwiseNPUConnector(VLLMBufferLayerwiseGPUConnector):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_dim_size, num_layers, use_gpu, use_double_buffer, **kwargs
        )
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED
        self.use_mla = bool(kwargs.get("use_mla", False))
        self.fused_rotary_emb: Any = None
        # Populated for MLA/DSA/DSA-C8 buffer sizing (transfer uses paged layerwise / V2).
        self.kv_lora_rank: int = 0
        self.qk_rope_head_dim: int = 0
        self.dsa_head_dim: int = 0
        self.dsa_c8_plane_bytes: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.page_buffer_size: int = 0
        self.block_size: int = 0

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)
            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError("Could not detect KV cache format.")

            ref_tensor = (
                kv_caches[0][0] if self.kv_format.is_tuple_format() else kv_caches[0]
            )
            self.kv_device = ref_tensor.device

            first_layer_cache = kv_caches[0]

            # flash attention: [num_layers, 2, num_blocks,
            # block_size, num_heads, head_size]
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]

                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )

                k_cache_shape_per_layer = key_tensor.shape

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )

                # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
                k_cache_shape_per_layer = first_layer_cache[0].shape
            elif self.kv_format in (
                KVCacheFormat.MLA_KV,
                KVCacheFormat.DSA_KV,
                KVCacheFormat.DSA_C8_KV,
            ):
                k_cache_shape_per_layer = _init_mla_dsa_connector_dims(
                    self, first_layer_cache, self.kv_format
                )
            else:
                raise ValueError(f"Unsupported KV cache format: {self.kv_format}")

            self.vllm_two_major = True

            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            if self.kv_format in (
                KVCacheFormat.MLA_KV,
                KVCacheFormat.DSA_KV,
                KVCacheFormat.DSA_C8_KV,
            ):
                if not isinstance(first_layer_cache, (tuple, list)):
                    raise TypeError("Expected tuple/list KV cache for MLA/DSA formats.")
                gpu_buffer_size = sum(
                    int(t.numel()) * int(t.element_size()) for t in first_layer_cache
                )
            else:
                num_elements = k_cache_shape_per_layer.numel() * 2
                gpu_buffer_size = num_elements * self.element_size

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024)} MB"
            )

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def _prepare_transfer_context(self, kwargs) -> torch.Tensor:
        """
        Initialize context for KV cache transfer, validate required
        parameters and lazy init buffer.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise ValueError("kvcaches should be provided in kwargs or initialized.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        self._lazy_initialize_buffer(self.kvcaches)
        return kwargs["slot_mapping"]

    def _get_full_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        starts: List[int],
        ends: List[int],
        mode: str = "slice",
    ) -> tuple[torch.Tensor, int]:
        """
        Generate full continuous slot mapping tensor and calculate total token count.
        Supports two modes for different transfer directions (to/from GPU).
        """
        if mode == "slice":
            slot_mapping_full = slot_mapping[starts[0] : ends[-1]]
        elif mode == "concat":
            slot_mapping_chunks = [
                slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
            ]
            slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        else:
            raise ValueError(
                f"Unsupported slot mapping mode: {mode}, only 'slice'/'concat' allowed"
            )

        num_tokens = len(slot_mapping_full)
        return slot_mapping_full, num_tokens

    def _allocate_gpu_buffers(
        self, num_tokens: int, count: int = 1
    ) -> Union[object, list[object]]:
        """
        Allocate specified number of GPU buffers for KV cache with shape
        calculated by token count. Performs strict assertion checks for
        valid buffer allocation.
        """
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None, (
            "GPU buffer allocator not initialized"
        )
        buffers = []
        for _ in range(count):
            buf_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_2TD
            )
            assert buf_obj is not None, "Failed to allocate GPU buffer in GPUConnector"
            assert buf_obj.tensor is not None, "GPU buffer object has no valid tensor"
            buffers.append(buf_obj)
        return buffers[0] if count == 1 else buffers

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """
        slot_mapping = self._prepare_transfer_context(kwargs)

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping_full, num_all_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="slice"
        )

        if _kv_tuple_formats_use_multi_layer_transfer(self.kv_format):
            raise NotImplementedError(
                "VLLMBufferLayerwiseNPUConnector does not support MLA/DSA/DSA-C8 "
                "(single-layer kernels accept only MERGED_KV/SEPARATE_KV). Use "
                "VLLMPagedMemLayerwiseNPUConnector or VLLMPagedMemNPUConnectorV2."
            )

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]
        load_gpu_buffer_obj: Any = None
        compute_gpu_buffer_obj: Any = None
        compute_gpu_buffer_obj, load_gpu_buffer_obj = self._allocate_gpu_buffers(
            num_all_tokens, count=2
        )

        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kv_device
            )
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kv_device
            )

        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.cuda.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        slot_mapping = self._prepare_transfer_context(kwargs)

        if _kv_tuple_formats_use_multi_layer_transfer(self.kv_format):
            raise NotImplementedError(
                "VLLMBufferLayerwiseNPUConnector does not support MLA/DSA/DSA-C8 "
                "(single-layer kernels accept only MERGED_KV/SEPARATE_KV). Use "
                "VLLMPagedMemLayerwiseNPUConnector or VLLMPagedMemNPUConnectorV2."
            )

        buf_start = 0
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(start, end, device=self.kv_device, dtype=torch.int64)
                )

        slot_mapping_full, num_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="concat"
        )

        tmp_gpu_buffer_obj = self._allocate_gpu_buffers(num_tokens, count=1)

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )

                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield


class VLLMPagedMemNPUConnectorV2(_V2KVTransferMixin, VLLMPagedMemGPUConnectorV2):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        # Initialize kv_format before calling super().__init__
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

        # Initialize MLA/DSA parameters
        self.kv_lora_rank: int = 0
        self.qk_rope_head_dim: int = 0
        self.dsa_head_dim: int = 0
        # Per-token byte spans for DSA_C8_KV (k, v, dsa_k, dsa_k_scale); lmc_chunk is uint8
        self.dsa_c8_plane_bytes: tuple[int, int, int, int] = (0, 0, 0, 0)
        # vLLM logical slots (nb*bs) when ``block_stride_elems`` is set on the group
        self._logical_page_slots: Optional[int] = None
        # Per-group NPU pointer tables and kernel params (multi-spec multi-group).
        self.group_kv_cache_pointers: Optional[list[torch.Tensor]] = None
        self.per_group_params: Optional[list[dict[str, Any]]] = None
        # True when per-layer entries have different tuple lengths (multi-spec mixed).
        self._is_mixed_format: bool = False

        # Set by ``from_metadata``. ``ensure_kv_layer_groups`` builds
        # ``metadata.kv_layer_groups_manager`` at KV registration time;
        # ``_initialize_pointers`` only builds pointer tables on first
        # store/retrieve. ``None`` for direct ``__init__`` callers
        # (unit tests, MP server) → legacy single-group fallback.
        self.metadata: Optional[LMCacheMetadata] = None

        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)

        if is_310p():
            assert "num_kv_head" in kwargs, ("num_kv_head should be provided in 310p",)
            assert "head_size" in kwargs, ("head_size should be provided in 310p",)
            self.num_kv_head = kwargs["num_kv_head"]
            self.head_size = kwargs["head_size"]
            self.dtype = kwargs["dtype"]
            self.device = kwargs["device"]

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional KV layout hints from the serving engine.

        Returns:
            A new instance of VLLMPagedMemNPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        connector = cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            num_kv_head=num_kv_head,
            head_size=head_size,
            layout_hints=layout_hints,
        )
        connector.metadata = metadata
        return connector

    # The NPU group index does not correspond 1:1 to the vLLM scheduler group
    # index (e.g. 6 NPU groups vs 11 scheduler groups on DSv4). Each kernel
    # call needs the correct slot_mapping, so we resolve the scheduler group
    # from the per-layer map and verify the flatten contract (one sched group
    # per NPU group).
    def _scheduler_slot_group_for_npu_group(
        self,
        group_idx: int,
        layer_indices: list[int],
    ) -> int:
        hints = self.layout_hints or {}
        sched_map = hints.get("scheduler_group_by_flat_layer")
        if sched_map is not None:
            candidates = {int(sched_map[i]) for i in layer_indices}
            if len(candidates) != 1:
                raise ValueError(
                    f"NPU group {group_idx} spans multiple scheduler groups "
                    f"{candidates}; flatten contract violated"
                )
            return next(iter(candidates))
        primary = int(hints.get("primary_kv_group_idx", 0))
        return primary

    def _compress_ratios_by_group(self) -> tuple[int, ...]:
        hints = self.layout_hints or {}
        ratios = hints.get("compress_ratios_by_group")
        if ratios:
            return tuple(int(r) for r in ratios)
        return (1,)

    # Called from register_kv_caches (adapter) so the group manager exists
    # before post_init / first store; without it, metadata.get_shapes() has
    # no group info and allocates a single MemoryObj instead of per-group slots.
    def ensure_kv_layer_groups(
        self, kv_caches: List[torch.Tensor]
    ) -> None:
        """Build ``metadata.kv_layer_groups_manager`` for allocation sizing.

        Idempotent. Must run before ``metadata.get_shapes()`` is used to
        allocate ``MemoryObj`` (i.e. before ``post_init`` / first store).
        Pointer tables remain lazy in ``_initialize_pointers``.
        """
        if self.metadata is None or self.metadata.kv_layer_groups_manager is not None:
            self._sync_logical_page_slots_from_manager()
            return

        # Third Party
        from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
        from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
        from lmcache.utils import EngineType

        from lmcache_ascend.v1.kv_layer_groups import build_kv_layer_groups

        first_entry = kv_caches[0]
        if isinstance(first_entry, torch.Tensor):
            num_blocks = int(first_entry.shape[0])
        elif isinstance(first_entry, (tuple, list)):
            num_blocks = int(first_entry[0].shape[0])
        else:
            num_blocks = int(kv_caches[0].shape[1])
        hints = self.layout_hints or {}
        # Bundled multi-spec layers keep 4-D sub-tensors that upstream
        # normalize_kv_and_discover_format cannot detect (it only handles
        # tensor_dim 3 or 5). Use the Ascend-local build_kv_layer_groups
        # which derives sizes dimensionality-agnostically instead.
        if hints.get("bundle_multi_spec"):
            mgr = KVLayerGroupsManager.__new__(KVLayerGroupsManager)
            build_kv_layer_groups(
                mgr,
                kv_caches,
                kv_format=KVCacheFormat.detect(kv_caches, use_mla=self.use_mla),
                num_blocks=num_blocks,
                is_310p=is_310p(),
                layout_hints=hints,
                lmcache_logical_chunk_size=self.metadata.chunk_size,
            )
            self.metadata.kv_layer_groups_manager = mgr
        else:
            gpu_kv_format, normalized = normalize_kv_and_discover_format(
                kv_caches,
                serving_engine=EngineType.VLLM,
                layout_hints=hints,
            )
            self.metadata.kv_layer_groups_manager = KVLayerGroupsManager(
                normalized,
                gpu_kv_format=gpu_kv_format,
                num_blocks=num_blocks,
                layout_hints=hints,
                lmcache_logical_chunk_size=self.metadata.chunk_size,
            )
        sched_map = hints.get("scheduler_group_by_flat_layer")
        if sched_map is not None:
            _split_kv_layer_groups_by_scheduler_slot(
                self.metadata.kv_layer_groups_manager,
                sched_map,
            )
        self._sync_logical_page_slots_from_manager()

    def _sync_logical_page_slots_from_manager(self) -> None:
        if (
            self.metadata is not None
            and self.metadata.kv_layer_groups_manager is not None
            and self.metadata.kv_layer_groups_manager.kv_layer_groups
        ):
            sd = self.metadata.kv_layer_groups_manager.kv_layer_groups[0].shape_desc
            if getattr(sd, "block_stride_elems", 0) > 0:
                self._logical_page_slots = int(sd.nb) * int(sd.bs)

    # For each NPU group, build a device-resident pointer tensor and kernel params.
    def _initialize_group_pointers_and_params(
        self, kv_caches: List[torch.Tensor]
    ) -> None:
        """Build per-group pointer tensors and kernel params for multi-group store."""
        klg_manager = (
            self.metadata.kv_layer_groups_manager
            if self.metadata is not None
            else None
        )
        if klg_manager is None or not klg_manager.kv_layer_groups:
            self.group_kv_cache_pointers = None
            self.per_group_params = None
            return

        group_pointers: list[torch.Tensor] = []
        group_params: list[dict[str, Any]] = []
        for group_idx, group in enumerate(klg_manager.kv_layer_groups):
            indices = group.layer_indices
            rep = kv_caches[indices[0]]
            if not _is_kernel_compatible_entry(rep):
                raise ValueError(
                    f"NPU KV layer group {group_idx} has non-transferable layout "
                    f"(layer {indices[0]}); expected flattened single-tensor entries"
                )
            entry_format = KVCacheFormat.detect([rep], use_mla=self.use_mla)
            multi_plane_ptrs = (
                entry_format == KVCacheFormat.SEPARATE_KV
                and int(group.shape_desc.kv_size) == 1
            )
            ptrs: list[int] = []
            for layer_idx in indices:
                ptrs.extend(
                    _pointers_for_entry(
                        kv_caches[layer_idx],
                        entry_format,
                        multi_plane=multi_plane_ptrs,
                    )
                )
            cpu_ptrs = torch.empty(len(ptrs), dtype=torch.int64, device="cpu")
            cpu_ptrs.numpy()[:] = ptrs
            gpu_ptrs = torch.empty(
                len(ptrs), dtype=torch.int64, device=self.kvcaches_device
            )
            gpu_ptrs.copy_(cpu_ptrs)
            group_pointers.append(gpu_ptrs)
            layer_hints = dict(self.layout_hints or {})
            flat_names = layer_hints.get("flat_layer_names")
            if flat_names and indices:
                layer_hints["_current_layer_name"] = flat_names[indices[0]]
            chunk_tokens = (
                int(self.metadata.chunk_size)
                if self.metadata is not None
                else None
            )
            params = _derive_group_params(
                rep,
                entry_format,
                group.shape_desc,
                logical_page_slots=self._logical_page_slots,
                layout_hints=layer_hints,
                num_tokens=chunk_tokens,
            )
            params["scheduler_slot_group"] = self._scheduler_slot_group_for_npu_group(
                group_idx, indices
            )
            params["layer_indices"] = list(indices)
            group_params.append(params)

        self.group_kv_cache_pointers = group_pointers
        self.per_group_params = group_params
        logger.info(
            "Initialized %d per-group KV cache pointer tables for multi-group store",
            len(group_pointers),
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)

        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError(
                "Undefined KV cache format detected. "
                "Unable to determine the format of input kv_caches."
            )

        ref_tensor = _first_layer_tensor(kv_caches)
        self.kvcaches_device = ref_tensor.device

        assert self.kvcaches_device.type == "npu", "The device should be Ascend NPU."
        idx = self.kvcaches_device.index

        self.ensure_kv_layer_groups(kv_caches)

        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]

        # Detect mixed-format entries (e.g. DSA 3-tuples alongside Mamba
        # 8-element state lists). A flat pointer table assumes one
        # KVCacheFormat for every layer, which is impossible here.
        tuple_lens = set()
        for entry in kv_caches:
            if isinstance(entry, (tuple, list)):
                tuple_lens.add(len(entry))
            else:
                tuple_lens.add(0)
        is_mixed_format = len(tuple_lens) > 1
        self._is_mixed_format = is_mixed_format

        # Mixed format: skip flat pointer table entirely; only per-group
        # pointers are used for store/retrieve.
        if is_mixed_format:
            logger.info(
                "Mixed KV entry lengths %s detected; skipping flat pointer "
                "table (per-group pointers will be used instead)",
                sorted(tuple_lens),
            )
            self.kv_cache_pointers = torch.empty(0, dtype=torch.int64, device="cpu")
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                0, dtype=torch.int64, device=self.kvcaches_device
            )
            self.page_buffer_size = 0
            self.block_size = 0
            self.kv_lora_rank = 0
            self.qk_rope_head_dim = 0
            self.dsa_head_dim = 0
            self.dsa_c8_plane_bytes = (0, 0, 0, 0)
            self._initialize_group_pointers_and_params(kv_caches)
            return self.kv_cache_pointers_on_gpu[idx]

        # --- Uniform format from here on: build flat pointer table ---
        self.kv_size = self.kv_format.get_kv_size()
        pointers_list: list[int] = []

        if self.kv_format == KVCacheFormat.SEPARATE_KV:
            if kv_caches and isinstance(kv_caches[0], torch.Tensor):
                # Flattened multi-spec page buffers: one tensor per logical layer.
                self.kv_size = 1
                flat_layer_count = len(kv_caches)
                if self.num_layers != flat_layer_count:
                    logger.info(
                        "Adjusting connector num_layers from metadata count %d "
                        "to flat KV cache count %d (post-flatten sub-layers)",
                        self.num_layers,
                        flat_layer_count,
                    )
                    self.num_layers = flat_layer_count
            else:
                self.kv_size = 2
        elif self.kv_format == KVCacheFormat.MERGED_KV:
            self.kv_size = 1
            flat_layer_count = len(kv_caches)
            if self.num_layers != flat_layer_count:
                logger.info(
                    "Adjusting connector num_layers from metadata count %d "
                    "to flat KV cache count %d",
                    self.num_layers,
                    flat_layer_count,
                )
                self.num_layers = flat_layer_count

        for entry in kv_caches:
            pointers_list.extend(_pointers_for_entry(entry, self.kv_format))

        self.kv_cache_pointers = torch.empty(
            len(pointers_list), dtype=torch.int64, device="cpu"
        )
        self.kv_cache_pointers.numpy()[:] = pointers_list

        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.kv_cache_pointers.shape, dtype=torch.int64, device=self.kvcaches_device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        # --- Extract connector-level kernel params from first layer ---
        first_entry = kv_caches[0]
        if isinstance(first_entry, (tuple, list)) and _is_shared_storage_blob(
            first_entry
        ):
            # Shared-storage blob: multiple dtype views over one allocation.
            first_tensor = first_entry[0]
        elif self.kv_format.is_tuple_format():
            # MLA/DSA/DSA-C8/SEPARATE tuple: first element is the k_cache tensor.
            first_tensor = kv_caches[0][0]
        else:
            # Single tensor per layer (MERGED_KV or flattened page buffer).
            first_tensor = kv_caches[0]

        if self.kv_format in (
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
            KVCacheFormat.DSA_C8_KV,
        ):
            _init_mla_dsa_connector_dims(
                self,
                kv_caches[0],
                self.kv_format,
                logical_page_slots=self._logical_page_slots,
            )
        else:
            # Non-MLA/DSA formats: derive block_size and page_buffer_size from tensor shape.
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                if isinstance(first_entry, (tuple, list)) and _is_shared_storage_blob(
                    first_entry
                ):
                    # Compress layers (DSv4): bf16+int8 views over one int8 blob;
                    # derive geometry from the primary (largest-byte) view.
                    primary = _get_primary_blob_view(first_entry)
                    effective_bs = int(primary.shape[1])
                    num_blocks = int(primary.shape[0])
                    if effective_bs <= 0 or num_blocks <= 0:
                        raise ValueError(
                            "Shared-storage KV blob requires a positive block_size "
                            "from the primary view tensor shape."
                        )
                    hidden_per_token = int(primary.numel()) // (
                        num_blocks * effective_bs
                    )
                    page_bytes = hidden_per_token * effective_bs
                    self.block_size = effective_bs
                    self.page_buffer_size = num_blocks * effective_bs
                    logger.debug(
                        "Shared-storage blob: num_blocks=%s effective_bs=%s "
                        "hidden_per_token=%s page_bytes=%s (primary_dtype=%s)",
                        num_blocks,
                        effective_bs,
                        hidden_per_token,
                        page_bytes,
                        primary.dtype,
                    )
                else:
                    # Standard separate (k, v) pair with independent allocations.
                    # 310P: [num_blocks, num_kv_heads * head_size // 16, block_size, 16]
                    # 910B: [num_blocks, block_size, num_kv_heads, head_size]
                    assert first_tensor.dim() >= 2
                    if is_310p():
                        self.block_size = first_tensor.shape[-2]
                        self.page_buffer_size = first_tensor.shape[0] * self.block_size
                    else:
                        self.block_size = int(first_tensor.shape[1])
                        self.page_buffer_size = (
                            first_tensor.shape[0] * self.block_size
                        )

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                # 310P: [2, num_blocks, num_kv_heads * head_size // 16, block_size, 16]
                # 910B: [2, num_blocks, block_size, num_kv_heads, head_size]
                assert first_tensor.dim() == 5
                if is_310p():
                    self.block_size = first_tensor.shape[-2]
                    self.page_buffer_size = first_tensor.shape[1] * self.block_size
                else:
                    self.block_size = int(first_tensor.shape[2])
                    self.page_buffer_size = (
                        first_tensor.shape[1] * self.block_size
                    )

        self._initialize_group_pointers_and_params(kv_caches)
        return self.kv_cache_pointers_on_gpu[idx]

    def to_gpu_310p(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemNPUConnector."
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format "
                    "in order to be processed by VLLMPagedMemNPUConnector."
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        tmp_gpu_buffer = torch.empty(
            memory_obj.tensor.size(), dtype=self.dtype, device=self.device
        )

        tmp_gpu_buffer.copy_(memory_obj.tensor)

        lmc_ops.multi_layer_kv_transfer_310p(
            tmp_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            False,
            self.use_mla,
            self.num_kv_head,
            self.head_size,
            self.block_size,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

    def from_gpu_310p(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        assert self.gpu_buffer.device == self.kvcaches_device

        tmp_gpu_buffer = torch.empty(
            memory_obj.tensor.size(), dtype=self.dtype, device=self.device
        )

        lmc_ops.multi_layer_kv_transfer_310p(
            tmp_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            True,
            self.use_mla,
            self.num_kv_head,
            self.head_size,
            self.block_size,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

        memory_obj.tensor.copy_(tmp_gpu_buffer)
        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        """Raise ValueError if memory_obj format doesn't match the connector's KV format."""
        if (
            (self.use_mla and memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT)
            or (self.kv_format == KVCacheFormat.DSA_C8_KV and memory_obj.tensor.dtype != torch.uint8)
            or (not self.use_mla and self.kv_format != KVCacheFormat.DSA_C8_KV
                and memory_obj.metadata.fmt != MemoryFormat.KV_2LTD)
        ):
            raise ValueError(
                f"Memory format mismatch: use_mla={self.use_mla}, "
                f"kv_format={self.kv_format}, "
                f"mem_fmt={memory_obj.metadata.fmt}, "
                f"tensor_dtype={memory_obj.tensor.dtype}"
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        if self._try_multi_group_dispatch(
            memory_obj, start, end, kwargs, is_store=False,
        ):
            return

        assert memory_obj.tensor is not None
        self._validate_memory_format(memory_obj)

        if self.kv_format == KVCacheFormat.DSA_C8_KV:
            self._invoke_multi_plane_kv_transfer(
                mem_tensor=memory_obj.tensor,
                group_ptrs=kv_cache_pointers,
                group_params=_build_dsa_c8_multi_plane_group_params(
                    self.dsa_c8_plane_bytes,
                    block_size=int(self.block_size),
                    page_buffer_size=int(self.page_buffer_size),
                    num_tokens=end - start,
                ),
                slot_mappings_by_group=(slot_mapping,),
                compress_ratios=(1,),
                g_start=start,
                g_end=end,
                is_store=False,
                npu_group_idx=0,
            )
            return

        k1, k2, k3, k4 = self.v2_plane_extras
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            False,
            self.use_mla,
            self.kv_format.value,
            k1,
            k2,
            k3,
            k4,
            int(self.block_size),
        )

    def _slot_mapping_for_npu_group(
        self,
        npu_group_idx: int,
        group_params: dict[str, Any],
        slot_mappings_by_group: Union[tuple[torch.Tensor, ...], list[torch.Tensor]],
    ) -> torch.Tensor:
        """Resolve scheduler slot mapping for one NPU layer group."""
        slot_g = int(group_params.get("scheduler_slot_group", npu_group_idx))
        if slot_g >= len(slot_mappings_by_group):
            raise IndexError(
                f"Scheduler slot group {slot_g} out of range "
                f"(num={len(slot_mappings_by_group)}) for NPU group {npu_group_idx}"
            )
        return slot_mappings_by_group[slot_g]

    def _try_multi_group_dispatch(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        kwargs: dict[str, Any],
        *,
        is_store: bool,
    ) -> bool:
        """Dispatch to multi-group transfer if per-group slot mappings are available.

        Returns True if the multi-group path handled the transfer, False for
        single-group fallback.  Raises on mixed-format caches that lack the
        required per-group infrastructure.
        """
        slot_mappings = kwargs.get("slot_mappings_npu_by_group")
        op = "store" if is_store else "retrieve"
        if slot_mappings is not None:
            if (
                self.group_kv_cache_pointers is not None
                and self.per_group_params is not None
                and len(self.group_kv_cache_pointers) > 0
            ):
                stream = self.store_stream if is_store else self.load_stream
                self._multi_group_kv_transfer(
                    memory_obj, start, end, slot_mappings, is_store=is_store,
                    stream=stream,
                )
                return True
            if self._is_mixed_format:
                raise ValueError(
                    "Mixed-format KV caches require initialized per-group "
                    f"pointers and slot_mappings_npu_by_group for {op}."
                )
        if self._is_mixed_format:
            raise ValueError(
                "Mixed-format KV caches require slot_mappings_npu_by_group "
                f"for {op}; single-group slot_mapping is invalid."
            )
        return False

    def _multi_group_kv_transfer(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        slot_mappings_by_group: Union[tuple[torch.Tensor, ...], list[torch.Tensor]],
        *,
        is_store: bool,
        stream: Any,
    ) -> None:
        """Run multi_layer_kv_transfer per NPU layer group (store or retrieve)."""
        assert self.group_kv_cache_pointers is not None
        assert self.per_group_params is not None

        compress_ratios = self._compress_ratios_by_group()
        n_memobj_groups = len(memory_obj.group_prefix_sum) - 1
        with torch.npu.stream(stream):
            for i, (group_ptrs, group_params) in enumerate(
                zip(self.group_kv_cache_pointers, self.per_group_params)
            ):
                if i >= n_memobj_groups:
                    raise RuntimeError(
                        f"NPU group {i} exceeds memory_obj group count "
                        f"{n_memobj_groups} (group_prefix_sum="
                        f"{memory_obj.group_prefix_sum}). "
                        f"Ensure per-NPU-group allocation is active — "
                        f"kv_layer_groups_manager must be initialised before "
                        f"the store pipeline allocates MemoryObj."
                    )
                mem_tensor = memory_obj.get_tensor(i)
                if mem_tensor is None:
                    logger.warning(
                        "Skipping multi-group %s for NPU group %d: no memory tensor",
                        "store" if is_store else "retrieve",
                        i,
                    )
                    continue

                # int8 / float32 lmc_chunk dtypes are not dispatched by the v2 kernel
                # (Char is unmapped in get_dtype_from_torch; FP32 has no case).
                # Reinterpret as uint8 for a byte-level copy on the INT8 path.
                # fp16 scale planes (s_extra > 0) use the same path as processScalePlane.
                scale_plane_bytes = int(group_params.get("s_extra", 0))
                if mem_tensor.dtype in (torch.int8, torch.float32):
                    mem_tensor = mem_tensor.view(torch.uint8)
                elif mem_tensor.dtype == torch.float16 and scale_plane_bytes > 0:
                    mem_tensor = mem_tensor.view(torch.uint8)

                if _uses_multi_plane_kv_transfer(group_params):
                    self._invoke_multi_plane_kv_transfer(
                        mem_tensor=mem_tensor,
                        group_ptrs=group_ptrs,
                        group_params=group_params,
                        slot_mappings_by_group=slot_mappings_by_group,
                        compress_ratios=compress_ratios,
                        g_start=start,
                        g_end=end,
                        is_store=is_store,
                        npu_group_idx=i,
                    )
                    continue

                sm = self._slot_mapping_for_npu_group(
                    i, group_params, slot_mappings_by_group
                )
                slot_g = int(group_params.get("scheduler_slot_group", 0))
                slot_start, slot_end = multi_plane_slot_slice_bounds(
                    start, end, slot_g, compress_ratios, int(sm.shape[0]),
                )

                lmc_ops.multi_layer_kv_transfer(
                    mem_tensor,
                    group_ptrs,
                    sm[slot_start:slot_end],
                    self.kvcaches_device,
                    group_params["page_buffer_size"],
                    is_store,
                    group_params["use_mla"],
                    group_params["kv_format"],
                    group_params["k_extra"],
                    group_params["v_extra"],
                    group_params["d_extra"],
                    group_params["s_extra"],
                    int(group_params["block_size"]),
                )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        with torch.npu.stream(self.store_stream):
            self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping_npu" in kwargs:
            slot_mapping: torch.Tensor = kwargs["slot_mapping_npu"]
        elif "slot_mapping" in kwargs:
            slot_mapping = kwargs["slot_mapping"]
            if not isinstance(slot_mapping, torch.Tensor):
                raise ValueError("'slot_mapping' should be a torch.Tensor.")
            # for Ascend kernels to keep test inputs backward compatible.
            if slot_mapping.device.type != "npu":
                with torch.npu.stream(self.store_stream):
                    slot_mapping = slot_mapping.to(
                        self.kvcaches_device,
                        non_blocking=True,
                    )
        else:
            raise ValueError(
                "'slot_mapping_npu' should be provided in kwargs "
                "(or 'slot_mapping' for compatibility)."
            )

        with torch.npu.stream(self.store_stream):
            kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError("KV cache format is not initialized!")

        if self._try_multi_group_dispatch(
            memory_obj, start, end, kwargs, is_store=True,
        ):
            no_sync = kwargs.get("no_sync", False)
            if not no_sync:
                self.store_stream.synchronize()
            if self.use_mla:
                memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            return

        assert memory_obj.tensor is not None
        self._validate_memory_format(memory_obj)

        if self.kv_format == KVCacheFormat.DSA_C8_KV:
            with torch.npu.stream(self.store_stream):
                self._invoke_multi_plane_kv_transfer(
                    mem_tensor=memory_obj.tensor,
                    group_ptrs=kv_cache_pointers,
                    group_params=_build_dsa_c8_multi_plane_group_params(
                        self.dsa_c8_plane_bytes,
                        block_size=int(self.block_size),
                        page_buffer_size=int(self.page_buffer_size),
                        num_tokens=end - start,
                    ),
                    slot_mappings_by_group=(slot_mapping,),
                    compress_ratios=(1,),
                    g_start=start,
                    g_end=end,
                    is_store=True,
                    npu_group_idx=0,
                )
            no_sync = kwargs.get("no_sync", False)
            if not no_sync:
                self.store_stream.synchronize()
            if self.use_mla:
                memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            return

        k1, k2, k3, k4 = self.v2_plane_extras
        with torch.npu.stream(self.store_stream):
            # No staging buffer or token count mismatch
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches_device,
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                    self.kv_format.value,
                    k1,
                    k2,
                    k3,
                    k4,
                    int(self.block_size),
                )
            else:
                assert self.gpu_buffer.device == self.kvcaches_device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.fused_multi_layer_kv_transfer(
                    memory_obj.tensor,  # dst: CPU buffer
                    tmp_gpu_buffer,  # staging cache
                    kv_cache_pointers,  # src: paged KV cache
                    slot_mapping[start:end],
                    self.kvcaches_device,
                    self.page_buffer_size,
                    True,  # from_gpu
                    self.use_mla,
                    self.kv_format.value,
                    k1,
                    k2,
                    k3,
                    k4,
                    int(self.block_size),
                )
        no_sync = kwargs.get("no_sync", False)
        if not no_sync and not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        # Check if any memory objects are ProxyMemoryObjs (deferred P2P fetch)
        has_proxy = any(isinstance(m, ProxyMemoryObj) for m in memory_objs)

        if has_proxy:
            assert not is_310p(), "Batched P2P transfer is not supported on 310P."

            self._remote_batched_to_gpu(memory_objs, starts, ends, **kwargs)

            # NOTE (gingfung): Ensure the compute stream waits for
            # load_stream's KV scatter to complete before attention
            # reads the same pages.
            # load_stream.synchronize() in _remote_batched_to_gpu is
            # host-side only, the compute stream has no knowledge of it
            # and can race ahead.
            torch.npu.current_stream().wait_stream(self.load_stream)
        else:
            with torch.cuda.stream(self.load_stream):
                for memory_obj, start, end in zip(
                    memory_objs, starts, ends, strict=False
                ):
                    if is_310p():
                        self.to_gpu_310p(memory_obj, start, end, **kwargs)
                    else:
                        self.to_gpu(memory_obj, start, end, **kwargs)
            self.load_stream.synchronize()

    def _clear_proxy_batch(self, batch) -> None:
        """Clear the backing objects of the proxy batch."""
        for proxy, _, _ in batch:
            proxy.clear_backing_obj()
        return None

    def _scatter_proxy_batch(self, batch, event, **kwargs):
        """Wait for a read event, scatter proxies to KV cache.

        Enqueues work on ``load_stream``.  The caller is responsible for
        recording a scatter-done event afterwards if needed for
        cross-stream synchronization.
        """
        if event is not None:
            self.load_stream.wait_event(event)
        with torch.cuda.stream(self.load_stream):
            for proxy, start, end in batch:
                self.to_gpu(proxy.backing_obj, start, end, **kwargs)

    def _remote_batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        """Handle batched_to_gpu when ProxyMemoryObjs are present.

        Uses a ping-pong pipeline with **event-based** cross-stream
        synchronization to overlap remote data fetching (on the HCCL
        transport stream) with KV cache scatter (on the load stream).


        Two pools of PIPELINE_DEPTH buffers are allocated from the
        transfer context's registered memory and alternated (ping-pong).
        This limits peak memory to 2 x PIPELINE_DEPTH chunks regardless
        of the total number of proxy objects.

        After all proxy objects are processed, sends the Done signal
        to release the remote peer's pinned resources.
        """
        transfer_contexts: Set[AscendBaseTransferContext] = set()

        # Separate proxy and non-proxy items
        proxy_items = []
        non_proxy_items = []
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if isinstance(memory_obj, ProxyMemoryObj):
                transfer_contexts.add(memory_obj.transfer_context)
                proxy_items.append((memory_obj, start, end))
            else:
                non_proxy_items.append((memory_obj, start, end))

        if proxy_items:
            # Get the transfer context for buffer allocation
            first_ctx = proxy_items[0][0].transfer_context

            # Derive pipeline depth from NPU buffer capacity so that
            # two full ping-pong pools fit in registered memory.
            pipeline_depth = first_ctx.max_pipeline_depth
            logger.debug(
                "P2P pipeline depth = %d (proxy_items=%d)",
                pipeline_depth,
                len(proxy_items),
            )

            # Allocate ping-pong buffer pools.
            # Initialized to None so the finally block can safely skip
            # release if allocation itself fails.
            pool_size = min(pipeline_depth, len(proxy_items))
            pool_a = None
            pool_b = None

            try:
                pool_a = first_ctx.allocate_buffers(pool_size)
                pool_b = first_ctx.allocate_buffers(pool_size)

                pools = [pool_a, pool_b]
                current_pool = 0

                # Group proxy items into micro-batches
                micro_batches = [
                    proxy_items[i : i + pipeline_depth]
                    for i in range(0, len(proxy_items), pipeline_depth)
                ]

                prev_read_event = None
                prev_batch = None

                # Per-pool scatter-done events: prevent the next RDMA
                # write into a pool from racing with a scatter that is
                # still reading from the same pool on load_stream.
                # Events are pre-allocated and re-recorded each iteration.
                channel = proxy_items[0][0]._transfer_channel
                transport_stream = getattr(channel, "transport_stream", None)
                pool_scatter_events = [
                    torch.npu.Event(),
                    torch.npu.Event(),
                ]
                pool_scatter_recorded = [False, False]

                for batch_idx, batch in enumerate(micro_batches):
                    pool = pools[current_pool]

                    # Ensure the previous scatter from this pool has
                    # finished before RDMA overwrites the pool buffers.
                    if (
                        pool_scatter_recorded[current_pool]
                        and transport_stream is not None
                    ):
                        transport_stream.wait_event(pool_scatter_events[current_pool])

                    # Assign backing buffers from current pool to proxies
                    for i, (proxy, _, _) in enumerate(batch):
                        proxy.set_backing_obj(pool[i])

                    proxies = [item[0] for item in batch]

                    # Submit RDMA read for current batch → transport_stream.
                    cur_read_event = ProxyMemoryObj.submit_resolve_batch(proxies)

                    # While the current batch is being read on
                    # transport_stream, scatter the previous batch on
                    # load_stream (waits for its RDMA read event).
                    if prev_batch is not None:
                        self._scatter_proxy_batch(
                            prev_batch,
                            prev_read_event,
                            **kwargs,
                        )
                        pool_scatter_events[1 - current_pool].record(self.load_stream)
                        pool_scatter_recorded[1 - current_pool] = True
                        self._clear_proxy_batch(prev_batch)

                    prev_read_event = cur_read_event
                    prev_batch = batch
                    current_pool = 1 - current_pool  # toggle ping-pong

                # Drain: scatter the last micro-batch.
                if prev_batch is not None:
                    self._scatter_proxy_batch(
                        prev_batch,
                        prev_read_event,
                        **kwargs,
                    )
                    self._clear_proxy_batch(prev_batch)
            finally:
                # Guarantee ping-pong buffers are returned and the Done
                # signal is sent even if the pipeline raises or
                # allocate_buffers itself fails.  Without this, an
                # exception would leak NPU pages and leave the sender's
                # pinned resources stuck until its TTL expires.
                self.load_stream.synchronize()
                if pool_a is not None:
                    first_ctx.release_buffers(pool_a)
                if pool_b is not None:
                    first_ctx.release_buffers(pool_b)

                for proxy, _, _ in proxy_items:
                    proxy.mark_consumed()

                for ctx in transfer_contexts:
                    ctx.send_done_now()

        # Process non-proxy items on load_stream (no pipelining needed)
        if non_proxy_items:
            with torch.cuda.stream(self.load_stream):
                for memory_obj, start, end in non_proxy_items:
                    self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        # NOTE (gingfung):
        # Since no_sync is only consumed by us, for now we modify the kwargs directly.
        # We avoid per-object synchronization during batch transfers.
        # A single synchronization is performed at the end of the batch.
        kwargs["no_sync"] = True

        ordering_event = kwargs.pop("ordering_event", None)
        current_stream = torch.npu.current_stream()
        if ordering_event is not None:
            self.store_stream.wait_event(ordering_event)
        else:
            self.store_stream.wait_stream(current_stream)

        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if is_310p():
                self.from_gpu_310p(memory_obj, start, end, **kwargs)
            else:
                self.from_gpu(memory_obj, start, end, **kwargs)
        self.store_stream.synchronize()

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.kv_format == KVCacheFormat.DSA_C8_KV:
            from lmcache_ascend.v1.kv_layer_groups import _multi_plane_lmc_row_bytes

            row_bytes = _multi_plane_lmc_row_bytes(
                list(self.dsa_c8_plane_bytes), num_tokens
            )
            return torch.Size([1, self.num_layers, num_tokens, row_bytes])
        if self.kv_format in (KVCacheFormat.MLA_KV, KVCacheFormat.DSA_KV):
            return torch.Size(
                [1, self.num_layers, num_tokens, self.v2_staging_hidden_dim]
            )
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size]
        )


class VLLMPagedMemLayerwiseNPUConnector(
    _V2KVTransferMixin, VLLMPagedMemLayerwiseGPUConnector
):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)

        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

        self.use_mla = bool(kwargs.get("use_mla", False))
        self.kv_lora_rank: int = 0
        self.qk_rope_head_dim: int = 0
        self.dsa_head_dim: int = 0
        self.dsa_c8_plane_bytes: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.page_buffer_size: int = 0
        self.block_size: int = 0

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.

        Supports both legacy formats and new SEPARATE_KV format:
        - Legacy MERGED_KV: [2, num_blocks, block_size, num_heads, head_size]
        - New SEPARATE_KV: tuple(key_tensor, value_tensor) where each is
          [num_blocks, block_size, num_heads, head_size]
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")

            self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)

            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError(
                    "Undefined KV cache format detected. "
                    "Unable to determine the format of input kv_caches."
                )

            logger.info(f"Detected KV cache format: {self.kv_format.name}")

            first_layer_cache = kv_caches[0]

            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]

                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )

                k_cache_shape_per_layer = key_tensor.shape
                self.vllm_two_major = False

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )

                self.vllm_two_major = first_layer_cache.shape[0] == 2

                if self.vllm_two_major:
                    # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
                    k_cache_shape_per_layer = first_layer_cache[0].shape
                else:
                    # Flash Infer: [num_blocks, 2, block_size, num_heads, head_size]
                    k_cache_shape_per_layer = first_layer_cache[:, 0].shape
            elif self.kv_format in (
                KVCacheFormat.MLA_KV,
                KVCacheFormat.DSA_KV,
                KVCacheFormat.DSA_C8_KV,
            ):
                k_cache_shape_per_layer = _init_mla_dsa_connector_dims(
                    self, first_layer_cache, self.kv_format
                )
            else:
                raise ValueError(f"Unsupported KV cache format: {self.kv_format}")

            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}"
            )

            if self.kv_format in (
                KVCacheFormat.MLA_KV,
                KVCacheFormat.DSA_KV,
                KVCacheFormat.DSA_C8_KV,
            ):
                if not isinstance(first_layer_cache, (tuple, list)):
                    raise TypeError("Expected tuple/list KV cache for MLA/DSA formats.")
                gpu_buffer_size = sum(
                    int(t.numel()) * int(t.element_size()) for t in first_layer_cache
                )
            else:
                num_elements_key = k_cache_shape_per_layer.numel()
                num_elements = num_elements_key * 2
                gpu_buffer_size = num_elements * self.element_size

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Staging buffer for one layer (``num_layers=1`` in transfer kernels)."""
        if self.kv_format == KVCacheFormat.DSA_C8_KV:
            from lmcache_ascend.v1.kv_layer_groups import _multi_plane_lmc_row_bytes

            row_bytes = _multi_plane_lmc_row_bytes(
                list(self.dsa_c8_plane_bytes), num_tokens
            )
            return torch.Size([1, 1, num_tokens, row_bytes])
        if self.kv_format in (KVCacheFormat.MLA_KV, KVCacheFormat.DSA_KV):
            return torch.Size([1, 1, num_tokens, self.v2_staging_hidden_dim])
        return super().get_shape(num_tokens)

    def _layerwise_v2_transfers(
        self,
        *,
        is_store: bool,
        memory_objs_layer: list,
        slot_mapping: torch.Tensor,
        slot_mapping_full: torch.Tensor,
        starts: List[int],
        ends: List[int],
        chunk_offsets: List[int],
        chunk_sizes: List[int],
        staging: Optional[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build (data_tensor, slot_mapping) pairs for one layerwise v2 transfer."""
        if is_store:
            if staging is not None:
                return [(staging, slot_mapping_full)]
            transfers = []
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                self._assert_memory_obj_format_ml(memory_obj)
                assert memory_obj.tensor is not None
                transfers.append((memory_obj.tensor, slot_mapping[start:end]))
            return transfers

        if staging is not None:
            staging.zero_()
            for i, memory_obj in enumerate(memory_objs_layer):
                self._assert_memory_obj_format_ml(memory_obj)
                assert memory_obj.tensor is not None
                off = chunk_offsets[i]
                cs = chunk_sizes[i]
                staging[0, 0, off : off + cs].copy_(
                    memory_obj.tensor[0, 0, :cs], non_blocking=True
                )
            return [(staging, slot_mapping_full)]

        transfers = []
        for start, end, memory_obj in zip(
            starts, ends, memory_objs_layer, strict=False
        ):
            self._assert_memory_obj_format_ml(memory_obj)
            assert memory_obj.tensor is not None
            transfers.append((memory_obj.tensor, slot_mapping[start:end]))
        return transfers

    def _run_layerwise_dsa_c8_transfers(
        self,
        layer_id: int,
        transfers: list[tuple[torch.Tensor, torch.Tensor, int, int]],
        *,
        slot_mapping_full: torch.Tensor,
        is_store: bool,
    ) -> None:
        ptrs = _layer_paged_kv_ptrs_tensor(self.kvcaches[layer_id], self.kv_format)
        for data_tensor, _sm, g_start, g_end in transfers:
            self._invoke_multi_plane_kv_transfer(
                mem_tensor=data_tensor,
                group_ptrs=ptrs,
                group_params=_build_dsa_c8_multi_plane_group_params(
                    self.dsa_c8_plane_bytes,
                    block_size=int(self.block_size),
                    page_buffer_size=int(self.page_buffer_size),
                    num_tokens=g_end - g_start,
                ),
                slot_mappings_by_group=(slot_mapping_full,),
                compress_ratios=(1,),
                g_start=g_start,
                g_end=g_end,
                is_store=is_store,
                npu_group_idx=0,
            )

    def _run_layerwise_v2_transfers(
        self,
        layer_id: int,
        transfers: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        is_store: bool,
    ) -> None:
        ptrs = _layer_paged_kv_ptrs_tensor(self.kvcaches[layer_id], self.kv_format)
        kdev = ptrs.device
        k1, k2, k3, k4 = self.v2_plane_extras
        for data_tensor, sm in transfers:
            lmc_ops.multi_layer_kv_transfer(
                data_tensor,
                ptrs,
                sm,
                kdev,
                self.page_buffer_size,
                is_store,
                self.use_mla,
                self.kv_format.value,
                k1,
                k2,
                k3,
                k4,
                int(self.block_size),
            )

    def _assert_memory_obj_format_ml(self, memory_obj: MemoryObj) -> None:
        if self.kv_format == KVCacheFormat.DSA_C8_KV:
            assert memory_obj.tensor is not None
            if memory_obj.tensor.dtype != torch.uint8:
                raise ValueError(
                    "DSA_C8_KV expects uint8 packed LMCache chunk tensors on memory objects."
                )
        elif memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
            raise ValueError(
                f"Expected MemoryFormat.KV_MLA_FMT for {self.kv_format.name}, "
                f"got {memory_obj.metadata.fmt}"
            )

    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            if _kv_tuple_formats_use_multi_layer_transfer(self.kv_format):
                buf_dtype = (
                    torch.uint8
                    if self.kv_format == KVCacheFormat.DSA_C8_KV
                    else self.dtype
                )
                tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                    buffer_shape, buf_dtype, MemoryFormat.KV_MLA_FMT
                )
            else:
                tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                    buffer_shape, self.dtype, MemoryFormat.KV_T2D
                )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate NPU buffer in NPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")
            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                if _kv_tuple_formats_use_multi_layer_transfer(self.kv_format):
                    staging = (
                        tmp_gpu_buffer_obj.tensor if self.use_gpu else None
                    )
                    if self.use_gpu:
                        assert tmp_gpu_buffer_obj is not None
                        assert staging is not None
                    transfers = self._layerwise_v2_transfers(
                        is_store=False,
                        memory_objs_layer=memory_objs_layer,
                        slot_mapping=slot_mapping,
                        slot_mapping_full=slot_mapping_full,
                        starts=starts,
                        ends=ends,
                        chunk_offsets=chunk_offsets,
                        chunk_sizes=chunk_sizes,
                        staging=staging,
                    )
                    self._run_layerwise_v2_transfers(
                        layer_id, transfers, is_store=False
                    )
                elif self.use_gpu:
                    cpu_tensors = []
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                        cpu_tensors.append(memory_obj.tensor)

                    # Fused transfer: N H2D memcpy + 1 scatter kernel
                    lmc_ops.batched_fused_single_layer_kv_transfer(
                        cpu_tensors,  # CPU memory objects
                        tmp_gpu_buffer_obj.tensor,  # GPU staging buffer
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        chunk_offsets,  # offset for each chunk
                        chunk_sizes,  # size for each chunk
                        False,  # to_gpu
                        self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                        True,  # token_major
                        self.vllm_two_major,
                    )

                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.tensor is not None

                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            False,
                            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                            True,
                            self.vllm_two_major,
                        )
                logger.debug(f"Finished loading layer {layer_id}")
        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            if _kv_tuple_formats_use_multi_layer_transfer(self.kv_format):
                buf_dtype = (
                    torch.uint8
                    if self.kv_format == KVCacheFormat.DSA_C8_KV
                    else self.dtype
                )
                tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                    buffer_shape, buf_dtype, MemoryFormat.KV_MLA_FMT
                )
            else:
                tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                    buffer_shape, self.dtype, MemoryFormat.KV_T2D
                )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate NPU buffer in NPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                if _kv_tuple_formats_use_multi_layer_transfer(self.kv_format):
                    staging = (
                        tmp_gpu_buffer_obj.tensor if self.use_gpu else None
                    )
                    if self.use_gpu:
                        assert tmp_gpu_buffer_obj is not None
                        assert staging is not None
                    transfers = self._layerwise_v2_transfers(
                        is_store=True,
                        memory_objs_layer=memory_objs_layer,
                        slot_mapping=slot_mapping,
                        slot_mapping_full=slot_mapping_full,
                        starts=starts,
                        ends=ends,
                        chunk_offsets=chunk_offsets,
                        chunk_sizes=chunk_sizes,
                        staging=staging,
                    )
                    self._run_layerwise_v2_transfers(
                        layer_id, transfers, is_store=True
                    )
                    if self.use_gpu:
                        assert staging is not None
                        for i, memory_obj in enumerate(memory_objs_layer):
                            self._assert_memory_obj_format_ml(memory_obj)
                            assert memory_obj.tensor is not None
                            off = chunk_offsets[i]
                            cs = chunk_sizes[i]
                            memory_obj.tensor[0, 0, :cs].copy_(
                                staging[0, 0, off : off + cs], non_blocking=True
                            )
                elif self.use_gpu:
                    cpu_tensors = []
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        cpu_tensors.append(memory_obj.tensor)

                    # Fused transfer: 1 scatter kernel + N D2H memcpy
                    lmc_ops.batched_fused_single_layer_kv_transfer(
                        cpu_tensors,
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        chunk_offsets,
                        chunk_sizes,
                        True,  # from_gpu
                        self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                        True,  # token_major
                        self.vllm_two_major,
                    )
                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.tensor is not None

                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            True,
                            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                            True,
                            self.vllm_two_major,
                        )
                logger.debug(f"Finished offloading layer {layer_id}")
            yield

            if sync:
                self.store_stream.synchronize()

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()
        yield


class SGLangNPUConnector(SGLangGPUConnector):
    pass


class SGLangLayerwiseNPUConnector(SGLangLayerwiseGPUConnector):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [num_blocks, block_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        # [2, self.layer_num, self.size // self.page_size + 1,
        # self.page_size, self.head_num, self.head_dim,]
        self.kv_format = KVCacheFormat.detect(kv_caches)
        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError("Could not detect KV cache format.")

        if self.use_gpu and self.gpu_buffer_allocator is None:
            k_cache_shape_per_layer = kv_caches[0][0].shape
            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}\n"
                f"  - num_elements: {num_elements}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024)} MB"
            )

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            current_layer_kv = (self.kvcaches[0][layer_id], self.kvcaches[1][layer_id])

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        current_layer_kv,
                        slot_mapping[start:end],
                        False,
                        self.kv_format.value,
                        True,
                        True,
                    )

            if self.use_gpu:
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    current_layer_kv,
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    True,
                    True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            current_layer_kv = (self.kvcaches[0][layer_id], self.kvcaches[1][layer_id])

            if self.use_gpu:
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    current_layer_kv,
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    True,
                    True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None

                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        current_layer_kv,
                        slot_mapping[start:end],
                        True,
                        self.kv_format.value,
                        True,
                        True,
                    )

            yield
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
