# SPDX-License-Identifier: Apache-2.0
"""Preprocess vllm-ascend multi-spec per-layer KV entries for the NPU connector."""

from __future__ import annotations

import os
from typing import Any, Sequence, Union

# Third Party
import torch

from lmcache_ascend.v1.kv_format import KVCacheFormat, _is_shared_storage_blob

_KVEntry = Union[torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor]]

_KERNEL_NATIVE_FORMATS = frozenset(
    {
        KVCacheFormat.MLA_KV,
        KVCacheFormat.DSA_KV,
        KVCacheFormat.DSA_C8_KV,
    }
)


def flatten_multi_spec_enabled() -> bool:
    val = os.environ.get("LMCACHE_ASCEND_FLATTEN_MULTI_SPEC", "1")
    return val != "0"


def bundle_multi_spec_enabled() -> bool:
    val = os.environ.get("LMCACHE_ASCEND_BUNDLE_MULTI_SPEC", "1")
    return val != "0"


def should_bundle_multi_spec(kv_cache_config: Any | None) -> bool:
    return flatten_multi_spec_enabled() and bundle_multi_spec_enabled() and has_multiple_scheduler_groups(
        kv_cache_config
    )


def _has_multiple_planes(entry: _KVEntry) -> bool:
    planes = _entry_planes(entry)
    return len(planes) > 1


def _keep_planes_bundled(entry: _KVEntry) -> bool:
    """Keep heterogeneous independent planes as one tuple; not shared-storage blobs."""
    if not _has_multiple_planes(entry):
        return False
    planes = _entry_planes(entry)
    # Same physical allocation exposed as multiple dtype views (e.g. bf16 + int8
    # over one int8 blob): one scheduler group, not independent multi-plane KV.
    # Bundling would misclassify this as MULTI_PLANE_KV and route to the wrong kernel.
    if _is_shared_storage_blob(planes):
        return False
    return True


def has_multiple_scheduler_groups(kv_cache_config: Any | None) -> bool:
    if kv_cache_config is None:
        return False
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    return bool(groups) and len(groups) > 1


def _entry_planes(entry: _KVEntry) -> list[torch.Tensor]:
    if isinstance(entry, torch.Tensor):
        return [entry]
    return [t for t in entry if isinstance(t, torch.Tensor)]


def _containing_groups_for_layer(
    layer_name: str, kv_cache_config: Any
) -> list[int]:
    """Return scheduler group indices that contain layer_name, in index order."""
    groups = kv_cache_config.kv_cache_groups
    return [g for g, grp in enumerate(groups) if layer_name in grp.layer_names]


def _is_kernel_native_tuple(entry: _KVEntry) -> bool:
    """True when the NPU connector transfers this tuple without exploding to 3-D."""
    if not isinstance(entry, tuple):
        return False
    return KVCacheFormat.detect([entry]) in _KERNEL_NATIVE_FORMATS


def _primary_scheduler_group_for_layer(
    layer_name: str,
    kv_cache_config: Any,
) -> int:
    """Return the single *primary* scheduler group index for a model layer.

    A layer can belong to several scheduler KV-cache groups (multi-spec
    configurations).  The **primary group** is the one chosen to represent
    the layer whenever a unique group assignment is needed — for example
    when every sub-tensor of an MLA/DSA tuple maps to the same group, or
    when a bundled multi-spec layer must record a single scheduler group.
    """
    containing = _containing_groups_for_layer(layer_name, kv_cache_config)
    if not containing:
        raise ValueError(f"Layer {layer_name!r} is not in any scheduler KV group")
    # ``ie_logical_block_size`` (global max block size) cannot pick the right
    # per-layer group in multi-group configs; groups have heterogeneous block sizes.
    # ``containing[0]`` is the lowest-index group listing this layer — usually the
    # main attention spec before companion or state specs in vLLM config order.
    return containing[0]


def ordered_scheduler_groups_for_layer(
    layer_name: str,
    entry: _KVEntry,
    kv_cache_config: Any,
) -> list[int]:
    """Return scheduler group indices in sub-tensor order for one model layer.

    The model runner produces sub-tensors by iterating kv_cache_groups in
    index order.  For MLA/DSA tuples (homogeneous K/V/indexer within one
    group), all sub-tensors map to the same primary group.  For multi-spec
    layers, each sub-tensor corresponds to a distinct scheduler group.
    """
    planes = _entry_planes(entry)
    if _is_kernel_native_tuple(entry):
        g = _primary_scheduler_group_for_layer(layer_name, kv_cache_config)
        return [g] * len(planes)

    containing = _containing_groups_for_layer(layer_name, kv_cache_config)
    if len(containing) == len(planes):
        return containing
    # More planes than groups: some groups contribute multiple tensors
    # (e.g. duplicate indexer views). Match by block_size from the tensors.
    if len(containing) < len(planes):
        groups = kv_cache_config.kv_cache_groups
        group_block_sizes = {g: int(groups[g].kv_cache_spec.block_size) for g in containing}
        result: list[int] = []
        for t in planes:
            t_bs = int(t.shape[1])
            # Find first unmatched group with matching block_size
            matched = None
            for g in containing:
                if group_block_sizes.get(g) == t_bs and result.count(g) < containing.count(g):
                    matched = g
                    break
            if matched is None:
                # Fallback: pick any group with matching block_size
                for g in containing:
                    if group_block_sizes.get(g) == t_bs:
                        matched = g
                        break
            if matched is None:
                raise ValueError(
                    f"Layer {layer_name!r}: sub-tensor with block_size={t_bs} "
                    f"cannot be matched to any scheduler group "
                    f"(available: {group_block_sizes})"
                )
            result.append(matched)
        return result
    raise ValueError(
        f"Layer {layer_name!r}: {len(planes)} planes vs "
        f"{len(containing)} scheduler groups containing this layer"
    )


def _collapse_to_mla_page_buffer(tensor: torch.Tensor) -> torch.Tensor:
    """Collapse vllm-ascend 4-D ``(nb, bs, nh, hs)`` to 3-D ``(nb, bs, nh*hs)``.

    Only needed for the non-bundled exploded path where standalone
    tensors feed into upstream ``normalize_kv_and_discover_format`` (which
    requires 3-D).  The bundled path keeps 4-D tensors as-is since the NPU
    kernel and _derive_group_params compute hidden_bytes dimensionality-
    agnostically via numel * elem_size // (nb * bs).
    
    Note: vllm-ascend page buffers were already 4-D (nb, bs, nh, hs); pre-flatten 
    Ascend saw them inside tuples. Flatten exposes standalone planes; upstream 
    normalize_kv_and_discover_format only accepts 3-D MLA or 5-D MHA.
    """
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4:
        nb, bs, nh, hs = tensor.shape
        return tensor.reshape(nb, bs, nh * hs)
    raise ValueError(
        f"Expected 3-D or 4-D KV sub-cache tensor, got ndim={tensor.ndim} "
        f"shape={tuple(tensor.shape)}"
    )


def build_layer_to_scheduler_groups(
    kv_cache_config: Any,
    layer_names: Sequence[str],
    kv_caches: dict[str, _KVEntry],
) -> dict[str, list[int]]:
    """Map each model layer to ordered scheduler group indices for its sub-caches."""
    return {
        layer_name: ordered_scheduler_groups_for_layer(
            layer_name,
            kv_caches[layer_name],
            kv_cache_config,
        )
        for layer_name in layer_names
    }


def build_flat_kv_caches(
    kv_caches: dict[str, _KVEntry],
    kv_cache_config: Any,
) -> tuple[dict[str, _KVEntry], tuple[int, ...], dict[str, list[int]], bool]:
    """Preprocess multi-spec KV caches for the NPU connector.

    Multi-spec layers pack multiple sub-tensors per model layer (one per
    spec / scheduler group).  This function builds the scheduler-group-per-
    layer mapping and either bundles or explodes the sub-tensors.

    With bundling (default): multi-spec layers stay as a tuple of their
    original 4-D sub-tensors under the model layer name.  The NPU connector
    handles the tuple via build_kv_layer_groups, deriving hidden_bytes
    dimensionality-agnostically (numel * elem_size // (nb * bs)).

    Without bundling (LMCACHE_ASCEND_BUNDLE_MULTI_SPEC=0): multi-spec
    layers are exploded into layer_name.sub0, .sub1, … each collapsed to
    3-D for upstream normalize_kv_and_discover_format compatibility.

    Returns:
        (flat_kv, sched_by_layer, layer_to_groups, bundled)
    """
    flat: dict[str, _KVEntry] = {}
    sched_by_layer: list[int] = []
    layer_to_groups = build_layer_to_scheduler_groups(
        kv_cache_config,
        kv_caches.keys(),
        kv_caches,
    )
    bundled = False
    bundle_multi_spec = should_bundle_multi_spec(kv_cache_config)
    for layer_name, entry in kv_caches.items():
        planes = _entry_planes(entry)
        groups = layer_to_groups[layer_name]
        if len(planes) != len(groups):
            raise ValueError(
                f"layer {layer_name}: {len(planes)} planes vs "
                f"{len(groups)} scheduler groups"
            )
        if _is_kernel_native_tuple(entry):
            flat[layer_name] = entry
            sched_by_layer.append(groups[0])
            bundled = True
            continue
        if bundle_multi_spec and _keep_planes_bundled(entry):
            flat[layer_name] = tuple(planes)
            sched_by_layer.append(groups[0])
            bundled = True
            continue
        for sub_idx, (sub_tensor, sched_g) in enumerate(zip(planes, groups)):
            flat[f"{layer_name}.sub{sub_idx}"] = _collapse_to_mla_page_buffer(
                sub_tensor
            )
            sched_by_layer.append(sched_g)
    return flat, tuple(sched_by_layer), layer_to_groups, bundled
