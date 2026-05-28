# SPDX-License-Identifier: Apache-2.0
"""Preprocess vllm-ascend multi-spec per-layer KV entries for the NPU connector."""

from __future__ import annotations

import os
from typing import Any, Sequence, Union

# Third Party
import torch

from lmcache_ascend.v1.kv_format import _is_shared_storage_blob

_KVEntry = Union[torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor]]


def flatten_multi_spec_enabled() -> bool:
    val = os.environ.get("LMCACHE_ASCEND_FLATTEN_MULTI_SPEC", "1")
    return val != "0"


def bundle_multi_spec_enabled() -> bool:
    val = os.environ.get("LMCACHE_ASCEND_BUNDLE_MULTI_SPEC", "1")
    return val != "0"


def should_bundle_multi_spec(kv_cache_config: Any | None) -> bool:
    return flatten_multi_spec_enabled() and bundle_multi_spec_enabled() and should_flatten_kv_caches(
        kv_cache_config
    )


def _is_multi_spec_layer(entry: _KVEntry) -> bool:
    subs = _listify_entry(entry)
    return len(subs) > 1 and not _is_mla_or_dsa_tuple(entry)


def _should_bundle_multi_spec_layer(entry: _KVEntry) -> bool:
    """Bundle heterogeneous multi-spec planes, not shared-storage blobs."""
    if not _is_multi_spec_layer(entry):
        return False
    subs = _listify_entry(entry)
    if len(subs) >= 2 and _is_shared_storage_blob(subs):
        return False
    return True


def should_flatten_kv_caches(kv_cache_config: Any | None) -> bool:
    if not flatten_multi_spec_enabled() or kv_cache_config is None:
        return False
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    return bool(groups) and len(groups) > 1


def _listify_entry(entry: _KVEntry) -> list[torch.Tensor]:
    if isinstance(entry, torch.Tensor):
        return [entry]
    return [t for t in entry if isinstance(t, torch.Tensor)]


def _containing_groups_for_layer(
    layer_name: str, kv_cache_config: Any
) -> list[int]:
    """Return scheduler group indices that contain layer_name, in index order."""
    groups = kv_cache_config.kv_cache_groups
    return [g for g, grp in enumerate(groups) if layer_name in grp.layer_names]


def _is_mla_or_dsa_tuple(entry: _KVEntry) -> bool:
    """True for MLA (2-tuple), DSA (3-tuple), or DSA_C8 (4-tuple with int8 indexer).

    Multi-spec compress-128 four-plane layers are *not* MLA/DSA: they use
    per-plane scheduler groups and are bundled as MULTI_PLANE_KV when enabled.
    """
    if not isinstance(entry, tuple):
        return False
    subs = _listify_entry(entry)
    if len(subs) == 2 and subs[0].shape != subs[1].shape:
        return True
    if len(subs) == 3 and subs[0].shape != subs[1].shape:
        return True
    if len(subs) == 4 and any(t.dtype == torch.int8 for t in subs):
        return True
    return False


def _primary_scheduler_group_for_layer(
    layer_name: str,
    kv_cache_config: Any,
    *,
    ie_logical_block_size: int | None = None,
) -> int:
    """Return the single *primary* scheduler group index for a model layer.

    A layer can belong to several scheduler KV-cache groups (multi-spec
    configurations).  The **primary group** is the one chosen to represent
    the layer whenever a unique group assignment is needed — for example
    when every sub-tensor of an MLA/DSA tuple maps to the same group, or
    when a bundled multi-spec layer must record a single scheduler group.

    Selection logic:
      1. If ``ie_logical_block_size`` is given (inter-engine scenarios),
         prefer the containing group whose ``block_size`` matches so that
         the block geometry stays consistent across engines.
      2. Otherwise fall back to the lowest-index containing group.
    """
    containing = _containing_groups_for_layer(layer_name, kv_cache_config)
    if not containing:
        raise ValueError(f"Layer {layer_name!r} is not in any scheduler KV group")
    if ie_logical_block_size is not None:
        groups = kv_cache_config.kv_cache_groups
        for g in containing:
            bs = int(groups[g].kv_cache_spec.block_size)
            if bs == ie_logical_block_size:
                return g
    return containing[0]


def ordered_scheduler_groups_for_layer(
    layer_name: str,
    entry: _KVEntry,
    kv_cache_config: Any,
    *,
    ie_logical_block_size: int | None = None,
) -> list[int]:
    """Return scheduler group indices in sub-tensor order for one model layer.

    The model runner produces sub-tensors by iterating kv_cache_groups in
    index order.  For MLA/DSA tuples (homogeneous K/V/indexer within one
    group), all sub-tensors map to the same primary group.  For multi-spec
    layers, each sub-tensor corresponds to a distinct scheduler group.
    """
    subs = _listify_entry(entry)
    if _is_mla_or_dsa_tuple(entry):
        g = _primary_scheduler_group_for_layer(
            layer_name,
            kv_cache_config,
            ie_logical_block_size=ie_logical_block_size,
        )
        return [g] * len(subs)

    containing = _containing_groups_for_layer(layer_name, kv_cache_config)
    if len(containing) == len(subs):
        return containing
    # More sub-tensors than groups: some groups contribute multiple tensors
    # (e.g. duplicate indexer views). Match by block_size from the tensors.
    if len(containing) < len(subs):
        groups = kv_cache_config.kv_cache_groups
        group_block_sizes = {g: int(groups[g].kv_cache_spec.block_size) for g in containing}
        result: list[int] = []
        for t in subs:
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
        f"Layer {layer_name!r}: {len(subs)} sub-tensors vs "
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
    *,
    ie_logical_block_size: int | None = None,
) -> dict[str, list[int]]:
    """Map each model layer to ordered scheduler group indices for its sub-caches."""
    return {
        layer_name: ordered_scheduler_groups_for_layer(
            layer_name,
            kv_caches[layer_name],
            kv_cache_config,
            ie_logical_block_size=ie_logical_block_size,
        )
        for layer_name in layer_names
    }


def build_flat_kv_caches(
    kv_caches: dict[str, _KVEntry],
    kv_cache_config: Any,
    *,
    ie_logical_block_size: int | None = None,
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
        ie_logical_block_size=ie_logical_block_size,
    )
    bundled = should_bundle_multi_spec(kv_cache_config)
    for layer_name, entry in kv_caches.items():
        subs = _listify_entry(entry)
        groups = layer_to_groups[layer_name]
        if len(subs) != len(groups):
            raise ValueError(
                f"layer {layer_name}: {len(subs)} sub-tensors vs "
                f"{len(groups)} scheduler groups"
            )
        if bundled and _should_bundle_multi_spec_layer(entry):
            flat[layer_name] = tuple(subs)
            sched_by_layer.append(groups[0])
            continue
        for sub_idx, (sub_tensor, sched_g) in enumerate(zip(subs, groups)):
            flat[f"{layer_name}.sub{sub_idx}"] = _collapse_to_mla_page_buffer(
                sub_tensor
            )
            sched_by_layer.append(sched_g)
    return flat, tuple(sched_by_layer), layer_to_groups, bundled
