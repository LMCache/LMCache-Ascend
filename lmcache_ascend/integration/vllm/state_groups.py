# SPDX-License-Identifier: Apache-2.0
"""GDN scheduler semantics, before any tensor-shape based KV interpretation."""

# Standard
from typing import Any, Sequence

# Third Party
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
import torch

# First Party
from lmcache_ascend.v1.state_layout import StateGroupLayout, build_state_group_layout


def layer_spec(group: Any, layer_name: str) -> Any:
    """Resolve a layer's spec, including vLLM uniform-type group wrappers."""
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name]
    return spec


def validate_gdn_spec(spec: MambaSpec) -> None:
    """Reject state kinds/modes outside the non-speculative GDN contract."""
    if spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN:
        raise ValueError(f"Unsupported state backend: {spec.mamba_type}")
    if spec.mamba_cache_mode != "align" or spec.num_speculative_blocks != 0:
        raise ValueError(
            "GDN checkpoints require align mode without speculative blocks"
        )
    if len(spec.shapes) != 2 or len(spec.dtypes) != 2:
        raise ValueError("GDN checkpoints require conv and SSM planes")


def state_group_index(kv_cache_config: Any, layer_name: str) -> int | None:
    """Return the unique state group for a layer, or None for existing KV paths."""
    containing = [
        (index, group)
        for index, group in enumerate(kv_cache_config.kv_cache_groups)
        if layer_name in group.layer_names
    ]
    states = [
        index
        for index, group in containing
        if isinstance(layer_spec(group, layer_name), MambaSpec)
    ]
    if not states:
        return None
    if len(containing) != 1:
        raise ValueError(f"State layer {layer_name!r} has ambiguous scheduler groups")
    validate_gdn_spec(layer_spec(containing[0][1], layer_name))
    return states[0]


def validate_state_planes(
    layer_name: str, spec: MambaSpec, planes: Sequence[torch.Tensor]
) -> None:
    """Check real state tensors against their spec, without converting data."""
    validate_gdn_spec(spec)
    if len(planes) != len(spec.shapes):
        raise ValueError(f"State layer {layer_name!r}: plane count does not match spec")
    num_blocks = None
    for index, (tensor, shape, dtype) in enumerate(
        zip(planes, spec.shapes, spec.dtypes, strict=True)
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != len(shape) + 1
            or tuple(tensor.shape[1:]) != tuple(shape)
            or tensor.dtype != dtype
            or tensor.shape[0] <= 0
        ):
            raise ValueError(
                f"State layer {layer_name!r}, plane {index}: spec mismatch"
            )
        if num_blocks is not None and tensor.shape[0] != num_blocks:
            raise ValueError(f"State layer {layer_name!r}: inconsistent block counts")
        num_blocks = tensor.shape[0]


def select_state_primary(kv_cache_config: Any) -> int | None:
    """Select one full-attention primary for GDN; leave non-GDN policy unchanged."""
    groups = kv_cache_config.kv_cache_groups
    has_state = any(
        isinstance(layer_spec(group, name), MambaSpec)
        for group in groups
        for name in group.layer_names
    )
    if not has_state:
        return None
    candidates = []
    for index, group in enumerate(groups):
        specs = [layer_spec(group, name) for name in group.layer_names]
        if (
            specs
            and all(
                isinstance(spec, FullAttentionSpec)
                and spec.sliding_window is None
                and spec.attention_chunk_size is None
                for spec in specs
            )
            and not group.is_eagle_group
        ):
            candidates.append(index)
    if len(candidates) != 1:
        raise ValueError("GDN requires one unambiguous full-attention primary group")
    return candidates[0]


def request_primary(
    block_ids: Sequence[Sequence[int]],
    block_sizes: Sequence[int],
    state_primary: int | None,
) -> int:
    """Use the GDN spec primary; preserve the existing policy for other models."""
    if state_primary is not None:
        if not 0 <= state_primary < len(block_ids):
            raise ValueError("The configured full-attention primary is missing")
        return state_primary
    return max(range(len(block_ids)), key=lambda i: len(block_ids[i]) * block_sizes[i])


def build_state_layouts(
    kv_cache_config: Any, kv_caches: dict[str, Sequence[torch.Tensor]]
) -> tuple[StateGroupLayout, ...]:
    """Build each GDN group's layout from specs and real registered tensors."""
    layouts = []
    for index, group in enumerate(kv_cache_config.kv_cache_groups):
        specs = [layer_spec(group, name) for name in group.layer_names]
        if not any(isinstance(spec, MambaSpec) for spec in specs):
            continue
        entries = []
        for name, spec in zip(group.layer_names, specs, strict=True):
            if state_group_index(kv_cache_config, name) != index:
                raise ValueError(f"State group {index}: inconsistent layer {name!r}")
            if name not in kv_caches:
                raise ValueError(f"State group {index}: missing tensors for {name!r}")
            entry = kv_caches[name]
            if not isinstance(entry, (tuple, list)):
                raise ValueError(f"State layer {name!r} requires a plane sequence")
            validate_state_planes(name, spec, entry)
            entries.append(entry)
        layouts.append(build_state_group_layout(index, group.layer_names, entries))
    return tuple(layouts)
