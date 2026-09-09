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
from lmcache_ascend.integration.vllm.skip_state_groups import (
    parse_skip_state_policy_from_env,
    should_skip_layer,
)
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
    names = set()
    policy = parse_skip_state_policy_from_env()
    for index, group in enumerate(groups):
        specs = [layer_spec(group, name) for name in group.layer_names]
        if group.is_eagle_group:
            raise ValueError("GDN checkpoints do not support MTP/eagle groups")
        for name, spec in zip(group.layer_names, specs, strict=True):
            if name in names:
                raise ValueError(
                    f"Hybrid layer {name!r} has ambiguous scheduler groups"
                )
            names.add(name)
            if should_skip_layer(layer_name=name, scheduler_group=group, policy=policy):
                raise ValueError(f"Cannot skip required hybrid layer {name!r}")
            if isinstance(spec, MambaSpec):
                validate_gdn_spec(spec)
            elif not (
                type(spec) is FullAttentionSpec
                and spec.sliding_window is None
                and spec.attention_chunk_size is None
            ):
                raise ValueError(
                    "GDN supports only full Attention and GDN state groups"
                )
        if any(isinstance(spec, MambaSpec) for spec in specs) and not all(
            isinstance(spec, MambaSpec) for spec in specs
        ):
            raise ValueError("Hybrid state and Attention require separate groups")
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


def validate_state_config(config: Any, vllm_config: Any) -> None:
    """The first hybrid path uses synchronous all-rank text/TP CPU/disk caching."""
    if vllm_config.speculative_config is not None:
        raise ValueError("GDN checkpoints do not support MTP/speculative decoding")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("GDN checkpoints do not support pipeline parallelism")
    # Qwen3.5 may expose a multimodal model config; request payloads are checked
    # before lookup and again in _attach_state_executions before transfer.
    for field in (
        "store_async",
        "enable_async_loading",
        "use_layerwise",
        "enable_blending",
        "enable_chunk_statistics",
        "enable_scheduler_bypass_lookup",
        "enable_pd",
        "enable_p2p",
        "enable_controller",
        "external_lookup_client",
        "remote_url",
        "storage_plugins",
        "remote_storage_plugins",
        "gds_path",
        "maru_path",
    ):
        if getattr(config, field):
            raise ValueError(f"GDN checkpoints do not support {field}")
    if config.hit_miss_ratio is not None:
        raise ValueError("GDN checkpoints do not support hit_miss_ratio")
    for field in (
        "save_only_first_rank",
        "enable_nixl_storage",
        "remove_after_retrieve",
        "audit_backend_enabled",
    ):
        if config.get_extra_config_value(field, False):
            raise ValueError(f"GDN checkpoints do not support {field}")
    world_size = vllm_config.parallel_config.tensor_parallel_size
    workers = config.get_lookup_server_worker_ids(False, world_size)
    if world_size <= 0 or (workers and sorted(workers) != list(range(world_size))):
        raise ValueError(
            "GDN checkpoints require all lookup workers; no worker subsets"
        )
    if config.lmcache_worker_ids and sorted(config.lmcache_worker_ids) != list(
        range(world_size)
    ):
        raise ValueError("GDN checkpoints do not support lmcache worker subsets")
    tiers = {"LocalCPUBackend", "LocalDiskBackend"}
    if (config.store_location is not None and config.store_location not in tiers) or (
        config.retrieve_locations is not None
        and (
            not config.retrieve_locations or not set(config.retrieve_locations) <= tiers
        )
    ):
        raise ValueError("GDN checkpoints support only local CPU/disk locations")
    if config.max_local_cpu_size <= 0 or not (
        config.local_cpu or (config.local_disk and config.max_local_disk_size > 0)
    ):
        raise ValueError("GDN checkpoints require a CPU allocator and a CPU/disk tier")
    if config.pin_timeout_sec <= 0:
        raise ValueError("GDN checkpoints require positive pin_timeout_sec")


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


def require_no_state_transfer(state_groups: Sequence[object]) -> None:
    """Keep raw state out of the KV-only flattening path."""
    if state_groups:
        raise NotImplementedError(
            "Raw GDN state requires hybrid registration, not KV flattening"
        )
