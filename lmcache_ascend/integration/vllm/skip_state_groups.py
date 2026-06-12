# SPDX-License-Identifier: Apache-2.0
"""Env-only policy helpers for skipping scheduler groups at registration time."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from lmcache.logging import init_logger

logger = init_logger(__name__)

_KVEntry = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]

# Default allowlist when LMCACHE_ASCEND_SKIP_STATE_SPEC_ALLOWLIST is unset.
DEFAULT_SKIP_STATE_SPEC_NAMES = (
    "C4AttnKVStateSpec",
    "C4AttnScoreStateSpec",
    "C4IndexerKVStateSpec",
    "C4IndexerScoreStateSpec",
    "C128AttnKVStateSpec",
    "C128AttnScoreStateSpec",
)

# Default layer-name suffix when LMCACHE_ASCEND_SKIP_STATE_LAYER_SUFFIX is unset.
DEFAULT_SKIP_STATE_SUFFIX = ".state_cache"


@dataclass(frozen=True)
class SkipStateGroupsPolicy:
    """Configuration for env-driven state-group skipping at registration."""

    enabled: bool
    spec_allowlist: frozenset[str] | None
    layer_name_suffix: str | None = None


def effective_spec_allowlist(policy: SkipStateGroupsPolicy | None) -> frozenset[str]:
    """Return the spec allowlist for one policy, applying defaults when unset."""
    if policy is None or not policy.enabled:
        return frozenset()
    if policy.spec_allowlist is not None:
        return policy.spec_allowlist
    return frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES)


def effective_layer_name_suffix(policy: SkipStateGroupsPolicy | None) -> str:
    """Return the layer-name suffix for one policy, applying defaults when unset."""
    if policy is None or not policy.enabled:
        return DEFAULT_SKIP_STATE_SUFFIX
    suffix = getattr(policy, "layer_name_suffix", None)
    if suffix is not None:
        return suffix
    return DEFAULT_SKIP_STATE_SUFFIX


def parse_skip_state_policy_from_env() -> SkipStateGroupsPolicy | None:
    """Read skip policy from env and return None when the feature is disabled."""
    if os.environ.get("LMCACHE_ASCEND_SKIP_STATE_GROUPS", "1") != "1":
        return None
    raw_allowlist = os.environ.get("LMCACHE_ASCEND_SKIP_STATE_SPEC_ALLOWLIST")
    raw_suffix = os.environ.get("LMCACHE_ASCEND_SKIP_STATE_LAYER_SUFFIX")
    if raw_allowlist is None:
        spec_allowlist: frozenset[str] | None = frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES)
    else:
        spec_allowlist = frozenset(
            token.strip() for token in raw_allowlist.split(",") if token.strip()
        )
    layer_name_suffix = None if raw_suffix is None else raw_suffix
    return SkipStateGroupsPolicy(
        enabled=True,
        spec_allowlist=spec_allowlist,
        layer_name_suffix=layer_name_suffix,
    )


def _spec_name(group: Any) -> str:
    """Return kv_cache_spec class name for one scheduler group entry."""
    spec = getattr(group, "kv_cache_spec", None)
    return type(spec).__name__ if spec is not None else "UnknownSpec"


def _inner_layer_names(group: Any) -> list[str]:
    spec = getattr(group, "kv_cache_spec", None)
    if spec is None or type(spec).__name__ != "UniformTypeKVCacheSpecs":
        return []
    inner = getattr(spec, "kv_cache_specs", None) or {}
    return list(inner.keys())


def should_skip_layer(
    *,
    layer_name: str | None = None,
    scheduler_group: Any | None = None,
    policy: SkipStateGroupsPolicy | None,
) -> bool:
    """Return True when a flat layer or scheduler group should be skipped.

    Matching uses either path from ``policy``:

    * **Layer name suffix** — v0.20 ``CompressorStateCache`` modules end with
      ``effective_layer_name_suffix(policy)`` (default ``DEFAULT_SKIP_STATE_SUFFIX``).
    * **Scheduler spec allowlist** — v0.18 typed group specs such as
      ``C4AttnKVStateSpec`` (see ``DEFAULT_SKIP_STATE_SPEC_NAMES``).
    * **v0.20 UniformType groups** — every ``layer_names`` entry, or every
      ``kv_cache_specs`` key, ends with ``effective_layer_name_suffix(policy)``.
    """
    if policy is None or not policy.enabled:
        return False

    suffix = effective_layer_name_suffix(policy)
    if layer_name is not None and layer_name.endswith(suffix):
        return True
    if scheduler_group is None:
        return False

    allowlist = effective_spec_allowlist(policy)
    if allowlist and _spec_name(scheduler_group) in allowlist:
        return True

    spec = getattr(scheduler_group, "kv_cache_spec", None)
    if spec is None or type(spec).__name__ != "UniformTypeKVCacheSpecs":
        return False

    layer_names = list(getattr(scheduler_group, "layer_names", []) or [])
    if layer_names and all(name.endswith(suffix) for name in layer_names):
        return True

    inner_names = _inner_layer_names(scheduler_group)
    return bool(inner_names) and all(name.endswith(suffix) for name in inner_names)


def resolve_skipped_scheduler_groups(
    kv_cache_config: Any,
    policy: SkipStateGroupsPolicy | None,
) -> frozenset[int]:
    """Resolve scheduler group indices to skip using the unified skip predicate."""
    if policy is None or not policy.enabled or kv_cache_config is None:
        return frozenset()
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    skipped: set[int] = set()
    for idx, group in enumerate(groups):
        if should_skip_layer(scheduler_group=group, policy=policy):
            skipped.add(idx)
    return frozenset(skipped)


def _filter_multi_plane_entry(
    entry: tuple[torch.Tensor, ...] | list[torch.Tensor],
    active_indices: Sequence[int],
) -> tuple[torch.Tensor, ...] | list[torch.Tensor]:
    """Slice tuple/list entries to active plane indices."""
    if isinstance(entry, tuple):
        return tuple(entry[i] for i in active_indices)
    return [entry[i] for i in active_indices]


def apply_skip_filter_to_flattened(
    flat_kv: Mapping[str, _KVEntry],
    sched_by_layer: Sequence[int],
    layer_to_scheduler_groups: Mapping[str, Sequence[int]],
    *,
    kv_cache_config: Any | None,
    bundled: bool,
    policy: SkipStateGroupsPolicy | None,
) -> tuple[dict[str, _KVEntry], tuple[int, ...], dict[str, list[int]]]:
    """Filter flattened registration artifacts so skipped groups never reach planning."""
    kept_layer_to_groups = {
        layer: [int(g) for g in groups]
        for layer, groups in layer_to_scheduler_groups.items()
    }
    kept_sched = tuple(sched_by_layer)

    if policy is None or not policy.enabled:
        return dict(flat_kv), kept_sched, kept_layer_to_groups

    skipped_groups = resolve_skipped_scheduler_groups(kv_cache_config, policy)
    if not skipped_groups and not any(
        should_skip_layer(layer_name=name, policy=policy) for name in flat_kv
    ):
        return dict(flat_kv), kept_sched, kept_layer_to_groups

    if len(flat_kv) != len(sched_by_layer):
        raise ValueError(
            "flat_kv and scheduler_group_by_flat_layer length mismatch: "
            f"{len(flat_kv)} vs {len(sched_by_layer)}"
        )

    kept_flat: dict[str, _KVEntry] = {}
    kept_sched_list: list[int] = []
    kept_layer_to_groups_out: dict[str, list[int]] = {}

    def _drop_flat_layer(layer_name: str, sched_g: int) -> bool:
        return should_skip_layer(
            layer_name=layer_name,
            policy=policy,
        ) or int(sched_g) in skipped_groups

    # Unbundled path: one flat entry maps to exactly one scheduler group.
    if not bundled:
        for (layer_name, entry), sched_g in zip(flat_kv.items(), sched_by_layer):
            if _drop_flat_layer(layer_name, int(sched_g)):
                continue
            kept_flat[layer_name] = entry
            kept_sched_list.append(int(sched_g))

        for layer_name, groups in layer_to_scheduler_groups.items():
            if should_skip_layer(layer_name=layer_name, policy=policy):
                continue
            filtered = [int(g) for g in groups if int(g) not in skipped_groups]
            if filtered:
                kept_layer_to_groups_out[layer_name] = filtered
        return kept_flat, tuple(kept_sched_list), kept_layer_to_groups_out

    # Bundled path: one logical layer may span multiple KV planes and scheduler groups.
    for flat_idx, (layer_name, entry) in enumerate(flat_kv.items()):
        if should_skip_layer(layer_name=layer_name, policy=policy):
            continue

        layer_groups = [int(g) for g in layer_to_scheduler_groups.get(layer_name, ())]
        fallback_sched = int(sched_by_layer[flat_idx])

        # Multi-plane entry: keep only planes whose scheduler groups were not skipped.
        if isinstance(entry, (tuple, list)) and layer_groups:
            keep_idx = [
                i
                for i, sched_g in enumerate(layer_groups)
                if int(sched_g) not in skipped_groups
            ]
            if not keep_idx:
                continue
            kept_flat[layer_name] = _filter_multi_plane_entry(entry, keep_idx)
            filtered_groups = [layer_groups[i] for i in keep_idx]
            kept_sched_list.append(filtered_groups[0])
            kept_layer_to_groups_out[layer_name] = filtered_groups
            continue

        if _drop_flat_layer(layer_name, fallback_sched):
            continue
        kept_flat[layer_name] = entry
        kept_sched_list.append(fallback_sched)
        filtered_groups = [
            int(g) for g in layer_groups if int(g) not in skipped_groups
        ]
        if filtered_groups:
            kept_layer_to_groups_out[layer_name] = filtered_groups

    return kept_flat, tuple(kept_sched_list), kept_layer_to_groups_out


def apply_skip_policy_from_env_to_flattened(
    kv_cache_config: Any,
    flat_kv: Mapping[str, _KVEntry],
    sched_by_layer: Sequence[int],
    layer_to_scheduler_groups: Mapping[str, Sequence[int]],
    *,
    bundled: bool,
) -> tuple[dict[str, _KVEntry], tuple[int, ...], dict[str, list[int]]]:
    """Apply env-driven skip policy to flattened registration artifacts."""
    policy = parse_skip_state_policy_from_env()
    before = len(flat_kv)
    flat_out, sched_out, layer_to_groups_out = apply_skip_filter_to_flattened(
        flat_kv,
        sched_by_layer,
        layer_to_scheduler_groups,
        kv_cache_config=kv_cache_config,
        bundled=bundled,
        policy=policy,
    )
    if policy is not None and policy.enabled and before != len(flat_out):
        skipped_groups = resolve_skipped_scheduler_groups(kv_cache_config, policy)
        logger.info(
            "Skip-state groups: skipped_scheduler_groups=%s "
            "flat_layers %d -> %d (bundled=%s)",
            sorted(skipped_groups),
            before,
            len(flat_out),
            bundled,
        )
    return flat_out, sched_out, layer_to_groups_out
