# SPDX-License-Identifier: Apache-2.0
"""Env-only policy helpers for skipping scheduler groups at registration time."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lmcache.logging import init_logger
import torch

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

logger = init_logger(__name__)


@dataclass(frozen=True)
class SkipStateGroupsPolicy:
    """Configuration for env-driven state-group skipping at registration."""

    enabled: bool
    spec_allowlist: frozenset[str] | None


def effective_spec_allowlist(policy: SkipStateGroupsPolicy | None) -> frozenset[str]:
    """Return the spec allowlist for one policy, applying defaults when unset."""
    if policy is None or not policy.enabled:
        return frozenset()
    if policy.spec_allowlist is not None:
        return policy.spec_allowlist
    return frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES)


def parse_skip_state_policy_from_env() -> SkipStateGroupsPolicy | None:
    """Read skip policy from env and return None when the feature is disabled."""
    if os.environ.get("LMCACHE_ASCEND_SKIP_STATE_GROUPS", "1") != "1":
        return None
    raw_allowlist = os.environ.get("LMCACHE_ASCEND_SKIP_STATE_SPEC_ALLOWLIST")
    if raw_allowlist is None:
        return SkipStateGroupsPolicy(
            enabled=True,
            spec_allowlist=frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES),
        )
    parsed = frozenset(
        token.strip() for token in raw_allowlist.split(",") if token.strip()
    )
    return SkipStateGroupsPolicy(enabled=True, spec_allowlist=parsed)


def _spec_name(group: Any) -> str:
    """Return kv_cache_spec class name for one scheduler group entry."""
    spec = getattr(group, "kv_cache_spec", None)
    return type(spec).__name__ if spec is not None else "UnknownSpec"


def resolve_skipped_scheduler_groups(
    kv_cache_config: Any,
    policy: SkipStateGroupsPolicy | None,
) -> frozenset[int]:
    """Resolve scheduler group indices to skip using explicit spec allowlist."""
    if policy is None or not policy.enabled or kv_cache_config is None:
        return frozenset()
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    skipped: set[int] = set()
    allowlist = effective_spec_allowlist(policy)
    if not allowlist:
        return frozenset()

    for idx, group in enumerate(groups):
        if _spec_name(group) in allowlist:
            skipped.add(idx)
    return frozenset(skipped)


def skipped_group_spec_names(
    kv_cache_config: Any,
    skipped_scheduler_groups: Sequence[int],
) -> tuple[str, ...]:
    """Return spec class names for skipped scheduler group indices."""
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    out: list[str] = []
    for idx in skipped_scheduler_groups:
        if 0 <= int(idx) < len(groups):
            out.append(_spec_name(groups[int(idx)]))
    return tuple(out)


def active_plane_indices(
    layer_name: str,
    layer_to_scheduler_groups: Mapping[str, Sequence[int]],
    skipped: set[int] | frozenset[int],
) -> list[int]:
    """Return per-layer plane indices whose scheduler groups are not skipped."""
    groups = list(layer_to_scheduler_groups.get(layer_name, ()))
    return [i for i, sched_g in enumerate(groups) if int(sched_g) not in skipped]


def filter_multi_plane_entry(
    entry: _KVEntry,
    active_indices: Sequence[int],
) -> _KVEntry:
    """Slice tuple/list entries to active indices; tensors are returned unchanged."""
    if isinstance(entry, tuple):
        return tuple(entry[i] for i in active_indices)
    if isinstance(entry, list):
        return [entry[i] for i in active_indices]
    return entry


def apply_skip_filter_to_flattened(
    flat_kv: Mapping[str, _KVEntry],
    sched_by_layer: Sequence[int] | None,
    layer_to_scheduler_groups: Mapping[str, Sequence[int]] | None,
    skipped_scheduler_groups: set[int] | frozenset[int],
    *,
    bundled: bool,
) -> tuple[
    dict[str, _KVEntry],
    tuple[int, ...] | None,
    dict[str, list[int]] | None,
]:
    """Filter flattened registration artifacts so skipped groups never reach planning."""
    # Fast path: no skipped groups means no filtering is needed.
    # Return normalized copies so downstream types stay stable.
    if not skipped_scheduler_groups:
        kept_layer_to_groups = (
            {
                layer: [int(g) for g in groups]
                for layer, groups in layer_to_scheduler_groups.items()
            }
            if layer_to_scheduler_groups is not None
            else None
        )
        kept_sched = tuple(sched_by_layer) if sched_by_layer is not None else None
        return dict(flat_kv), kept_sched, kept_layer_to_groups

    # Defensive path: sched map unavailable, so only metadata can be pruned.
    # Keep flat entries untouched and remove skipped IDs from layer mapping.
    if sched_by_layer is None:
        kept_layer_to_groups = (
            {
                layer: [
                    int(g) for g in groups if int(g) not in skipped_scheduler_groups
                ]
                for layer, groups in (layer_to_scheduler_groups or {}).items()
                if any(int(g) not in skipped_scheduler_groups for g in groups)
            }
            if layer_to_scheduler_groups is not None
            else None
        )
        return dict(flat_kv), None, kept_layer_to_groups

    if len(flat_kv) != len(sched_by_layer):
        raise ValueError(
            "flat_kv and scheduler_group_by_flat_layer length mismatch: "
            f"{len(flat_kv)} vs {len(sched_by_layer)}"
        )

    kept_flat: dict[str, _KVEntry] = {}
    kept_sched: list[int] = []
    kept_layer_to_groups: dict[str, list[int]] = {}

    # Exploded flattening: one flat entry maps to one scheduler group.
    # Drop skipped entries directly and rebuild aligned outputs.
    if not bundled:
        for (layer_name, entry), sched_g in zip(flat_kv.items(), sched_by_layer):
            g = int(sched_g)
            if g in skipped_scheduler_groups:
                continue
            kept_flat[layer_name] = entry
            kept_sched.append(g)
        if layer_to_scheduler_groups is not None:
            for layer_name, groups in layer_to_scheduler_groups.items():
                filtered = [
                    int(g) for g in groups if int(g) not in skipped_scheduler_groups
                ]
                if filtered:
                    kept_layer_to_groups[layer_name] = filtered
        return (
            kept_flat,
            tuple(kept_sched),
            kept_layer_to_groups if layer_to_scheduler_groups is not None else None,
        )

    # Bundled flattening: one flat entry can include multiple scheduler groups.
    # Keep only active planes and drop entries with zero remaining planes.
    layer_groups_map = layer_to_scheduler_groups or {}
    for flat_idx, (layer_name, entry) in enumerate(flat_kv.items()):
        layer_groups = [int(g) for g in layer_groups_map.get(layer_name, ())]
        fallback_sched = int(sched_by_layer[flat_idx])

        # Tuple/list entries carry planes; filter by active plane indices.
        # The first remaining scheduler group is the flat entry fallback group.
        if isinstance(entry, (tuple, list)) and layer_groups:
            keep_idx = active_plane_indices(
                layer_name,
                layer_groups_map,
                skipped_scheduler_groups,
            )
            if not keep_idx:
                continue
            filtered_entry = filter_multi_plane_entry(entry, keep_idx)
            filtered_groups = [layer_groups[i] for i in keep_idx]
            kept_flat[layer_name] = filtered_entry
            kept_sched.append(filtered_groups[0])
            kept_layer_to_groups[layer_name] = filtered_groups
            continue

        # Non-tuple entries behave like single-group flat items.
        # Keep only if their fallback scheduler group is not skipped.
        if fallback_sched in skipped_scheduler_groups:
            continue
        kept_flat[layer_name] = entry
        kept_sched.append(fallback_sched)
        if layer_groups:
            filtered_groups = [
                int(g) for g in layer_groups if int(g) not in skipped_scheduler_groups
            ]
            if filtered_groups:
                kept_layer_to_groups[layer_name] = filtered_groups

    return (
        kept_flat,
        tuple(kept_sched),
        kept_layer_to_groups if layer_to_scheduler_groups is not None else None,
    )


def _log_skip_state_filter_details(
    kv_cache_config: Any,
    policy: SkipStateGroupsPolicy | None,
    skipped_groups: frozenset[int],
    flat_kv: Mapping[str, _KVEntry],
    sched_by_layer: Sequence[int] | None,
    layer_to_scheduler_groups: Mapping[str, Sequence[int]] | None,
    filtered_flat: Mapping[str, _KVEntry],
    filtered_sched: Sequence[int] | None,
    filtered_layer_to_groups: Mapping[str, list[int]] | None,
    *,
    bundled: bool,
) -> None:
    """Emit debug logs describing which scheduler groups and planes were skipped."""
    if policy is None:
        logger.debug(
            "Skip-state groups: disabled via LMCACHE_ASCEND_SKIP_STATE_GROUPS"
        )
        return

    if not skipped_groups:
        allowlist = effective_spec_allowlist(policy)
        logger.info(
            "Skip-state groups: allowlist=%s matched 0 scheduler groups",
            sorted(allowlist),
        )
        return

    skipped_sorted = tuple(sorted(int(g) for g in skipped_groups))
    skipped_specs = skipped_group_spec_names(kv_cache_config, skipped_sorted)
    logger.info(
        "Skip-state groups enabled: skipped_groups=%s skipped_specs=%s bundled=%s",
        skipped_sorted,
        skipped_specs,
        bundled,
    )

    if layer_to_scheduler_groups is not None:
        for layer_name, groups in layer_to_scheduler_groups.items():
            sched_groups = [int(g) for g in groups]
            skipped_planes = [
                plane_idx
                for plane_idx, sched_g in enumerate(sched_groups)
                if sched_g in skipped_groups
            ]
            kept_planes = active_plane_indices(
                layer_name,
                layer_to_scheduler_groups,
                skipped_groups,
            )
            kept_sched_groups = [sched_groups[i] for i in kept_planes]
            skipped_sched_groups = [sched_groups[i] for i in skipped_planes]
            if not skipped_planes:
                continue
            logger.info(
                "Skip-state layer=%s skipped_planes=%s skipped_sched_groups=%s "
                "kept_planes=%s kept_sched_groups=%s",
                layer_name,
                skipped_planes,
                skipped_sched_groups,
                kept_planes,
                kept_sched_groups,
            )

    if not bundled and sched_by_layer is not None:
        dropped_layers = [
            layer_name
            for layer_name, sched_g in zip(flat_kv.keys(), sched_by_layer)
            if int(sched_g) in skipped_groups
        ]
        if dropped_layers:
            logger.info(
                "Skip-state exploded flatten dropped %d flat entries: %s",
                len(dropped_layers),
                dropped_layers,
            )

    logger.info(
        "Skip-state filter result: flat_layers %d -> %d sched_by_layer=%s -> %s",
        len(flat_kv),
        len(filtered_flat),
        len(sched_by_layer) if sched_by_layer is not None else None,
        len(filtered_sched) if filtered_sched is not None else None,
    )
    if filtered_layer_to_groups is not None and layer_to_scheduler_groups is not None:
        for layer_name in layer_to_scheduler_groups:
            before = [int(g) for g in layer_to_scheduler_groups.get(layer_name, ())]
            after = [int(g) for g in filtered_layer_to_groups.get(layer_name, ())]
            if before != after:
                logger.debug(
                    "Skip-state layer=%s scheduler_groups %s -> %s",
                    layer_name,
                    before,
                    after,
                )


def apply_skip_policy_from_env_to_flattened(
    kv_cache_config: Any,
    flat_kv: Mapping[str, _KVEntry],
    sched_by_layer: Sequence[int] | None,
    layer_to_scheduler_groups: Mapping[str, Sequence[int]] | None,
    *,
    bundled: bool,
) -> tuple[
    dict[str, _KVEntry],
    tuple[int, ...] | None,
    dict[str, list[int]] | None,
]:
    """Apply env-driven skip policy to flattened artifacts and emit optional logs."""
    policy = parse_skip_state_policy_from_env()
    skipped_groups = resolve_skipped_scheduler_groups(kv_cache_config, policy)

    filtered = apply_skip_filter_to_flattened(
        flat_kv,
        sched_by_layer,
        layer_to_scheduler_groups,
        skipped_groups,
        bundled=bundled,
    )
    filtered_flat, filtered_sched, filtered_layer_to_groups = filtered
    _log_skip_state_filter_details(
        kv_cache_config,
        policy,
        skipped_groups,
        flat_kv,
        sched_by_layer,
        layer_to_scheduler_groups,
        filtered_flat,
        filtered_sched,
        filtered_layer_to_groups,
        bundled=bundled,
    )
    return filtered
