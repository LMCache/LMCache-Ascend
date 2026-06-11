# SPDX-License-Identifier: Apache-2.0
"""Env-only policy helpers for skipping scheduler groups at registration time."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
    skipped_scheduler_groups: set[int] | frozenset[int],
    *,
    bundled: bool,
) -> tuple[dict[str, _KVEntry], tuple[int, ...], dict[str, list[int]]]:
    """Filter flattened registration artifacts so skipped groups never reach planning."""
    kept_layer_to_groups = {
        layer: [int(g) for g in groups]
        for layer, groups in layer_to_scheduler_groups.items()
    }
    kept_sched = tuple(sched_by_layer)

    # Nothing to skip: pass through flattened artifacts with normalized metadata copies.
    # Downstream layout hints and group planning still expect concrete dict/tuple types.
    if not skipped_scheduler_groups:
        return dict(flat_kv), kept_sched, kept_layer_to_groups

    if len(flat_kv) != len(sched_by_layer):
        raise ValueError(
            "flat_kv and scheduler_group_by_flat_layer length mismatch: "
            f"{len(flat_kv)} vs {len(sched_by_layer)}"
        )

    kept_flat: dict[str, _KVEntry] = {}
    kept_sched_list: list[int] = []
    kept_layer_to_groups_out: dict[str, list[int]] = {}

    # Exploded flattening: each flat entry owns one scheduler group.
    # Drop skipped entries outright and rebuild the parallel sched map.
    if not bundled:
        for (layer_name, entry), sched_g in zip(flat_kv.items(), sched_by_layer):
            g = int(sched_g)
            if g in skipped_scheduler_groups:
                continue
            kept_flat[layer_name] = entry
            kept_sched_list.append(g)

        for layer_name, groups in layer_to_scheduler_groups.items():
            filtered = [
                int(g) for g in groups if int(g) not in skipped_scheduler_groups
            ]
            if filtered:
                kept_layer_to_groups_out[layer_name] = filtered
        return kept_flat, tuple(kept_sched_list), kept_layer_to_groups_out

    # Bundled flattening: one flat entry may hold multiple sub-tensor planes.
    # Slice tuple planes to kept groups, or drop single-tensor entries whose group is skipped.
    for flat_idx, (layer_name, entry) in enumerate(flat_kv.items()):
        layer_groups = [int(g) for g in layer_to_scheduler_groups.get(layer_name, ())]
        fallback_sched = int(sched_by_layer[flat_idx])

        # Multi-plane layer: keep only planes whose scheduler group is not skipped.
        # Rebuild the per-layer group list to match the shortened tuple.
        if isinstance(entry, (tuple, list)) and layer_groups:
            keep_idx = [
                i
                for i, sched_g in enumerate(layer_groups)
                if int(sched_g) not in skipped_scheduler_groups
            ]
            if not keep_idx:
                continue
            kept_flat[layer_name] = _filter_multi_plane_entry(entry, keep_idx)
            filtered_groups = [layer_groups[i] for i in keep_idx]
            # 0 because it is the same convention for the primary group selection as in the multi_spec_flatten.py
            kept_sched_list.append(filtered_groups[0])
            kept_layer_to_groups_out[layer_name] = filtered_groups
            continue

        # Single-tensor flat entry (e.g. dense layer.sub0): keep or drop by its sched group.
        # Prune skipped group IDs from the layer mapping when any planes remain.
        if fallback_sched in skipped_scheduler_groups:
            continue
        kept_flat[layer_name] = entry
        kept_sched_list.append(fallback_sched)
        filtered_groups = [
            int(g) for g in layer_groups if int(g) not in skipped_scheduler_groups
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
    skipped_groups = resolve_skipped_scheduler_groups(kv_cache_config, policy)
    return apply_skip_filter_to_flattened(
        flat_kv,
        sched_by_layer,
        layer_to_scheduler_groups,
        skipped_groups,
        bundled=bundled,
    )
