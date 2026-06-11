# SPDX-License-Identifier: Apache-2.0
"""Tests for registration-time skip-state scheduler group filtering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lmcache_ascend.integration.vllm.multi_spec_flatten import build_flat_kv_caches
from lmcache_ascend.integration.vllm.skip_state_groups import (
    DEFAULT_SKIP_STATE_SPEC_NAMES,
    apply_skip_filter_to_flattened,
    parse_skip_state_policy_from_env,
    resolve_skipped_scheduler_groups,
)

from .conftest_ds4 import (
    make_ds4_kv_caches_dict,
    set_bundle_multi_spec_env,
    set_skip_state_groups_env,
)
from .test_layer_to_scheduler_group_mapping import L2, _make_ds4_kv_cache_config


def test_parse_skip_policy_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    set_skip_state_groups_env(monkeypatch, enabled=False)
    assert parse_skip_state_policy_from_env() is None


def test_parse_skip_policy_enabled_without_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_skip_state_groups_env(monkeypatch, enabled=True, allowlist=None)
    policy = parse_skip_state_policy_from_env()
    assert policy is not None
    assert policy.enabled is True
    assert policy.spec_allowlist == frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES)


def test_parse_skip_policy_enabled_empty_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_skip_state_groups_env(monkeypatch, enabled=True, allowlist="")
    policy = parse_skip_state_policy_from_env()
    assert policy is not None
    assert policy.enabled is True
    assert policy.spec_allowlist == frozenset()


def test_parse_skip_policy_enabled_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_skip_state_groups_env(
        monkeypatch,
        enabled=True,
        allowlist="C4AttnKVStateSpec, C4AttnScoreStateSpec",
    )
    policy = parse_skip_state_policy_from_env()
    assert policy is not None
    assert policy.enabled is True
    assert policy.spec_allowlist == frozenset(
        {"C4AttnKVStateSpec", "C4AttnScoreStateSpec"}
    )


def test_resolve_skipped_scheduler_groups_allowlist() -> None:
    cfg = _make_ds4_kv_cache_config()
    policy = SimpleNamespace(
        enabled=True,
        spec_allowlist=frozenset({"C4AttnKVStateSpec", "C4AttnScoreStateSpec"}),
    )
    skipped = resolve_skipped_scheduler_groups(cfg, policy)
    assert skipped == frozenset({4, 5})


def test_resolve_skipped_scheduler_groups_default_allowlist() -> None:
    cfg = _make_ds4_kv_cache_config()
    policy = SimpleNamespace(
        enabled=True,
        spec_allowlist=frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES),
    )
    skipped = resolve_skipped_scheduler_groups(cfg, policy)
    assert skipped == frozenset({4, 5, 6, 7, 10, 11})


def test_resolve_skipped_scheduler_groups_empty_allowlist() -> None:
    cfg = _make_ds4_kv_cache_config()
    policy = SimpleNamespace(enabled=True, spec_allowlist=frozenset())
    skipped = resolve_skipped_scheduler_groups(cfg, policy)
    assert skipped == frozenset()


def test_apply_skip_filter_bundled_reduces_planes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_bundle_multi_spec_env(monkeypatch, enabled=True)
    dev = torch.device("cpu")
    kv_dict = make_ds4_kv_caches_dict(dev, num_blocks=8)
    cfg = _make_ds4_kv_cache_config()
    flat, sched, layer_to_groups, bundled = build_flat_kv_caches(kv_dict, cfg)
    assert bundled is True

    skipped = frozenset({4, 5, 6, 7})
    filtered = apply_skip_filter_to_flattened(
        flat,
        sched,
        layer_to_groups,
        skipped,
        bundled=bundled,
    )
    filtered_flat, filtered_sched, filtered_layer_to_groups = filtered

    assert L2 in filtered_flat
    assert isinstance(filtered_flat[L2], tuple)
    assert len(filtered_flat[L2]) == 4
    assert filtered_layer_to_groups[L2] == [0, 2, 3, 3]
    assert len(filtered_sched) == len(filtered_flat)


def test_apply_skip_filter_exploded_drops_flat_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_bundle_multi_spec_env(monkeypatch, enabled=False)
    dev = torch.device("cpu")
    kv_dict = make_ds4_kv_caches_dict(dev, num_blocks=8)
    cfg = _make_ds4_kv_cache_config()
    flat, sched, layer_to_groups, bundled = build_flat_kv_caches(kv_dict, cfg)
    assert bundled is False

    skipped = frozenset({4, 5, 6, 7})
    filtered = apply_skip_filter_to_flattened(
        flat,
        sched,
        layer_to_groups,
        skipped,
        bundled=bundled,
    )
    filtered_flat, filtered_sched, filtered_layer_to_groups = filtered

    assert len(filtered_flat) == len(flat) - 4
    assert len(filtered_sched) == len(filtered_flat)
    assert all(int(g) not in skipped for g in filtered_sched)
    assert filtered_layer_to_groups[L2] == [0, 2, 3, 3]
