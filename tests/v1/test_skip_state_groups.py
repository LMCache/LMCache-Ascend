# SPDX-License-Identifier: Apache-2.0
"""Tests for registration-time skip-state scheduler group filtering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lmcache_ascend.integration.vllm.multi_spec_flatten import build_flat_kv_caches
from lmcache_ascend.integration.vllm.skip_state_groups import (
    DEFAULT_SKIP_STATE_SPEC_NAMES,
    DEFAULT_SKIP_STATE_SUFFIX,
    SkipStateGroupsPolicy,
    apply_skip_filter_to_flattened,
    effective_layer_name_suffix,
    parse_skip_state_policy_from_env,
    resolve_skipped_scheduler_groups,
    should_skip_layer,
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
    assert policy.layer_name_suffix is None
    assert effective_layer_name_suffix(policy) == DEFAULT_SKIP_STATE_SUFFIX


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


def test_resolve_skipped_v020_empty_allowlist_still_skips_state_cache() -> None:
    class UniformTypeKVCacheSpecs:
        def __init__(self, specs: dict, block_size: int = 8) -> None:
            self.kv_cache_specs = specs
            self.block_size = block_size

    cfg = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    {"model.layers.0.self_attn.compressor.state_cache": object()},
                    block_size=8,
                ),
                layer_names=["model.layers.0.self_attn.compressor.state_cache"],
            ),
        ]
    )
    policy = SimpleNamespace(enabled=True, spec_allowlist=frozenset())
    skipped = resolve_skipped_scheduler_groups(cfg, policy)
    assert skipped == frozenset({0})


def test_apply_skip_filter_drops_state_cache_layer_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev = torch.device("cpu")
    kv_dict = {
        "model.layers.0.state_cache": (
            torch.zeros(2, 4, 1, 8, device=dev),
            torch.zeros(2, 4, 1, 8, device=dev),
        ),
        "model.layers.0.self_attn": (
            torch.zeros(2, 4, 1, 8, device=dev),
            torch.zeros(2, 4, 1, 8, device=dev),
        ),
    }
    flat = kv_dict
    sched = (0, 0)
    layer_to_groups = {
        "model.layers.0.state_cache": [4, 4],
        "model.layers.0.self_attn": [0, 0],
    }
    filtered_flat, filtered_sched, _ = apply_skip_filter_to_flattened(
        flat,
        sched,
        layer_to_groups,
        kv_cache_config=None,
        bundled=True,
        policy=SkipStateGroupsPolicy(enabled=True, spec_allowlist=frozenset()),
    )
    assert "model.layers.0.state_cache" not in filtered_flat
    assert "model.layers.0.self_attn" in filtered_flat
    assert len(filtered_flat) == 1


def test_apply_skip_filter_bundled_reduces_planes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_bundle_multi_spec_env(monkeypatch, enabled=True)
    dev = torch.device("cpu")
    kv_dict = make_ds4_kv_caches_dict(dev, num_blocks=8)
    cfg = _make_ds4_kv_cache_config()
    flat, sched, layer_to_groups, bundled = build_flat_kv_caches(kv_dict, cfg)
    assert bundled is True

    policy = SkipStateGroupsPolicy(
        enabled=True,
        spec_allowlist=frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES),
    )
    filtered = apply_skip_filter_to_flattened(
        flat,
        sched,
        layer_to_groups,
        kv_cache_config=cfg,
        bundled=bundled,
        policy=policy,
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

    policy = SkipStateGroupsPolicy(
        enabled=True,
        spec_allowlist=frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES),
    )
    filtered = apply_skip_filter_to_flattened(
        flat,
        sched,
        layer_to_groups,
        kv_cache_config=cfg,
        bundled=bundled,
        policy=policy,
    )
    filtered_flat, filtered_sched, filtered_layer_to_groups = filtered

    skipped = resolve_skipped_scheduler_groups(cfg, policy)
    assert len(filtered_flat) == sum(1 for g in sched if int(g) not in skipped)
    assert len(filtered_sched) == len(filtered_flat)
    assert all(int(g) not in skipped for g in filtered_sched)
    assert filtered_layer_to_groups[L2] == [0, 2, 3, 3]


def test_parse_skip_policy_enabled_custom_layer_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_skip_state_groups_env(
        monkeypatch,
        enabled=True,
        allowlist="",
        layer_suffix=".custom_skip",
    )
    policy = parse_skip_state_policy_from_env()
    assert policy is not None
    assert policy.layer_name_suffix == ".custom_skip"
    assert effective_layer_name_suffix(policy) == ".custom_skip"


def test_should_skip_layer_layer_suffix() -> None:
    policy = SkipStateGroupsPolicy(enabled=True, spec_allowlist=frozenset())
    assert should_skip_layer(
        layer_name="model.layers.0.self_attn.compressor.state_cache",
        policy=policy,
    )
    assert not should_skip_layer(
        layer_name="model.layers.0.self_attn",
        policy=policy,
    )


def test_should_skip_layer_custom_suffix() -> None:
    policy = SkipStateGroupsPolicy(
        enabled=True,
        spec_allowlist=frozenset(),
        layer_name_suffix=".custom_skip",
    )
    assert should_skip_layer(
        layer_name="model.layers.0.custom_skip",
        policy=policy,
    )
    assert not should_skip_layer(
        layer_name="model.layers.0.self_attn.compressor.state_cache",
        policy=policy,
    )


def test_should_skip_layer_v018_spec_name() -> None:
    group = SimpleNamespace(kv_cache_spec=type("C4AttnKVStateSpec", (), {})())
    policy = SkipStateGroupsPolicy(
        enabled=True,
        spec_allowlist=frozenset({"C4AttnKVStateSpec"}),
    )
    assert should_skip_layer(scheduler_group=group, policy=policy)


def test_resolve_skipped_v020_state_group_by_layer_names() -> None:
    class UniformTypeKVCacheSpecs:
        def __init__(self, specs: dict) -> None:
            self.kv_cache_specs = specs

    cfg = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    {
                        "model.layers.0.self_attn.compressor.state_cache": object(),
                        "model.layers.1.self_attn.compressor.state_cache": object(),
                    }
                ),
                layer_names=[
                    "model.layers.0.self_attn.compressor.state_cache",
                    "model.layers.1.self_attn.compressor.state_cache",
                ],
            ),
            SimpleNamespace(
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    {"model.layers.2.self_attn.swa": object()}
                ),
                layer_names=["model.layers.2.self_attn.swa"],
            ),
        ]
    )
    policy = SimpleNamespace(
        enabled=True,
        spec_allowlist=frozenset(DEFAULT_SKIP_STATE_SPEC_NAMES),
    )
    skipped = resolve_skipped_scheduler_groups(cfg, policy)
    assert 0 in skipped
    assert 1 not in skipped


def test_resolve_skipped_v020_state_group_by_inner_keys_only() -> None:
    class UniformTypeKVCacheSpecs:
        def __init__(self, specs: dict) -> None:
            self.kv_cache_specs = specs

    cfg = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    {"model.layers.0.self_attn.compressor.state_cache": object()}
                ),
                layer_names=[],
            ),
        ]
    )
    policy = SimpleNamespace(enabled=True, spec_allowlist=frozenset())
    skipped = resolve_skipped_scheduler_groups(cfg, policy)
    assert skipped == frozenset({0})
