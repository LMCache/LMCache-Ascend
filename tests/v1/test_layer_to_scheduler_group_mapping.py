# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-spec flatten layer → scheduler group ordering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lmcache_ascend.integration.vllm.multi_spec_flatten import (
    build_flat_kv_caches,
    build_layer_to_scheduler_groups,
    ordered_scheduler_groups_for_layer,
)

# Test data: spec schedules used to generate multi-spec sub-tensor fixtures.
DSV4_CR4_SCHEDULE = (
    ("Compress4AttentionSpec", 128),
    ("SWAAttentionSpec", 128),
    ("C4IndexerSpec", 1024),
    ("C4IndexerSpec", 1024),
    ("C4AttnKVStateSpec", 32),
    ("C4AttnScoreStateSpec", 32),
    ("C4IndexerKVStateSpec", 128),
    ("C4IndexerScoreStateSpec", 128),
)
DSV4_CR128_SCHEDULE = (
    ("Compress128AttentionSpec", 128),
    ("SWAAttentionSpec", 128),
    ("C128AttnKVStateSpec", 64),
    ("C128AttnScoreStateSpec", 64),
)

DSV4_IE_LOGICAL_BLOCK_SIZE = 1024


class SWAAttentionSpec:
    def __init__(self, block_size: int = 128) -> None:
        self.block_size = block_size


class Compress4AttentionSpec:
    def __init__(self, block_size: int = 128) -> None:
        self.block_size = block_size


class Compress128AttentionSpec:
    def __init__(self, block_size: int = 128) -> None:
        self.block_size = block_size


class C4IndexerSpec:
    def __init__(self, block_size: int = 1024) -> None:
        self.block_size = block_size


class C4AttnKVStateSpec:
    def __init__(self, block_size: int = 32) -> None:
        self.block_size = block_size


class C4AttnScoreStateSpec:
    def __init__(self, block_size: int = 32) -> None:
        self.block_size = block_size


class C4IndexerKVStateSpec:
    def __init__(self, block_size: int = 128) -> None:
        self.block_size = block_size


class C4IndexerScoreStateSpec:
    def __init__(self, block_size: int = 128) -> None:
        self.block_size = block_size


class C128AttnKVStateSpec:
    def __init__(self, block_size: int = 64) -> None:
        self.block_size = block_size


class C128AttnScoreStateSpec:
    def __init__(self, block_size: int = 64) -> None:
        self.block_size = block_size


L0, L1, L2, L3, L4 = (
    "model.layers.0",
    "model.layers.1",
    "model.layers.2",
    "model.layers.3",
    "model.layers.4",
)


def _make_ds4_kv_cache_config() -> SimpleNamespace:
    """Synthetic 11-group config aligned with DS4RandomQuarterLayers."""
    groups = [
        (Compress4AttentionSpec, 128, [L2]),
        (SWAAttentionSpec, 128, [L0, L1, L4]),
        (SWAAttentionSpec, 128, [L2]),
        (C4IndexerSpec, 1024, [L2]),
        (C4AttnKVStateSpec, 32, [L2]),
        (C4AttnScoreStateSpec, 32, [L2]),
        (C4IndexerKVStateSpec, 128, [L2]),
        (C4IndexerScoreStateSpec, 128, [L2]),
        (Compress128AttentionSpec, 128, [L3]),
        (SWAAttentionSpec, 128, [L3]),
        (C128AttnKVStateSpec, 64, [L3]),
        (C128AttnScoreStateSpec, 64, [L3]),
    ]
    kv_cache_groups = []
    for spec_cls, bs, layer_names in groups:
        kv_cache_groups.append(
            SimpleNamespace(
                kv_cache_spec=spec_cls(bs),
                layer_names=layer_names,
            )
        )
    return SimpleNamespace(kv_cache_groups=kv_cache_groups)


def _tensor(bs: int, hidden: int = 512, num_blocks: int = 4) -> torch.Tensor:
    return torch.zeros(num_blocks, bs, hidden)


@pytest.fixture
def ds4_config():
    return _make_ds4_kv_cache_config()


def test_dense_layer_spec_order(ds4_config) -> None:
    groups = ordered_scheduler_groups_for_layer(
        L0, _tensor(128), ds4_config, ie_logical_block_size=128
    )
    assert groups == [1]


def test_compress4_layer_spec_order(ds4_config) -> None:
    subs = [_tensor(block_size) for _, block_size in DSV4_CR4_SCHEDULE]
    groups = ordered_scheduler_groups_for_layer(
        L2, subs, ds4_config, ie_logical_block_size=128
    )
    assert groups == [0, 2, 3, 3, 4, 5, 6, 7]


def test_dsa_tuple_maps_all_subs_to_one_scheduler_group(ds4_config) -> None:
    k = torch.zeros(4, 128, 1, 512)
    v = torch.zeros(4, 128, 1, 64)
    dsa_k = torch.zeros(4, 128, 1, 128, dtype=torch.int8)
    dsa_scale = torch.zeros(4, 128, 1, 1, dtype=torch.float16)
    groups = ordered_scheduler_groups_for_layer(
        L3, (k, v, dsa_k, dsa_scale), ds4_config, ie_logical_block_size=128
    )
    assert groups == [8, 8, 8, 8]


def test_compress128_layer_spec_order() -> None:
    """Four sub-cache list matching CR128 spec order (non-DSA path)."""
    layer = "model.layers.x"
    groups_cfg = [
        (Compress128AttentionSpec, 128, [layer]),
        (SWAAttentionSpec, 128, [layer]),
        (C128AttnKVStateSpec, 64, [layer]),
        (C128AttnScoreStateSpec, 64, [layer]),
    ]
    kv_cache_groups = [
        SimpleNamespace(kv_cache_spec=spec_cls(bs), layer_names=names)
        for spec_cls, bs, names in groups_cfg
    ]
    config = SimpleNamespace(kv_cache_groups=kv_cache_groups)
    subs = [_tensor(block_size) for _, block_size in DSV4_CR128_SCHEDULE]
    groups = ordered_scheduler_groups_for_layer(
        layer, subs, config, ie_logical_block_size=128
    )
    assert groups == [0, 1, 2, 3]


def test_flatten_single_tensor(ds4_config) -> None:
    kv = {L0: _tensor(128)}
    flat, sched, _, _ = build_flat_kv_caches(kv, ds4_config, ie_logical_block_size=128)
    assert list(flat.keys()) == [f"{L0}.sub0"]
    assert sched == (1,)


def test_flatten_compress4_eight_subs(
    ds4_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .conftest_ds4 import set_bundle_multi_spec_env

    set_bundle_multi_spec_env(monkeypatch, enabled=False)
    subs = [_tensor(block_size) for _, block_size in DSV4_CR4_SCHEDULE]
    kv = {L2: subs}
    flat, sched, _, _ = build_flat_kv_caches(kv, ds4_config, ie_logical_block_size=128)
    assert len(flat) == 8
    assert sched == (0, 2, 3, 3, 4, 5, 6, 7)


def test_build_flat_kv_caches_collapses_4d_to_3d(
    ds4_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4-D vllm-ascend pages collapse to 3-D MLA buffers for upstream normalize."""
    from .conftest_ds4 import set_bundle_multi_spec_env

    set_bundle_multi_spec_env(monkeypatch, enabled=False)
    k = torch.zeros(4, 128, 1, 512)
    v = torch.zeros(4, 128, 1, 64)
    kv = {L3: (k, v)}
    flat, sched, _, _ = build_flat_kv_caches(
        kv, ds4_config, ie_logical_block_size=DSV4_IE_LOGICAL_BLOCK_SIZE
    )
    assert len(flat) == 2
    assert sched == (8, 8)
    assert flat[f"{L3}.sub0"].shape == (4, 128, 512)
    assert flat[f"{L3}.sub1"].shape == (4, 128, 64)
    assert all(t.ndim == 3 for t in flat.values())


def test_build_layer_to_scheduler_groups(ds4_config) -> None:
    kv = {
        L0: _tensor(128),
        L2: [_tensor(block_size) for _, block_size in DSV4_CR4_SCHEDULE],
    }
    mapping = build_layer_to_scheduler_groups(
        ds4_config, kv.keys(), kv, ie_logical_block_size=128
    )
    assert mapping[L0] == [1]
    assert mapping[L2] == [0, 2, 3, 3, 4, 5, 6, 7]


def test_bundle_flatten_preserves_multi_spec_layers(ds4_config) -> None:
    """Bundled flatten keeps L2 eight-tuple and L3 four-tuple (5 flat layers)."""
    from .conftest_ds4 import DS4_IE_LOGICAL_BLOCK_SIZE, make_ds4_kv_caches_dict

    dev = torch.device("cpu")
    kv_dict = make_ds4_kv_caches_dict(dev, num_blocks=8)
    flat, sched, layer_to_groups, _ = build_flat_kv_caches(
        kv_dict,
        ds4_config,
        ie_logical_block_size=DS4_IE_LOGICAL_BLOCK_SIZE,
    )
    assert len(flat) == 5
    assert isinstance(flat[L2], tuple)
    assert isinstance(flat[L3], tuple)
    assert len(flat[L2]) == 8
    assert len(flat[L3]) == 4
    assert len(sched) == 5
    assert len(layer_to_groups[L2]) == 8
