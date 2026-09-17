# SPDX-License-Identifier: Apache-2.0
"""Wrap ``get_tokens_per_block`` for Ascend compressed MLA (physical block)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import lmcache.integration.vllm.kv_cache_groups as kg
import lmcache_ascend.integration.vllm.logical_block_size as lbs
from lmcache_ascend.integration.vllm.logical_block_size import (
    _block_size_is_physical,
    install_on,
    install_overrides,
    tokens_per_block_id,
)


class AttentionSpec:
    """Name used by ``get_tokens_per_block`` DCP MRO detection."""


@dataclass(frozen=True)
class AscendMLAAttentionSpec(AttentionSpec):
    block_size: int
    compress_ratio: int = 1
    model_version: str | None = None

    @classmethod
    def merge(cls, specs: list["AscendMLAAttentionSpec"]) -> "AscendMLAAttentionSpec":
        return cls(block_size=specs[0].block_size)


@dataclass(frozen=True)
class AscendSlidingWindowMLASpec:
    block_size: int
    compress_ratio: int = 1
    sliding_window: int = 128


@dataclass
class UniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MLAAttentionSpec(AttentionSpec):
    """Upstream GPU MLA: ``block_size`` is already logical tokens."""

    block_size: int
    compress_ratio: int = 1


def _gpu_prop_spec(block: int, ratio: int) -> Any:
    """Leaf named ``AscendMLAAttentionSpec`` that still has GPU storage math."""

    @dataclass(frozen=True)
    class AscendMLAAttentionSpec(AttentionSpec):
        block_size: int
        compress_ratio: int = 1

        @property
        def storage_block_size(self) -> int:
            return self.block_size // self.compress_ratio

    return AscendMLAAttentionSpec(block_size=block, compress_ratio=ratio)


def _make_spec(kind: str, block: int, ratio: int) -> Any:
    if kind == "mla":
        return AscendMLAAttentionSpec(block_size=block, compress_ratio=ratio)
    if kind == "swa":
        return AscendSlidingWindowMLASpec(block_size=block, compress_ratio=ratio)
    if kind == "gpu":
        return MLAAttentionSpec(block_size=block, compress_ratio=ratio)
    if kind == "gpu_prop":
        return _gpu_prop_spec(block, ratio)
    if kind == "uniform":
        leaf = AscendMLAAttentionSpec(block_size=block, compress_ratio=ratio)
        return UniformTypeKVCacheSpecs(
            block_size=block, kv_cache_specs={"l0": leaf, "l1": leaf}
        )
    raise ValueError(kind)


def _gtpb(spec: object, dcp_size: int) -> int:
    install_overrides()
    return kg.get_tokens_per_block(spec, dcp_size)


@pytest.mark.parametrize(
    "kind,block,ratio,dcp,expected",
    [
        ("mla", 32, 128, 1, 4096),
        ("mla", 32, 128, 2, 8192),
        ("swa", 32, 4, 1, 128),
        ("uniform", 32, 128, 1, 4096),
        ("uniform", 32, 128, 2, 8192),
        ("gpu", 32, 128, 1, 32),
        ("gpu", 32, 128, 2, 64),
        ("gpu_prop", 32, 128, 1, 4096),
        ("gpu_prop", 32, 4, 1, 128),
    ],
)
def test_tokens_per_block(
    kind: str, block: int, ratio: int, dcp: int, expected: int
) -> None:
    spec = _make_spec(kind, block, ratio)
    per_id = tokens_per_block_id(spec)
    if dcp <= 1:
        assert per_id == expected
    else:
        assert per_id == expected // dcp
    assert _gtpb(spec, dcp) == expected


def test_merge_restores_dropped_compress_ratio() -> None:
    install_on((AscendMLAAttentionSpec,))
    merged = AscendMLAAttentionSpec.merge(
        [
            AscendMLAAttentionSpec(
                block_size=32, compress_ratio=128, model_version="deepseek_v4"
            )
            for _ in range(2)
        ]
    )
    assert merged.compress_ratio == 128
    assert merged.model_version == "deepseek_v4"
    assert tokens_per_block_id(merged) == 4096


def test_install_overrides_is_idempotent() -> None:
    install_overrides()
    install_overrides()
    spec = AscendMLAAttentionSpec(block_size=32, compress_ratio=128)
    assert _gtpb(spec, 1) == 4096
    assert kg.get_tokens_per_block is not tokens_per_block_id
    assert getattr(kg.get_tokens_per_block, "_lmcache_ascend_logical_block_size", False)


class _PrePRPool:
    def max_memory_usage_bytes(self, vllm_config: object) -> int:
        return self.block_size * self.compress_ratio


class _PostPRPool:
    def max_memory_usage_bytes(self, vllm_config: object) -> int:
        return self.block_size


class _NoPoolMethod:
    pass


@pytest.mark.parametrize(
    "cls,expected",
    [
        (_PrePRPool, True),
        (_PostPRPool, False),
        (_NoPoolMethod, True),
        (None, True),
    ],
    ids=["pre_pr", "post_pr", "no_method", "none"],
)
def test_block_size_is_physical_from_pool_math(
    cls: type | None, expected: bool
) -> None:
    assert _block_size_is_physical(cls) is expected


def test_post_pr_skips_get_tokens_wrap() -> None:
    prev = kg.get_tokens_per_block
    orig = lbs._UNPATCHED_GET_TOKENS
    if orig is None and not getattr(prev, "_lmcache_ascend_logical_block_size", False):
        orig = prev
    lbs._INSTALLED = False
    lbs._GET_TOKENS_WRAPPED = False
    if getattr(prev, "_lmcache_ascend_logical_block_size", False) and orig is not None:
        lbs._rebind_get_tokens_per_block(prev, orig)
    try:
        lbs.install_overrides(scale_physical=False)
        spec = AscendMLAAttentionSpec(block_size=4096, compress_ratio=128)
        assert kg.get_tokens_per_block(spec, 1) == 4096
        assert not getattr(
            kg.get_tokens_per_block, "_lmcache_ascend_logical_block_size", False
        )
    finally:
        lbs._GET_TOKENS_WRAPPED = False
        lbs._INSTALLED = False
        lbs.install_overrides(scale_physical=True)
