# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``AscendLMCacheEngine._estimate_shard_size`` ,
when save_only_first_rank is True in MLA scenario.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

GB = 1024**3
MB = 1024**2


def _make_engine(monkeypatch, *, total_mem, allocated, per_chunk_bytes):
    """Mock NPU state."""
    # First Party
    from lmcache_ascend.v1 import cache_engine as ce_mod

    engine = object.__new__(ce_mod.AscendLMCacheEngine)

    # Fake metadata consumed by _estimate_shard_size
    engine.metadata = SimpleNamespace(
        chunk_size=256,
        worker_id=0,
        get_shapes=lambda _cs: ("unused",),  # bypassed by get_size_bytes patch
        get_dtypes=lambda: ("unused",),
    )

    # --- Patch module-level get_size_bytes ---
    monkeypatch.setattr(
        ce_mod,
        "get_size_bytes",
        lambda _shapes, _dtypes: per_chunk_bytes,
    )

    # --- Patch torch.npu device APIs ---
    fake_props = SimpleNamespace(total_memory=total_mem)
    monkeypatch.setattr(
        ce_mod.torch.npu,
        "get_device_properties",
        lambda _device: fake_props,
    )
    monkeypatch.setattr(
        ce_mod.torch.npu,
        "memory_allocated",
        lambda _device: allocated,
    )

    return engine


class TestEstimateShardSize:
    """Verify ``_estimate_shard_size`` under different NPU memory states."""

    def test_normal_case(self, monkeypatch):
        """Abundant free memory → clamp at upper bound 16."""
        engine = _make_engine(
            monkeypatch,
            total_mem=80 * GB,
            allocated=30 * GB,  # 50 GB free
            per_chunk_bytes=10 * MB,  # budget = 12.5 GB → max_shard=640
        )
        assert engine._estimate_shard_size() == 16

    def test_moderate_memory(self, monkeypatch):
        """Limited free memory → value within (1, 16)."""
        engine = _make_engine(
            monkeypatch,
            total_mem=40 * GB,
            allocated=30 * GB,  # 10 GB free, budget = 2.5 GB
            per_chunk_bytes=100 * MB,  # max_shard = 2.5GB / 200MB = 12
        )
        assert engine._estimate_shard_size() == 12

    def test_zero_free_memory(self, monkeypatch):
        """No free memory at all → RuntimeError (fail-fast)."""
        engine = _make_engine(
            monkeypatch,
            total_mem=10 * GB,
            allocated=10 * GB,  # available = 0
            per_chunk_bytes=10 * MB,
        )
        with pytest.raises(RuntimeError, match="insufficient"):
            engine._estimate_shard_size()

    def test_tiny_free_memory(self, monkeypatch):
        """Free memory too small to hold one shard → RuntimeError."""
        engine = _make_engine(
            monkeypatch,
            total_mem=10 * GB,
            allocated=10 * GB - 1 * MB,  # only 1 MB free
            per_chunk_bytes=10 * MB,
        )
        with pytest.raises(RuntimeError, match="insufficient"):
            engine._estimate_shard_size()

    def test_huge_chunk_size(self, monkeypatch):
        """per-chunk bytes larger than pool budget → RuntimeError."""
        engine = _make_engine(
            monkeypatch,
            total_mem=10 * GB,
            allocated=0,  # 10 GB free, budget = 2.5 GB
            per_chunk_bytes=5 * GB,  # max_shard = 2.5GB / 10GB = 0
        )
        with pytest.raises(RuntimeError, match="insufficient"):
            engine._estimate_shard_size()

    def test_per_chunk_larger_than_available(self, monkeypatch):
        """per-chunk bytes larger than total free memory → RuntimeError."""
        engine = _make_engine(
            monkeypatch,
            total_mem=1 * GB,
            allocated=900 * MB,  # 100 MB free
            per_chunk_bytes=1 * GB,  # one chunk > available
        )
        with pytest.raises(RuntimeError, match="insufficient"):
            engine._estimate_shard_size()

    def test_per_chunk_zero_raises(self, monkeypatch):
        """per_chunk_bytes == 0 → RuntimeError (no ZeroDivisionError)."""
        engine = _make_engine(
            monkeypatch,
            total_mem=80 * GB,
            allocated=0,
            per_chunk_bytes=0,
        )
        with pytest.raises(RuntimeError, match="Invalid per-chunk size"):
            engine._estimate_shard_size()

    @pytest.mark.parametrize(
        "total,alloc,per_chunk,expected",
        [
            # budget = 40GB/4 = 10GB, divisor = 2GB, max_shard = 5
            (40 * GB, 0, 1 * GB, 5),
            # budget = 24GB/4 = 6GB, divisor = 2GB, max_shard = 3
            (24 * GB, 0, 1 * GB, 3),
            # budget = 32GB/4 = 8GB, divisor = 0.5GB, max_shard = 16 → clamp at 16
            (32 * GB, 0, 256 * MB, 16),
            # max_shard exactly 16, no clamping needed
            (132 * GB, 4 * GB, 1 * GB, 16),
        ],
    )
    def test_boundary_values(self, monkeypatch, total, alloc, per_chunk, expected):
        engine = _make_engine(
            monkeypatch,
            total_mem=total,
            allocated=alloc,
            per_chunk_bytes=per_chunk,
        )
        assert engine._estimate_shard_size() == expected
