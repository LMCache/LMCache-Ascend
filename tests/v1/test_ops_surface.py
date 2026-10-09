# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the curated ops binding surface."""

# Standard
import sys
import types

# Third Party
from lmcache.v1.platform import torch_ops
import pytest

# First Party
import lmcache_ascend.c_ops as real_c_ops

_NATIVE_ONLY_NAMES = (
    "TransferDirection",
    "EngineKVFormat",
    "GPUKVFormat",
    "is_cross_layer",
    "is_kv_list",
    "is_layer_list",
    "is_mla",
)


@pytest.fixture()
def surface(monkeypatch: pytest.MonkeyPatch):
    """Fresh ``lmcache_ascend.ops`` rebuilt against the real c_ops."""
    monkeypatch.delitem(sys.modules, "lmcache_ascend.ops", raising=False)
    # First Party
    import lmcache_ascend.ops as ops_surface

    return ops_surface


def test_surface_excludes_native_only_names(surface) -> None:
    for name in _NATIVE_ONLY_NAMES:
        assert not hasattr(surface, name)
    assert hasattr(surface, "PageBufferShapeDesc")
    assert surface.multi_layer_block_kv_transfer is (
        real_c_ops.multi_layer_block_kv_transfer
    )


def test_torch_fallback_reexports_are_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("lmcache_ascend.c_ops")
    fake.PageBufferShapeDesc = real_c_ops.PageBufferShapeDesc
    fake.record_completion_on_stream = torch_ops.record_completion_on_stream
    monkeypatch.setitem(sys.modules, "lmcache_ascend.c_ops", fake)
    # ``import a.b as c`` resolves through the parent attribute, so patch
    # both the sys.modules entry and the real package's attribute.
    monkeypatch.setattr(sys.modules["lmcache_ascend"], "c_ops", fake)
    monkeypatch.delitem(sys.modules, "lmcache_ascend.ops", raising=False)

    # First Party
    import lmcache_ascend.ops as ops_surface

    assert not hasattr(ops_surface, "record_completion_on_stream")
    assert hasattr(ops_surface, "PageBufferShapeDesc")


def test_genuine_native_symbols_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    def _native_recorder(stream_ptr: int, kind: str, payload: bytes) -> None:
        pass

    fake = types.ModuleType("lmcache_ascend.c_ops")
    fake.PageBufferShapeDesc = real_c_ops.PageBufferShapeDesc
    fake.record_completion_on_stream = _native_recorder
    monkeypatch.setitem(sys.modules, "lmcache_ascend.c_ops", fake)
    # ``import a.b as c`` resolves through the parent attribute, so patch
    # both the sys.modules entry and the real package's attribute.
    monkeypatch.setattr(sys.modules["lmcache_ascend"], "c_ops", fake)
    monkeypatch.delitem(sys.modules, "lmcache_ascend.ops", raising=False)

    # First Party
    import lmcache_ascend.ops as ops_surface

    assert ops_surface.record_completion_on_stream is _native_recorder
