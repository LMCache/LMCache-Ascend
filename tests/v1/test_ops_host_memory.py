# SPDX-License-Identifier: Apache-2.0
"""Exercise host allocation startup without an NPU or compiled Ascend ops."""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import importlib.util
import sys

# Third Party
import pytest
import torch


@pytest.fixture
def host_ops(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Load the real ops surface with controlled runtime/native boundaries."""
    # Resolve the real backend before replacing torch.npu for this test.
    importlib.import_module("lmcache.v1.platform.torch_ops")
    initialized = False

    def initialize() -> None:
        nonlocal initialized
        initialized = True

    def allocate(size: int, flags: int) -> int:
        if not initialized:
            raise RuntimeError("ACL context is not initialized")
        return 0x1000

    npu = SimpleNamespace(init=Mock(side_effect=initialize))
    native = SimpleNamespace(alloc_pinned_ptr=Mock(side_effect=allocate))
    package = ModuleType("lmcache_ascend")
    package.__path__ = []
    monkeypatch.setattr(package, "c_ops", native, raising=False)
    monkeypatch.setitem(sys.modules, "lmcache_ascend", package)
    monkeypatch.setitem(sys.modules, "lmcache_ascend.c_ops", native)
    monkeypatch.setattr(torch, "npu", npu, raising=False)

    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/ops.py"
    spec = importlib.util.spec_from_file_location("host_ops_under_test", path)
    assert spec is not None and spec.loader is not None
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    return SimpleNamespace(
        ops=ops, initialize=npu.init, allocate=native.alloc_pinned_ptr
    )


def test_host_allocation_initializes_runtime_on_demand(
    host_ops: SimpleNamespace,
) -> None:
    """Import stays lazy; the first allocation prepares runtime before native use."""
    host_ops.initialize.assert_not_called()
    assert host_ops.ops.alloc_pinned_ptr(512 * 1024 * 1024, 0) == 0x1000
    host_ops.allocate.assert_called_once_with(512 * 1024 * 1024, 0)


def test_runtime_initialization_failure_prevents_allocation(
    host_ops: SimpleNamespace,
) -> None:
    """Propagate initialization failure without attempting a native allocation."""
    error = RuntimeError("NPU initialization failed")
    host_ops.initialize.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        host_ops.ops.alloc_pinned_ptr(512 * 1024 * 1024, 0)
    assert caught.value is error
    host_ops.allocate.assert_not_called()
