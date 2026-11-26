# SPDX-License-Identifier: Apache-2.0
"""Check native allocation integration with Mixed and benchmark configuration."""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ctypes
import importlib
import sys

# Third Party
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
import lmcache
import pytest
import torch


@pytest.fixture
def source_package(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    for name, path in [("lmcache_ascend", root), ("lmcache_ascend.v1", root / "v1")]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    name = "lmcache_ascend.v1.mp_transfer_channel_benchmark"
    monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    sys.modules.pop(name, None)


@pytest.mark.parametrize("fail_release", [False, True])
def test_native_pinned_pair_works_with_normal_mixed(monkeypatch, fail_release):
    buffers, events = {}, []

    def allocate(size, flags):
        assert flags == 0
        buffer = (ctypes.c_uint8 * size)()
        ptr = ctypes.addressof(buffer)
        buffers[ptr] = buffer
        events.append("allocate")
        return ptr

    def free(ptr):
        nonlocal fail_release
        if fail_release:
            fail_release = False
            raise RuntimeError("ACL release failed")
        del buffers[ptr]
        events.append("free")

    # The native test exercises these C++ functions; only their ACL allocation
    # boundary is substituted here, leaving Mixed's construction/close intact.
    monkeypatch.setattr(lmcache.device_ops, "alloc_pinned_ptr", allocate)
    monkeypatch.setattr(lmcache.device_ops, "free_pinned_ptr", free)
    pool = MixedMemoryAllocator(4096, align_bytes=64)
    obj = pool.allocate(torch.Size([32]), torch.uint8, MemoryFormat.KV_2LTD)
    assert obj is not None
    obj.tensor.fill_(17)
    assert list(buffers[pool.buffer.data_ptr()])[:32] == [17] * 32
    pool.free(obj)
    assert events == ["allocate"]
    if fail_release:
        with pytest.raises(RuntimeError, match="ACL release failed"):
            pool.close()
        assert len(buffers) == 1
    pool.close()
    pool.close()
    assert events == ["allocate", "free"]
    assert buffers == {}


@pytest.mark.parametrize(
    ("use_lazy", "outcome"), [(False, "success"), (True, "failure"), (False, "error")]
)
def test_benchmark_without_allocator_setting(
    source_package, monkeypatch, use_lazy, outcome
):
    # Third Party
    from lmcache.tools.transfer_channel_benchmark import benchmark
    from lmcache.v1.distributed import config as distributed_config

    monkeypatch.delenv("LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR", raising=False)
    monkeypatch.setattr(
        distributed_config,
        "current_device_spec",
        SimpleNamespace(is_pin_supported=True),
    )
    entry = importlib.import_module("lmcache_ascend.v1.mp_transfer_channel_benchmark")
    config_type = benchmark.L1MemoryManagerConfig
    # Inspect the real benchmark's configuration before SHM or ACL allocation.
    monkeypatch.setattr(benchmark, "L1MemoryManager", lambda config: config)
    args = ["benchmark", "--role", "client"]
    if use_lazy:
        args.append("--use-lazy")
    monkeypatch.setattr(sys, "argv", args)

    def run_client(cfg):
        config = benchmark._create_l1_manager(cfg, 1048576)
        assert config.shm_name == ""
        assert config.use_lazy is use_lazy
        if outcome == "error":
            raise RuntimeError("benchmark failed")
        return outcome == "success"

    monkeypatch.setattr(benchmark, "client_main", run_client)
    if outcome == "error":
        with pytest.raises(RuntimeError, match="benchmark failed"):
            entry.main()
    else:
        assert entry.main() == (0 if outcome == "success" else 1)
    assert benchmark.L1MemoryManagerConfig is config_type
    assert config_type(size_in_bytes=1048576, use_lazy=False).shm_name
