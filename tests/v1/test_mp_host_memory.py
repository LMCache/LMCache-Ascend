# SPDX-License-Identifier: Apache-2.0
"""CPU ownership and startup tests against the unmodified upstream Mixed allocator.

Only Ascend package activation and the native ACL boundary are substituted;
these tests do not establish CANN module loading or hardware correctness.
"""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ctypes
import importlib
import sys

# Third Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager import l1_memory_manager
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
import lmcache
import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    # Source-only testing must not activate unrelated installed NPU patches.
    for name, path in [("lmcache_ascend", root), ("lmcache_ascend.v1", root / "v1")]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    for name in ("mp_host_memory",):
        monkeypatch.delitem(sys.modules, "lmcache_ascend.v1." + name, raising=False)

    native = ModuleType("lmcache_ascend.c_ops")
    buffers, events = {}, []

    def allocate(size):
        events.append("allocate")
        buffer = (ctypes.c_uint8 * size)()
        ptr = ctypes.addressof(buffer)
        buffers[ptr] = buffer
        return ptr

    def free(ptr):
        events.append("free")
        del buffers[ptr]

    native.alloc_host_ptr = allocate
    native.free_host_ptr = free
    monkeypatch.setitem(sys.modules, "lmcache_ascend.c_ops", native)
    monkeypatch.setattr(sys.modules["lmcache_ascend"], "c_ops", native, raising=False)
    monkeypatch.delenv("LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR", raising=False)
    mp = importlib.import_module("lmcache_ascend.v1.mp_host_memory")
    yield SimpleNamespace(mp=mp, native=native, buffers=buffers, events=events)
    for name in ("mp_host_memory",):
        sys.modules.pop("lmcache_ascend.v1." + name, None)


def config(**kwargs):
    options = dict(size_in_bytes=4096, use_lazy=False, shm_name="", align_bytes=64)
    options.update(kwargs)
    return L1MemoryManagerConfig(**options)


def test_acl_tensor_aliases_original_allocation_and_keeps_free_function(
    modules, monkeypatch
):
    allocation = modules.mp.allocate_host_memory(32)
    ptr = allocation.buffer.data_ptr()
    allocation.buffer.fill_(23)
    assert list(modules.buffers[ptr]) == [23] * 32
    monkeypatch.setattr(
        modules.native, "free_host_ptr", lambda ptr: pytest.fail("wrong free")
    )
    allocation.release()
    allocation.release()
    assert allocation.released
    assert modules.buffers == {}
    assert modules.events == ["allocate", "free"]


def test_failed_release_keeps_allocation_unreleased(modules, monkeypatch):
    free = modules.native.free_host_ptr
    fail = True

    def sometimes_free(ptr):
        if fail:
            raise RuntimeError("release failed")
        free(ptr)

    monkeypatch.setattr(modules.native, "free_host_ptr", sometimes_free)
    allocation = modules.mp.allocate_host_memory(32)
    with pytest.raises(RuntimeError, match="release failed"):
        allocation.release()
    assert not allocation.released
    fail = False
    allocation.release()
    assert allocation.released
    assert modules.buffers == {}


def test_mixed_construction_failure_reclaims_acl_memory(modules, monkeypatch):
    # Third Party
    from lmcache.v1.memory_allocators import mixed_memory_allocator as mixed

    def fail_pool(*args, **kwargs):
        raise ValueError("pool failed")

    monkeypatch.setattr(mixed, "TensorMemoryAllocator", fail_pool)
    with pytest.raises(ValueError, match="pool failed"):
        modules.mp.create_acl_mixed_allocator(config(), "npu")
    assert modules.buffers == {}
    assert modules.events == ["allocate", "free"]


def test_mixed_rollback_failure_preserves_construction_error(
    modules, monkeypatch, caplog
):
    # Third Party
    from lmcache.v1.memory_allocators import mixed_memory_allocator as mixed

    def fail_pool(*args, **kwargs):
        raise ValueError("pool failed")

    def fail_free(ptr):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(mixed, "TensorMemoryAllocator", fail_pool)
    monkeypatch.setattr(modules.native, "free_host_ptr", fail_free)
    with pytest.raises(ValueError, match="pool failed"):
        modules.mp.create_acl_mixed_allocator(config(), "npu")
    assert "cleanup failed" in caplog.text


def test_close_wrapper_delegates_default_backing(modules):
    original = MixedMemoryAllocator.close
    closed = []

    def close_default(pool):
        closed.append(pool)
        original(pool)

    close = modules.mp.make_mixed_close(close_default)
    pool = MixedMemoryAllocator(4096)
    close(pool)
    close(pool)
    assert modules.events == []
    assert closed == [pool, pool]


def test_close_wrapper_synchronizes_before_free_once(modules, monkeypatch):
    # Third Party
    from lmcache.v1.memory_allocators import mixed_memory_allocator as mixed

    pool = modules.mp.create_acl_mixed_allocator(config(), "npu")
    monkeypatch.setattr(mixed.torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(
        mixed.torch_dev, "synchronize", lambda: modules.events.append("sync")
    )
    close = modules.mp.make_mixed_close(MixedMemoryAllocator.close)
    close(pool)
    close(pool)
    assert modules.events == ["allocate", "sync", "free"]


def test_acl_mixed_exposes_l1_address_and_pool_ownership(modules, monkeypatch):
    monkeypatch.setattr(
        l1_memory_manager,
        "create_memory_allocator",
        lambda cfg: modules.mp.create_acl_mixed_allocator(cfg, "npu"),
    )
    monkeypatch.setattr(
        MixedMemoryAllocator,
        "close",
        modules.mp.make_mixed_close(MixedMemoryAllocator.close),
    )
    manager = l1_memory_manager.L1MemoryManager(config())
    desc = manager.get_l1_memory_desc()
    assert desc.ptr in modules.buffers
    assert desc.size == 4096
    assert desc.align_bytes == 64
    assert desc.stable_registration_size == 4096
    manager.close()
    assert modules.buffers == {}


@pytest.mark.parametrize("size", [0, -1, sys.maxsize + 1])
def test_invalid_size_fails_before_native_allocation(modules, size):
    with pytest.raises(ValueError):
        modules.mp.allocate_host_memory(size)
    assert modules.events == []


def test_missing_free_symbol_fails_before_allocating(modules, monkeypatch):
    monkeypatch.delattr(modules.native, "free_host_ptr")
    with pytest.raises(RuntimeError, match="free_host_ptr"):
        modules.mp.allocate_host_memory(32)
    assert modules.events == []


def test_view_failure_returns_raw_allocation(modules, monkeypatch):
    def fail_view(*args, **kwargs):
        raise ValueError("view failed")

    monkeypatch.setattr(torch, "frombuffer", fail_view)
    with pytest.raises(ValueError, match="view failed"):
        modules.mp.allocate_host_memory(32)
    assert modules.buffers == {}
    assert modules.events == ["allocate", "free"]


def test_acl_mp_path_initializes_runtime_then_uses_mixed(modules, monkeypatch):
    monkeypatch.setattr(lmcache, "torch_device_type", "npu")
    monkeypatch.setattr(
        lmcache.torch_dev, "init", lambda: modules.events.append("init"), raising=False
    )
    create = modules.mp.make_mp_memory_allocator(
        l1_memory_manager.create_memory_allocator
    )
    monkeypatch.setattr(
        MixedMemoryAllocator,
        "close",
        modules.mp.make_mixed_close(MixedMemoryAllocator.close),
    )
    pool = create(config())
    assert isinstance(pool, MixedMemoryAllocator)
    assert modules.events == ["init", "allocate"]
    obj = pool.allocate(torch.Size([32]), torch.uint8, MemoryFormat.KV_2LTD)
    assert obj is not None
    obj.tensor.fill_(17)
    assert list(modules.buffers[pool.buffer.data_ptr()])[:32] == [17] * 32
    pool.free(obj)
    assert "free" not in modules.events
    pool.close()
    pool.close()
    assert modules.events == ["init", "allocate", "free"]


def test_runtime_failure_never_allocates_or_falls_back(modules, monkeypatch):
    monkeypatch.setattr(lmcache, "torch_device_type", "npu")

    def fail_init():
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(lmcache.torch_dev, "init", fail_init, raising=False)
    create = modules.mp.make_mp_memory_allocator(
        l1_memory_manager.create_memory_allocator
    )
    with pytest.raises(RuntimeError, match="runtime failed"):
        create(config())
    assert modules.events == []


@pytest.mark.parametrize(
    "option", ["use_lazy", "shm_name", "use_hugepages", "devdax_path"]
)
def test_acl_configuration_rejected_before_runtime(modules, monkeypatch, option):
    monkeypatch.setattr(lmcache, "torch_device_type", "npu")
    monkeypatch.setattr(
        lmcache.torch_dev,
        "init",
        lambda: pytest.fail("runtime initialized before configuration validation"),
        raising=False,
    )
    settings = config()
    setattr(
        settings, option, "unsupported" if option.endswith(("name", "path")) else True
    )
    create = modules.mp.make_mp_memory_allocator(
        l1_memory_manager.create_memory_allocator
    )
    with pytest.raises(ValueError):
        create(settings)
    assert modules.events == []


def test_invalid_selection_is_rejected_at_installation(modules):
    with pytest.raises(ValueError):
        modules.mp.install_mp_host_memory("typo")


def test_installer_default_leaves_upstream_untouched(modules):
    create = l1_memory_manager.create_memory_allocator
    close = MixedMemoryAllocator.close
    modules.mp.install_mp_host_memory("default")
    assert l1_memory_manager.create_memory_allocator is create
    assert MixedMemoryAllocator.close is close


def test_install_failure_leaves_upstream_untouched(modules, monkeypatch):
    # Third Party
    from lmcache.v1.multiprocess import server

    create = l1_memory_manager.create_memory_allocator
    close = MixedMemoryAllocator.close
    start = server._start_cache_server

    def fail(original):
        raise RuntimeError("wrapper failed")

    # Fail after the other wrappers have been constructed, before publication.
    monkeypatch.setattr(modules.mp, "_validated_start", fail)
    with pytest.raises(RuntimeError, match="wrapper failed"):
        modules.mp.install_mp_host_memory("aclrt_malloc_host")
    assert not modules.mp._installed
    assert l1_memory_manager.create_memory_allocator is create
    assert MixedMemoryAllocator.close is close
    assert server._start_cache_server is start
    assert modules.events == []


def test_installed_acl_path_keeps_validation_and_creation(modules, monkeypatch):
    # Third Party
    from lmcache.v1.multiprocess import server

    monkeypatch.setattr(lmcache, "torch_device_type", "npu")
    monkeypatch.setattr(lmcache.torch_dev, "init", lambda: None, raising=False)
    original_create = l1_memory_manager.create_memory_allocator
    monkeypatch.setattr(l1_memory_manager, "create_memory_allocator", original_create)
    close = MixedMemoryAllocator.close
    monkeypatch.setattr(MixedMemoryAllocator, "close", close)
    started = []

    def start(mp_config, storage_manager_config, *, marker):
        started.append(marker)
        return l1_memory_manager.create_memory_allocator(
            storage_manager_config.l1_manager_configs[0].memory_config
        )

    monkeypatch.setattr(server, "_start_cache_server", start)
    mp_config = SimpleNamespace(supported_transfer_mode="auto", shm_name="bad_shm")
    storage = SimpleNamespace(
        l1_manager_configs=[SimpleNamespace(memory_config=config(), gds_l1_config=None)]
    )
    modules.mp.install_mp_host_memory("aclrt_malloc_host")
    installed = l1_memory_manager.create_memory_allocator
    installed_close = MixedMemoryAllocator.close
    installed_start = server._start_cache_server
    assert installed is not original_create
    assert installed_close is not close
    # Installation fixes this process's path; subsequent calls must not reselect.
    for later_selection in ("default", "invalid"):
        modules.mp.install_mp_host_memory(later_selection)
        assert l1_memory_manager.create_memory_allocator is installed
        assert MixedMemoryAllocator.close is installed_close
        assert server._start_cache_server is installed_start
    with pytest.raises(ValueError, match="shm_name"):
        server._start_cache_server(mp_config, storage, marker="invalid")
    assert started == []
    assert modules.events == []
    mp_config.shm_name = ""
    pool = server._start_cache_server(mp_config, storage, marker="valid")
    assert l1_memory_manager.create_memory_allocator is installed
    assert MixedMemoryAllocator.close is installed_close
    assert started == ["valid"]
    assert pool.buffer.data_ptr() in modules.buffers
    pool.close()
    assert modules.buffers == {}
