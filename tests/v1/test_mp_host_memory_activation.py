# SPDX-License-Identifier: Apache-2.0
"""Exercise actual package activation and upstream CLI with CPU native substitutes."""

# Standard
from pathlib import Path
import os
import subprocess
import sys

# Third Party
import pytest

_ACTIVATION = r"""
import ctypes
import sys
from types import ModuleType, SimpleNamespace

import torch

order, scenario = sys.argv[1:]
acl_mode = scenario in ("acl", "benchmark")
events, buffers = [], {}
version = ModuleType("lmcache_ascend._version")
version.__version__ = "0.0.0"
version.__version_tuple__ = (0, 0, 0)
sys.modules[version.__name__] = version
native = ModuleType("lmcache_ascend.c_ops")

def allocate(size, *args, kind="acl"):
    events.append(kind + "_allocate")
    buffer = (ctypes.c_uint8 * size)()
    ptr = ctypes.addressof(buffer)
    buffers[ptr] = buffer
    return ptr

def free(ptr, *args, kind="acl"):
    events.append(kind + "_free")
    del buffers[ptr]

native.alloc_host_ptr = allocate
native.free_host_ptr = free
native.alloc_pinned_ptr = lambda size, *args: allocate(size, kind="default")
native.free_pinned_ptr = lambda ptr, *args: free(ptr, kind="default")
# Match the current Ascend surface: object-group transfer, no batch direct-copy.
native.execute_object_group_transfer = lambda *args: None
sys.modules[native.__name__] = native
torch.npu = SimpleNamespace(
    is_available=lambda: True,
    device_count=lambda: 1,
    init=lambda: events.append("init"),
    synchronize=lambda: events.append("sync"),
)
torch.utils.rename_privateuse1_backend("npu")

if scenario == "install_failure":
    def fail_install(frame, event, arg):
        if (event == "call" and frame.f_code.co_name == "make_mixed_close"
                and frame.f_code.co_filename.endswith("/mp_host_memory.py")):
            sys.setprofile(None)
            raise RuntimeError("wrapper installation failed")
    sys.setprofile(fail_install)
elif scenario == "import_failure":
    import builtins
    original_import = builtins.__import__
    def fail_import(name, *args, **kwargs):
        if name == "lmcache_ascend.v1.mp_host_memory":
            raise ImportError("host memory module import failed")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = fail_import

# Leave failures uncaught: process exit must bypass LMCache's Exception handler.
if order == "ascend_first":
    import lmcache_ascend
import lmcache
import lmcache_ascend
assert scenario not in ("invalid", "install_failure", "import_failure")
assert events == []
assert lmcache.torch_device_type == "npu"
from lmcache.v1.platform.devices.npu.device_ops import NpuDeviceOps
assert isinstance(lmcache.device_ops, NpuDeviceOps)

if scenario in ("default", "unset"):
    assert "lmcache_ascend.v1.mp_host_memory" not in sys.modules
    assert "lmcache.v1.multiprocess.server" not in sys.modules
else:
    assert "lmcache.v1.multiprocess.server" in sys.modules

from lmcache.cli.main import main
from lmcache.v1.multiprocess import http_server, server, object_group_transfer
from lmcache.v1.distributed.memory_manager import l1_memory_manager
assert object_group_transfer._HAS_NATIVE_OBJECT_GROUP_TRANSFER
assert not object_group_transfer._HAS_BATCH_MEMCPY_ASYNC
class ReachedAllocator(BaseException):
    pass

original_init = l1_memory_manager.L1MemoryManager.__init__
def inspect_allocator(self, config):
    original_init(self, config)
    pool = self._allocator
    assert hasattr(pool, "ascend_host_allocation") == acl_mode
    ptr = pool.buffer.data_ptr()
    assert ptr in buffers
    self.close()
    assert buffers == {}
    raise ReachedAllocator()
l1_memory_manager.L1MemoryManager.__init__ = inspect_allocator

# Exercise actual cache-server startup through L1 creation. Stop before network
# listeners/transfer workers start; no claim about Uvicorn's exit semantics.
def run_http(**kwargs):
    server.run_cache_server(
        mp_config=kwargs["mp_config"],
        storage_manager_config=kwargs["storage_manager_config"],
        obs_config=kwargs["obs_config"],
        coordinator_config=kwargs["coordinator_config"],
        return_engine=True,
        start_prometheus_http_server=False,
    )
http_server.run_http_server = run_http
try:
    if scenario == "benchmark":
        import runpy
        from lmcache.tools.transfer_channel_benchmark import benchmark
        config_type = benchmark.L1MemoryManagerConfig
        sys.argv = ["benchmark", "--role", "server", "--buffer-size", "1MB",
                    "--page-size", "4KB", "--object-size", "4KB",
                    "--num-objects", "1", "--num-source-objects", "1"]
        runpy.run_module("lmcache_ascend.v1.mp_transfer_channel_benchmark",
                         run_name="__main__")
    else:
        main(["server", "--no-l1-use-lazy", "--shm-name", "",
              "--l1-size-gb", "0.000004", "--eviction-policy", "LRU",
              "--disable-observability"])
except ReachedAllocator:
    pass
else:
    raise AssertionError("must reach L1 construction")
if scenario == "benchmark":
    assert benchmark.L1MemoryManagerConfig is config_type
if acl_mode:
    assert events == ["init", "acl_allocate", "sync", "acl_free"]
else:
    assert events.count("default_allocate") == 1
    assert events.count("default_free") == 1
    assert "init" not in events and "acl_allocate" not in events
print("activation checks passed")
"""


@pytest.mark.parametrize(
    ("order", "scenario"),
    [
        ("lmcache_first", "acl"),
        ("ascend_first", "acl"),
        ("ascend_first", "benchmark"),
        ("lmcache_first", "unset"),
        ("lmcache_first", "default"),
        ("lmcache_first", "invalid"),
        ("ascend_first", "invalid"),
        ("lmcache_first", "install_failure"),
        ("lmcache_first", "import_failure"),
    ],
)
def test_package_activation_and_server_cli(order, scenario):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    env["LMCACHE_DEVICE_BACKEND"] = "npu"
    env["LMCACHE_TRACK_USAGE"] = "false"
    env["LMCACHE_DISABLE_BANNER"] = "1"
    env["LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR"] = {
        "default": "default",
        "invalid": "typo",
    }.get(scenario, "aclrt_malloc_host")
    if scenario == "unset":
        env.pop("LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR")
    result = subprocess.run(
        [sys.executable, "-c", _ACTIVATION, order, scenario],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if scenario in ("invalid", "install_failure", "import_failure"):
        assert result.returncode == 1, result.stdout + result.stderr
        assert "activation checks passed" not in result.stdout
        assert "Failed to install Ascend MP host memory" in result.stderr
        expected = {
            "invalid": "LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR",
            "install_failure": "wrapper installation failed",
            "import_failure": "host memory module import failed",
        }[scenario]
        assert expected in result.stderr
        assert "No compute backend loaded" not in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "activation checks passed" in result.stdout
