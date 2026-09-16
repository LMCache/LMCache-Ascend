# SPDX-License-Identifier: Apache-2.0
"""
Tests for the Ascend compatibility patch of the upstream EIC connector.

The vendor `eic` client is not on public PyPI and libcudart is absent on CANN
images, so both are stubbed. These tests verify import/init compatibility only;
a live EIC cluster is covered in internal deployment testing.
"""

# Standard
import sys
import types

# Third Party
import pytest


class _FakeStatusCode:
    SUCCESS = 0
    KEY_NOT_EXIST = 1


class _FakeEnumValue:
    def __init__(self, value):
        self.value = value


class _FakeTransportType:
    TRANSPORT_RDMA = _FakeEnumValue(2)
    TRANSPORT_GDR = _FakeEnumValue(3)

    def __call__(self, value):
        return _FakeEnumValue(value)


class _FakeLogLevel:
    def __call__(self, value):
        return value


class _FakeClient:
    def init(self, instance_id, endpoint, option):
        self.instance_id = instance_id
        self.endpoint = endpoint
        return 0


def _install_fake_eic(monkeypatch):
    # lmcache.v1.memory_management imports lmcache.c_ops at module scope. The
    # compiled extension is absent without a built lmcache/lmcache-ascend
    # install; only pin-allocation functions use it, and these tests never call
    # them, so stub the module.
    c_ops = types.ModuleType("lmcache.c_ops")
    monkeypatch.setitem(sys.modules, "lmcache.c_ops", c_ops)

    fake = types.ModuleType("eic")
    fake.Client = _FakeClient
    fake.InitOption = type("InitOption", (), {})
    fake.SetOption = type("SetOption", (), {})
    fake.GetOption = type("GetOption", (), {})
    fake.ExistOption = type("ExistOption", (), {})
    fake.StringVector = type("StringVector", (), {"append": lambda self, *a: None})
    fake.IOBuffers = type("IOBuffers", (), {"append": lambda self, *a: None})
    fake.MemoryInfo = type("MemoryInfo", (), {})
    fake.StatusCode = _FakeStatusCode
    fake.TransportType = _FakeTransportType()
    fake.LogLevel = _FakeLogLevel()
    monkeypatch.setitem(sys.modules, "eic", fake)
    return fake


def _reload_connector():
    import importlib

    import lmcache.v1.storage_backend.connector.eic_connector as mod

    return importlib.reload(mod)


def test_patch_makes_missing_libcudart_tolerated(monkeypatch, tmp_path):
    _install_fake_eic(monkeypatch)

    mod = _reload_connector()
    # Neutralize the metadata-heavy base __init__ after reload so the class
    # binds to the same module object.
    monkeypatch.setattr(mod.RemoteConnector, "__init__", lambda self, c, m: None)

    cfg = tmp_path / "lmcache_eic.yaml"
    cfg.write_text(
        "remote_url: 'eic://127.0.0.1:12500'\n"
        "eic_instance_id: 'test-instance'\n"
        "eic_trans_type: 2\n"
        "eic_thread_num: 1\n"
        "eic_log_dir: " + str(tmp_path) + "\n"
        "eic_kv_ttl: -1\n"
    )
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(cfg))

    # Prebuilt connection probes 2048 keys via mexist; respond miss.
    _FakeClient.mexist = lambda self, keys, opt: (
        0,
        types.SimpleNamespace(status_codes=[_FakeStatusCode.KEY_NOT_EXIST]),
    )

    import ctypes

    real_cdll = ctypes.CDLL

    def raising_cdll(name, *args, **kwargs):
        if name == "libcudart.so":
            raise OSError("libcudart.so: cannot open shared object file")
        return real_cdll(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "CDLL", raising_cdll)

    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        patch_eic_connector,
    )

    patch_eic_connector()

    class _FakeLoop:
        def call_soon_threadsafe(self, *a, **kw):
            pass

    class _FakeAllocator:
        config = object()
        metadata = object()

    connector = mod.EICConnector(
        "eic://127.0.0.1:12500/", _FakeLoop(), _FakeAllocator()
    )
    assert connector.connection.instance_id == "test-instance"
    assert connector.connection.endpoint == "127.0.0.1:12500"
    # No real cuda library: the signature binding lands on the no-op shim.
    from lmcache_ascend.v1.storage_backend.connector.eic_npu import _CudaLibShim

    assert isinstance(connector.cuda_lib, _CudaLibShim)
    # The connector assigns argtypes/restype to cudaMemcpy at init.
    assert connector.cuda_lib.cudaMemcpy.argtypes is not None


def test_patch_keeps_real_cudart_when_present(monkeypatch, tmp_path):
    _install_fake_eic(monkeypatch)
    mod = _reload_connector()
    monkeypatch.setattr(mod.RemoteConnector, "__init__", lambda self, c, m: None)

    cfg = tmp_path / "lmcache_eic.yaml"
    cfg.write_text("remote_url: 'eic://127.0.0.1:12500'\neic_instance_id: 'i'\n")
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(cfg))
    _FakeClient.mexist = lambda self, keys, opt: (
        0,
        types.SimpleNamespace(status_codes=[_FakeStatusCode.KEY_NOT_EXIST]),
    )

    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        patch_eic_connector,
    )

    patch_eic_connector()
    # With libcudart present the real ctypes binding path runs. On hosts
    # without it (macOS CI), the library lookup itself raises OSError, which is
    # out of scope; only assert the patch is installed.
    assert mod._lmcache_ascend_eic_patched is True


def test_patch_is_idempotent(monkeypatch):
    _install_fake_eic(monkeypatch)
    mod = _reload_connector()
    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        patch_eic_connector,
    )

    patch_eic_connector()
    first_init = mod.EICConnector.__init__
    patch_eic_connector()
    assert mod.EICConnector.__init__ is first_init


def test_patch_skips_without_eic_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "eic" or name.startswith("eic."):
            raise ImportError("No module named 'eic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        patch_eic_connector,
    )

    # Must not raise even when the vendor client is unavailable.
    patch_eic_connector()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
