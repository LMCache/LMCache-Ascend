# SPDX-License-Identifier: Apache-2.0
"""
Tests for the Ascend compatibility patch of the upstream EIC connector.

These tests exercise only the package boundary and never reload the real
connector module (which binds the vendor-only ``eic`` package at import and is
expensive to reconstruct). A throwaway module stand-in plays the connector
role, and every sys.modules / builtins override goes through ``monkeypatch``,
so nothing leaks into the process-global state of later tests.
"""

# Standard
import ctypes
import sys
import threading
import types

# Third Party
import pytest

CONNECTOR_PATH = "lmcache.v1.storage_backend.connector.eic_connector"


def _legacy_connector_module():
    """A connector module as it looks before core made libcudart optional."""
    mod = types.ModuleType(CONNECTOR_PATH)
    mod.ctypes = ctypes
    return mod


def _install_dummy_connector(monkeypatch, mod):
    # Replace both the sys.modules leaf and the parent-package attribute. The
    # patch imports the module as ``import a.b.c as x``, which resolves to the
    # parent's already-bound ``c`` attribute even when sys.modules[c] was
    # swapped, so overriding only sys.modules silently targets a pre-imported
    # real leaf. monkeypatch restores both after the test.
    import lmcache.v1.storage_backend.connector as parent

    monkeypatch.setattr(parent, "eic_connector", mod, raising=False)
    monkeypatch.setitem(sys.modules, CONNECTOR_PATH, mod)
    return mod


def test_patch_is_noop_when_core_marks_cudart_optional(monkeypatch):
    # Core LMCache#5141 loads libcudart defensively. The Ascend patch must not
    # wrap ctypes in that case: a truthy CDLL stand-in would defeat core's
    # ``cuda_lib is None`` GDR guard.
    from lmcache_ascend.v1.storage_backend.connector.eic_npu import _GuardedCtypes

    mod = _legacy_connector_module()
    mod._LMCACHE_EIC_CUDART_OPTIONAL = True
    _install_dummy_connector(monkeypatch, mod)

    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        patch_eic_connector,
    )

    patch_eic_connector()
    assert mod.ctypes is ctypes
    assert not isinstance(mod.ctypes, _GuardedCtypes)


def test_patch_applies_proxy_on_legacy_core(monkeypatch):
    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        _GuardedCtypes,
        patch_eic_connector,
    )

    mod = _install_dummy_connector(monkeypatch, _legacy_connector_module())

    patch_eic_connector()
    assert isinstance(mod.ctypes, _GuardedCtypes)
    # The process-global ctypes module is never replaced.
    assert not isinstance(ctypes, _GuardedCtypes)
    assert callable(ctypes.CDLL)


def test_patch_is_idempotent_on_legacy_core(monkeypatch):
    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        _GuardedCtypes,
        patch_eic_connector,
    )

    mod = _install_dummy_connector(monkeypatch, _legacy_connector_module())

    patch_eic_connector()
    first = mod.ctypes
    patch_eic_connector()
    assert mod.ctypes is first
    assert isinstance(mod.ctypes, _GuardedCtypes)


def test_guarded_ctypes_only_shims_cudart_and_passes_other_attributes():
    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        _CudaLibShim,
        _GuardedCtypes,
    )

    real_cdll = ctypes.CDLL

    def fake_cdll(name, *args, **kwargs):
        if name == "libcudart.so":
            raise OSError("cannot open shared object file")
        return real_cdll(name, *args, **kwargs)

    guarded = _GuardedCtypes(ctypes)
    guarded._real_cdll = fake_cdll

    # The one library CANN images lack becomes a persisted no-op shim.
    shim = guarded.CDLL("libcudart.so")
    assert isinstance(shim, _CudaLibShim)
    shim.cudaMemcpy.argtypes = [ctypes.c_void_p]
    shim.cudaMemcpy.restype = ctypes.c_int
    with pytest.raises(RuntimeError):
        shim.cudaMemcpy(0, 0, 0, 0)
    # Unknown symbols on the shim are still callable no-ops that raise.
    with pytest.raises(RuntimeError):
        shim.cudaSomethingElse()

    # Non-CDLL attributes delegate to the real ctypes module.
    assert guarded.c_int is ctypes.c_int


def test_guarded_ctypes_concurrent_cudart_lookups_never_touch_global():
    # Regression for the review's deterministic race: the old patch swapped the
    # process-wide ctypes.CDLL and restored it in a finally, so a constructor
    # interleaved in that window got OSError on libcudart, and unrelated CDLL
    # callers saw the swap. The module-level proxy has no global mutation and no
    # restore window, so N concurrent lookups must all get a shim while the
    # real global CDLL stays intact and callable.
    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        _CudaLibShim,
        _GuardedCtypes,
    )

    real_cdll = ctypes.CDLL

    def fake_cdll(name, *args, **kwargs):
        if name == "libcudart.so":
            raise OSError("cannot open shared object file")
        return real_cdll(name, *args, **kwargs)

    guarded = _GuardedCtypes(ctypes)
    guarded._real_cdll = fake_cdll

    results = []
    errors = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait()
            results.append(guarded.CDLL("libcudart.so"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, [repr(e) for e in errors]
    assert len(results) == 8
    assert all(isinstance(r, _CudaLibShim) for r in results)
    # The global CDLL is never replaced, even mid-call from other threads.
    assert ctypes.CDLL is real_cdll


def test_patch_skips_without_connector_module(monkeypatch):
    # An unimportable leaf makes the dotted import raise ImportError, exactly
    # as when the optional connector / vendor `eic` package is absent. Block
    # both the sys.modules entry and the parent attribute, since the import
    # resolves to a bound parent attribute first. patch_eic_connector must
    # swallow the ImportError, not raise.
    import lmcache.v1.storage_backend.connector as parent

    monkeypatch.setitem(sys.modules, CONNECTOR_PATH, None)
    monkeypatch.delattr(parent, "eic_connector", raising=False)

    from lmcache_ascend.v1.storage_backend.connector.eic_npu import (
        patch_eic_connector,
    )

    patch_eic_connector()  # no exception == the asserted behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
