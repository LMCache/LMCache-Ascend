# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``NDSLibrary`` (mocked; no ``dlopen`` required)."""

# Standard
from unittest.mock import MagicMock, patch
import importlib

# Third Party
import pytest

MODULE = "lmcache_ascend.v1.gds.nds_library"


@pytest.fixture(autouse=True)
def _clear_nds_singleton():
    mod = importlib.import_module(MODULE)
    mod.get_nds_library.cache_clear()
    yield
    mod.get_nds_library.cache_clear()


def test_library_import_does_not_dlopen_nds(monkeypatch):
    monkeypatch.setenv("NDS_STACK", "dataturbo")
    mod = importlib.reload(importlib.import_module(MODULE))
    mod.get_nds_library.cache_clear()
    lib = mod.get_nds_library("/nonexistent/for/import/test.so")
    assert not lib.is_loaded


def test_get_nds_library_openeuler_stack(monkeypatch):
    monkeypatch.setenv("NDS_STACK", "openeuler")
    mod = importlib.reload(importlib.import_module(MODULE))
    oe = importlib.import_module("lmcache_ascend.v1.gds.openeuler_nds_library")
    mod.get_nds_library.cache_clear()
    oe.get_openeuler_nds_library.cache_clear()
    try:
        with patch(
            "lmcache_ascend.v1.gds.openeuler_nds_library.OpenEulerNDSLibrary.load"
        ):
            lib = mod.get_nds_library()
        assert type(lib).__name__ == "OpenEulerNDSLibrary"
    finally:
        mod.get_nds_library.cache_clear()
        oe.get_openeuler_nds_library.cache_clear()


def test_load_rejects_missing_required_symbol(tmp_path, monkeypatch):
    monkeypatch.setenv("NDS_STACK", "dataturbo")
    mod = importlib.reload(importlib.import_module(MODULE))
    mod.get_nds_library.cache_clear()
    fake = tmp_path / "libdpc_nds.so"
    fake.write_bytes(b"\x7fELF")
    fake.chmod(0o444)
    lib = mod.NDSLibrary(str(fake))
    with patch.object(mod.ctypes, "CDLL") as cdll:
        mock = MagicMock()
        del mock.NdsFileRead
        mock.NdsFileDriverOpen = MagicMock()
        mock.NdsFileDriverClose = MagicMock()
        mock.NdsFileHandleRegister = MagicMock()
        mock.NdsFileHandleDeregister = MagicMock()
        cdll.return_value = mock
        with pytest.raises(AttributeError, match="NdsFileRead"):
            lib.load()


def test_check_byte_count_raises():
    mod = importlib.import_module(MODULE)
    with pytest.raises(RuntimeError, match="short I/O"):
        mod._check_byte_count(1, 4096, "NdsFileRead")
