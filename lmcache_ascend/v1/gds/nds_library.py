# SPDX-License-Identifier: Apache-2.0
"""NDS library factory: openEuler ``libnds.so`` (local NVMe) or DataTurbo ``libdpc_nds.so``."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import ctypes
import os
import stat
import threading
from typing import Any

from lmcache_ascend.v1.gds.npu_host_callback import (
    NpuStreamHostCallbackQueue,
    exception_to_submission_result,
)

NDS_STACK = os.environ.get("NDS_STACK", "openeuler").strip().lower()

DATATURBO_NDS_LIBRARY_PATH = os.environ.get(
    "NDS_LIBRARY_PATH",
    "/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so",
)
DEFAULT_NDS_LIBRARY_PATH = DATATURBO_NDS_LIBRARY_PATH

NDS_FILE_SUCCESS = 0
NDS_FILE_DRIVER_ERROR = -1

_MAX_IO_BYTES_DEFAULT = 128 * 1024 * 1024
_ALIGNMENT_DEFAULT = 4096


class _NdsFileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int)]


class _NdsFileDescr(ctypes.Structure):
    _fields_ = [("fd", ctypes.c_int)]


@dataclass(frozen=True)
class NDSCapabilities:
    """Symbol discovery result; refine fields from PR 0 probe artifacts."""

    can_read: bool
    can_write: bool
    requires_buffer_registration: bool
    supports_stream_ordering: bool
    alignment_bytes: int
    max_io_bytes: int
    write_symbol: str | None = None


def _check_driver_error(error: _NdsFileError, operation: str) -> None:
    if int(error.err) != NDS_FILE_SUCCESS:
        raise RuntimeError(f"{operation} failed: NdsFileError(err={int(error.err)})")


def _check_byte_count(actual: int, expected: int, operation: str) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{operation} short I/O: got {actual} bytes, expected {expected}"
        )


class NDSLibrary:
    """Process-local lazy owner of the NDS shared library."""

    def __init__(self, library_path: str = DEFAULT_NDS_LIBRARY_PATH) -> None:
        self._library_path = library_path
        self._lib: ctypes.CDLL | None = None
        self._lock = threading.Lock()
        self._driver_open = False
        self._write_symbol_name: str | None = None
        self._NdsFileDriverOpen = None
        self._NdsFileDriverClose = None
        self._NdsFileHandleRegister = None
        self._NdsFileHandleDeregister = None
        self._NdsFileRead = None
        self._NdsFileWrite = None
        self._stream_callbacks = NpuStreamHostCallbackQueue()

    @property
    def is_loaded(self) -> bool:
        return self._lib is not None

    def load(self) -> None:
        with self._lock:
            if self._lib is not None:
                return
            if not os.path.isfile(self._library_path):
                raise FileNotFoundError(
                    f"NDS library not found at {self._library_path}"
                )
            mode = stat.S_IMODE(os.stat(self._library_path).st_mode)
            if mode != stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH:
                raise PermissionError(
                    f"NDS library must be mode 444, got {oct(mode)}: "
                    f"{self._library_path}"
                )
            self._lib = ctypes.CDLL(self._library_path, mode=ctypes.RTLD_LOCAL)
            self._resolve_symbols()
            self._declare_signatures()

    def unload(self) -> None:
        with self._lock:
            if self._driver_open:
                self.driver_close()
            self._lib = None

    def _resolve_symbols(self) -> None:
        assert self._lib is not None
        lib = self._lib
        self._NdsFileDriverOpen = self._required(lib, "NdsFileDriverOpen")
        self._NdsFileDriverClose = self._required(lib, "NdsFileDriverClose")
        self._NdsFileHandleRegister = self._required(lib, "NdsFileHandleRegister")
        self._NdsFileHandleDeregister = self._required(lib, "NdsFileHandleDeregister")
        self._NdsFileRead = self._required(lib, "NdsFileRead")
        for candidate in ("NdsFileWrite", "NdsFileWriteSync"):
            sym = getattr(lib, candidate, None)
            if sym is not None:
                self._NdsFileWrite = sym
                self._write_symbol_name = candidate
                break

    @staticmethod
    def _required(lib: ctypes.CDLL, name: str):
        try:
            return getattr(lib, name)
        except AttributeError as exc:
            raise AttributeError(f"NDS library missing required symbol {name!r}") from exc

    def _declare_signatures(self) -> None:
        assert self._NdsFileDriverOpen is not None
        self._NdsFileDriverOpen.argtypes = []
        self._NdsFileDriverOpen.restype = _NdsFileError

        assert self._NdsFileDriverClose is not None
        self._NdsFileDriverClose.argtypes = []
        self._NdsFileDriverClose.restype = _NdsFileError

        assert self._NdsFileHandleRegister is not None
        self._NdsFileHandleRegister.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_NdsFileDescr),
        ]
        self._NdsFileHandleRegister.restype = _NdsFileError

        assert self._NdsFileHandleDeregister is not None
        self._NdsFileHandleDeregister.argtypes = [ctypes.c_void_p]
        self._NdsFileHandleDeregister.restype = None

        assert self._NdsFileRead is not None
        self._NdsFileRead.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self._NdsFileRead.restype = ctypes.c_ssize_t

        if self._NdsFileWrite is not None:
            self._NdsFileWrite.argtypes = self._NdsFileRead.argtypes
            self._NdsFileWrite.restype = ctypes.c_ssize_t

    def capabilities(self) -> NDSCapabilities:
        if self._lib is None:
            return NDSCapabilities(
                can_read=False,
                can_write=False,
                requires_buffer_registration=False,
                supports_stream_ordering=False,
                alignment_bytes=_ALIGNMENT_DEFAULT,
                max_io_bytes=_MAX_IO_BYTES_DEFAULT,
            )
        return NDSCapabilities(
            can_read=self._NdsFileRead is not None,
            can_write=self._NdsFileWrite is not None,
            requires_buffer_registration=False,
            supports_stream_ordering=False,
            alignment_bytes=_ALIGNMENT_DEFAULT,
            max_io_bytes=_MAX_IO_BYTES_DEFAULT,
            write_symbol=self._write_symbol_name,
        )

    def driver_open(self) -> None:
        self.load()
        assert self._NdsFileDriverOpen is not None
        err = self._NdsFileDriverOpen()
        _check_driver_error(err, "NdsFileDriverOpen")
        self._driver_open = True

    def driver_close(self) -> None:
        if not self._driver_open or self._NdsFileDriverClose is None:
            self._driver_open = False
            return
        err = self._NdsFileDriverClose()
        _check_driver_error(err, "NdsFileDriverClose")
        self._driver_open = False

    def file_register(self, fd: int) -> ctypes.c_void_p:
        self.load()
        assert self._NdsFileHandleRegister is not None
        native = ctypes.c_void_p()
        descr = _NdsFileDescr()
        descr.fd = fd
        err = self._NdsFileHandleRegister(
            ctypes.byref(native), ctypes.byref(descr)
        )
        _check_driver_error(err, "NdsFileHandleRegister")
        return native

    def file_deregister(self, native_handle: ctypes.c_void_p) -> None:
        if self._NdsFileHandleDeregister is None or not native_handle:
            return
        self._NdsFileHandleDeregister(native_handle)

    def read(
        self,
        native_handle: ctypes.c_void_p,
        ptr_base: int,
        size: int,
        file_offset: int,
        ptr_offset: int,
    ) -> int:
        self.load()
        assert self._NdsFileRead is not None
        if size <= 0:
            raise ValueError("read size must be positive")
        if size > _MAX_IO_BYTES_DEFAULT:
            raise ValueError(f"read size {size} exceeds max {_MAX_IO_BYTES_DEFAULT}")
        result = self._NdsFileRead(
            native_handle,
            ctypes.c_void_p(ptr_base),
            ctypes.c_size_t(size),
            ctypes.c_int64(file_offset),
            ctypes.c_int64(ptr_offset),
        )
        if result < 0:
            raise RuntimeError(f"NdsFileRead failed with return code {result}")
        return int(result)

    def write(
        self,
        native_handle: ctypes.c_void_p,
        ptr_base: int,
        size: int,
        file_offset: int,
        ptr_offset: int,
    ) -> int:
        self.load()
        if self._NdsFileWrite is None:
            raise RuntimeError("NDS write symbol is not available in this library")
        if size <= 0:
            raise ValueError("write size must be positive")
        result = self._NdsFileWrite(
            native_handle,
            ctypes.c_void_p(ptr_base),
            ctypes.c_size_t(size),
            ctypes.c_int64(file_offset),
            ctypes.c_int64(ptr_offset),
        )
        if result < 0:
            raise RuntimeError(f"{self._write_symbol_name} failed with return code {result}")
        return int(result)

    def wait_callbacks(self) -> None:
        self._stream_callbacks.wait_callbacks()

    def read_async(
        self,
        native_handle: Any,
        ptr_base: int,
        size: int,
        file_offset: int,
        ptr_offset: int,
        result: ctypes.c_int64,
        raw_stream: int,
    ) -> None:
        def host_work() -> None:
            try:
                done = self.read(
                    native_handle, ptr_base, size, file_offset, ptr_offset
                )
                if done != size:
                    raise RuntimeError(
                        f"NDS read short I/O: got {done} bytes, expected {size}"
                    )
                result.value = size
            except BaseException as exc:
                result.value = exception_to_submission_result(exc)

        self._stream_callbacks.launch(raw_stream, host_work)

    def write_async(
        self,
        native_handle: Any,
        ptr_base: int,
        segments: list[tuple[int, int, int]],
        result: ctypes.c_int64,
        raw_stream: int,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        def host_work() -> None:
            try:
                done_total = 0
                for device_off, seg_len, ptr_off in segments:
                    done = self.write(
                        native_handle, ptr_base, seg_len, device_off, ptr_off
                    )
                    if done != seg_len:
                        raise RuntimeError(
                            f"NDS write short I/O: got {done} bytes, expected {seg_len}"
                        )
                    done_total += seg_len
                if on_complete is not None:
                    on_complete()
                result.value = done_total
            except BaseException as exc:
                result.value = exception_to_submission_result(exc)

        self._stream_callbacks.launch(raw_stream, host_work)

    def register_memory(self, addr: int, size: int, device_index: int = 0) -> int:
        """No-op for the DataTurbo client. The openEuler adapter overrides this."""
        del addr, size, device_index
        return 0

    def unregister_memory(self, addr: int, size: int, handle: int = 0) -> None:
        del addr, size, handle


@lru_cache(maxsize=1)
def get_nds_library(library_path: str | None = None):
    """Return process-global NDS adapter (lazy load). Default stack: openEuler local NVMe."""
    stack = NDS_STACK
    if stack in ("openeuler", "oe", "p2p", "local"):
        from lmcache_ascend.v1.gds.openeuler_nds_library import (
            DEFAULT_OPENEULER_NDS_LIBRARY_PATH,
            get_openeuler_nds_library,
        )

        path = library_path or DEFAULT_OPENEULER_NDS_LIBRARY_PATH
        return get_openeuler_nds_library(path)
    if stack in ("dataturbo", "dpc", "mindio", "oceanstor"):
        path = library_path or DATATURBO_NDS_LIBRARY_PATH
        return NDSLibrary(library_path=path)
    raise ValueError(
        f"Unknown NDS_STACK={stack!r}; use openeuler or dataturbo"
    )
