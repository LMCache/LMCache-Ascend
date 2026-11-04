# SPDX-License-Identifier: Apache-2.0
"""ctypes adapter for openEuler ``libnds.so`` (kernel NDS / nds_p2p on local NVMe).

Source: openEuler kernel ``tools/lib/nds`` ([openEuler-26.09](https://gitcode.com/openeuler/kernel/tree/openEuler-26.09)).
User API: ``nds_api.h``; device node ``/dev/p2p_device`` (``drivers/nds``).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from functools import lru_cache
import ctypes
import os
import threading

from lmcache_ascend.v1.gds.nds_library import NDSCapabilities, _check_byte_count

DEFAULT_OPENEULER_NDS_LIBRARY_PATH = os.environ.get(
    "NDS_LIBRARY_PATH",
    "/usr/local/lib/libnds.so",
)

NDS_IO_OP_PREAD = 0
NDS_IO_OP_PWRITE = 1
NDS_IO_F_REGISTERED_MEM = 1 << 0
NDS_API_VERSION = 1
_ALIGNMENT = 512
_MAX_IO_BYTES = 128 * 1024 * 1024


class _NdsInitParam(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
        ("version", ctypes.c_uint32),
    ]


class _NdsFsDesc(ctypes.Structure):
    _fields_ = [
        ("fs_fd", ctypes.POINTER(ctypes.c_int32)),
        ("fs_fd_cnt", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _NdsIoCtxParam(ctypes.Structure):
    _fields_ = [
        ("max_io_cnt", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]


class _NdsIoObj(ctypes.Structure):
    class _Data(ctypes.Union):
        _fields_ = [("_data", ctypes.c_uint64), ("fd", ctypes.c_int32)]

    _fields_ = [("reserved", ctypes.c_uint64), ("u", _Data)]


class _NdsIoVec(ctypes.Structure):
    _fields_ = [
        ("buf_addr", ctypes.c_uint64),
        ("buf_len", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _NdsIoCb(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint32),
        ("rw_flags", ctypes.c_uint32),
        ("obj", _NdsIoObj),
        ("offset", ctypes.c_uint64),
        ("user_data", ctypes.c_uint64),
        ("iov", ctypes.POINTER(_NdsIoVec)),
        ("iov_cnt", ctypes.c_uint32),
        ("host_pid", ctypes.c_int32),
        ("reserved", ctypes.c_uint64 * 2),
    ]


class _NdsIoEvent(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_uint64),
        ("res", ctypes.c_int64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _NdsIoCtx(ctypes.Structure):
    pass


def _posix_errno(ret: int, operation: str) -> None:
    if ret < 0:
        raise OSError(-ret, operation)


class OpenEulerNDSLibrary:
    """Process-local owner of openEuler ``libnds.so`` (P2P NVMe ↔ NPU HBM)."""

    def __init__(
        self, library_path: str = DEFAULT_OPENEULER_NDS_LIBRARY_PATH
    ) -> None:
        self._library_path = library_path
        self._lib: ctypes.CDLL | None = None
        self._lock = threading.Lock()
        self._driver_open = False
        self._io_ctx: ctypes.POINTER(_NdsIoCtx) | None = None

    @property
    def is_loaded(self) -> bool:
        return self._lib is not None

    def load(self) -> None:
        with self._lock:
            if self._lib is not None:
                return
            if not os.path.isfile(self._library_path):
                raise FileNotFoundError(
                    f"openEuler NDS library not found at {self._library_path}"
                )
            self._lib = ctypes.CDLL(self._library_path, mode=ctypes.RTLD_LOCAL)
            self._declare_signatures()

    def unload(self) -> None:
        with self._lock:
            if self._driver_open:
                self.driver_close()
            self._lib = None

    def _declare_signatures(self) -> None:
        assert self._lib is not None
        lib = self._lib
        for name in (
            "nds_init",
            "nds_exit",
            "nds_register_fs",
            "nds_unregister_fs",
            "nds_register_mem",
            "nds_unregister_mem",
            "nds_io_new_ctx",
            "nds_io_destroy_ctx",
            "nds_io_submit",
            "nds_io_getevents",
        ):
            if not hasattr(lib, name):
                raise AttributeError(
                    f"openEuler libnds missing required symbol {name!r}"
                )

        lib.nds_init.argtypes = [ctypes.POINTER(_NdsInitParam)]
        lib.nds_init.restype = ctypes.c_int
        lib.nds_exit.argtypes = []
        lib.nds_exit.restype = ctypes.c_int
        lib.nds_register_fs.argtypes = [ctypes.POINTER(_NdsFsDesc)]
        lib.nds_register_fs.restype = ctypes.c_int
        lib.nds_unregister_fs.argtypes = [ctypes.POINTER(_NdsFsDesc)]
        lib.nds_unregister_fs.restype = ctypes.c_int
        lib.nds_register_mem.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        lib.nds_register_mem.restype = ctypes.c_int
        lib.nds_unregister_mem.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        lib.nds_unregister_mem.restype = ctypes.c_int
        lib.nds_io_new_ctx.argtypes = [
            ctypes.POINTER(_NdsIoCtxParam),
            ctypes.POINTER(ctypes.POINTER(_NdsIoCtx)),
        ]
        lib.nds_io_new_ctx.restype = ctypes.c_int
        lib.nds_io_destroy_ctx.argtypes = [ctypes.POINTER(_NdsIoCtx)]
        lib.nds_io_destroy_ctx.restype = ctypes.c_int
        lib.nds_io_submit.argtypes = [
            ctypes.POINTER(_NdsIoCtx),
            ctypes.c_int,
            ctypes.POINTER(_NdsIoCb),
        ]
        lib.nds_io_submit.restype = ctypes.c_int
        lib.nds_io_getevents.argtypes = [
            ctypes.POINTER(_NdsIoCtx),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_NdsIoEvent),
            ctypes.c_void_p,
        ]
        lib.nds_io_getevents.restype = ctypes.c_int

    def capabilities(self) -> NDSCapabilities:
        if self._lib is None:
            return NDSCapabilities(
                can_read=False,
                can_write=False,
                requires_buffer_registration=True,
                supports_stream_ordering=False,
                alignment_bytes=_ALIGNMENT,
                max_io_bytes=_MAX_IO_BYTES,
            )
        return NDSCapabilities(
            can_read=True,
            can_write=True,
            requires_buffer_registration=True,
            supports_stream_ordering=False,
            alignment_bytes=_ALIGNMENT,
            max_io_bytes=_MAX_IO_BYTES,
            write_symbol="nds_io_submit(PWRITE)",
        )

    def driver_open(self) -> None:
        self.load()
        assert self._lib is not None
        param = _NdsInitParam()
        ret = self._lib.nds_init(ctypes.byref(param))
        _posix_errno(ret, "nds_init")
        if param.version != NDS_API_VERSION:
            raise RuntimeError(
                f"nds_init returned unexpected API version {param.version}"
            )
        ctx = ctypes.POINTER(_NdsIoCtx)()
        ctx_param = _NdsIoCtxParam(max_io_cnt=64, flags=0, reserved=0)
        ret = self._lib.nds_io_new_ctx(ctypes.byref(ctx_param), ctypes.byref(ctx))
        _posix_errno(ret, "nds_io_new_ctx")
        self._io_ctx = ctx
        self._driver_open = True

    def driver_close(self) -> None:
        if not self._driver_open or self._lib is None:
            self._driver_open = False
            self._io_ctx = None
            return
        if self._io_ctx is not None:
            ret = self._lib.nds_io_destroy_ctx(self._io_ctx)
            if ret < 0:
                pass
            self._io_ctx = None
        ret = self._lib.nds_exit()
        _posix_errno(ret, "nds_exit")
        self._driver_open = False

    def file_register(self, fd: int) -> int:
        self.load()
        assert self._lib is not None
        fd_arr = (ctypes.c_int32 * 1)(fd)
        desc = _NdsFsDesc(fd_arr, 1, 0)
        ret = self._lib.nds_register_fs(ctypes.byref(desc))
        _posix_errno(ret, "nds_register_fs")
        return fd

    def file_deregister(self, native_handle: int) -> None:
        del native_handle

    def register_memory(self, addr: int, size: int) -> None:
        self.load()
        assert self._lib is not None
        ret = self._lib.nds_register_mem(
            ctypes.c_void_p(addr), ctypes.c_uint64(size), 0
        )
        _posix_errno(ret, "nds_register_mem")

    def unregister_memory(self, addr: int, size: int) -> None:
        if self._lib is None:
            return
        ret = self._lib.nds_unregister_mem(
            ctypes.c_void_p(addr), ctypes.c_uint64(size), 0
        )
        if ret == -2:  # ENOENT if already gone
            return
        _posix_errno(ret, "nds_unregister_mem")

    def read(
        self,
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        ptr_offset: int,
    ) -> int:
        return self._io(file_fd, ptr_base, size, file_offset, ptr_offset, False)

    def write(
        self,
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        ptr_offset: int,
    ) -> int:
        return self._io(file_fd, ptr_base, size, file_offset, ptr_offset, True)

    def _io(
        self,
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        ptr_offset: int,
        is_write: bool,
    ) -> int:
        self.load()
        assert self._lib is not None and self._io_ctx is not None
        if size <= 0:
            raise ValueError("I/O size must be positive")
        if size > _MAX_IO_BYTES:
            raise ValueError(f"I/O size {size} exceeds max {_MAX_IO_BYTES}")
        buf_addr = ptr_base + ptr_offset
        iov = _NdsIoVec(
            buf_addr=buf_addr, buf_len=size, reserved=0
        )
        cb = _NdsIoCb()
        cb.opcode = NDS_IO_OP_PWRITE if is_write else NDS_IO_OP_PREAD
        cb.rw_flags = NDS_IO_F_REGISTERED_MEM
        cb.obj.reserved = 0
        cb.obj.u.fd = file_fd
        cb.offset = file_offset
        cb.user_data = 0
        cb.iov = ctypes.pointer(iov)
        cb.iov_cnt = 1
        cb.host_pid = 0
        cb.reserved = (0, 0)

        ret = self._lib.nds_io_submit(self._io_ctx, 1, ctypes.byref(cb))
        if ret != 1:
            _posix_errno(ret if ret < 0 else -1, "nds_io_submit")

        event = _NdsIoEvent()
        ret = self._lib.nds_io_getevents(
            self._io_ctx, 1, 1, ctypes.byref(event), None
        )
        if ret < 0:
            _posix_errno(ret, "nds_io_getevents")
        if event.res != 0:
            raise OSError(-event.res, "nds_io completion")
        _check_byte_count(size, size, "nds_io")
        return size


@lru_cache(maxsize=1)
def get_openeuler_nds_library(
    library_path: str = DEFAULT_OPENEULER_NDS_LIBRARY_PATH,
) -> OpenEulerNDSLibrary:
    return OpenEulerNDSLibrary(library_path=library_path)
