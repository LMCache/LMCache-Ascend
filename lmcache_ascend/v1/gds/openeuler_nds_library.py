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

from lmcache.logging import init_logger
from lmcache_ascend.v1.gds.nds_library import NDSCapabilities, _check_byte_count

logger = init_logger(__name__)

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

# Reaper thread: how many completions one nds_io_getevents call drains, and
# how long it blocks (via the timespec timeout) before re-checking the stop
# flag. Measured against real hardware: a non-null timeout does NOT appear
# to wake nds_io_getevents early on completion -- it sleeps out close to the
# full window per call regardless (a 0.2s poll turned a ~0.1s store into
# ~1.2s). Keep this small so that quantization is the dominant latency term
# only down to single-digit milliseconds, not hundreds.
_REAP_BATCH = 64
_REAP_POLL_SECONDS = 0.001
_USER_DATA_MAX = 0xFFFFFFFFFFFFFFFF


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


class _Timespec(ctypes.Structure):
    """``struct timespec`` for the ``nds_io_getevents`` timeout argument."""

    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


def _posix_errno(ret: int, operation: str) -> None:
    if ret < 0:
        raise OSError(-ret, operation)


class _PendingIO:
    """One in-flight ``nds_io_submit`` awaiting its completion from the reaper."""

    __slots__ = ("event", "res")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.res = 0


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

        # Tier A async-submission plumbing: nds_io_submit only enqueues (fast,
        # safe to call concurrently guarded by a short-held lock); a single
        # dedicated reaper thread is the sole caller of nds_io_getevents and
        # demuxes completions by user_data to the waiting caller. This
        # replaces one global submit+wait lock (which fully serialized every
        # NDS I/O backend-wide) with per-call waits, so unrelated reads/writes
        # from different streams/handles can be in flight concurrently.
        self._submit_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _PendingIO] = {}
        self._next_user_data = 1
        self._reaper_thread: threading.Thread | None = None
        self._reaper_stop = threading.Event()

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
        self._start_reaper()

    def driver_close(self) -> None:
        if not self._driver_open or self._lib is None:
            self._driver_open = False
            self._io_ctx = None
            return
        # Stop and join the reaper before destroying the io_ctx it reads from.
        self._stop_reaper()
        if self._io_ctx is not None:
            ret = self._lib.nds_io_destroy_ctx(self._io_ctx)
            if ret < 0:
                pass
            self._io_ctx = None
        ret = self._lib.nds_exit()
        _posix_errno(ret, "nds_exit")
        self._driver_open = False
        # Fail any submission that was in flight when the driver closed
        # rather than leaving its waiter blocked forever.
        with self._pending_lock:
            stale = list(self._pending.values())
            self._pending.clear()
        for pending in stale:
            pending.res = -5  # EIO
            pending.event.set()

    def _start_reaper(self) -> None:
        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(
            target=self._reap_loop, name="nds-io-reaper", daemon=True
        )
        self._reaper_thread.start()

    def _stop_reaper(self) -> None:
        self._reaper_stop.set()
        thread = self._reaper_thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("nds-io-reaper did not stop within 5s")
        self._reaper_thread = None

    def _reap_loop(self) -> None:
        """Sole caller of ``nds_io_getevents``; demuxes by ``user_data``.

        Runs with a short timeout so ``driver_close`` can stop it promptly;
        a timeout with zero events is the normal, frequent case while idle.
        """
        assert self._lib is not None and self._io_ctx is not None
        events = (_NdsIoEvent * _REAP_BATCH)()
        timeout = _Timespec(
            tv_sec=0, tv_nsec=int(_REAP_POLL_SECONDS * 1_000_000_000)
        )
        while not self._reaper_stop.is_set():
            # min_nr=1 so the call actually blocks up to the timeout instead
            # of returning immediately with zero events (min_nr=0 would spin
            # this thread at 100% CPU).
            ret = self._lib.nds_io_getevents(
                self._io_ctx, 1, _REAP_BATCH, events, ctypes.byref(timeout)
            )
            if ret <= 0:
                if ret < 0:
                    logger.error("nds_io_getevents reaper error: %d", ret)
                continue
            for i in range(ret):
                self._complete(events[i].user_data, events[i].res)

    def _complete(self, user_data: int, res: int) -> None:
        with self._pending_lock:
            pending = self._pending.pop(user_data, None)
        if pending is None:
            logger.warning(
                "nds-io-reaper: completion for unknown user_data=%d (res=%d)",
                user_data,
                res,
            )
            return
        pending.res = res
        pending.event.set()

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

    def _next_user_data_locked(self) -> int:
        """Allocate a user_data id; call with ``self._pending_lock`` held."""
        user_data = self._next_user_data
        self._next_user_data = (
            1 if user_data >= _USER_DATA_MAX else user_data + 1
        )
        return user_data

    def _submit(
        self,
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        ptr_offset: int,
        is_write: bool,
    ) -> _PendingIO:
        """Enqueue one op via ``nds_io_submit`` and return its waitable handle.

        Does not block on completion. ``nds_io_submit`` itself is held under
        a short lock (unknown thread-safety of the native ring buffer for
        concurrent submitters); the wait is not, so independent callers can
        have I/O in flight at the same time instead of serializing behind
        one submit+wait critical section.
        """
        self.load()
        assert self._lib is not None and self._io_ctx is not None
        if size <= 0:
            raise ValueError("I/O size must be positive")
        if size > _MAX_IO_BYTES:
            raise ValueError(f"I/O size {size} exceeds max {_MAX_IO_BYTES}")
        buf_addr = ptr_base + ptr_offset
        iov = _NdsIoVec(buf_addr=buf_addr, buf_len=size, reserved=0)
        pending = _PendingIO()
        with self._pending_lock:
            user_data = self._next_user_data_locked()
            self._pending[user_data] = pending
        cb = _NdsIoCb()
        cb.opcode = NDS_IO_OP_PWRITE if is_write else NDS_IO_OP_PREAD
        cb.rw_flags = NDS_IO_F_REGISTERED_MEM
        cb.obj.reserved = 0
        cb.obj.u.fd = file_fd
        cb.offset = file_offset
        cb.user_data = user_data
        cb.iov = ctypes.pointer(iov)
        cb.iov_cnt = 1
        cb.host_pid = 0
        cb.reserved = (0, 0)

        with self._submit_lock:
            ret = self._lib.nds_io_submit(self._io_ctx, 1, ctypes.byref(cb))
        if ret != 1:
            with self._pending_lock:
                self._pending.pop(user_data, None)
            _posix_errno(ret if ret < 0 else -1, "nds_io_submit")
        return pending

    def _wait(self, pending: _PendingIO, size: int) -> int:
        """Block until the reaper reports this submission's completion."""
        pending.event.wait()
        if pending.res != 0:
            raise OSError(-pending.res, "nds_io completion")
        _check_byte_count(size, size, "nds_io")
        return size

    def _io(
        self,
        file_fd: int,
        ptr_base: int,
        size: int,
        file_offset: int,
        ptr_offset: int,
        is_write: bool,
    ) -> int:
        pending = self._submit(file_fd, ptr_base, size, file_offset, ptr_offset, is_write)
        return self._wait(pending, size)


@lru_cache(maxsize=1)
def get_openeuler_nds_library(
    library_path: str = DEFAULT_OPENEULER_NDS_LIBRARY_PATH,
) -> OpenEulerNDSLibrary:
    return OpenEulerNDSLibrary(library_path=library_path)
