# SPDX-License-Identifier: Apache-2.0
"""Enqueue host work on an Ascend NPU stream (``aclrtLaunchHostFunc`` shim)."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
import ctypes
import threading

# Third Party
import torch

_ACL_HOST_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_ACL_RT_LIB: ctypes.CDLL | None = None
_ACL_TRAMPOLINE_REFS: list[_ACL_HOST_FUNC] = []


def exception_to_submission_result(exc: BaseException) -> int:
    """Map an exception to a negative ``Submission.result`` for deferred I/O."""
    if isinstance(exc, OSError) and exc.errno is not None:
        return -int(exc.errno)
    return -1


def _acl_rt_library() -> ctypes.CDLL:
    global _ACL_RT_LIB
    if _ACL_RT_LIB is None:
        lib = ctypes.CDLL("libascendcl.so")
        lib.aclrtLaunchHostFunc.argtypes = [
            ctypes.c_void_p,
            _ACL_HOST_FUNC,
            ctypes.c_void_p,
        ]
        lib.aclrtLaunchHostFunc.restype = ctypes.c_int
        _ACL_RT_LIB = lib
    return _ACL_RT_LIB


def launch_npu_stream_host_callback(
    raw_stream: int, callback: Callable[[], None]
) -> None:
    """Enqueue ``callback`` on the current NPU stream after prior stream work."""
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError("NDS stream-ordered I/O requires torch.npu")
    stream = npu.current_stream()
    handle = getattr(stream, "npu_stream", None)
    if not isinstance(handle, int) or handle != raw_stream:
        current = handle if isinstance(handle, int) else 0
        raise RuntimeError(
            f"NDS I/O must run on the current NPU stream "
            f"(registered 0x{raw_stream:x}, current 0x{current:x})"
        )
    launch_host_func = getattr(stream, "launch_host_func", None)
    if callable(launch_host_func):
        launch_host_func(callback)
        return

    @_ACL_HOST_FUNC
    def trampoline(_arg: int) -> None:
        callback()

    _ACL_TRAMPOLINE_REFS.append(trampoline)
    err = _acl_rt_library().aclrtLaunchHostFunc(
        ctypes.c_void_p(raw_stream),
        trampoline,
        None,
    )
    if int(err) != 0:
        _ACL_TRAMPOLINE_REFS.pop()
        raise RuntimeError(f"aclrtLaunchHostFunc failed with error {err}")


class NpuStreamHostCallbackQueue:
    """Track in-flight stream host callbacks for ``wait_callbacks`` on close."""

    def __init__(self) -> None:
        self._inflight = 0
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)

    def wait_callbacks(self) -> None:
        with self._lock:
            while self._inflight > 0:
                self._done.wait()

    def launch(self, raw_stream: int, work: Callable[[], None]) -> None:
        with self._lock:
            self._inflight += 1

        def wrapped() -> None:
            try:
                work()
            finally:
                with self._lock:
                    self._inflight -= 1
                    self._done.notify_all()

        launch_npu_stream_host_callback(raw_stream, wrapped)
