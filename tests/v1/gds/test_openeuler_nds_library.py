# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OpenEulerNDSLibrary's async submit/reap/wait plumbing (mocked).

Exercises the Tier A redesign: nds_io_submit only enqueues; a dedicated
reaper thread is the sole caller of nds_io_getevents and demuxes completions
by user_data to the waiting caller. No real libnds.so / NPU is involved;
``_lib`` and ``_io_ctx`` are replaced with mocks.
"""

# Standard
import threading
import time
from unittest.mock import MagicMock, patch

# Third Party
import pytest

from lmcache_ascend.v1.gds.openeuler_nds_library import OpenEulerNDSLibrary


def _make_lib() -> OpenEulerNDSLibrary:
    lib = OpenEulerNDSLibrary(library_path="/nonexistent")
    lib._lib = MagicMock()
    lib._io_ctx = MagicMock()
    lib._driver_open = True
    return lib


def _pending_ids(lib: OpenEulerNDSLibrary) -> list:
    with lib._pending_lock:
        return list(lib._pending.keys())


def test_submit_then_reap_completes_pending():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 1

    def fake_getevents(ctx, min_nr, max_nr, events_out, timeout_ptr):
        del ctx, min_nr, max_nr, timeout_ptr
        ids = _pending_ids(lib)
        if not ids:
            time.sleep(0.005)
            return 0
        events_out[0].user_data = ids[0]
        events_out[0].res = 0
        return 1

    lib._lib.nds_io_getevents.side_effect = fake_getevents
    lib._start_reaper()
    try:
        pending = lib._submit(
            file_fd=3,
            ptr_base=0x1000,
            size=4096,
            file_offset=0,
            ptr_offset=0,
            is_write=False,
        )
        assert lib._wait(pending, 4096) == 4096
    finally:
        lib._stop_reaper()


def test_wait_raises_on_negative_completion():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 1

    def fake_getevents(ctx, min_nr, max_nr, events_out, timeout_ptr):
        del ctx, min_nr, max_nr, timeout_ptr
        ids = _pending_ids(lib)
        if not ids:
            time.sleep(0.005)
            return 0
        events_out[0].user_data = ids[0]
        events_out[0].res = -5  # -EIO
        return 1

    lib._lib.nds_io_getevents.side_effect = fake_getevents
    lib._start_reaper()
    try:
        pending = lib._submit(
            file_fd=3,
            ptr_base=0x1000,
            size=4096,
            file_offset=0,
            ptr_offset=0,
            is_write=False,
        )
        with pytest.raises(OSError):
            lib._wait(pending, 4096)
    finally:
        lib._stop_reaper()


def test_submits_overlap_without_serializing_on_completion():
    """Two submissions can be outstanding at once; submit never blocks on

    the other's completion, unlike the old single submit+wait-under-lock
    design.
    """
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 1
    release = threading.Event()

    def fake_getevents(ctx, min_nr, max_nr, events_out, timeout_ptr):
        del ctx, min_nr, timeout_ptr
        if not release.is_set():
            time.sleep(0.005)
            return 0
        ids = _pending_ids(lib)
        n = min(len(ids), max_nr)
        for i in range(n):
            events_out[i].user_data = ids[i]
            events_out[i].res = 0
        return n

    lib._lib.nds_io_getevents.side_effect = fake_getevents
    lib._start_reaper()
    try:
        p1 = lib._submit(3, 0x1000, 4096, 0, 0, False)
        p2 = lib._submit(3, 0x2000, 4096, 4096, 0, False)
        # Both submits returned without waiting on either's completion.
        assert not p1.event.is_set()
        assert not p2.event.is_set()
        release.set()
        assert lib._wait(p1, 4096) == 4096
        assert lib._wait(p2, 4096) == 4096
    finally:
        lib._stop_reaper()


def test_driver_close_fails_outstanding_pending():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 1
    lib._lib.nds_io_getevents.return_value = 0  # never completes on its own
    lib._lib.nds_io_destroy_ctx.return_value = 0
    lib._lib.nds_exit.return_value = 0
    lib._start_reaper()
    pending = lib._submit(3, 0x1000, 4096, 0, 0, False)
    lib.driver_close()
    with pytest.raises(OSError):
        lib._wait(pending, 4096)


def test_submit_failure_removes_pending_entry():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 0  # anything != 1 is a failure
    with pytest.raises(OSError):
        lib._submit(3, 0x1000, 4096, 0, 0, False)
    assert _pending_ids(lib) == []


def _run_launch_immediately(raw_stream: int, callback) -> None:
    callback()


def _capture_launch(raw_stream: int, callback, captured: list) -> None:
    captured.append(callback)


def test_read_async_defers_result_until_host_callback():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 1
    result = __import__("ctypes").c_int64(0)
    captured: list = []

    def fake_getevents(ctx, min_nr, max_nr, events_out, timeout_ptr):
        del ctx, min_nr, timeout_ptr
        ids = _pending_ids(lib)
        if not ids:
            return 0
        events_out[0].user_data = ids[0]
        events_out[0].res = 0
        return 1

    lib._lib.nds_io_getevents.side_effect = fake_getevents
    lib._start_reaper()
    try:
        with patch(
            "lmcache_ascend.v1.gds.npu_host_callback.launch_npu_stream_host_callback",
            side_effect=lambda stream, cb: _capture_launch(stream, cb, captured),
        ):
            lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)
        assert result.value == 0
        lib._lib.nds_io_submit.assert_not_called()
        captured[0]()
        assert result.value == 4096
    finally:
        lib._stop_reaper()


def test_read_async_rejects_wrong_current_stream():
    lib = _make_lib()
    result = __import__("ctypes").c_int64(0)
    npu_mod = MagicMock()
    stream = MagicMock()
    stream.npu_stream = 8
    npu_mod.is_available.return_value = True
    npu_mod.current_stream.return_value = stream
    with patch.object(__import__("torch"), "npu", npu_mod, create=True):
        with pytest.raises(RuntimeError, match="current NPU stream"):
            lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)


def test_read_async_uses_acl_when_stream_lacks_launch_host_func():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 1
    lib._start_reaper()
    result = __import__("ctypes").c_int64(0)

    def fake_getevents(ctx, min_nr, max_nr, events_out, timeout_ptr):
        del ctx, min_nr, timeout_ptr
        ids = _pending_ids(lib)
        if not ids:
            time.sleep(0.005)
            return 0
        events_out[0].user_data = ids[0]
        events_out[0].res = 0
        return 1

    lib._lib.nds_io_getevents.side_effect = fake_getevents
    stream = MagicMock()
    stream.npu_stream = 7
    del stream.launch_host_func
    npu_mod = MagicMock()
    npu_mod.is_available.return_value = True
    npu_mod.current_stream.return_value = stream
    acl_lib = MagicMock()
    acl_lib.aclrtLaunchHostFunc.return_value = 0
    try:
        with patch.object(__import__("torch"), "npu", npu_mod, create=True):
            with patch(
                "lmcache_ascend.v1.gds.npu_host_callback._acl_rt_library",
                return_value=acl_lib,
            ):
                lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)
        time.sleep(0.05)
        assert result.value == 4096
        acl_lib.aclrtLaunchHostFunc.assert_called_once()
    finally:
        lib._stop_reaper()


def test_wait_callbacks_blocks_until_inflight_host_work_finishes():
    lib = _make_lib()
    lib.read = MagicMock(return_value=4096)  # type: ignore[method-assign]
    result = __import__("ctypes").c_int64(0)
    gate = threading.Event()

    def slow_launch(raw_stream: int, callback) -> None:
        del raw_stream

        def deferred() -> None:
            gate.wait()
            callback()

        threading.Thread(target=deferred, daemon=True).start()

    with patch(
        "lmcache_ascend.v1.gds.npu_host_callback.launch_npu_stream_host_callback",
        side_effect=slow_launch,
    ):
        lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)
    done = threading.Event()

    def wait_in_thread() -> None:
        lib.wait_callbacks()
        done.set()

    thread = threading.Thread(target=wait_in_thread)
    thread.start()
    time.sleep(0.05)
    assert not done.is_set()
    gate.set()
    thread.join(timeout=2.0)
    assert done.is_set()
    assert result.value == 4096


def test_read_async_maps_io_failure_to_negative_result():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 1

    def fake_getevents(ctx, min_nr, max_nr, events_out, timeout_ptr):
        del ctx, min_nr, timeout_ptr
        ids = _pending_ids(lib)
        if not ids:
            time.sleep(0.005)
            return 0
        events_out[0].user_data = ids[0]
        events_out[0].res = -5
        return 1

    lib._lib.nds_io_getevents.side_effect = fake_getevents
    lib._start_reaper()
    result = __import__("ctypes").c_int64(0)
    with patch(
        "lmcache_ascend.v1.gds.npu_host_callback.launch_npu_stream_host_callback",
        side_effect=_run_launch_immediately,
    ):
        lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)
    assert result.value == -5
    lib._stop_reaper()
