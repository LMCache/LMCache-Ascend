# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OpenEulerNDSLibrary's per-stream io_ctx plumbing (mocked).

Each registered stream owns one ``nds_io_ctx``. The host callback submits
and then blocks in ``nds_io_getevents`` on that context. No real libnds.so
or NPU is involved; ``_lib`` is a mock.
"""

# Standard
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest

from lmcache_ascend.v1.gds.openeuler_nds_library import OpenEulerNDSLibrary


def _make_lib() -> OpenEulerNDSLibrary:
    with patch.object(OpenEulerNDSLibrary, "load", lambda self: None):
        lib = OpenEulerNDSLibrary(library_path="/nonexistent")
    lib._lib = MagicMock()
    lib._driver_open = True
    return lib


def _complete(events_out, res: int = 0) -> int:
    events_out[0].res = res
    return 1


def test_register_stream_creates_one_context():
    lib = _make_lib()
    first = MagicMock(name="ctx-7")
    lib._new_io_ctx = MagicMock(return_value=first)  # type: ignore[method-assign]
    lib.register_stream(7)
    lib.register_stream(7)
    assert lib._new_io_ctx.call_count == 1
    assert lib._stream_ctxs[7] is first


def test_register_stream_requires_open_driver():
    lib = _make_lib()
    lib._driver_open = False
    with pytest.raises(RuntimeError, match="driver is not open"):
        lib.register_stream(7)


def test_deregister_stream_destroys_context_once():
    lib = _make_lib()
    ctx = MagicMock(name="ctx-7")
    destroyed: list = []
    lib._stream_ctxs[7] = ctx
    lib._destroy_io_ctx = destroyed.append  # type: ignore[method-assign]
    lib.deregister_stream(7)
    lib.deregister_stream(7)
    assert destroyed == [ctx]
    assert 7 not in lib._stream_ctxs


def test_driver_close_destroys_remaining_contexts():
    lib = _make_lib()
    ctx_a, ctx_b = MagicMock(name="a"), MagicMock(name="b")
    lib._stream_ctxs = {1: ctx_a, 2: ctx_b}
    lib._lib.nds_io_destroy_ctx.return_value = 0
    lib._lib.nds_exit.return_value = 0
    lib.driver_close()
    destroyed = {call.args[0] for call in lib._lib.nds_io_destroy_ctx.call_args_list}
    assert destroyed == {ctx_a, ctx_b}
    assert lib._stream_ctxs == {}
    assert lib._driver_open is False
    lib._lib.nds_exit.assert_called_once()


def test_io_submits_and_reaps_the_same_context():
    lib = _make_lib()
    ctx = MagicMock(name="ctx")
    lib._lib.nds_io_submit.return_value = 1
    lib._lib.nds_io_getevents.side_effect = lambda *args: _complete(args[3])
    assert lib._io(ctx, 3, 0x1000, 4096, 0, 0, False) == 4096
    assert lib._lib.nds_io_submit.call_args.args[0] is ctx
    assert lib._lib.nds_io_getevents.call_args.args[0] is ctx
    assert lib._lib.nds_io_getevents.call_args.args[4] is None


def test_io_raises_on_negative_completion():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 1
    lib._lib.nds_io_getevents.side_effect = lambda *args: _complete(args[3], -5)
    with pytest.raises(OSError):
        lib._io(MagicMock(), 3, 0x1000, 4096, 0, 0, False)


def test_submit_failure_does_not_call_getevents():
    lib = _make_lib()
    lib._lib.nds_io_submit.return_value = 0
    with pytest.raises(OSError):
        lib._io(MagicMock(), 3, 0x1000, 4096, 0, 0, False)
    lib._lib.nds_io_getevents.assert_not_called()


def test_two_streams_block_on_separate_contexts():
    lib = _make_lib()
    ctx_a, ctx_b = MagicMock(name="a"), MagicMock(name="b")
    release = threading.Event()
    entered = {id(ctx_a): threading.Event(), id(ctx_b): threading.Event()}

    def fake_getevents(ctx, min_nr, max_nr, events_out, timeout_ptr):
        del min_nr, max_nr, timeout_ptr
        entered[id(ctx)].set()
        assert release.wait(timeout=2.0)
        return _complete(events_out)

    lib._lib.nds_io_submit.return_value = 1
    lib._lib.nds_io_getevents.side_effect = fake_getevents
    results: dict[int, int] = {}

    def run(stream: int, ctx) -> None:
        results[stream] = lib._io(ctx, 3, 0x1000, 4096, 0, 0, False)

    threads = [
        threading.Thread(target=run, args=(1, ctx_a)),
        threading.Thread(target=run, args=(2, ctx_b)),
    ]
    for thread in threads:
        thread.start()
    assert entered[id(ctx_a)].wait(timeout=2.0)
    assert entered[id(ctx_b)].wait(timeout=2.0)
    release.set()
    for thread in threads:
        thread.join(timeout=2.0)
    assert results == {1: 4096, 2: 4096}
    seen = {call.args[0] for call in lib._lib.nds_io_getevents.call_args_list}
    assert seen == {ctx_a, ctx_b}


def test_sync_read_uses_a_private_context():
    lib = _make_lib()
    ctx = MagicMock(name="ephemeral")
    lib._new_io_ctx = MagicMock(return_value=ctx)  # type: ignore[method-assign]
    destroyed: list = []
    lib._destroy_io_ctx = destroyed.append  # type: ignore[method-assign]
    lib._lib.nds_io_submit.return_value = 1
    lib._lib.nds_io_getevents.side_effect = lambda *args: _complete(args[3])
    assert lib.read(3, 0x1000, 4096, 0, 0) == 4096
    assert lib._lib.nds_io_submit.call_args.args[0] is ctx
    assert destroyed == [ctx]


def _run_launch_immediately(raw_stream: int, callback) -> None:
    del raw_stream
    callback()


def _capture_launch(raw_stream: int, callback, captured: list) -> None:
    del raw_stream
    captured.append(callback)


def test_read_async_defers_result_until_host_callback():
    lib = _make_lib()
    ctx = MagicMock(name="ctx-7")
    lib._stream_ctxs[7] = ctx
    lib._lib.nds_io_submit.return_value = 1
    lib._lib.nds_io_getevents.side_effect = lambda *args: _complete(args[3])
    result = __import__("ctypes").c_int64(0)
    captured: list = []
    with patch(
        "lmcache_ascend.v1.gds.npu_host_callback.launch_npu_stream_host_callback",
        side_effect=lambda stream, cb: _capture_launch(stream, cb, captured),
    ):
        lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)
    assert result.value == 0
    lib._lib.nds_io_submit.assert_not_called()
    captured[0]()
    assert result.value == 4096
    assert lib._lib.nds_io_submit.call_args.args[0] is ctx


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
    lib._stream_ctxs[7] = MagicMock(name="ctx-7")
    lib._lib.nds_io_submit.return_value = 1
    lib._lib.nds_io_getevents.side_effect = lambda *args: _complete(args[3])
    result = __import__("ctypes").c_int64(0)
    stream = MagicMock()
    stream.npu_stream = 7
    del stream.launch_host_func
    npu_mod = MagicMock()
    npu_mod.is_available.return_value = True
    npu_mod.current_stream.return_value = stream
    acl_lib = MagicMock()

    def _run_host_func(_stream, func, arg) -> int:
        func(arg)
        return 0

    acl_lib.aclrtLaunchHostFunc.side_effect = _run_host_func
    with patch.object(__import__("torch"), "npu", npu_mod, create=True):
        with patch(
            "lmcache_ascend.v1.gds.npu_host_callback._acl_rt_library",
            return_value=acl_lib,
        ):
            lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)
    assert result.value == 4096
    acl_lib.aclrtLaunchHostFunc.assert_called_once()


def test_wait_callbacks_blocks_until_inflight_host_work_finishes():
    lib = _make_lib()
    lib._stream_ctxs[7] = MagicMock(name="ctx-7")
    lib._io = MagicMock(return_value=4096)  # type: ignore[method-assign]
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


def test_read_async_maps_missing_context_to_negative_result():
    lib = _make_lib()
    result = __import__("ctypes").c_int64(0)
    with patch(
        "lmcache_ascend.v1.gds.npu_host_callback.launch_npu_stream_host_callback",
        side_effect=_run_launch_immediately,
    ):
        lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)
    assert result.value < 0


def test_write_async_reads_segment_fields():
    lib = _make_lib()
    ctx = MagicMock(name="ctx-7")
    lib._stream_ctxs[7] = ctx
    lib._io = MagicMock(return_value=4096)  # type: ignore[method-assign]
    result = __import__("ctypes").c_int64(0)
    completed: list[bool] = []
    segment = SimpleNamespace(device_offset=0xA000, length=4096, buf_offset=128)
    with patch(
        "lmcache_ascend.v1.gds.npu_host_callback.launch_npu_stream_host_callback",
        side_effect=_run_launch_immediately,
    ):
        lib.write_async(
            9,
            0x1000,
            [segment],
            result,
            7,
            on_complete=lambda: completed.append(True),
        )
    lib._io.assert_called_once_with(ctx, 9, 0x1000, 4096, 0xA000, 128, True)
    assert result.value == 4096
    assert completed == [True]


def test_read_async_maps_io_failure_to_negative_result():
    lib = _make_lib()
    lib._stream_ctxs[7] = MagicMock(name="ctx-7")
    lib._lib.nds_io_submit.return_value = 1
    lib._lib.nds_io_getevents.side_effect = lambda *args: _complete(args[3], -5)
    result = __import__("ctypes").c_int64(0)
    with patch(
        "lmcache_ascend.v1.gds.npu_host_callback.launch_npu_stream_host_callback",
        side_effect=_run_launch_immediately,
    ):
        lib.read_async(3, 0x1000, 4096, 0, 0, result, 7)
    assert result.value == -5
