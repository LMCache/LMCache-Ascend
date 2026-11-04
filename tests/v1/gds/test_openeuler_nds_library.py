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
from unittest.mock import MagicMock

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
