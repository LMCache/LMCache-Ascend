# SPDX-License-Identifier: Apache-2.0

"""Tests for the Ascend C++ EventRecorder (record_event_on_stream / drain).

Ported from lmcache tests/v1/mp_observability/test_event_recorder.py; the
cupy stream is replaced by a torch_npu stream adapter exposing the same
`.ptr` / `.synchronize()` surface.
"""

# Standard
import threading
import time

# Third Party
import pytest
import torch

np_ops = pytest.importorskip(
    "lmcache_ascend.c_ops", reason="lmcache_ascend.c_ops not built"
)

if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("requires available NPU runtime", allow_module_level=True)

if not hasattr(np_ops, "record_event_on_stream"):
    pytest.skip("record_event_on_stream not available", allow_module_level=True)


class _NpuStreamHandle:
    """Adapt a torch_npu stream to the `.ptr` / `.synchronize()` shape used
    by lmcache.v1.multiprocess helpers (cupy-compatible)."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self.ptr = stream.npu_stream

    def synchronize(self) -> None:
        self._stream.synchronize()


def _make_stream() -> _NpuStreamHandle:
    return _NpuStreamHandle(torch.npu.Stream())


def _drain():
    return np_ops.drain_recorded_events()


@pytest.fixture(autouse=True)
def _clear_event_buffer():
    _drain()
    yield
    _drain()


@pytest.fixture()
def stream():
    s = _make_stream()
    yield s
    s.synchronize()


def test_record_on_stream_then_drain(stream):
    """Records on one stream become visible after that stream reaches the host
    callback. Drain returns every payload once, in order, with str/int
    metadata and monotonic timestamps; a second drain is empty."""
    n = 5
    for i in range(n):
        np_ops.record_event_on_stream(
            stream.ptr,
            f"mp.test.{i}",
            f"sess-{i}",
            {"device": "npu:0"},
            {"idx": i},
        )
    stream.synchronize()

    events = _drain()
    assert len(events) == n
    timestamps = []
    for i, (name, sid, ts, str_meta, int_meta) in enumerate(events):
        assert name == f"mp.test.{i}"
        assert sid == f"sess-{i}"
        assert ts > 0.0
        assert str_meta == {"device": "npu:0"}
        assert int_meta == {"idx": i}
        timestamps.append(ts)
    assert timestamps == sorted(timestamps)
    assert _drain() == []


def test_two_streams_both_drained():
    """Host callbacks from two streams land in the same buffer; each stream's
    records stay ordered."""
    s1 = _make_stream()
    s2 = _make_stream()
    n = 5
    for i in range(n):
        np_ops.record_event_on_stream(s1.ptr, "mp.test", "sess-1", {}, {"i": i})
        np_ops.record_event_on_stream(s2.ptr, "mp.test", "sess-2", {}, {"i": i})
    s1.synchronize()
    s2.synchronize()

    events = _drain()
    assert len(events) == 2 * n
    per_session = {"sess-1": [], "sess-2": []}
    for _, sid, _, _, int_meta in events:
        per_session[sid].append(int_meta["i"])
    assert per_session["sess-1"] == list(range(n))
    assert per_session["sess-2"] == list(range(n))


def test_null_stream_ptr_enqueues_immediately():
    """stream_ptr==0 skips aclrtLaunchHostFunc and runs the callback inline."""
    np_ops.record_event_on_stream(
        0, "mp.store.start", "sess-null", {"k": "v"}, {"n": 1}
    )
    events = _drain()
    assert len(events) == 1
    name, sid, ts, str_meta, int_meta = events[0]
    assert name == "mp.store.start"
    assert sid == "sess-null"
    assert ts > 0.0
    assert str_meta == {"k": "v"}
    assert int_meta == {"n": 1}


def test_concurrent_record_and_drain(stream):
    """Drain while the producer is still recording: every event is observed
    exactly once (mutex-protected buffer, GIL released inside record)."""
    total = 500
    drained: list[int] = []

    def drainer():
        while len(drained) < total:
            events = _drain()
            if events:
                drained.extend(e[4]["i"] for e in events)
            else:
                time.sleep(0.001)

    t = threading.Thread(target=drainer)
    t.start()
    try:
        for i in range(total):
            np_ops.record_event_on_stream(
                stream.ptr, "mp.concurrent", "s", {}, {"i": i}
            )
        stream.synchronize()
    finally:
        t.join(timeout=10.0)
    assert not t.is_alive()
    assert sorted(drained) == list(range(total))
