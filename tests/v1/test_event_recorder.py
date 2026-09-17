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


@pytest.fixture()
def stream():
    s = _make_stream()
    yield s
    s.synchronize()


class TestRecordAndDrain:
    """Low-level tests for c_ops.record_event_on_stream / drain."""

    def test_drain_empty(self):
        events = np_ops.drain_recorded_events()
        assert events == []

    def test_single_event(self, stream):
        np_ops.record_event_on_stream(
            stream.ptr,
            "mp.store.start",
            "sess-1",
            {"device": "npu:0"},
            {},
        )
        stream.synchronize()

        events = np_ops.drain_recorded_events()
        assert len(events) == 1
        name, sid, ts, str_meta, int_meta = events[0]
        assert name == "mp.store.start"
        assert sid == "sess-1"
        assert ts > 0.0
        assert str_meta == {"device": "npu:0"}
        assert int_meta == {}

    def test_int_metadata_preserved(self, stream):
        np_ops.record_event_on_stream(
            stream.ptr,
            "mp.store.end",
            "sess-2",
            {"device": "npu:0"},
            {"stored_count": 42},
        )
        stream.synchronize()

        events = np_ops.drain_recorded_events()
        assert len(events) == 1
        _, _, _, str_meta, int_meta = events[0]
        assert str_meta == {"device": "npu:0"}
        assert int_meta == {"stored_count": 42}

    def test_multiple_events_ordered(self, stream):
        for i in range(5):
            np_ops.record_event_on_stream(
                stream.ptr,
                f"mp.test.{i}",
                f"sess-{i}",
                {},
                {"idx": i},
            )
        stream.synchronize()

        events = np_ops.drain_recorded_events()
        assert len(events) == 5
        for i, (name, sid, ts, _, int_meta) in enumerate(events):
            assert name == f"mp.test.{i}"
            assert sid == f"sess-{i}"
            assert int_meta["idx"] == i
            assert ts > 0.0

    def test_drain_clears_buffer(self, stream):
        np_ops.record_event_on_stream(stream.ptr, "mp.store.start", "s", {}, {})
        stream.synchronize()

        first = np_ops.drain_recorded_events()
        assert len(first) == 1

        second = np_ops.drain_recorded_events()
        assert second == []

    def test_timestamps_monotonic(self, stream):
        for _ in range(3):
            np_ops.record_event_on_stream(stream.ptr, "mp.store.start", "s", {}, {})
        stream.synchronize()

        events = np_ops.drain_recorded_events()
        timestamps = [e[2] for e in events]
        assert timestamps == sorted(timestamps)

    def test_null_stream_ptr_enqueues_immediately(self):
        np_ops.record_event_on_stream(
            0, "mp.store.start", "sess-null", {"k": "v"}, {"n": 1}
        )

        events = np_ops.drain_recorded_events()
        assert len(events) == 1
        name, sid, ts, str_meta, int_meta = events[0]
        assert name == "mp.store.start"
        assert sid == "sess-null"
        assert ts > 0.0
        assert str_meta == {"k": "v"}
        assert int_meta == {"n": 1}

    def test_two_streams_both_drained(self):
        s1 = _make_stream()
        s2 = _make_stream()
        for i in range(5):
            np_ops.record_event_on_stream(s1.ptr, "mp.test", "sess-1", {}, {"i": i})
            np_ops.record_event_on_stream(s2.ptr, "mp.test", "sess-2", {}, {"i": i})
        s1.synchronize()
        s2.synchronize()

        events = np_ops.drain_recorded_events()
        assert len(events) == 10
        per_session = {"sess-1": [], "sess-2": []}
        for _, sid, _, _, int_meta in events:
            per_session[sid].append(int_meta["i"])
        assert per_session["sess-1"] == [0, 1, 2, 3, 4]
        assert per_session["sess-2"] == [0, 1, 2, 3, 4]

    def test_concurrent_record_and_drain(self, stream):
        """A drainer thread must observe every event exactly once while the
        main thread keeps recording (mutex-protected buffer, GIL released
        inside record)."""
        total = 500
        drained: list[int] = []

        def drainer():
            while len(drained) < total:
                events = np_ops.drain_recorded_events()
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
