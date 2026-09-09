# SPDX-License-Identifier: Apache-2.0
"""Core regressions for sparse common-R selection and ownership."""

# Standard
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock
import json

# Third Party
from lmcache.utils import CacheEngineKey
import pytest
import torch

# First Party
from lmcache_ascend.v1 import state_lookup as lookup
from lmcache_ascend.v1.state_checkpoint import CheckpointRef, state_checkpoint_key


def client_for(sets):
    calls = []
    config = SimpleNamespace(
        enable_blending=False,
        enable_async_loading=False,
        use_layerwise=False,
        hit_miss_ratio=None,
        enable_chunk_statistics=False,
        get_lookup_server_worker_ids=lambda *args: [],
    )

    def exchange(frames):
        operation = json.loads(frames[-1])[lookup.OP_KEY]
        calls.append(operation)
        if operation["op"] == "release":
            return [bytes(4)] * len(sets)
        return [
            max(
                (r for r in available if operation["C"] < r <= operation["upper"]),
                default=0,
            ).to_bytes(4, "big")
            for available in sets
        ]

    client = SimpleNamespace(
        config=config,
        transport=SimpleNamespace(world_size=len(sets), send_and_recv_all=exchange),
        token_database=SimpleNamespace(process_tokens=lambda *a, **kw: [(0, 3072, 1)]),
        reqs_status={"same": 3072},
        clear_lookup_status=Mock(),
    )
    return client, calls


@pytest.mark.parametrize(
    "sets, expected",
    [
        ([{1024, 3072}, {1024, 2048}], 1024),
        ([{1024}, {2048}], 0),
        ([{1024}, set()], 0),
        ([{1024}], 1024),
    ],
)
def test_sparse_intersection(sets, expected):
    client, calls = client_for(sets)
    assert (
        lookup.lookup_state(client, range(3072), "same", 0, 3072, len(sets)) == expected
    )
    if expected and len(sets) == 2:
        assert [x["upper"] for x in calls] == [3072, 2048, 1024]
    if not expected:
        assert calls[-1] == {"op": "release"}


def test_same_request_new_upper_cancel_and_preemption():
    client, calls = client_for([{1024, 3072}, {1024, 2048}])
    args = (client, range(3072), "same")
    assert lookup.lookup_state(*args, 0, 3072, 2) == 1024
    client.clear_lookup_status("same")  # Allocation clears scalar status only.
    assert calls[-1]["op"] == "probe"
    lookup.cancel_state_lookup(client, "same")
    assert calls[-1]["op"] == "release"
    assert lookup.lookup_state(*args, 1024, 2048, 2) == 0
    assert lookup.lookup_state(*args, 0, 1024, 2) == 1024
    assert client.reqs_status["same"] == 3072  # Stale integer never read or replaced.


@pytest.mark.parametrize("reply", [[], [bytes(4)], [b"x", bytes(4)]])
def test_missing_or_malformed_rank_releases(reply):
    client, _ = client_for([{1024}, {1024}])
    client.transport.send_and_recv_all = Mock(return_value=reply)
    assert lookup.lookup_state(client, range(3072), "same", 0, 3072, 2) == 0
    assert client.transport.send_and_recv_all.call_count == 2


def test_timeout_releases_then_retry():
    client, _ = client_for([{1024}, {1024}])
    original = client.transport.send_and_recv_all
    client.transport.send_and_recv_all = Mock(
        side_effect=[TimeoutError(), [bytes(4)] * 2]
    )
    with pytest.raises(TimeoutError):
        lookup.lookup_state(client, range(3072), "same", 0, 3072, 2)
    assert client.transport.send_and_recv_all.call_count == 2
    client.transport.send_and_recv_all = original
    assert lookup.lookup_state(client, range(3072), "same", 0, 3072, 2) == 1024


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_async_loading", True),
        ("use_layerwise", True),
        ("hit_miss_ratio", 0.5),
        ("enable_chunk_statistics", True),
    ],
)
def test_unsupported_options(field, value):
    client, _ = client_for([{1024}, {1024}])
    setattr(client.config, field, value)
    with pytest.raises(ValueError):
        lookup.lookup_state(client, range(3072), "same", 0, 3072, 2)


def test_subset_and_reserved_collision():
    client, _ = client_for([{1024}])
    with pytest.raises(ValueError):
        lookup.lookup_state(client, range(3072), "same", 0, 3072, 2)
    with pytest.raises(ValueError, match="reserved"):
        lookup.lookup_state(
            client, range(3072), "same", 0, 3072, 1, {lookup.OP_KEY: "user"}
        )


class Backend:
    def __init__(self, objects):
        self.objects = objects

    def get_blocking(self, key):
        obj = self.objects.get(key)
        if obj:
            obj.refs += 1
        return obj


class Buffer:
    def __init__(self, obj):
        self.obj = obj

    def close(self):
        self.obj.refs -= 1


def local_fixture(monkeypatch, states, attention):
    chunks = [
        (
            r - 1024,
            r,
            CacheEngineKey(
                "model", 2, 0, r, torch.bfloat16, {"lmcache.tag.tenant": "a"}
            ),
        )
        for r in (1024, 2048, 3072)
    ]
    objects = {}
    for boundary, group in states:
        prefix = chunks[boundary // 1024 - 1][2]
        key = state_checkpoint_key(
            CheckpointRef.from_chunk(
                prefix,
                chunk_end=boundary,
                boundary=boundary,
                chunk_size=1024,
                group_index=group,
            )
        )
        objects[key] = SimpleNamespace(refs=1, compatible=True)
    pins = []

    def contains(key, locations, pin):
        if key.chunk_hash in attention:
            pins.append(key)
            return "LocalCPUBackend"
        return None

    def unpin(keys, locations):
        for key in keys:
            pins.remove(key)

    def adopt(layout, obj):
        if not obj.compatible:
            obj.refs -= 1
            raise ValueError("incompatible")
        return Buffer(obj)

    monkeypatch.setattr(lookup, "adopt_state_checkpoint", adopt)
    manager = SimpleNamespace(
        storage_backends={"LocalCPUBackend": Backend(objects)},
        contains=contains,
        batched_unpin=unpin,
    )
    return manager, chunks, objects, pins


def test_local_requires_every_group_and_attention_above_c(monkeypatch):
    manager, chunks, objects, pins = local_fixture(
        monkeypatch, {(3072, 1), (2048, 1), (2048, 3)}, {2048}
    )
    layouts = [SimpleNamespace(group_index=g) for g in (1, 3)]
    selection = lookup.local_candidate(manager, chunks, layouts, 1024, 1024, 3072, None)
    assert selection.boundary == 2048
    assert set(selection.buffers) == {1, 3}
    assert [key.chunk_hash for key in pins] == [2048]
    selection.close(manager)
    assert not pins
    assert all(obj.refs == 1 for obj in objects.values())
    assert lookup.local_candidate(manager, chunks, layouts, 1024, 0, 3072, None) is None
    assert not pins
    assert all(obj.refs == 1 for obj in objects.values())


def test_incompatible_owned_get_is_released(monkeypatch):
    manager, chunks, objects, pins = local_fixture(monkeypatch, {(1024, 1)}, {1024})
    next(iter(objects.values())).compatible = False
    assert (
        lookup.local_candidate(
            manager, chunks, [SimpleNamespace(group_index=1)], 1024, 0, 3072, None
        )
        is None
    )
    assert all(obj.refs == 1 for obj in objects.values())
    assert not pins


def test_probe_replacement_release_and_lease(monkeypatch):
    manager, chunks, objects, pins = local_fixture(
        monkeypatch, {(1024, 1), (3072, 1)}, {1024, 2048, 3072}
    )
    timers = []

    class Timer:
        def __init__(self, seconds, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(lookup.threading, "Timer", Timer)
    seen = []

    def process(**kwargs):
        seen.append(kwargs["request_configs"])
        return chunks

    engine = SimpleNamespace(
        storage_manager=manager,
        token_database=SimpleNamespace(process_tokens=process),
        state_layouts=[SimpleNamespace(group_index=1)],
        retrieve_locations=None,
        config=SimpleNamespace(
            chunk_size=1024,
            enable_blending=False,
            enable_async_loading=False,
            pin_timeout_sec=300,
        ),
        use_layerwise=False,
        is_healthy=lambda: True,
        lookup_pins={},
        _engine_state_lock=Lock(),
    )
    configs = {"lmcache.tag.tenant": "a", "request_option": 9}
    assert (
        lookup.dispatch_state_lookup(
            engine, "req", {"op": "probe", "C": 0, "upper": 3072}, configs
        )
        == 3072
    )
    assert len(pins) == 3
    assert (
        lookup.dispatch_state_lookup(
            engine, "req", {"op": "probe", "C": 0, "upper": 1024}, configs
        )
        == 1024
    )
    assert len(pins) == 1
    timers[0].callback()  # A replaced timer cannot release the current selection.
    assert len(pins) == 1
    assert seen == [configs, configs]
    timers[-1].callback()
    assert not pins and not engine.lookup_pins
    assert all(obj.refs == 1 for obj in objects.values())
    assert (
        lookup.dispatch_state_lookup(
            engine, "req", {"op": "probe", "C": 0, "upper": 1024}, configs
        )
        == 1024
    )
    lookup.dispatch_state_lookup(engine, "req", {"op": "release"}, {})
    assert not pins
    assert all(obj.refs == 1 for obj in objects.values())


def test_attention_probe_exception_releases_partial_pins_and_state(monkeypatch):
    manager, chunks, objects, pins = local_fixture(
        monkeypatch, {(3072, 1)}, {1024, 2048, 3072}
    )
    contains = manager.contains

    def fail(key, locations, pin):
        if key.chunk_hash == 2048:
            raise RuntimeError("backend failure")
        return contains(key, locations, pin)

    manager.contains = fail
    with pytest.raises(RuntimeError, match="backend failure"):
        lookup.local_candidate(
            manager, chunks, [SimpleNamespace(group_index=1)], 1024, 0, 3072, None
        )
    assert not pins
    assert all(obj.refs == 1 for obj in objects.values())


def test_keyboard_cancellation_releases_remote_probe():
    client, _ = client_for([{1024}, {1024}])
    client.transport.send_and_recv_all = Mock(
        side_effect=[KeyboardInterrupt(), [bytes(4)] * 2]
    )
    with pytest.raises(KeyboardInterrupt):
        lookup.lookup_state(client, range(3072), "same", 0, 3072, 2)
    assert client.transport.send_and_recv_all.call_count == 2


def test_user_configs_preserved_in_wire():
    client, _ = client_for([{1024}])
    exchange = client.transport.send_and_recv_all
    client.transport.send_and_recv_all = Mock(side_effect=exchange)
    configs = {"lmcache.tag.tenant": "a", "other": 3}
    assert lookup.lookup_state(client, range(3072), "same", 0, 3072, 1, configs) == 1024
    wire = json.loads(client.transport.send_and_recv_all.call_args.args[0][-1])
    wire.pop(lookup.OP_KEY)
    assert wire == configs
    assert lookup.OP_KEY not in configs


def test_cleanup_attempts_every_pin_location_and_buffer():
    error = RuntimeError("unpin failure")
    manager = SimpleNamespace(batched_unpin=Mock(side_effect=[error, None]))
    first, second = Mock(), Mock()
    selection = lookup.StateLookupSelection(
        1024, {1: first, 3: second}, {"LocalCPUBackend": [1], "LocalDiskBackend": [2]}
    )
    with pytest.raises(RuntimeError, match="unpin failure"):
        selection.close(manager)
    assert manager.batched_unpin.call_count == 2
    first.close.assert_called_once()
    second.close.assert_called_once()
    selection.close(manager)
    first.close.assert_called_once()
