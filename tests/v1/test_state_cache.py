# SPDX-License-Identifier: Apache-2.0
"""Independent state storage identity, managed allocation and ownership contracts."""

# Standard
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import asyncio
import json
import os

# Third Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import (
    LocalDiskBackend,
    LocalDiskWorker,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.memory_management import (
    MemoryFormat,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryAllocator,
)
import pytest
import torch

# First Party
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import StateExecution
from lmcache_ascend.v1.state_cache import StateCache
from lmcache_ascend.v1.state_checkpoint import CheckpointRef, state_checkpoint_key
from lmcache_ascend.v1.state_layout import build_state_group_layout
from lmcache_ascend.v1.state_memory import (
    adopt_state_checkpoint,
    allocate_state_checkpoint,
    state_checkpoint_metadata,
)
from lmcache_ascend.v1.storage_backend.storage_manager import state_store_locations


def layout():
    return build_state_group_layout(
        1, ["gdn.0"], [(torch.empty(2, 3, dtype=torch.bfloat16), torch.empty(2, 2))]
    )


def prefix():
    return CacheEngineKey(
        "model",
        4,
        2,
        123,
        torch.bfloat16,
        {
            "lmcache.tag.tenant": "tenant-a",
            "request_option": "keep",
        },
    )


def test_state_key_namespace_and_round_trip():
    key = prefix()
    configs = dict(key.request_configs)
    ref = CheckpointRef.from_chunk(
        key, chunk_end=32, boundary=32, chunk_size=16, group_index=1
    )
    state = state_checkpoint_key(ref)
    assert state != key
    assert state != state_checkpoint_key(replace(ref, group_index=2))
    assert state != state_checkpoint_key(replace(ref, boundary=64))
    assert CacheEngineKey.from_string(state.to_string()) == state
    assert CacheEngineKey.from_dict(state.to_dict()) == state
    assert state.tags[:1] == key.tags
    assert state.request_configs["request_option"] == "keep"
    assert (state.model_name, state.world_size, state.worker_id, state.chunk_hash) == (
        key.model_name,
        key.world_size,
        key.worker_id,
        key.chunk_hash,
    )
    assert key.request_configs == configs
    assert state.request_configs is not key.request_configs


@pytest.mark.parametrize("tag", ["state.domain", "state.group", "state.boundary"])
def test_reserved_state_tags_cannot_be_overwritten(tag):
    key = prefix()
    key.request_configs["lmcache.tag." + tag] = "user"
    with pytest.raises(ValueError, match="reserved"):
        state_checkpoint_key(CheckpointRef(key, 32, 1))


@pytest.fixture
def mixed():
    # Exercise the installed production routes with a real tensor pool; avoid
    # platform-specific host registration in MixedMemoryAllocator.__init__.
    allocator = object.__new__(MixedMemoryAllocator)
    allocator._unregistered = True
    allocator.host_mem_lock = Lock()
    allocator.pin_allocator = TensorMemoryAllocator(
        torch.zeros(16384, dtype=torch.uint8)
    )
    yield allocator
    assert allocator.pin_allocator.num_active_allocations == 0


def test_managed_binary_padding_and_adopt_owned_reference(mixed):
    expected = layout()
    owner = allocate_state_checkpoint(expected, mixed)
    obj = owner.memory_obj
    assert obj.meta.shapes == [torch.Size([1, 3]), torch.Size([2]), torch.Size([1, 2])]
    assert obj.meta.dtypes == [torch.bfloat16, torch.uint8, torch.float32]
    owner.planes[1].fill_(7)
    obj.ref_count_up()  # CPU backend get supplies one owned reference.
    loaded = adopt_state_checkpoint(expected, obj)
    assert loaded.memory_obj is obj
    assert obj.meta.ref_count == 2
    assert loaded.planes[1].data_ptr() == owner.planes[1].data_ptr()
    owner.close()
    assert torch.all(loaded.planes[1] == 7)
    loaded.close()
    loaded.close()


@pytest.mark.parametrize(
    "damage", ["group", "signature", "shapes", "dtype", "format", "missing"]
)
def test_adopt_rejects_incompatible_metadata_and_releases(mixed, damage):
    expected = layout()
    owner = allocate_state_checkpoint(expected, mixed)
    obj = owner.memory_obj
    obj.ref_count_up()
    if damage == "group":
        obj.meta.state_checkpoint["group_index"] = 3
    elif damage == "signature":
        obj.meta.state_checkpoint["layout"][0] = 99
    elif damage == "shapes":
        obj.meta.shapes = [torch.Size([expected.nbytes])]
    elif damage == "dtype":
        obj.meta.dtypes = [torch.uint8]
    elif damage == "format":
        obj.meta.fmt = MemoryFormat.KV_2LTD
    else:
        del obj.meta.state_checkpoint
    with pytest.raises(ValueError, match="metadata"):
        adopt_state_checkpoint(expected, obj)
    assert obj.meta.ref_count == 1
    owner.close()


def test_metadata_survives_portable_serialization(mixed):
    expected = layout()
    with allocate_state_checkpoint(expected, mixed) as owner:
        obj = owner.memory_obj
        obj.meta.state_checkpoint = json.loads(
            json.dumps(state_checkpoint_metadata(expected))
        )
        obj.ref_count_up()
        with adopt_state_checkpoint(expected, obj):
            assert obj.meta.ref_count == 2


def test_binary_batched_routes_and_existing_attention_route(mixed):
    objects = mixed.batched_allocate(
        [torch.Size([16])], [torch.uint8], 2, MemoryFormat.BINARY
    )
    assert objects is not None
    mixed.batched_free(objects)
    obj = mixed.allocate(
        torch.Size([2, 1, 16, 4]), torch.bfloat16, MemoryFormat.KV_2LTD
    )
    assert obj is not None
    mixed.free(obj)


@pytest.mark.parametrize("batched", [False, True])
def test_oversized_paged_binary_rejected_before_pool_mutation(mixed, batched):
    mixed.pin_allocator = PagedTensorMemoryAllocator(
        torch.zeros(16, dtype=torch.uint8), [torch.Size([8])], [torch.uint8]
    )
    with pytest.raises(ValueError, match="page size"):
        if batched:
            mixed.batched_allocate(
                [torch.Size([24])], [torch.uint8], 2, MemoryFormat.BINARY
            )
        else:
            allocate_state_checkpoint(layout(), mixed)
    assert len(mixed.pin_allocator.free_blocks) == 2


@pytest.mark.parametrize(
    "change", ["group_index", "layer_names", "alignment", "version"]
)
def test_adopt_checks_expected_runtime_layout(mixed, change):
    expected = layout()
    values = {"group_index": 2, "layer_names": ("other",), "alignment": 8, "version": 2}
    with allocate_state_checkpoint(expected, mixed) as owner:
        owner.memory_obj.ref_count_up()
        with pytest.raises(ValueError):
            adopt_state_checkpoint(
                replace(expected, **{change: values[change]}), owner.memory_obj
            )
        assert owner.memory_obj.meta.ref_count == 1


def test_undersized_loaded_payload_releases_owned_reference(mixed):
    with allocate_state_checkpoint(layout(), mixed) as owner:
        obj = owner.memory_obj
        raw = obj.raw_data
        obj.raw_data = raw[:4]
        obj.ref_count_up()
        with pytest.raises(ValueError, match="undersized"):
            adopt_state_checkpoint(layout(), obj)
        assert obj.meta.ref_count == 1
        obj.raw_data = raw


def test_binary_routes_hold_existing_host_lock(mixed, monkeypatch):
    for name in ("allocate", "batched_allocate", "free", "batched_free"):
        original = getattr(mixed.pin_allocator, name)

        def checked(*args, _original=original, **kwargs):
            assert mixed.host_mem_lock.locked()
            return _original(*args, **kwargs)

        monkeypatch.setattr(mixed.pin_allocator, name, checked)
    obj = mixed.allocate([torch.Size([16])], [torch.uint8], MemoryFormat.BINARY)
    mixed.free(obj)
    objects = mixed.batched_allocate(
        [torch.Size([16])], [torch.uint8], 2, MemoryFormat.BINARY
    )
    mixed.batched_free(objects)


@pytest.mark.parametrize("failure", [None, "allocate", "copy", "put", "admitted"])
def test_sync_save_publication_and_manager_reference_handoff(
    mixed, monkeypatch, failure
):
    events, objects, published = [], [], {}

    class CPUBackend:
        use_hot = True

        def contains(self, key):
            return key in published

        def get_allocator_backend(self):
            return self

        def allocate(self, shapes, dtypes, *, fmt, busy_loop):
            assert busy_loop is False
            if failure == "allocate":
                return None
            obj = mixed.allocate(shapes, dtypes, fmt)
            objects.append(obj)
            return obj

        def batched_submit_put_task(self, keys, objs, transfer_spec=None):
            assert events == ["wait", "copy"]
            if failure == "put":
                raise RuntimeError("put")
            for key, obj in zip(keys, objs, strict=True):
                obj.ref_count_up()
                published[key] = obj
            if failure == "admitted":
                raise RuntimeError("admitted")

    cpu = CPUBackend()
    manager = SimpleNamespace(
        allocator_backend=cpu,
        storage_backends={"LocalCPUBackend": cpu},
        _bypass_lock=Lock(),
        _bypassed_backends=set(),
    )
    manager.batched_put = lambda *a, **kw: StorageManager.batched_put(manager, *a, **kw)
    execution = StateExecution(
        "r",
        tuple(range(32)),
        16,
        32,
        32,
        ((), (1, 1)),
        ((1, 16),),
        False,
        True,
        {"lmcache.tag.tenant": "tenant-a"},
    )

    def process_tokens(tokens, request_configs):
        assert tokens == list(execution.token_ids)
        assert request_configs == execution.request_configs
        yield 16, 32, prefix()

    def copy(operation):
        assert not published
        events.append("copy")
        assert operation.runtime.block_id == 1
        assert operation.checkpoint.boundary == 32
        if failure == "copy":
            raise RuntimeError("copy")
        operation.buffer.planes[1].fill_(9)

    monkeypatch.setattr("lmcache_ascend.v1.state_cache.transfer_state", copy)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    stream = SimpleNamespace(wait_event=lambda event: events.append("wait"))
    cache = StateCache(manager, SimpleNamespace(process_tokens=process_tokens), 16)
    runtime = {"gdn.0": (torch.empty(2, 3, dtype=torch.bfloat16), torch.empty(2, 2))}
    if failure in (None, "allocate"):
        cache.save(execution, (layout(),), runtime, stream, object())
    else:
        with pytest.raises((MemoryError, RuntimeError)):
            cache.save(execution, (layout(),), runtime, stream, object())
    if failure in (None, "admitted"):
        assert len(published) == 1
        obj = next(iter(published.values()))
        assert obj.meta.ref_count == 1
        obj.ref_count_up()  # Simulate the owned CPU get contract.
        with adopt_state_checkpoint(layout(), obj) as loaded:
            assert torch.all(loaded.planes[1] == 9)
        obj.ref_count_down()  # Evict the backend reference.
    else:
        assert not published
        assert all(obj.meta.ref_count == 0 for obj in objects)


@pytest.fixture
def disk_tiers(mixed, tmp_path, monkeypatch):
    """Real CPU/disk methods and managed memory; deterministic queued disk I/O."""
    stats = SimpleNamespace(
        update_local_storage_usage=lambda _: None,
        update_local_cpu_evict_failed_count=lambda _: None,
        update_local_cpu_evict_metrics=lambda _: None,
    )
    cpu = object.__new__(LocalCPUBackend)
    cpu.memory_allocator = mixed
    cpu.use_hot = True
    cpu.cpu_lock = Lock()
    cpu.cache_policy = get_cache_policy("LRU")
    cpu.hot_cache = cpu.cache_policy.init_mutable_mapping()
    cpu.batched_msg_sender = None
    cpu.stats_monitor = stats
    cpu.keys_in_request = []
    disk = object.__new__(LocalDiskBackend)
    disk.local_cpu_backend = cpu
    disk.disk_lock = Lock()
    disk.path = str(tmp_path)
    disk.max_cache_size = 16384
    disk.current_cache_size = 0
    disk.usage = 0
    disk.use_odirect = False
    disk.os_disk_bs = 4096
    disk.batched_msg_sender = None
    disk.stats_monitor = stats
    disk.cache_policy = get_cache_policy("LRU")
    disk.dict = disk.cache_policy.init_mutable_mapping()
    disk.keys_in_request = []
    disk.loop = object()
    worker = object.__new__(LocalDiskWorker)
    worker.put_lock = Lock()
    worker.put_tasks = []

    async def submit_task(kind, fn, *args, **kwargs):
        assert kind == "put"
        return fn(*args, **kwargs)

    worker.submit_task = submit_task
    disk.disk_worker = worker
    pending = []

    def schedule(coro, loop):
        assert loop is disk.loop
        pending.append(coro)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", schedule)
    manager = SimpleNamespace(
        allocator_backend=cpu,
        storage_backends={"LocalCPUBackend": cpu, "LocalDiskBackend": disk},
        _bypass_lock=Lock(),
        _bypassed_backends=set(),
    )
    manager.batched_put = lambda *a, **kw: StorageManager.batched_put(manager, *a, **kw)
    yield cpu, disk, manager, pending
    for coro in pending:
        coro.close()
    for key in list(cpu.hot_cache):
        cpu.remove(key)


def checkpoint_key():
    return state_checkpoint_key(CheckpointRef(prefix(), 32, 1))


def complete_disk_write(pending):
    assert len(pending) == 1
    asyncio.run(pending.pop())


def test_disk_round_trip_after_actual_cpu_eviction(disk_tiers):
    cpu, disk, manager, pending = disk_tiers
    key = checkpoint_key()
    with allocate_state_checkpoint(layout(), cpu, busy_loop=False) as owner:
        original = owner.memory_obj
        original.tensor.fill_(
            0xA5
        )  # Include both bytes of composite alignment padding.
        owner.planes[0].fill_(3)
        owner.planes[1].fill_(7)
        expected = bytes(original.byte_array)
        shapes, dtypes = list(original.meta.shapes), list(original.meta.dtypes)
        original.ref_count_up()
        manager.batched_put([key], [original])
        assert cpu.contains(key)
        assert not disk.contains(key)
        assert original.meta.ref_count == 3  # Owner, CPU and queued disk writer.
        complete_disk_write(pending)
        assert original.meta.ref_count == 2
        assert disk.current_cache_size == len(expected)
        assert disk.usage == len(expected)
        assert disk.dict[key].size == len(expected)
    assert cpu.remove(key)
    assert original.meta.ref_count == 0
    assert cpu.get_blocking(key) is None
    assert disk.dict[key].shapes == shapes
    assert disk.dict[key].dtypes == dtypes
    with adopt_state_checkpoint(layout(), disk.get_blocking(key)) as loaded:
        assert loaded.memory_obj is not original
        assert bytes(loaded.memory_obj.byte_array) == expected
        assert bytes(loaded.memory_obj.byte_array)[6:8] == b"\xa5\xa5"
        assert torch.all(loaded.planes[0] == 3)
        assert torch.all(loaded.planes[1] == 7)
        assert loaded.planes[0].dtype == torch.bfloat16
        assert loaded.planes[1].dtype == torch.float32
        assert loaded.memory_obj.meta.state_checkpoint == state_checkpoint_metadata(
            layout()
        )
        assert (
            loaded.memory_obj.meta.state_checkpoint
            is not disk.dict[key].state_checkpoint
        )
        assert disk.dict[key].pin_count == 0


def test_disk_callback_sees_complete_metadata_after_owner_release(disk_tiers):
    cpu, disk, _, pending = disk_tiers
    key = checkpoint_key()
    callbacks = []
    owner = allocate_state_checkpoint(layout(), cpu, busy_loop=False)
    obj = owner.memory_obj

    def completed(actual):
        assert actual == key
        assert obj.meta.ref_count == 0
        assert not disk.exists_in_put_tasks(key)
        with adopt_state_checkpoint(layout(), disk.get_blocking(key)) as loaded:
            assert loaded.layout == layout()
        callbacks.append(actual)
        raise RuntimeError("callback bookkeeping")

    disk.submit_put_task(key, obj, completed)
    owner.close()
    complete_disk_write(pending)
    assert callbacks == [key]
    assert disk.contains(key)


@pytest.mark.parametrize(
    "failure", ["write", "short_write", "schedule", "executor", "capacity"]
)
def test_failed_disk_write_releases_reservation_pending_and_ref(
    disk_tiers, monkeypatch, failure
):
    cpu, disk, _, pending = disk_tiers
    key = checkpoint_key()
    owner = allocate_state_checkpoint(layout(), cpu, busy_loop=False)
    obj = owner.memory_obj
    completed = []

    def fail(*args, **kwargs):
        raise OSError(failure)

    if failure == "write":
        monkeypatch.setattr(disk, "write_file", fail)
    elif failure == "short_write":

        def short_write(buffer, path):
            with open(path, "wb") as file:
                file.write(buffer[:1])

        monkeypatch.setattr(disk, "write_file", short_write)
    elif failure == "schedule":
        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fail)
    elif failure == "executor":

        async def failed_submit(*args, **kwargs):
            raise OSError("executor")

        monkeypatch.setattr(disk.disk_worker, "submit_task", failed_submit)
    else:
        disk.max_cache_size = 0
    if failure == "schedule":
        with pytest.raises(OSError):
            disk.submit_put_task(key, obj, completed.append)
    else:
        disk.submit_put_task(key, obj, completed.append)
        if failure != "capacity":
            owner.close()  # Writer is now the sole remaining owner.
            with pytest.raises(OSError):
                complete_disk_write(pending)
    owner.close()
    assert obj.meta.ref_count == 0
    assert not disk.contains(key)
    assert not disk.exists_in_put_tasks(key)
    assert disk.current_cache_size == 0
    assert disk.usage == 0
    assert not completed
    assert not list(Path(disk.path).iterdir())


@pytest.mark.parametrize(
    "failure", ["missing", "truncated", "read", "allocate", "metadata"]
)
def test_disk_read_failure_releases_allocation_and_temporary_pin(
    disk_tiers, mixed, monkeypatch, failure
):
    cpu, disk, _, pending = disk_tiers
    key = checkpoint_key()
    with allocate_state_checkpoint(layout(), cpu, busy_loop=False) as owner:
        disk.submit_put_task(key, owner.memory_obj)
        complete_disk_write(pending)
    disk_meta = disk.dict[key]
    if failure == "missing":
        os.remove(disk_meta.path)
    elif failure == "truncated":
        with open(disk_meta.path, "wb") as file:
            file.write(b"x")
    elif failure == "metadata":
        del disk_meta.state_checkpoint
    elif failure == "read":

        def fail_read(*args):
            assert disk_meta.pin_count == 1
            assert not disk.cache_policy.get_evict_candidates(disk.dict)
            raise OSError("read")

        monkeypatch.setattr(
            "lmcache_ascend.v1.storage_backend.local_disk_backend._read_bytes_exact",
            fail_read,
        )
    else:

        def no_memory(*args, **kwargs):
            assert kwargs["busy_loop"] is False
            assert disk_meta.pin_count == 1
            return None

        monkeypatch.setattr(cpu, "allocate", no_memory)
    if failure in ("allocate", "metadata"):
        assert disk.get_blocking(key) is None
    else:
        with pytest.raises(OSError):
            disk.get_blocking(key)
    assert disk_meta.pin_count == 0
    assert mixed.pin_allocator.num_active_allocations == 0


def test_disk_duplicate_preserves_valid_entry(disk_tiers):
    cpu, disk, _, pending = disk_tiers
    key = checkpoint_key()
    with allocate_state_checkpoint(layout(), cpu, busy_loop=False) as owner:
        owner.planes[1].fill_(4)
        disk.submit_put_task(key, owner.memory_obj)
        disk.submit_put_task(key, owner.memory_obj)
        assert len(pending) == 1
        complete_disk_write(pending)
        reserved = disk.current_cache_size
        disk.submit_put_task(key, owner.memory_obj)
        assert not pending
        assert disk.current_cache_size == reserved
        assert owner.memory_obj.meta.ref_count == 1
    with adopt_state_checkpoint(layout(), disk.get_blocking(key)) as loaded:
        assert torch.all(loaded.planes[1] == 4)


def test_disk_capacity_rejection_can_retry_after_unpin(disk_tiers):
    cpu, disk, _, pending = disk_tiers
    first = checkpoint_key()
    second = state_checkpoint_key(CheckpointRef(prefix(), 64, 1))
    with allocate_state_checkpoint(layout(), cpu, busy_loop=False) as owner:
        size = owner.memory_obj.get_size()
        disk.max_cache_size = size
        disk.submit_put_task(first, owner.memory_obj)
        complete_disk_write(pending)
        assert disk.pin(first)
        disk.submit_put_task(second, owner.memory_obj)
        assert not pending
        assert not disk.exists_in_put_tasks(second)
        assert owner.memory_obj.meta.ref_count == 1
        assert disk.current_cache_size == size
        assert disk.unpin(first)
        disk.submit_put_task(second, owner.memory_obj)
        complete_disk_write(pending)
        assert not disk.contains(first)
        assert disk.contains(second)
        assert disk.current_cache_size == disk.usage == size


def test_state_eviction_missing_file_keeps_capacity_reusable(disk_tiers):
    cpu, disk, _, pending = disk_tiers
    first = checkpoint_key()
    second = state_checkpoint_key(CheckpointRef(prefix(), 64, 1))
    with allocate_state_checkpoint(layout(), cpu, busy_loop=False) as owner:
        size = owner.memory_obj.get_size()
        disk.max_cache_size = size
        disk.submit_put_task(first, owner.memory_obj)
        complete_disk_write(pending)
        os.remove(disk.dict[first].path)
        with pytest.raises(FileNotFoundError):
            disk.submit_put_task(second, owner.memory_obj)
        assert not disk.dict and not pending
        assert disk.current_cache_size == disk.usage == 0
        assert not disk.exists_in_put_tasks(second)
        assert owner.memory_obj.meta.ref_count == 1
        disk.submit_put_task(second, owner.memory_obj)
        complete_disk_write(pending)
        assert disk.contains(second)
        assert disk.current_cache_size == disk.usage == size


def test_state_tiers_support_disk_only_staging_and_explicit_location(disk_tiers):
    cpu, _, manager, _ = disk_tiers
    assert state_store_locations(manager) == ["LocalCPUBackend", "LocalDiskBackend"]
    assert state_store_locations(manager, "LocalDiskBackend") == ["LocalDiskBackend"]
    cpu.use_hot = False
    assert state_store_locations(manager) == ["LocalDiskBackend"]
    assert state_store_locations(manager, "LocalCPUBackend") == []
    with pytest.raises(ValueError, match="unavailable"):
        state_store_locations(manager, "RemoteBackend")


@pytest.mark.parametrize("mode", ["both", "cpu_hit", "disk_only", "explicit_disk"])
def test_completed_save_reaches_each_missing_selected_tier(
    disk_tiers, monkeypatch, mode
):
    cpu, disk, manager, pending = disk_tiers
    key = checkpoint_key()
    if mode == "cpu_hit":
        with allocate_state_checkpoint(layout(), cpu, busy_loop=False) as owner:
            cpu.submit_put_task(key, owner.memory_obj)
    if mode == "disk_only":
        cpu.use_hot = False
    execution = StateExecution(
        "r",
        tuple(range(32)),
        16,
        32,
        32,
        ((), (1, 1)),
        ((1, 16),),
        False,
        True,
        {},
    )
    token_db = SimpleNamespace(process_tokens=lambda *a, **kw: [(16, 32, prefix())])
    events = []

    def copy(operation):
        assert not disk.contains(key)
        assert not pending
        assert events == ["wait"]
        operation.buffer.planes[1].fill_(11)
        events.append("copy")

    monkeypatch.setattr("lmcache_ascend.v1.state_cache.transfer_state", copy)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    stream = SimpleNamespace(wait_event=lambda _: events.append("wait"))
    StateCache(manager, token_db, 16).save(
        execution,
        (layout(),),
        {"gdn.0": (torch.empty(2, 3, dtype=torch.bfloat16), torch.empty(2, 2))},
        stream,
        object(),
        location="LocalDiskBackend" if mode == "explicit_disk" else None,
    )
    assert events == ["wait", "copy"]
    assert cpu.contains(key) == (mode in ("both", "cpu_hit"))
    assert not disk.contains(key)
    complete_disk_write(pending)
    if cpu.contains(key):
        cpu.remove(key)
    with adopt_state_checkpoint(layout(), disk.get_blocking(key)) as loaded:
        assert torch.all(loaded.planes[1] == 11)


@pytest.mark.parametrize("failure", ["missing", "truncated", "allocate", None])
def test_attention_blocking_disk_read_is_exact_and_finite(
    disk_tiers, mixed, monkeypatch, failure
):
    cpu, disk, _, _ = disk_tiers
    key = prefix()
    obj = cpu.allocate(
        torch.Size([2, 1, 2, 4]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
        busy_loop=False,
    )
    obj.tensor.fill_(5)
    expected = bytes(obj.byte_array)
    # Exercise the ordinary writer with its existing submit-owned reference.
    disk.async_save_bytes_to_disk(key, obj)
    meta = disk.dict[key]
    if failure == "missing":
        os.remove(meta.path)
    elif failure == "truncated":
        Path(meta.path).write_bytes(b"x")
    elif failure == "allocate":

        def no_memory(*args, **kwargs):
            assert kwargs["busy_loop"] is False
            return None

        monkeypatch.setattr(cpu, "allocate", no_memory)
    if failure in ("missing", "truncated"):
        with pytest.raises(OSError):
            disk.get_blocking(key)
    elif failure == "allocate":
        assert disk.get_blocking(key) is None
    else:
        loaded = disk.get_blocking(key)
        try:
            assert bytes(loaded.byte_array) == expected
        finally:
            loaded.ref_count_down()
    assert meta.pin_count == 0
    assert mixed.pin_allocator.num_active_allocations == 0
