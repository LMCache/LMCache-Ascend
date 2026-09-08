# SPDX-License-Identifier: Apache-2.0
"""Independent state storage identity, managed allocation and ownership contracts."""

# Standard
from contextlib import nullcontext
from dataclasses import replace
from threading import Lock
from types import SimpleNamespace
import json

# Third Party
from lmcache.utils import CacheEngineKey
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
