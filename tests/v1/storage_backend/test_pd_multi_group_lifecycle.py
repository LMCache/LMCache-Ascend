# SPDX-License-Identifier: Apache-2.0
"""PD request ownership with heterogeneous KV groups; transport is mocked."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import threading

# Third Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.pd_backend import AllocRequest
import pytest
import torch

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.storage_backend import storage_manager as sm
from lmcache_ascend.v1.storage_backend.pd import receiver_mixin
from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend
from lmcache_ascend.v1.storage_backend.pd.messages import PullReadyNotif

DTYPES = [torch.bfloat16, torch.float32]
FMT = MemoryFormat.KV_2LTD


def _shapes(num_tokens=8):
    return [torch.Size([2, 1, num_tokens, 4]), torch.Size([1, 1, num_tokens // 4, 4])]


def _memory_obj(fill=0):
    shapes = _shapes()
    size = sum(s.numel() * d.itemsize for s, d in zip(shapes, DTYPES, strict=True))
    metadata = MemoryObjMetadata(
        shape=shapes[0],
        dtype=DTYPES[0],
        address=0,
        phy_size=size,
        ref_count=1,
        fmt=FMT,
        shapes=shapes,
        dtypes=list(DTYPES),
    )
    return TensorMemoryObj(
        torch.full((size,), fill, dtype=torch.uint8, device="cpu"),
        metadata,
        parent_allocator=None,
    )


def _key(index):
    return CacheEngineKey("multi-group", 1, 0, index, torch.bfloat16, None)


def _backend():
    backend = object.__new__(AscendPDBackend)
    backend.data = {}
    backend._pd_entries = {}
    backend._pd_request_keys = {}
    backend.data_lock = threading.Lock()
    backend._metadata = MagicMock()
    backend._metadata.get_num_groups.return_value = 2
    backend._metadata.get_shapes.side_effect = _shapes
    backend._metadata.get_dtypes.return_value = list(DTYPES)
    backend.transfer_channel = MagicMock()
    backend.memory_allocator = MagicMock()
    backend._peer_alloc_backoff_ttl = 0
    backend._fmt = FMT
    backend._send_pull_done_to_sender = MagicMock()
    return backend


@pytest.mark.parametrize("failure", [None, "out_of_memory", "missing_tensor"])
def test_multi_group_copy_preserves_layout_bytes_and_key_alignment(failure):
    sources = [_memory_obj(fill=i + 1) for i in range(3)]
    targets = [_memory_obj(), _memory_obj()]
    invalid = MagicMock(tensor=None)
    second = {None: targets[1], "out_of_memory": None, "missing_tensor": invalid}[
        failure
    ]
    allocator = MagicMock()
    allocator.contains.side_effect = lambda key: key == "k0"
    allocator.allocate.side_effect = [targets[0], second]
    stream = MagicMock()

    with patch.object(sm.torch_dev, "stream", return_value=nullcontext()):
        keys, objects = sm.allocate_and_copy_objects(
            allocator,
            ["k0", "k1", "k2"],
            sources,
            stream,
        )

    assert keys == (["k1", "k2"] if failure is None else ["k1"])
    assert objects == (targets if failure is None else targets[:1])
    for source, target in zip(sources[1:], objects, strict=False):
        assert target.get_shapes() == source.get_shapes() == _shapes()
        assert target.get_dtypes() == source.get_dtypes() == DTYPES
        assert torch.equal(target.raw_data, source.raw_data)
        for group in range(2):
            assert torch.equal(target.get_tensor(group), source.get_tensor(group))
    for allocation in allocator.allocate.call_args_list:
        assert allocation.args == (_shapes(), DTYPES)
    stream.synchronize.assert_called_once()
    if failure == "missing_tensor":
        invalid.ref_count_down.assert_called_once()


def test_multi_group_shared_entry_releases_all_groups_after_last_request():
    backend = _backend()
    key = _key(0)
    obj = _memory_obj()
    allocator = MagicMock()
    allocator.free.side_effect = lambda memory_obj: memory_obj.invalidate()
    obj.parent_allocator = allocator
    backend.put(key, obj)
    for request in ("req-1", "req-2"):
        assert backend.batched_contains_and_lease([key], request) == 1
    first = backend.batched_get_blocking_for_request([key], "req-1")[0]
    second = backend.batched_get_blocking_for_request([key], "req-2")[0]
    assert first is second is obj
    assert obj.get_ref_count() == 3
    first.ref_count_down()
    backend.release_request_lease("req-1")
    assert obj.get_ref_count() == 2
    allocator.free.assert_not_called()
    assert backend.data[key] is obj
    assert second.get_shapes() == _shapes()
    second.ref_count_down()
    backend.release_request_lease("req-2")
    assert obj.get_ref_count() == 0
    allocator.free.assert_called_once_with(obj)
    assert not obj.is_valid()
    assert backend.data == backend._pd_entries == backend._pd_request_keys == {}


def test_multi_group_delay_pull_clones_keep_partial_layout_and_request_ownership():
    backend = _backend()
    keys = [_key(0), _key(1)]
    message = PullReadyNotif(
        pull_id="pull-1",
        keys=[key.to_string() for key in keys],
        sender_buffer_uuids=["buffer-0", "buffer-1"],
        sender_mem_indexes=[10, 11],
        sender_id="sender-1",
        sender_done_url="tcp://localhost:9901",
        fmt=FMT.value,
        shape=list(_shapes()[0]),
        dtype="bfloat16",
        last_chunk_toks=4,
    )
    ack, callback = backend._handle_pull_delay(message, "sender-1")
    assert ack.already_sent_indexes == []
    assert callback is None
    prototypes = [backend.data[key] for key in keys]
    context = prototypes[0].transfer_context
    assert prototypes[1].transfer_context is context

    # #274: temporary deduplication pins must not consume the prototype.
    pinned = backend._contains_and_pin(keys[0])
    assert pinned.get_ref_count() == 2
    pinned.ref_count_down()
    assert pinned.get_ref_count() == 1
    backend._send_pull_done_to_sender.assert_not_called()

    for request in ("req-1", "req-2"):
        assert backend.batched_contains_and_lease(keys, request) == 2
        assert backend.batched_contains_and_lease(keys, request) == 2
    assert context._active_lease_count == 4
    first = backend.batched_get_blocking_for_request(keys, "req-1")
    second = backend.batched_get_blocking_for_request(keys, "req-2")
    for index, (one, two, prototype) in enumerate(
        zip(first, second, prototypes, strict=True)
    ):
        assert one is not two and two is not prototype
        expected_shapes = _shapes(8 if index == 0 else 4)
        assert one.get_shapes() == two.get_shapes() == expected_shapes
        assert one.get_dtypes() == two.get_dtypes() == DTYPES
        assert two._remote_buffer_uuid == message.sender_buffer_uuids[index]
        assert two._remote_mem_index == message.sender_mem_indexes[index]
        one.mark_consumed()
        assert not two.consumed and not prototype.consumed

    context.allocate_buffers(2)
    backend.memory_allocator.batched_allocate.assert_called_once_with(
        _shapes(),
        DTYPES,
        2,
        FMT,
        "gpu",
    )
    backend.release_request_lease("req-1")
    context.send_done_now()
    backend._send_pull_done_to_sender.assert_not_called()
    assert all(key in backend.data for key in keys)
    backend.release_request_lease("req-2")
    backend._send_pull_done_to_sender.assert_called_once_with("sender-1", "pull-1")
    assert backend.data == backend._pd_entries == backend._pd_request_keys == {}
    backend.release_request_lease("req-2")
    context.send_done_now()
    backend._send_pull_done_to_sender.assert_called_once()


def test_multi_group_push_rollback_clears_new_entries_and_releases_existing_pin():
    backend = _backend()
    keys = [_key(i) for i in range(3)]
    existing, allocated = _memory_obj(), _memory_obj()
    backend.put(keys[0], existing)
    backend.transfer_channel.get_local_buffer_refs.return_value = (["buffer"], [10])
    request = AllocRequest(
        keys=[key.to_string() for key in keys],
        fmt=FMT.value,
        shape=list(_shapes()[0]),
        dtype="bfloat16",
        last_chunk_toks=4,
    )
    with patch.object(
        receiver_mixin, "allocate_with_retry", side_effect=[allocated, None]
    ) as allocate:
        response = backend._allocate_and_put(request)
    assert response.alloc_failed
    assert response.already_sent_indexes == [0]
    assert response.remote_buffer_uuids == response.remote_indexes == []
    assert allocate.call_args_list[0].args[1:3] == (_shapes(), DTYPES)
    assert allocate.call_args_list[1].args[1:3] == (_shapes(4), DTYPES)
    assert set(backend.data) == set(backend._pd_entries) == {keys[0]}
    assert backend._pd_request_keys == {}
    assert existing.get_ref_count() == 1
    assert allocated.get_ref_count() == 0


@pytest.mark.parametrize("pd_receiver", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_retrieve_forwards_group_mappings_and_restores_request_context(
    pd_receiver, fails
):
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(enable_pd=pd_receiver, pd_role="receiver")
    backend = MagicMock()
    engine.storage_manager = SimpleNamespace(storage_backends={"PDBackend": backend})
    tokens = list(range(8))
    mappings = (torch.arange(8), torch.arange(2))
    result = torch.ones(8, dtype=torch.bool)

    def retrieve_impl(actual_tokens, mask=None, **kwargs):
        assert actual_tokens is tokens
        assert kwargs["slot_mappings_by_group"] is mappings
        assert kwargs["slot_mappings_npu_by_group"] is mappings
        assert sm._current_pd_retrieve_id.get() == ("req-1" if pd_receiver else "outer")
        if fails:
            raise RuntimeError("retrieve failed")
        return result

    engine._retrieve_impl = retrieve_impl
    token = sm.set_current_pd_retrieve_id("outer")
    try:
        kwargs = dict(
            req_id="req-1",
            slot_mappings_by_group=mappings,
            slot_mappings_npu_by_group=mappings,
        )
        if fails:
            with pytest.raises(RuntimeError, match="retrieve failed"):
                engine.retrieve(tokens, **kwargs)
        else:
            assert engine.retrieve(tokens, **kwargs) is result
        assert sm._current_pd_retrieve_id.get() == "outer"
    finally:
        sm.reset_current_pd_retrieve_id(token)
    if pd_receiver:
        backend.release_request_lease.assert_called_once_with("req-1")
    else:
        backend.release_request_lease.assert_not_called()
