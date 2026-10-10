# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Tests for AscendPDBackend and its sender/receiver mixins.

Unit tests use mocks (no NPU required).  Integration tests require NPU
hardware and are gated behind ``@pytest.mark.skipif``.
"""

# Standard
from typing import Tuple
from unittest.mock import MagicMock, call, patch
import threading
import time

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.storage_backend.pd_backend import AllocRequest
import msgspec
import pytest
import torch

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.storage_backend.pd.messages import (
    AscendAllocResponse,
    AscendPDMsg,
    PullDoneSignal,
    PullReadyDoneAck,
    PullReadyNotif,
)

logger = init_logger(__name__)


def _make_key(key_id: str = "test_key") -> CacheEngineKey:
    return CacheEngineKey("test_model", 2, 0, hash(key_id), torch.bfloat16, None)


DEFAULT_SHAPE = torch.Size([2, 2, 256, 512])
DEFAULT_DTYPE = torch.bfloat16


def _make_mock_mem_obj(
    shape: torch.Size = DEFAULT_SHAPE,
    dtype: torch.dtype = DEFAULT_DTYPE,
    address: int = 0,
) -> MagicMock:
    mock = MagicMock(spec=MemoryObj)
    mock.tensor = MagicMock()
    mock.data_ptr = 0xDEAD
    mock.meta = MagicMock(spec=MemoryObjMetadata)
    mock.meta.address = address
    mock.meta.shape = shape
    mock.meta.dtype = dtype
    mock.meta.fmt = MemoryFormat.KV_2LTD
    mock.ref_count_down = MagicMock()
    mock.ref_count_up = MagicMock()
    mock.unpin = MagicMock()
    mock.get_ref_count = MagicMock(return_value=1)
    return mock


def _make_proxy(
    context: MagicMock | None = None,
    chunk_index: int = 0,
) -> ProxyMemoryObj:
    if context is None:
        context = MagicMock()
    proxy = ProxyMemoryObj(
        backing_obj=None,
        transfer_channel=MagicMock(),
        target_peer_url="fake_url",
        remote_buffer_uuid=f"fake_uuid_{chunk_index}",
        remote_mem_index=chunk_index,
        transfer_context=context,
        chunk_index=chunk_index,
        shapes=[DEFAULT_SHAPE],
        dtypes=[DEFAULT_DTYPE],
        fmt=MemoryFormat.KV_2LTD,
    )
    return proxy


def _make_consumed_proxy() -> ProxyMemoryObj:
    """Create a ProxyMemoryObj that is already consumed."""
    proxy = _make_proxy()
    proxy.mark_consumed()
    return proxy


def _make_pd_backend_stub(
    role: str = "receiver",
    buffer_device: str = "npu:0",
    use_cpu_offload: bool = False,
    pull_mode: bool = False,
    delay_pull: bool = False,
    chunk_size: int = 256,
    kv_shape: Tuple[int, ...] = DEFAULT_SHAPE,
    kv_dtype: torch.dtype = DEFAULT_DTYPE,
):
    """Create a mock object with the minimal attributes needed by PD backend methods."""
    # First Party
    from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend, PDEntry
    from lmcache_ascend.v1.storage_backend.pd.sender_mixin import AscendPDSenderMixin

    backend = MagicMock()
    backend._pd_entries = {}
    backend._pd_request_keys = {}
    backend._pd_handoff_deadlines = {}
    backend._pd_handoff_lease_ttl = 300.0
    backend.tp_rank = 0

    class _PDDataDict(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            backend._pd_entries[key] = PDEntry(base_obj=value)

        def pop(self, key, default=None):
            backend._pd_entries.pop(key, None)
            return super().pop(key, default)

    backend.data = _PDDataDict()
    backend.data_lock = threading.Lock()
    backend.pd_config = MagicMock()
    backend.pd_config.role = role
    backend.pd_config.buffer_device = buffer_device
    backend.use_cpu_offload = use_cpu_offload
    backend.pull_mode = pull_mode
    backend.delay_pull = delay_pull
    backend.running = True
    backend.transfer_channel = MagicMock()
    backend.memory_allocator = MagicMock()
    backend.full_chunk_size = chunk_size
    backend._fmt = MemoryFormat.KV_2LTD
    backend._kv_shapes = [DEFAULT_SHAPE]
    backend._kv_dtypes = [kv_dtype]
    backend._metadata = MagicMock()
    backend._metadata.get_num_groups.return_value = 1
    backend._metadata.get_shapes.return_value = [DEFAULT_SHAPE]
    backend._metadata.get_dtypes.return_value = [kv_dtype]
    backend._peer_alloc_backoff_ttl = 0.01

    # Wire internal delegation methods to their real implementations so tests
    # that call e.g. AscendPDBackend.contains(backend, ...) actually exercise
    # the eviction / partition logic instead of hitting auto-mocked no-ops.
    backend._wire_shape_dtype_and_last_chunk_toks = lambda memory_objs: (
        AscendPDSenderMixin._wire_shape_dtype_and_last_chunk_toks(backend, memory_objs)
    )
    backend._lookup = lambda key, pin=False: AscendPDBackend._lookup(
        backend, key, pin=pin
    )
    backend._contains_and_pin = lambda key: AscendPDBackend._contains_and_pin(
        backend, key
    )
    backend._partition_keys = lambda keys: AscendPDBackend._partition_keys(
        backend, keys
    )
    backend._ensure_request_lease_locked = (
        lambda key, entry, request_id: AscendPDBackend._ensure_request_lease_locked(
            backend,
            key,
            entry,
            request_id,
        )
    )
    backend._release_context_leases = (
        lambda contexts, request_id: AscendPDBackend._release_context_leases(
            contexts,
            request_id,
        )
    )
    backend._detach_request_lease_locked = (
        lambda key, entry, request_id: AscendPDBackend._detach_request_lease_locked(
            backend,
            key,
            entry,
            request_id,
        )
    )
    backend._refresh_handoff_deadline_locked = lambda lease_id: (
        AscendPDBackend._refresh_handoff_deadline_locked(backend, lease_id)
    )
    backend._delete_pd_entry_locked = (
        lambda key, entry, release_obj: AscendPDBackend._delete_pd_entry_locked(
            backend,
            key,
            entry,
            release_obj,
        )
    )
    backend._partition_keys_with_handoff = lambda keys, handoff_id: (
        AscendPDBackend._partition_keys_with_handoff(backend, keys, handoff_id)
    )
    backend.put_with_handoff_lease = lambda key, obj, handoff_id: (
        AscendPDBackend.put_with_handoff_lease(backend, key, obj, handoff_id)
    )
    backend.release_request_lease = lambda request_id, expected_handoff_deadline=None: (
        AscendPDBackend.release_request_lease(
            backend,
            request_id,
            expected_handoff_deadline=expected_handoff_deadline,
        )
    )
    backend.release_handoff_lease = lambda handoff_id: (
        AscendPDBackend.release_handoff_lease(backend, handoff_id)
    )
    backend.release_expired_handoff_leases = lambda: (
        AscendPDBackend.release_expired_handoff_leases(backend)
    )
    backend.promote_handoff_lease = lambda handoff_id, request_id: (
        AscendPDBackend.promote_handoff_lease(backend, handoff_id, request_id)
    )

    return backend


class TestAscendPDBackend:
    """Mock-based unit tests for AscendPDBackend logic."""

    def test_pd_message_types(self):
        """All Ascend PD message types roundtrip through msgspec."""
        msgs = [
            AllocRequest(
                keys=["k1", "k2"],
                fmt=MemoryFormat.KV_2LTD.value,
                shape=list(DEFAULT_SHAPE),
                dtype="bfloat16",
                last_chunk_toks=256,
            ),
            AscendAllocResponse(
                already_sent_indexes=[0],
                remote_indexes=[1, 2],
                remote_buffer_uuids=["uuid-a", "uuid-b"],
                alloc_failed=False,
            ),
            PullReadyNotif(
                pull_id="pull_1",
                keys=["k1"],
                sender_buffer_uuids=["suuid-1"],
                sender_mem_indexes=[0],
                sender_id="sender_1",
                sender_done_url="tcp://sender:9999",
                fmt=MemoryFormat.KV_2LTD.value,
                shape=list(DEFAULT_SHAPE),
                dtype="bfloat16",
                last_chunk_toks=256,
                handoff_id="handoff_1",
            ),
            PullReadyDoneAck(
                already_sent_indexes=[],
                alloc_failed=False,
            ),
            PullDoneSignal(pull_id="pull_1"),
        ]
        for msg in msgs:
            encoded = msgspec.msgpack.encode(msg)
            decoded = msgspec.msgpack.decode(encoded, type=AscendPDMsg)
            assert type(decoded) is type(msg)
            assert decoded == msg

    def test_allocate_receiver_uses_gpu(self):
        """Receiver allocates on GPU (NPU)."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub(
            role="receiver",
            buffer_device="npu:0",
            kv_shape=DEFAULT_SHAPE,
            kv_dtype=DEFAULT_DTYPE,
            chunk_size=256,
            pull_mode=False,
            delay_pull=False,
            use_cpu_offload=False,
        )
        backend.memory_allocator.allocate = MagicMock(return_value="gpu_obj")

        result = AscendPDBackend.allocate(
            backend,
            DEFAULT_SHAPE,
            DEFAULT_DTYPE,
            MemoryFormat.KV_2LTD,
        )

        backend.memory_allocator.allocate.assert_called_once()
        call_kwargs = backend.memory_allocator.allocate.call_args
        assert call_kwargs.kwargs.get("allocator_type") == "gpu"
        assert result == "gpu_obj"

    def test_allocate_sender_with_offload_uses_cpu(self):
        """Sender with cpu_offload allocates on CPU."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub(
            role="sender",
            buffer_device="npu:0",
            kv_shape=DEFAULT_SHAPE,
            kv_dtype=DEFAULT_DTYPE,
            chunk_size=256,
            pull_mode=False,
            delay_pull=False,
            use_cpu_offload=True,
        )
        backend.memory_allocator.allocate = MagicMock(return_value="cpu_obj")

        result = AscendPDBackend.allocate(
            backend,
            DEFAULT_SHAPE,
            DEFAULT_DTYPE,
            MemoryFormat.KV_2LTD,
        )

        call_kwargs = backend.memory_allocator.allocate.call_args
        assert call_kwargs.kwargs.get("allocator_type") == "cpu"
        assert result == "cpu_obj"

    def test_contains_evicts_consumed_proxy(self):
        """Consumed ProxyMemoryObj is evicted from data on contains()."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("consumed_key")
        backend.data[key] = _make_consumed_proxy()

        result = AscendPDBackend.contains(backend, key, pin=False)

        assert result is False
        assert key not in backend.data

    def test_contains_normal_obj_returns_true(self):
        """Regular MemoryObj is found by contains()."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("normal_key")
        backend.data[key] = _make_mock_mem_obj()

        result = AscendPDBackend.contains(backend, key, pin=False)
        assert result is True

    def test_contains_missing_key(self):
        """Missing key returns False."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("missing")

        result = AscendPDBackend.contains(backend, key, pin=False)
        assert result is False

    def test_contains_pin_calls_ref_count_up(self):
        """Pinning a key calls ref_count_up on the object."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("pin_key")
        mock_obj = _make_mock_mem_obj()
        backend.data[key] = mock_obj

        result = AscendPDBackend.contains(backend, key, pin=True)

        assert result is True
        mock_obj.ref_count_up.assert_called_once()

    def test_partition_keys(self):
        """Keys are partitioned into already-sent and new indexes."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key0 = _make_key("k0")
        key1 = _make_key("k1")
        key2 = _make_key("k2")

        mock_obj0 = _make_mock_mem_obj()
        backend.data[key0] = mock_obj0

        str_keys = [key0.to_string(), key1.to_string(), key2.to_string()]

        already_sent_idx, already_sent_objs, new_idx = AscendPDBackend._partition_keys(
            backend, str_keys
        )

        assert already_sent_idx == [0]
        assert len(already_sent_objs) == 1
        assert already_sent_objs[0] is mock_obj0
        assert new_idx == [1, 2]
        mock_obj0.ref_count_up.assert_called_once()

    def test_partition_keys_proxy_pin_release_preserves_transfer_owner(self):
        """Already-sent Proxy lookup release must not complete its transfer."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend
        from lmcache_ascend.v1.storage_backend.utils import release_memory_objects

        backend = _make_pd_backend_stub()
        key = _make_key("proxy-key")
        context = MagicMock()
        proxy = ProxyMemoryObj(
            backing_obj=None,
            transfer_channel=MagicMock(),
            target_peer_url="sender_1",
            remote_buffer_uuid="suuid-0",
            remote_mem_index=0,
            transfer_context=context,
            chunk_index=0,
            shapes=[DEFAULT_SHAPE],
            dtypes=[DEFAULT_DTYPE],
            fmt=MemoryFormat.KV_2LTD,
        )
        backend.data[key] = proxy

        already_sent_idx, already_sent_objs, new_idx = AscendPDBackend._partition_keys(
            backend, [key.to_string()]
        )
        release_memory_objects(already_sent_objs)

        assert already_sent_idx == [0]
        assert already_sent_objs == [proxy]
        assert new_idx == []
        context.decref.assert_not_called()

        proxy.ref_count_down()
        context.decref.assert_called_once()

    def test_pd_backend_keeps_shared_entry_until_last_request_lease_released(self):
        """A shared PD hit is deleted only after every request lease is released."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("shared_lease")
        mem_obj = _make_mock_mem_obj()
        AscendPDBackend.put(backend, key, mem_obj)

        assert AscendPDBackend.batched_contains_and_lease(backend, [key], "req-1") == 1
        assert AscendPDBackend.batched_contains_and_lease(backend, [key], "req-2") == 1

        AscendPDBackend.release_request_lease(backend, "req-1")

        assert key in backend.data
        assert key in backend._pd_entries
        assert backend._pd_entries[key].owners == {"req-2"}
        mem_obj.ref_count_down.assert_not_called()

        AscendPDBackend.release_request_lease(backend, "req-2")

        assert key not in backend.data
        assert key not in backend._pd_entries
        assert backend._pd_request_keys == {}
        mem_obj.ref_count_down.assert_called_once()

    def test_pd_backend_proxy_leases_are_request_local(self):
        """Consuming one request's proxy clone must not poison shared PD hits."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("shared_proxy")
        transfer_context = MagicMock()
        base_proxy = _make_proxy(transfer_context)
        AscendPDBackend.put(backend, key, base_proxy)

        assert AscendPDBackend.batched_contains_and_lease(backend, [key], "req-1") == 1
        assert AscendPDBackend.batched_contains_and_lease(backend, [key], "req-2") == 1

        entry = backend._pd_entries[key]
        req1_proxy = entry.proxy_leases["req-1"]
        req2_proxy = entry.proxy_leases["req-2"]

        assert req1_proxy is not base_proxy
        assert req2_proxy is not base_proxy
        assert req1_proxy is not req2_proxy
        transfer_context.acquire_request.assert_any_call("req-1")
        transfer_context.acquire_request.assert_any_call("req-2")

        req1_proxy.mark_consumed()

        assert req1_proxy.consumed is True
        assert base_proxy.consumed is False
        assert req2_proxy.consumed is False
        assert AscendPDBackend.batched_get_blocking_for_request(
            backend,
            [key],
            "req-2",
        ) == [req2_proxy]

    def test_push_mode_allocate_and_put(self):
        """Push-mode allocate_and_put returns UUID-based refs."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        mock_obj = _make_mock_mem_obj()
        backend.allocate = MagicMock(return_value=mock_obj)
        backend.put = MagicMock()
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["uuid-alloc"],
            [42],
        )

        alloc_req = AllocRequest(
            keys=[_make_key("k1").to_string()],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        resp = AscendPDReceiverMixin._allocate_and_put(backend, alloc_req)

        assert isinstance(resp, AscendAllocResponse)
        assert resp.alloc_failed is False
        assert resp.remote_buffer_uuids == ["uuid-alloc"]
        assert resp.remote_indexes == [42]
        assert resp.already_sent_indexes == []
        backend.put.assert_called_once()

    def test_push_mode_alloc_failure(self):
        """Push-mode allocation failure returns alloc_failed=True."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        backend.allocate = MagicMock(return_value=None)
        backend.put = MagicMock()

        alloc_req = AllocRequest(
            keys=[_make_key("k1").to_string()],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            return_value=None,
        ):
            resp = AscendPDReceiverMixin._allocate_and_put(backend, alloc_req)

        assert isinstance(resp, AscendAllocResponse)
        assert resp.alloc_failed is True
        backend.put.assert_not_called()

    def test_push_mode_partial_alloc_failure_cleans_pd_entries(self):
        """Partial push allocation rollback removes both receiver indexes."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        backend.data = {}
        backend._pd_entries = {}
        backend.put = lambda key, mem_obj: AscendPDBackend.put(backend, key, mem_obj)
        allocated_obj = _make_mock_mem_obj()
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["uuid-alloc"],
            [42],
        )
        key0 = _make_key("partial-alloc-0")
        key1 = _make_key("partial-alloc-1")

        alloc_req = AllocRequest(
            keys=[key0.to_string(), key1.to_string()],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            side_effect=[allocated_obj, None],
        ):
            resp = AscendPDReceiverMixin._allocate_and_put(backend, alloc_req)

        assert isinstance(resp, AscendAllocResponse)
        assert resp.alloc_failed is True
        assert key0 not in backend.data
        assert key0 not in backend._pd_entries
        assert key1 not in backend.data
        assert key1 not in backend._pd_entries
        allocated_obj.ref_count_down.assert_called_once()

    def test_pull_eager_flow(self):
        """Pull-eager: allocates, reads from sender, returns ack + callback."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        mock_obj = _make_mock_mem_obj()
        backend.allocate = MagicMock(return_value=mock_obj)
        backend.transfer_channel.batched_read = MagicMock(return_value=1)
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_eager_1",
            handoff_id="handoff_eager_1",
            keys=[_make_key("k1").to_string()],
            sender_buffer_uuids=["suuid-1"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            return_value=mock_obj,
        ):
            ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_eager(
                backend, msg, "sender_1"
            )

        assert isinstance(ack, PullReadyDoneAck)
        assert ack.alloc_failed is False
        assert ack.already_sent_indexes == []
        backend.transfer_channel.batched_read.assert_called_once()
        key = _make_key("k1")
        assert backend._pd_entries[key].base_obj is mock_obj
        assert backend._pd_entries[key].owners == {
            "__lmcache_pd_handoff__:handoff_eager_1"
        }

        # Post-ack callback sends Done signal
        assert post_ack_fn is not None
        post_ack_fn()
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_1", "pull_eager_1"
        )

    def test_pull_eager_alloc_failure(self):
        """Pull-eager with alloc failure returns alloc_failed=True."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        backend.allocate = MagicMock(return_value=None)
        backend.put = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_fail",
            handoff_id="handoff_fail",
            keys=[_make_key("k1").to_string()],
            sender_buffer_uuids=["suuid-1"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            return_value=None,
        ):
            ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_eager(
                backend, msg, "sender_1"
            )

        assert ack.alloc_failed is True
        assert post_ack_fn is None

    def test_pull_delay_flow(self):
        """Pull-delay creates ProxyMemoryObj instances in data store."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(
            delay_pull=True,
            buffer_device="npu:0",
            kv_shape=DEFAULT_SHAPE,
            kv_dtype=torch.bfloat16,
            chunk_size=256,
            pull_mode=True,
            use_cpu_offload=True,
        )
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_delay_1",
            handoff_id="handoff_delay_1",
            keys=[_make_key("k1").to_string(), _make_key("k2").to_string()],
            sender_buffer_uuids=["suuid-0", "suuid-1"],
            sender_mem_indexes=[0, 1],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_delay(
            backend, msg, "sender_1"
        )

        assert isinstance(ack, PullReadyDoneAck)
        assert ack.alloc_failed is False
        assert post_ack_fn is None
        assert len(backend._pd_entries) == 2
        for entry in backend._pd_entries.values():
            assert isinstance(entry.base_obj, ProxyMemoryObj)
            assert entry.owners == {"__lmcache_pd_handoff__:handoff_delay_1"}

    def test_pull_delay_transfer_context_done_callback_is_idempotent(self):
        """Delay-pull transfer context sends done signal at most once."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(
            delay_pull=True,
            buffer_device="npu",
            kv_shape=DEFAULT_SHAPE,
            kv_dtype=torch.bfloat16,
            chunk_size=256,
            pull_mode=True,
            use_cpu_offload=True,
        )
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_delay_done_once",
            handoff_id="handoff_delay_done_once",
            keys=[_make_key("k1").to_string()],
            sender_buffer_uuids=["suuid-0"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_delay(
            backend, msg, "sender_1"
        )
        assert isinstance(ack, PullReadyDoneAck)
        assert post_ack_fn is None
        assert len(backend._pd_entries) == 1

        proxy_obj = next(iter(backend._pd_entries.values())).base_obj
        assert isinstance(proxy_obj, ProxyMemoryObj)

        transfer_ctx = proxy_obj.transfer_context
        backend.release_handoff_lease("handoff_delay_done_once")
        transfer_ctx.send_done_now()
        transfer_ctx.send_done_now()
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_1", "pull_delay_done_once"
        )

    def test_partition_keys_with_handoff_reserves_existing_hit(self):
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("shared_handoff_hit")
        mem_obj = _make_mock_mem_obj()
        AscendPDBackend.put(backend, key, mem_obj)

        already_indexes, already_objs, new_indexes = (
            backend._partition_keys_with_handoff([key.to_string()], "shared-handoff")
        )

        assert already_indexes == [0]
        assert already_objs == [mem_obj]
        assert new_indexes == []
        assert backend._pd_entries[key].owners == {
            "__lmcache_pd_handoff__:shared-handoff"
        }
        assert "__lmcache_pd_handoff__:shared-handoff" in (
            backend._pd_handoff_deadlines
        )

    def test_pull_ready_without_handoff_id_returns_typed_failure(self):
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(pull_mode=True)
        msg = PullReadyNotif(
            pull_id="pull_without_handoff",
            keys=[_make_key("missing_handoff").to_string()],
            sender_buffer_uuids=["suuid-0"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        ack, post_ack_fn = AscendPDReceiverMixin._handle_pull_ready(
            backend, msg, "sender_1"
        )

        assert ack == PullReadyDoneAck(already_sent_indexes=[], alloc_failed=True)
        assert post_ack_fn is None

    def test_promote_handoff_lease_acquires_request_before_releasing_synthetic(self):
        backend = _make_pd_backend_stub()
        context = MagicMock()
        key = _make_key("promote_handoff")
        proxy = _make_proxy(context=context)

        inserted, existing = backend.put_with_handoff_lease(
            key, proxy, "handoff-promote"
        )
        claimed = backend.promote_handoff_lease("handoff-promote", "decoder-request")

        assert inserted is True
        assert existing is None
        assert claimed == 1
        assert backend._pd_entries[key].owners == {"decoder-request"}
        assert context.acquire_request.call_args_list == [
            call("__lmcache_pd_handoff__:handoff-promote"),
            call("decoder-request"),
        ]
        context.release_request.assert_called_once_with(
            "__lmcache_pd_handoff__:handoff-promote"
        )

    def test_pull_delay_handoff_closes_pullready_to_lookup_race(self):
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(
            delay_pull=True,
            buffer_device="npu:0",
            pull_mode=True,
        )
        backend._send_pull_done_to_sender = MagicMock()
        key = _make_key("shared_handoff_race")

        first = PullReadyNotif(
            pull_id="pull_old",
            handoff_id="handoff_old",
            keys=[key.to_string()],
            sender_buffer_uuids=["old-uuid"],
            sender_mem_indexes=[0],
            sender_id="sender_old",
            sender_done_url="tcp://sender-old:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )
        first_ack, _ = AscendPDReceiverMixin._handle_pull_delay(
            backend, first, "sender_old"
        )
        assert first_ack.already_sent_indexes == []
        assert backend.promote_handoff_lease("handoff_old", "request_old") == 1

        second = PullReadyNotif(
            pull_id="pull_new",
            handoff_id="handoff_new",
            keys=[key.to_string()],
            sender_buffer_uuids=["new-uuid"],
            sender_mem_indexes=[1],
            sender_id="sender_new",
            sender_done_url="tcp://sender-new:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )
        second_ack, _ = AscendPDReceiverMixin._handle_pull_delay(
            backend, second, "sender_new"
        )
        assert second_ack.already_sent_indexes == [0]

        # The previous owner releases after PullReady ACK but before the new
        # decoder lookup. The synthetic owner must keep the entry and sender
        # context alive across this exact interleaving.
        backend.release_request_lease("request_old")
        assert key in backend._pd_entries
        backend._send_pull_done_to_sender.assert_not_called()

        assert backend.promote_handoff_lease("handoff_new", "request_new") == 1
        assert backend._pd_entries[key].owners == {"request_new"}
        backend.release_request_lease("request_new")

        assert key not in backend._pd_entries
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_old", "pull_old"
        )

    def test_expired_delay_pull_handoff_releases_entry_and_sender(self):
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(
            delay_pull=True,
            buffer_device="npu:0",
            pull_mode=True,
        )
        backend._send_pull_done_to_sender = MagicMock()
        key = _make_key("expired_handoff")
        msg = PullReadyNotif(
            pull_id="pull_expired",
            handoff_id="handoff_expired",
            keys=[key.to_string()],
            sender_buffer_uuids=["expired-uuid"],
            sender_mem_indexes=[0],
            sender_id="sender_expired",
            sender_done_url="tcp://sender-expired:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=list(DEFAULT_SHAPE),
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        ack, _ = AscendPDReceiverMixin._handle_pull_delay(
            backend, msg, "sender_expired"
        )
        lease_id = "__lmcache_pd_handoff__:handoff_expired"
        backend._pd_handoff_deadlines[lease_id] = time.monotonic() - 1

        assert ack.alloc_failed is False
        assert backend.release_expired_handoff_leases() == 1
        assert key not in backend._pd_entries
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_expired", "pull_expired"
        )

    def test_stale_expiry_snapshot_does_not_release_refreshed_handoff(self):
        backend = _make_pd_backend_stub()
        key = _make_key("refreshed_handoff")
        mem_obj = _make_mock_mem_obj()
        backend.put_with_handoff_lease(key, mem_obj, "handoff-refreshed")
        lease_id = "__lmcache_pd_handoff__:handoff-refreshed"
        stale_deadline = backend._pd_handoff_deadlines[lease_id]

        with backend.data_lock:
            backend._refresh_handoff_deadline_locked(lease_id)

        released = backend.release_request_lease(
            lease_id,
            expected_handoff_deadline=stale_deadline,
        )

        assert released is False
        assert key in backend._pd_entries
        assert lease_id in backend._pd_entries[key].owners

    def test_promote_handoff_rolls_back_partial_real_owner_acquisition(self):
        backend = _make_pd_backend_stub()
        context = MagicMock()
        keys = [_make_key("rollback-0"), _make_key("rollback-1")]
        for index, key in enumerate(keys):
            backend.put_with_handoff_lease(
                key,
                _make_proxy(context=context, chunk_index=index),
                "handoff-rollback",
            )

        original_ensure = backend._ensure_request_lease_locked
        real_owner_attempts = 0

        def fail_second_real_owner(key, entry, request_id):
            nonlocal real_owner_attempts
            if request_id == "decoder-rollback":
                real_owner_attempts += 1
                if real_owner_attempts == 2:
                    raise RuntimeError("injected promotion failure")
            return original_ensure(key, entry, request_id)

        backend._ensure_request_lease_locked = fail_second_real_owner

        claimed = backend.promote_handoff_lease("handoff-rollback", "decoder-rollback")

        lease_id = "__lmcache_pd_handoff__:handoff-rollback"
        assert claimed == 0
        assert "decoder-rollback" not in backend._pd_request_keys
        assert backend._pd_request_keys[lease_id] == set(keys)
        for key in keys:
            assert backend._pd_entries[key].owners == {lease_id}
        context.release_request.assert_called_once_with("decoder-rollback")

    def test_proxy_submit_resolve_batch_fallback_uses_sync_batched_read(self):
        """No submit_batched_read: fallback uses synchronous batched_read."""

        class _NoSubmitChannel:
            def __init__(self):
                self.batched_read = MagicMock(return_value=1)

        transfer_channel = _NoSubmitChannel()

        proxy = ProxyMemoryObj(
            backing_obj=None,
            transfer_channel=transfer_channel,
            target_peer_url="sender_1",
            remote_buffer_uuid="suuid-0",
            remote_mem_index=0,
            transfer_context=MagicMock(_loop=None),
            chunk_index=0,
            shapes=[DEFAULT_SHAPE],
            dtypes=[DEFAULT_DTYPE],
            fmt=MemoryFormat.KV_2LTD,
        )
        backing_obj = _make_mock_mem_obj()
        proxy.set_backing_obj(backing_obj)

        event = ProxyMemoryObj.submit_resolve_batch([proxy])

        assert event is None
        assert proxy.resolved is True
        transfer_channel.batched_read.assert_called_once()

    def test_proxy_submit_resolve_batch_uses_submit_when_supported(self):
        """submit_batched_read path returns event and marks proxies resolved."""
        transfer_channel = MagicMock()
        expected_event = MagicMock()
        transfer_channel.submit_batched_read = MagicMock(return_value=expected_event)
        transfer_channel.batched_read = MagicMock()

        proxy = ProxyMemoryObj(
            backing_obj=None,
            transfer_channel=transfer_channel,
            target_peer_url="sender_1",
            remote_buffer_uuid="suuid-0",
            remote_mem_index=0,
            transfer_context=MagicMock(_loop=None),
            chunk_index=0,
            shapes=[DEFAULT_SHAPE],
            dtypes=[DEFAULT_DTYPE],
            fmt=MemoryFormat.KV_2LTD,
        )
        backing_obj = _make_mock_mem_obj()
        proxy.set_backing_obj(backing_obj)

        event = ProxyMemoryObj.submit_resolve_batch([proxy])

        assert event is expected_event
        assert proxy.resolved is True
        transfer_channel.submit_batched_read.assert_called_once()
        transfer_channel.batched_read.assert_not_called()

    def test_circuit_breaker_skips_backed_off_peer(self):
        """When peer is backed off, put task is skipped."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = _make_pd_backend_stub(role="sender")
        backend._peer_alloc_backoff = {
            "receiver_1234": time.monotonic() + 60,
        }
        backend._peer_alloc_backoff_lock = threading.Lock()
        backend.tp_rank = 0
        backend.proxy_side_channel = MagicMock()
        backend._ensure_peer_connection = MagicMock()
        backend._remote_allocate = MagicMock()

        transfer_spec = MagicMock()
        transfer_spec.receiver_init_port = [1234]
        transfer_spec.receiver_host = "receiver_"
        transfer_spec.is_last_prefill = True
        transfer_spec.req_id = "req_1"

        mock_objs = [_make_mock_mem_obj()]

        AscendPDSenderMixin.batched_submit_put_task(
            backend, [_make_key("k1")], mock_objs, transfer_spec
        )

        # Should NOT have called _ensure_peer_connection or _remote_allocate
        backend._ensure_peer_connection.assert_not_called()
        backend._remote_allocate.assert_not_called()
        # Should still send proxy notification for last prefill
        backend.proxy_side_channel.send.assert_called_once()

    def test_pull_sender_propagates_logical_handoff_id(self):
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = _make_pd_backend_stub(role="sender", pull_mode=True)
        backend.tp_rank = 0
        backend.local_id = "prefiller_7700"
        backend._wait_for_backpressure = MagicMock()
        backend._ensure_peer_connection = MagicMock()
        backend._sender_done_url = "tcp://sender:9999"
        backend._pull_pending = {}
        backend._early_pull_done = set()
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_pinned_count = 0
        backend._pull_pending_ttl = 360.0
        backend.proxy_side_channel = MagicMock()

        side_channel = MagicMock()
        side_channel.recv.return_value = msgspec.msgpack.encode(
            PullReadyDoneAck(already_sent_indexes=[0])
        )
        backend.mem_alloc_sockets = {"decoder_7710": side_channel}
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["sender-uuid"],
            [7],
        )

        transfer_spec = MagicMock()
        transfer_spec.req_id = "logical-handoff-123"
        transfer_spec.receiver_host = "decoder_"
        transfer_spec.receiver_init_port = [7710]
        transfer_spec.receiver_alloc_port = [7810]
        transfer_spec.is_last_prefill = False

        AscendPDSenderMixin._batched_submit_put_task_pull(
            backend,
            [_make_key("sender-propagation")],
            [_make_mock_mem_obj()],
            transfer_spec,
        )

        encoded = side_channel.send.call_args.args[0]
        decoded = msgspec.msgpack.decode(encoded, type=AscendPDMsg)
        assert isinstance(decoded, PullReadyNotif)
        assert decoded.handoff_id == "logical-handoff-123"

    def test_pull_sender_rejects_empty_handoff_before_pinning(self):
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = _make_pd_backend_stub(role="sender", pull_mode=True)
        transfer_spec = MagicMock()
        transfer_spec.req_id = ""
        mem_obj = _make_mock_mem_obj()

        with pytest.raises(ValueError, match="req_id must not be empty"):
            AscendPDSenderMixin._batched_submit_put_task_pull(
                backend,
                [_make_key("empty-handoff")],
                [mem_obj],
                transfer_spec,
            )

        mem_obj.ref_count_up.assert_not_called()

    def test_handle_pull_done_releases_resources(self):
        """_handle_pull_done releases pinned MemObjs."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = MagicMock()
        mock_obj = _make_mock_mem_obj()
        backend._pull_pending = {"pull_1": (time.monotonic(), [mock_obj])}
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_pinned_count = 1
        backend._early_pull_done = set()

        AscendPDSenderMixin._handle_pull_done(backend, "pull_1")

        assert "pull_1" not in backend._pull_pending
        mock_obj.ref_count_down.assert_called_once()
        assert backend._pull_pending_pinned_count == 0

    def test_handle_pull_done_early_signal(self):
        """Early Done signal is buffered for later processing."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = MagicMock()
        backend._pull_pending = {}
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_pinned_count = 0
        backend._early_pull_done = set()

        AscendPDSenderMixin._handle_pull_done(backend, "pull_early")

        assert "pull_early" in backend._early_pull_done

    def test_backpressure_blocks_when_above_hwm(self):
        """_wait_for_backpressure blocks until count drops below HWM."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = MagicMock()
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_hwm = 5
        # Start above HWM, then release in background
        backend._pull_pending_pinned_count = 10

        released = threading.Event()

        def release_after_delay():
            time.sleep(0.05)
            with backend._pull_pending_lock:
                backend._pull_pending_pinned_count = 0
            released.set()

        t = threading.Thread(target=release_after_delay, daemon=True)
        t.start()

        # This should block until count drops
        AscendPDSenderMixin._wait_for_backpressure(backend, 2)

        assert released.is_set()
        t.join(timeout=2)

    def test_sweep_expired_pull_pending(self):
        """Expired entries are released by the sweep."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.sender_mixin import (
            AscendPDSenderMixin,
        )

        backend = MagicMock()
        backend._pull_pending_lock = threading.Lock()
        backend._pull_pending_ttl = 0.001

        mock_obj = _make_mock_mem_obj()
        # Entry pinned well in the past
        backend._pull_pending = {
            "expired_pull": (0.0, [mock_obj]),
        }
        backend._pull_pending_pinned_count = 1

        time.sleep(0.01)
        AscendPDSenderMixin._sweep_expired_pull_pending(backend)

        assert "expired_pull" not in backend._pull_pending
        mock_obj.ref_count_down.assert_called_once()
        assert backend._pull_pending_pinned_count == 0

    def test_allocate_and_put_with_already_sent(self):
        """Already-sent keys are identified and not re-allocated."""
        # First Party
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()

        key0 = _make_key("existing")
        existing_obj = _make_mock_mem_obj()
        backend.data[key0] = existing_obj

        new_obj = _make_mock_mem_obj(address=1)
        backend.allocate = MagicMock(return_value=new_obj)
        backend.put = MagicMock()
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["uuid-new"],
            [1],
        )

        alloc_req = AllocRequest(
            keys=[key0.to_string(), _make_key("new_key").to_string()],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
            dtype="bfloat16",
            last_chunk_toks=256,
        )

        with patch(
            "lmcache_ascend.v1.storage_backend.pd.receiver_mixin.allocate_with_retry",
            return_value=new_obj,
        ):
            resp = AscendPDReceiverMixin._allocate_and_put(backend, alloc_req)

        assert resp.already_sent_indexes == [0]
        assert len(resp.remote_buffer_uuids) == 1
        assert resp.alloc_failed is False
        # Only the new key was put
        backend.put.assert_called_once()
        # Already-sent obj was unpinned
        existing_obj.ref_count_down.assert_called()
