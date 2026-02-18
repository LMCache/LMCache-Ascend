# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Tests for AscendPDBackend and its sender/receiver mixins.

Unit tests use mocks (no NPU required).  Integration tests require NPU
hardware and are gated behind ``@pytest.mark.skipif``.
"""

# Standard
from typing import Union, Tuple
from unittest.mock import MagicMock, patch
import threading
import time

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.storage_backend.pd_backend import AllocRequest
import msgspec
import pytest
import torch
import torch_npu

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


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _make_key(key_id: str = "test_key") -> CacheEngineKey:
    return CacheEngineKey(
        "vllm", "test_model", 2, 0, hash(key_id), torch.bfloat16, None
    )


def _make_mock_mem_obj(
    shape: torch.Size = torch.Size([2, 2, 256, 512]),
    dtype: torch.dtype = torch.bfloat16,
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


def _make_consumed_proxy() -> ProxyMemoryObj:
    """Create a ProxyMemoryObj that is already consumed."""
    proxy = ProxyMemoryObj(
        backing_obj=None,
        transfer_channel=MagicMock(),
        target_peer_url="fake_url",
        remote_buffer_uuid="fake_uuid",
        remote_mem_index=0,
        transfer_context=MagicMock(),
        chunk_index=0,
        shapes=[torch.Size([2, 2, 256, 512])],
        dtypes=[torch.bfloat16],
        fmt=MemoryFormat.KV_2LTD,
    )
    proxy.mark_consumed()
    return proxy


def _make_pd_backend_stub(
    role: str = "receiver", 
    buffer_device: str = "npu:0",
    use_cpu_offload: bool = False,
    pull_mode: bool = False,
    delay_pull: bool = False,
    chunk_size: int = 256,
    kv_shape: Tuple[int, ...] = (2, 2, 256, 512),
    kv_dtype: torch.dtype = torch.bfloat16,
):
    """Create a mock object with the minimal attributes needed by PD backend methods."""
    from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

    backend = MagicMock()
    backend.data = {}
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
    backend._kv_shapes = [torch.Size(kv_shape)]
    backend._kv_dtypes = [kv_dtype]

    # Wire internal delegation methods to their real implementations so tests
    # that call e.g. AscendPDBackend.contains(backend, ...) actually exercise
    # the eviction / partition logic instead of hitting auto-mocked no-ops.
    backend._lookup = lambda key, pin=False: AscendPDBackend._lookup(backend, key, pin=pin)
    backend._contains_and_pin = lambda key: AscendPDBackend._contains_and_pin(backend, key)
    backend._partition_keys = lambda keys: AscendPDBackend._partition_keys(backend, keys)

    return backend


# ──────────────────────────────────────────────────────────
# Unit tests (mock-based, no NPU required)
# ──────────────────────────────────────────────────────────


class TestAscendPDBackendUnit:
    """Mock-based unit tests for AscendPDBackend logic."""

    # ── Message encode/decode ──────────────────────────────

    def test_pd_message_types(self):
        """All Ascend PD message types roundtrip through msgspec."""
        msgs = [
            AllocRequest(
                keys=["k1", "k2"],
                fmt=MemoryFormat.KV_2LTD.value,
                shape=[2, 2, 256, 512],
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
                shape=[2, 2, 256, 512],
                dtype="bfloat16",
                last_chunk_toks=256,
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

    # ── allocate() role-aware placement ────────────────────

    def test_allocate_receiver_uses_gpu(self):
        """Receiver allocates on GPU (NPU)."""
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub(
            role="receiver", buffer_device="npu:0", 
            kv_shape=(2, 2, 256, 512), 
            kv_dtype=torch.bfloat16,
            chunk_size=256,
            pull_mode=False,
            delay_pull=False,
            use_cpu_offload=False,
        )
        backend.memory_allocator.allocate = MagicMock(return_value="gpu_obj")

        result = AscendPDBackend.allocate(
            backend,
            torch.Size([2, 2, 256, 512]),
            torch.bfloat16,
            MemoryFormat.KV_2LTD,
        )

        backend.memory_allocator.allocate.assert_called_once()
        call_kwargs = backend.memory_allocator.allocate.call_args
        assert call_kwargs.kwargs.get("allocator_type") == "gpu"
        assert result == "gpu_obj"

    def test_allocate_sender_with_offload_uses_cpu(self):
        """Sender with cpu_offload allocates on CPU."""
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub(
            role="sender", buffer_device="npu:0", 
            kv_shape=(2, 2, 256, 512), 
            kv_dtype=torch.bfloat16,
            chunk_size=256,
            pull_mode=False,
            delay_pull=False,
            use_cpu_offload=True,
        )
        backend.memory_allocator.allocate = MagicMock(return_value="cpu_obj")

        result = AscendPDBackend.allocate(
            backend,
            torch.Size([2, 2, 256, 512]),
            torch.bfloat16,
            MemoryFormat.KV_2LTD,
        )

        call_kwargs = backend.memory_allocator.allocate.call_args
        assert call_kwargs.kwargs.get("allocator_type") == "cpu"
        assert result == "cpu_obj"


    def test_contains_evicts_consumed_proxy(self):
        """Consumed ProxyMemoryObj is evicted from data on contains()."""
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("consumed_key")
        backend.data[key] = _make_consumed_proxy()

        result = AscendPDBackend.contains(backend, key, pin=False)

        assert result is False
        assert key not in backend.data

    def test_contains_normal_obj_returns_true(self):
        """Regular MemoryObj is found by contains()."""
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("normal_key")
        backend.data[key] = _make_mock_mem_obj()

        result = AscendPDBackend.contains(backend, key, pin=False)
        assert result is True

    def test_contains_missing_key(self):
        """Missing key returns False."""
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key = _make_key("missing")

        result = AscendPDBackend.contains(backend, key, pin=False)
        assert result is False

    def test_contains_pin_calls_ref_count_up(self):
        """Pinning a key calls ref_count_up on the object."""
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
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        backend = _make_pd_backend_stub()
        key0 = _make_key("k0")
        key1 = _make_key("k1")
        key2 = _make_key("k2")

        mock_obj0 = _make_mock_mem_obj()
        backend.data[key0] = mock_obj0

        str_keys = [key0.to_string(), key1.to_string(), key2.to_string()]

        already_sent_idx, already_sent_objs, new_idx = (
            AscendPDBackend._partition_keys(backend, str_keys)
        )

        assert already_sent_idx == [0]
        assert len(already_sent_objs) == 1
        assert already_sent_objs[0] is mock_obj0
        assert new_idx == [1, 2]
        mock_obj0.ref_count_up.assert_called_once()

    # ── _allocate_and_put (push mode) ──────────────────────

    def test_push_mode_allocate_and_put(self):
        """Push-mode allocate_and_put returns UUID-based refs."""
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
            shape=[2, 2, 256, 512],
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
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        backend.allocate = MagicMock(return_value=None)
        backend.put = MagicMock()

        alloc_req = AllocRequest(
            keys=[_make_key("k1").to_string()],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
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

    # ── _handle_pull_eager ─────────────────────────────────

    def test_pull_eager_flow(self):
        """Pull-eager: allocates, reads from sender, returns ack + callback."""
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        mock_obj = _make_mock_mem_obj()
        backend.allocate = MagicMock(return_value=mock_obj)
        backend.put = MagicMock()
        backend.transfer_channel.batched_read = MagicMock(return_value=1)
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_eager_1",
            keys=[_make_key("k1").to_string()],
            sender_buffer_uuids=["suuid-1"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
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
        backend.put.assert_called_once()

        # Post-ack callback sends Done signal
        assert post_ack_fn is not None
        post_ack_fn()
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_1", "pull_eager_1"
        )

    def test_pull_eager_alloc_failure(self):
        """Pull-eager with alloc failure returns alloc_failed=True."""
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub()
        backend.allocate = MagicMock(return_value=None)
        backend.put = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_fail",
            keys=[_make_key("k1").to_string()],
            sender_buffer_uuids=["suuid-1"],
            sender_mem_indexes=[0],
            sender_id="sender_1",
            sender_done_url="tcp://sender:9999",
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[2, 2, 256, 512],
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

    # ── _handle_pull_delay ─────────────────────────────────

    def test_pull_delay_flow(self):
        """Pull-delay creates ProxyMemoryObj instances in data store."""
        from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import (
            AscendPDReceiverMixin,
        )

        backend = _make_pd_backend_stub(
            delay_pull=True,
            buffer_device="npu:0",
            kv_shape=(2, 2, 256, 512),
            kv_dtype=torch.bfloat16,
            chunk_size=256,
            pull_mode=True,
            use_cpu_offload=True,
        )
        backend.put = MagicMock()
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_delay_1",
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
        # Two ProxyMemoryObjs should have been put()
        assert backend.put.call_count == 2
        for call in backend.put.call_args_list:
            _, mem_obj = call.args
            assert isinstance(mem_obj, ProxyMemoryObj)

    # ── Circuit breaker ────────────────────────────────────

    def test_circuit_breaker_skips_backed_off_peer(self):
        """When peer is backed off, put task is skipped."""
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

    # ── Pull done listener ─────────────────────────────────

    def test_handle_pull_done_releases_resources(self):
        """_handle_pull_done releases pinned MemObjs."""
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

    # ── Backpressure ───────────────────────────────────────

    def test_backpressure_blocks_when_above_hwm(self):
        """_wait_for_backpressure blocks until count drops below HWM."""
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

    # ── Sweep expired pull pending ─────────────────────────

    def test_sweep_expired_pull_pending(self):
        """Expired entries are released by the sweep."""
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

    # ── _allocate_and_put with already_sent keys ───────────

    def test_allocate_and_put_with_already_sent(self):
        """Already-sent keys are identified and not re-allocated."""
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
