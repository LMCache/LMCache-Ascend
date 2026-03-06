# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Tests for AscendP2PBackend.

Unit tests use mocks (no NPU required).  Integration tests require at least 2
NPU devices and are gated with ``@pytest.mark.skipif``.
"""

# Standard
from unittest.mock import AsyncMock, MagicMock
import asyncio
import threading

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.storage_backend.p2p_backend import (
    P2PErrorCode,
    P2PErrorMsg,
    PeerInfo,
)
import msgspec
import pytest
import torch
import zmq.asyncio

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.storage_backend.p2p_backend import (
    AscendBatchedLookupAndGetDoneMsg,
    AscendBatchedLookupAndGetDoneRetMsg,
    AscendBatchedLookupAndGetMsg,
    AscendBatchedLookupAndGetRetMsg,
    AscendBatchedLookupAndPutMsg,
    AscendP2PMsg,
)

logger = init_logger(__name__)


def _make_key(key_id: str = "test_key") -> CacheEngineKey:
    return CacheEngineKey(
        "vllm", "test_model", 2, 0, hash(key_id), torch.bfloat16, None
    )


DEFAULT_SHAPE = torch.Size([2, 2, 256, 512])
DEFAULT_DTYPE = torch.bfloat16


def _make_mock_mem_obj(
    fill_val: float = 1.0,
    shape: torch.Size = DEFAULT_SHAPE,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> MagicMock:
    """Create a mock MemoryObj with the minimal interface."""
    mock = MagicMock(spec=MemoryObj)
    mock.tensor = MagicMock()
    mock.data_ptr = 0xDEAD
    mock.meta = MagicMock(spec=MemoryObjMetadata)
    mock.meta.address = 0
    mock.meta.shape = shape
    mock.meta.dtype = dtype
    mock.meta.fmt = MemoryFormat.KV_2LTD
    mock.ref_count_down = MagicMock()
    mock.ref_count_up = MagicMock()
    mock.unpin = MagicMock()
    return mock


def _run_coroutine(loop: asyncio.AbstractEventLoop, coro):
    """Submit coroutine to background loop and wait for result."""
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=10)


def _make_p2p_backend_stub(
    pull_mode: bool = False,
    delay_pull: bool = False,
    use_npu: bool = False,
    peer_init_url: str = "localhost:5000",
    kv_shapes: list = None,
    kv_dtypes: list = None,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    save_unfull_chunk: bool = False,
) -> MagicMock:
    """Create a MagicMock stub with the minimal attributes for AscendP2PBackend.

    Wires ``_allocate_memory_for_keys`` to the real implementation via a
    delegation lambda (matching the ``_make_pd_backend_stub`` pattern in
    ``test_ascend_pd_backend.py``), so tests that call
    ``AscendP2PBackend.batched_get_non_blocking(backend, ...)`` exercise the
    real allocation logic rather than hitting an auto-mocked no-op that returns
    an un-unpackable MagicMock.

    Code-path coverage notes for ``_allocate_memory_for_keys``
    ----------------------------------------------------------
    Current callers of this stub all use ``use_npu=False`` and
    ``save_unfull_chunk=False``, so only the ``local_cpu_backend.allocate``
    branch is exercised through ``batched_get_non_blocking``.

    * ``use_npu=True``: calls ``memory_allocator.gpu_allocator.allocate``.
      This branch is covered directly by
      ``test_allocate_memory_for_keys_npu_path``.
    * ``save_unfull_chunk=True`` (last chunk): calls ``_get_unfull_chunk_shapes``.
      Not yet covered.  If that path is tested via this stub, add a delegation
      lambda for ``_get_unfull_chunk_shapes`` here.
    """
    # First Party
    from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

    if kv_shapes is None:
        kv_shapes = [DEFAULT_SHAPE]
    if kv_dtypes is None:
        kv_dtypes = [DEFAULT_DTYPE]

    backend = MagicMock()
    backend.pull_mode = pull_mode
    backend.delay_pull = delay_pull
    backend.use_npu = use_npu
    backend.peer_init_url = peer_init_url
    backend.config = MagicMock()
    backend.config.save_unfull_chunk = save_unfull_chunk
    backend.full_size_shapes = kv_shapes
    backend.dtypes = kv_dtypes
    backend.fmt = fmt

    backend._allocate_memory_for_keys = (
        lambda keys, cum_chunk_lengths: AscendP2PBackend._allocate_memory_for_keys(
            backend, keys, cum_chunk_lengths
        )
    )

    return backend


@pytest.fixture
def async_loop():
    """Background asyncio event loop."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert ready.wait(timeout=5), "Event loop failed to start"

    yield loop

    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=3)
    if not loop.is_closed():
        try:
            for task in asyncio.all_tasks(loop):
                task.cancel()
        except Exception:
            pass
        loop.close()


class TestAscendP2PBackendUnit:
    """Mock-based unit tests for AscendP2PBackend logic."""

    def test_message_types_encode_decode(self):
        """Verify all Ascend P2P message types roundtrip through msgspec."""
        msgs = [
            AscendBatchedLookupAndGetMsg(
                lookup_id="lu_1",
                receiver_id="peer_1",
                keys=["k1", "k2"],
                buffer_uuids=["uuid-a", "uuid-b"],
                mem_indexes=[0, 1],
                pull_mode=True,
            ),
            AscendBatchedLookupAndGetRetMsg(
                num_hit_chunks=2,
                remote_buffer_uuids=["uuid-c"],
                remote_mem_indexes=[3],
            ),
            AscendBatchedLookupAndPutMsg(
                sender_id="peer_1",
                keys=["k3"],
                offsets=[0],
                mem_indexes=[0],
                buffer_uuids=["uuid-d"],
            ),
            AscendBatchedLookupAndGetDoneMsg(lookup_id="lu_3"),
            AscendBatchedLookupAndGetDoneRetMsg(),
            P2PErrorMsg(error_code=P2PErrorCode.P2P_SERVER_ERROR),
        ]
        for msg in msgs:
            encoded = msgspec.msgpack.encode(msg)
            decoded = msgspec.msgpack.decode(encoded, type=AscendP2PMsg)
            assert type(decoded) is type(msg)

    def test_handle_batched_lookup_and_get_push_mode(self, async_loop):
        """Handler writes to client buffers and returns num_hit_chunks."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.chunk_size = 256
        backend.transfer_channel = MagicMock()
        backend.transfer_channel.remote_xfer_handler_exists.return_value = True
        backend.transfer_channel.async_batched_write = AsyncMock()

        mock_objs = [_make_mock_mem_obj()]
        backend.local_cpu_backend = MagicMock()
        backend.local_cpu_backend.batched_async_contains = AsyncMock(return_value=1)
        backend.local_cpu_backend.batched_get_non_blocking = AsyncMock(
            return_value=mock_objs
        )

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id="lu_push",
            receiver_id="peer_1",
            keys=[_make_key("k1").to_string()],
            buffer_uuids=["remote-uuid"],
            mem_indexes=[0],
            pull_mode=False,
        )

        ret = _run_coroutine(
            async_loop,
            AscendP2PBackend._handle_batched_lookup_and_get(backend, msg),
        )

        assert isinstance(ret, AscendBatchedLookupAndGetRetMsg)
        assert ret.num_hit_chunks == 1
        backend.transfer_channel.async_batched_write.assert_awaited_once()

    def test_handle_batched_lookup_and_get_pull_mode(self, async_loop):
        """Pull mode returns buffer refs; stores pending resources."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.chunk_size = 256
        backend.transfer_channel = MagicMock()
        backend.transfer_channel.remote_xfer_handler_exists.return_value = True
        backend.transfer_channel.async_batched_write = AsyncMock()

        mock_objs = [_make_mock_mem_obj()]
        backend.local_cpu_backend = MagicMock()
        backend.local_cpu_backend.batched_async_contains = AsyncMock(return_value=1)
        backend.local_cpu_backend.batched_get_non_blocking = AsyncMock(
            return_value=mock_objs
        )
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["server-uuid"],
            [42],
        )
        backend.pending_pull_resources = {}

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id="lu_pull",
            receiver_id="peer_1",
            keys=[_make_key("k1").to_string()],
            buffer_uuids=[],
            mem_indexes=[],
            pull_mode=True,
        )

        ret = _run_coroutine(
            async_loop,
            AscendP2PBackend._handle_batched_lookup_and_get(backend, msg),
        )

        assert isinstance(ret, AscendBatchedLookupAndGetRetMsg)
        assert ret.num_hit_chunks == 1
        assert ret.remote_buffer_uuids == ["server-uuid"]
        assert ret.remote_mem_indexes == [42]
        # Should NOT have called batched_write in pull mode
        backend.transfer_channel.async_batched_write.assert_not_awaited()
        # Should have stored pending resources
        assert "lu_pull" in backend.pending_pull_resources

    def test_handle_batched_lookup_and_get_done(self, async_loop):
        """Done signal releases pending pull resources."""
        backend = MagicMock()
        backend.loop = async_loop

        mock_obj = _make_mock_mem_obj()
        backend.pending_pull_resources = {
            "lu_done": (0.0, [mock_obj]),
        }

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        msg = AscendBatchedLookupAndGetDoneMsg(lookup_id="lu_done")
        ret = _run_coroutine(
            async_loop,
            AscendP2PBackend._handle_batched_lookup_and_get_done(backend, msg),
        )

        assert isinstance(ret, AscendBatchedLookupAndGetDoneRetMsg)
        assert "lu_done" not in backend.pending_pull_resources
        mock_obj.ref_count_down.assert_called_once()
        mock_obj.unpin.assert_called_once()

    def test_handle_done_missing_lookup_id(self, async_loop):
        """Done signal for unknown lookup_id doesn't crash."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.pending_pull_resources = {}

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        msg = AscendBatchedLookupAndGetDoneMsg(lookup_id="nonexistent")
        ret = _run_coroutine(
            async_loop,
            AscendP2PBackend._handle_batched_lookup_and_get_done(backend, msg),
        )
        assert isinstance(ret, AscendBatchedLookupAndGetDoneRetMsg)

    def test_sweep_expired_pending_pull_resources(self, async_loop):
        """Expired entries are cleaned up by the sweep coroutine."""
        backend = MagicMock()
        backend.loop = async_loop
        backend._pull_pending_ttl = 0.01
        backend.running = asyncio.Event()
        backend.running.set()

        mock_obj = _make_mock_mem_obj()
        # Timestamp far in the past relative to loop.time()
        backend.pending_pull_resources = {
            "expired_1": (0.0, [mock_obj]),
        }

        # First Party

        async def _sweep_once():
            """Run one iteration of the sweep logic."""
            now = async_loop.time()
            expired_ids = [
                pid
                for pid, (ts, _) in backend.pending_pull_resources.items()
                if now - ts > backend._pull_pending_ttl
            ]
            for pid in expired_ids:
                entry = backend.pending_pull_resources.pop(pid, None)
                if entry is not None:
                    # First Party
                    from lmcache_ascend.v1.storage_backend.utils import (
                        release_memory_objects,
                    )

                    _, mem_objs = entry
                    release_memory_objects(mem_objs, unpin=True)

        _run_coroutine(async_loop, _sweep_once())

        assert "expired_1" not in backend.pending_pull_resources
        mock_obj.ref_count_down.assert_called_once()
        mock_obj.unpin.assert_called_once()

    def test_allocate_memory_for_keys_oom(self, async_loop):
        """OOM during allocation releases partial results."""
        backend = MagicMock()
        backend.use_npu = False
        backend.config = MagicMock()
        backend.config.save_unfull_chunk = False
        backend.full_size_shapes = [torch.Size([2, 2, 256, 512])]
        backend.dtypes = [torch.bfloat16]
        backend.fmt = MemoryFormat.KV_2LTD

        good_obj = _make_mock_mem_obj()
        # First call succeeds, second fails
        backend.local_cpu_backend = MagicMock()
        backend.local_cpu_backend.allocate = MagicMock(side_effect=[good_obj, None])

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        keys = [_make_key("k1"), _make_key("k2")]
        cum_chunk_lengths = [0, 256, 512]

        mem_objs, str_keys = AscendP2PBackend._allocate_memory_for_keys(
            backend, keys, cum_chunk_lengths
        )

        assert mem_objs == []
        assert str_keys == []
        good_obj.ref_count_down.assert_called_once()

    def test_allocate_memory_for_keys_npu_path(self):
        """NPU path uses memory_allocator.gpu_allocator.allocate,
        not local_cpu_backend."""
        backend = MagicMock()
        backend.use_npu = True
        backend.config = MagicMock()
        backend.config.save_unfull_chunk = False
        backend.full_size_shapes = [DEFAULT_SHAPE]
        backend.dtypes = [DEFAULT_DTYPE]
        backend.fmt = MemoryFormat.KV_2LTD

        mock_obj1 = _make_mock_mem_obj()
        mock_obj2 = _make_mock_mem_obj()
        backend.memory_allocator.gpu_allocator.allocate = MagicMock(
            side_effect=[mock_obj1, mock_obj2]
        )

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        keys = [_make_key("k1"), _make_key("k2")]
        cum_chunk_lengths = [0, 256, 512]

        mem_objs, str_keys = AscendP2PBackend._allocate_memory_for_keys(
            backend, keys, cum_chunk_lengths
        )

        assert mem_objs == [mock_obj1, mock_obj2]
        assert len(str_keys) == 2
        assert backend.memory_allocator.gpu_allocator.allocate.call_count == 2
        backend.local_cpu_backend.allocate.assert_not_called()

    def test_send_lookup_request_with_retry_zmq_error(self, async_loop):
        """ZMQ errors trigger retries and peer reconnection."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.max_retry_count = 2
        backend._ensure_peer_connection = AsyncMock()

        mock_socket = AsyncMock()
        mock_socket.send = AsyncMock(side_effect=zmq.ZMQError("connection lost"))
        mock_socket.recv = AsyncMock()

        mock_lock = asyncio.Lock()
        peer_info = MagicMock(spec=PeerInfo)
        peer_info.lookup_socket = mock_socket
        peer_info.lookup_lock = mock_lock
        backend.target_peer_info_mapping = {"peer_url": peer_info}

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id="lu_retry",
            receiver_id="peer_1",
            keys=["k1"],
            buffer_uuids=[],
            mem_indexes=[],
            pull_mode=False,
        )

        ret = _run_coroutine(
            async_loop,
            AscendP2PBackend._send_lookup_request_with_retry(
                backend, "lu_retry", "peer_url", msg
            ),
        )

        assert ret is None
        assert mock_socket.send.await_count == 2
        assert backend._ensure_peer_connection.await_count == 2

    def test_send_lookup_request_with_retry_success(self, async_loop):
        """Successful response is returned directly."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.max_retry_count = 3
        backend._ensure_peer_connection = AsyncMock()

        good_ret = AscendBatchedLookupAndGetRetMsg(num_hit_chunks=2)
        encoded = msgspec.msgpack.encode(good_ret)

        mock_socket = AsyncMock()
        mock_socket.send = AsyncMock()
        mock_socket.recv = AsyncMock(return_value=encoded)

        mock_lock = asyncio.Lock()
        peer_info = MagicMock(spec=PeerInfo)
        peer_info.lookup_socket = mock_socket
        peer_info.lookup_lock = mock_lock
        backend.target_peer_info_mapping = {"peer_url": peer_info}

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id="lu_ok",
            receiver_id="peer_1",
            keys=["k1"],
            buffer_uuids=[],
            mem_indexes=[],
            pull_mode=False,
        )

        ret = _run_coroutine(
            async_loop,
            AscendP2PBackend._send_lookup_request_with_retry(
                backend, "lu_ok", "peer_url", msg
            ),
        )

        assert isinstance(ret, AscendBatchedLookupAndGetRetMsg)
        assert ret.num_hit_chunks == 2

    def test_batched_get_non_blocking_push_mode(self, async_loop):
        """Push mode allocates memory, sends request, returns hit objects."""
        backend = _make_p2p_backend_stub(
            pull_mode=False, delay_pull=False, use_npu=False
        )
        backend.loop = async_loop
        backend.lookup_id_to_peer_mapping = {"lu_push": ("target_peer_url", "cpu")}

        mock_obj1 = _make_mock_mem_obj()
        mock_obj2 = _make_mock_mem_obj()
        backend.local_cpu_backend.allocate = MagicMock(
            side_effect=[mock_obj1, mock_obj2]
        )

        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["uuid-1", "uuid-2"],
            [0, 1],
        )

        ret_msg = AscendBatchedLookupAndGetRetMsg(num_hit_chunks=2)
        backend._send_lookup_request_with_retry = AsyncMock(return_value=ret_msg)

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        keys = [_make_key("k1"), _make_key("k2")]
        transfer_spec = {"cum_chunk_lengths": [0, 256, 512]}

        result = _run_coroutine(
            async_loop,
            AscendP2PBackend.batched_get_non_blocking(
                backend, "lu_push", keys, transfer_spec
            ),
        )

        assert len(result) == 2
        assert result[0] is mock_obj1
        assert result[1] is mock_obj2

    def test_batched_get_non_blocking_pull_delay(self, async_loop):
        """Delay pull returns ProxyMemoryObj instances."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.pull_mode = True
        backend.delay_pull = True
        backend.use_npu = True
        backend.peer_init_url = "localhost:5000"
        backend.config = MagicMock()
        backend.full_size_shapes = [torch.Size([2, 2, 256, 512])]
        backend.dtypes = [torch.bfloat16]
        backend.fmt = MemoryFormat.KV_2LTD
        backend.memory_allocator = MagicMock()
        backend.lookup_id_to_peer_mapping = {"lu_delay": ("target_peer_url", "npu")}
        backend.transfer_channel = MagicMock()

        ret_msg = AscendBatchedLookupAndGetRetMsg(
            num_hit_chunks=2,
            remote_buffer_uuids=["ruuid-0", "ruuid-1"],
            remote_mem_indexes=[10, 11],
        )
        backend._send_lookup_request_with_retry = AsyncMock(return_value=ret_msg)

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        keys = [_make_key("k1"), _make_key("k2")]
        transfer_spec = {"cum_chunk_lengths": [0, 256, 512]}

        result = _run_coroutine(
            async_loop,
            AscendP2PBackend.batched_get_non_blocking(
                backend, "lu_delay", keys, transfer_spec
            ),
        )

        assert len(result) == 2
        for obj in result:
            assert isinstance(obj, ProxyMemoryObj)
            assert obj.is_proxy

    def test_batched_get_non_blocking_error_response(self, async_loop):
        """Error response from peer returns empty list."""
        backend = _make_p2p_backend_stub(
            pull_mode=False, delay_pull=False, use_npu=False
        )
        backend.loop = async_loop
        backend.lookup_id_to_peer_mapping = {"lu_err": ("target_peer_url", "cpu")}

        mock_obj = _make_mock_mem_obj()
        backend.local_cpu_backend.allocate = MagicMock(return_value=mock_obj)
        backend.transfer_channel.get_local_buffer_refs.return_value = (
            ["uuid-1"],
            [0],
        )

        error_ret = P2PErrorMsg(error_code=P2PErrorCode.P2P_SERVER_ERROR)
        backend._send_lookup_request_with_retry = AsyncMock(return_value=error_ret)

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        keys = [_make_key("k1")]
        transfer_spec = {"cum_chunk_lengths": [0, 256]}

        result = _run_coroutine(
            async_loop,
            AscendP2PBackend.batched_get_non_blocking(
                backend, "lu_err", keys, transfer_spec
            ),
        )

        assert result == []
        backend._cleanup_memory_objects.assert_called_once()

    def test_handle_pull_mode_transfer_success(self, async_loop):
        """Pull mode transfer reads data and sends done signal."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.transfer_channel = MagicMock()
        backend.transfer_channel.async_batched_read = AsyncMock()
        backend._send_done_signal = AsyncMock()

        mock_objs = [_make_mock_mem_obj()]

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        success = _run_coroutine(
            async_loop,
            AscendP2PBackend._handle_pull_mode_transfer(
                backend,
                "lu_pull_xfer",
                "target_peer_url",
                mock_objs,
                ["ruuid-0"],
                [10],
            ),
        )

        assert success is True
        backend.transfer_channel.async_batched_read.assert_awaited_once()
        backend._send_done_signal.assert_awaited_once_with(
            "lu_pull_xfer", "target_peer_url"
        )

    def test_handle_pull_mode_transfer_empty_uuids(self, async_loop):
        """Pull transfer with empty buffer uuids returns False."""
        backend = MagicMock()
        backend.loop = async_loop
        backend._send_done_signal = AsyncMock()

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        success = _run_coroutine(
            async_loop,
            AscendP2PBackend._handle_pull_mode_transfer(
                backend,
                "lu_empty",
                "target_peer_url",
                [],
                [],
                [],
            ),
        )

        assert success is False

    def test_handle_pull_mode_transfer_read_failure_still_sends_done(self, async_loop):
        """Done signal is sent even when the read operation fails."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.transfer_channel = MagicMock()
        backend.transfer_channel.async_batched_read = AsyncMock(
            side_effect=RuntimeError("RDMA read failed")
        )
        backend._send_done_signal = AsyncMock()

        mock_objs = [_make_mock_mem_obj()]

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        success = _run_coroutine(
            async_loop,
            AscendP2PBackend._handle_pull_mode_transfer(
                backend,
                "lu_fail",
                "target_peer_url",
                mock_objs,
                ["ruuid-0"],
                [10],
            ),
        )

        assert success is False
        # Done signal MUST be sent even on failure
        backend._send_done_signal.assert_awaited_once()

    def test_handle_get_xfer_not_initialized(self, async_loop):
        """Returns error when transfer handler doesn't exist for receiver."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.chunk_size = 256
        backend.transfer_channel = MagicMock()
        backend.transfer_channel.remote_xfer_handler_exists.return_value = False

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id="lu_no_xfer",
            receiver_id="unknown_peer",
            keys=["k1"],
            buffer_uuids=[],
            mem_indexes=[],
            pull_mode=False,
        )

        ret = _run_coroutine(
            async_loop,
            AscendP2PBackend._handle_batched_lookup_and_get(backend, msg),
        )

        assert isinstance(ret, P2PErrorMsg)
        assert ret.error_code == P2PErrorCode.REMOTE_XFER_HANDLER_NOT_INITIALIZED
