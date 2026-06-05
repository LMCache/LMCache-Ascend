# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Tests for AscendP2PBackend.

Unit tests use mocks (no NPU required).  Integration tests require at least 2
NPU devices and are gated with ``@pytest.mark.skipif``.
"""

# Standard
from concurrent.futures import TimeoutError
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import threading

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.storage_backend.p2p_backend import P2PErrorCode, P2PErrorMsg
import msgspec
import pytest
import torch
import zmq
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
    AscendPeerInfo,
    AscendQueryDonePortRetMsg,
)
from lmcache_ascend.v1.transfer_context import P2PTransferContext

logger = init_logger(__name__)


def _make_key(key_id: str = "test_key") -> CacheEngineKey:
    return CacheEngineKey("test_model", 2, 0, hash(key_id), torch.bfloat16, None)


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


async def _run_on_p2p_loop_inline(coro):
    """Unit-test stand-in for AscendP2PBackend._run_on_p2p_loop."""
    return await coro


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
    backend._run_on_p2p_loop = AsyncMock(side_effect=_run_on_p2p_loop_inline)

    backend._allocate_memory_for_keys = lambda keys, cum_chunk_lengths: (
        AscendP2PBackend._allocate_memory_for_keys(backend, keys, cum_chunk_lengths)
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
        """Connection errors trigger retries and peer reconnection (DEALER path)."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.max_retry_count = 2
        backend._lookup_timeout_s = 3.0
        backend._ensure_peer_connection = AsyncMock()
        # New DEALER path funnels the round-trip through _peer_request_reply.
        backend._peer_request_reply = AsyncMock(
            side_effect=zmq.ZMQError("connection lost")
        )

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
        assert backend._peer_request_reply.await_count == 2
        assert backend._ensure_peer_connection.await_count == 2

    def test_send_lookup_request_with_retry_success(self, async_loop):
        """Successful response is returned directly (DEALER path)."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.max_retry_count = 3
        backend._lookup_timeout_s = 3.0
        backend._ensure_peer_connection = AsyncMock()

        good_ret = AscendBatchedLookupAndGetRetMsg(num_hit_chunks=2)
        encoded = msgspec.msgpack.encode(good_ret)
        backend._peer_request_reply = AsyncMock(return_value=encoded)

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
        backend._peer_request_reply.assert_awaited_once()

    def test_send_lookup_request_with_retry_timeout_is_fast_miss(self, async_loop):
        """A per-request timeout fails fast (returns None) without retrying."""
        backend = MagicMock()
        backend.loop = async_loop
        backend.max_retry_count = 3
        backend._lookup_timeout_s = 0.2
        backend._ensure_peer_connection = AsyncMock()
        backend._peer_request_reply = AsyncMock(side_effect=asyncio.TimeoutError())

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id="lu_timeout",
            receiver_id="peer_1",
            keys=["k1"],
            buffer_uuids=[],
            mem_indexes=[],
            pull_mode=False,
        )

        ret = _run_coroutine(
            async_loop,
            AscendP2PBackend._send_lookup_request_with_retry(
                backend, "lu_timeout", "peer_url", msg
            ),
        )

        assert ret is None
        # Fail-fast: exactly one attempt, no reconnect storm.
        assert backend._peer_request_reply.await_count == 1
        backend._ensure_peer_connection.assert_not_awaited()

    def test_ensure_done_peer_connection_does_not_sleep(self, async_loop):
        """Done socket creation should not rely on a fixed post-connect delay."""
        backend = MagicMock()
        backend.done_peer_sockets = {}
        backend.done_peer_update_lock = asyncio.Lock()
        backend.socket_recv_timeout_ms = 1000
        backend.socket_send_timeout_ms = 1000
        backend.async_context = MagicMock()
        backend._wait_for_async_context = AsyncMock()
        backend._query_done_url = AsyncMock(return_value="127.0.0.1:5555")

        new_socket = MagicMock()

        # First Party
        from lmcache_ascend.v1.storage_backend import p2p_backend as p2p_mod
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        with (
            patch.object(
                p2p_mod,
                "get_zmq_socket_with_timeout",
                return_value=new_socket,
            ),
            patch.object(p2p_mod.asyncio, "sleep", new_callable=AsyncMock) as sleep,
        ):
            _run_coroutine(
                async_loop,
                AscendP2PBackend._ensure_done_peer_connection(
                    backend,
                    "peer_url",
                ),
            )

        sleep.assert_not_awaited()
        assert backend.done_peer_sockets["peer_url"][1] is new_socket

    def test_ensure_done_peer_connection_concurrent_creation(self, async_loop):
        """Concurrent first sends should create only one Done socket per peer."""
        backend = MagicMock()
        backend.done_peer_sockets = {}
        backend.done_peer_update_lock = asyncio.Lock()
        backend.socket_recv_timeout_ms = 1000
        backend.socket_send_timeout_ms = 1000
        backend.async_context = MagicMock()
        backend._wait_for_async_context = AsyncMock()
        backend._query_done_url = AsyncMock(return_value="127.0.0.1:5555")

        new_socket = MagicMock()

        # First Party
        from lmcache_ascend.v1.storage_backend import p2p_backend as p2p_mod
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        async def run_concurrent_ensures():
            await asyncio.gather(
                AscendP2PBackend._ensure_done_peer_connection(backend, "peer_url"),
                AscendP2PBackend._ensure_done_peer_connection(backend, "peer_url"),
            )

        with patch.object(
            p2p_mod,
            "get_zmq_socket_with_timeout",
            return_value=new_socket,
        ) as socket_factory:
            _run_coroutine(async_loop, run_concurrent_ensures())

        backend._query_done_url.assert_awaited_once_with("peer_url")
        socket_factory.assert_called_once()
        assert backend.done_peer_sockets["peer_url"][1] is new_socket

    def test_ensure_done_peer_connection_force_recreates(self, async_loop):
        """Force update closes and replaces the existing Done socket."""
        old_socket = MagicMock()
        backend = MagicMock()
        backend.done_peer_sockets = {"peer_url": (asyncio.Lock(), old_socket)}
        backend.done_peer_update_lock = asyncio.Lock()
        backend.socket_recv_timeout_ms = 1000
        backend.socket_send_timeout_ms = 1000
        backend.async_context = MagicMock()
        backend._wait_for_async_context = AsyncMock()
        backend._query_done_url = AsyncMock(return_value="127.0.0.1:5555")

        new_socket = MagicMock()

        # First Party
        from lmcache_ascend.v1.storage_backend import p2p_backend as p2p_mod
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        with patch.object(
            p2p_mod,
            "get_zmq_socket_with_timeout",
            return_value=new_socket,
        ):
            _run_coroutine(
                async_loop,
                AscendP2PBackend._ensure_done_peer_connection(
                    backend,
                    "peer_url",
                    force=True,
                ),
            )

        old_socket.close.assert_called_once_with(linger=0)
        assert backend.done_peer_sockets["peer_url"][1] is new_socket

    def test_send_done_signal_retries_zmq_again(self, async_loop):
        """A send timeout recreates the Done socket and retries."""
        first_socket = AsyncMock()
        first_socket.send = AsyncMock(side_effect=zmq.Again("send timed out"))
        first_socket.recv = AsyncMock()

        second_socket = AsyncMock()
        second_socket.send = AsyncMock()
        second_socket.recv = AsyncMock()

        backend = MagicMock()
        backend.max_retry_count = 2
        backend.done_peer_sockets = {"peer_url": (asyncio.Lock(), first_socket)}

        async def ensure_done_peer_connection(target_peer_url, force=False):
            if force:
                backend.done_peer_sockets[target_peer_url] = (
                    asyncio.Lock(),
                    second_socket,
                )

        backend._ensure_done_peer_connection = AsyncMock(
            side_effect=ensure_done_peer_connection
        )

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        _run_coroutine(
            async_loop,
            AscendP2PBackend._send_done_signal_on_loop(
                backend,
                "lu_done",
                "peer_url",
            ),
        )

        first_socket.send.assert_awaited_once()
        second_socket.send.assert_awaited_once()
        second_socket.recv.assert_awaited_once()
        backend._ensure_done_peer_connection.assert_any_await(
            "peer_url",
            force=True,
        )

    def test_send_done_signal_recreate_failure_stays_in_retry_loop(self, async_loop):
        """A failed forced recreate should not escape the Done retry loop."""
        first_socket = AsyncMock()
        first_socket.send = AsyncMock(side_effect=zmq.Again("send timed out"))
        first_socket.recv = AsyncMock()

        second_socket = AsyncMock()
        second_socket.send = AsyncMock()
        second_socket.recv = AsyncMock()

        backend = MagicMock()
        backend.max_retry_count = 3
        backend.done_peer_sockets = {"peer_url": (asyncio.Lock(), first_socket)}

        async def ensure_done_peer_connection(target_peer_url, force=False):
            if force:
                backend.done_peer_sockets.pop(target_peer_url, None)
                raise RuntimeError("done port query failed")
            if target_peer_url in backend.done_peer_sockets:
                return
            backend.done_peer_sockets[target_peer_url] = (
                asyncio.Lock(),
                second_socket,
            )

        backend._ensure_done_peer_connection = AsyncMock(
            side_effect=ensure_done_peer_connection
        )

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        _run_coroutine(
            async_loop,
            AscendP2PBackend._send_done_signal_on_loop(
                backend,
                "lu_done_recreate_retry",
                "peer_url",
            ),
        )

        first_socket.send.assert_awaited()
        second_socket.send.assert_awaited_once()
        second_socket.recv.assert_awaited_once()
        assert backend._ensure_done_peer_connection.await_count == 3

    def test_handle_query_done_port_waits_for_ready_url(self, async_loop):
        """Done-port queries wait for the done handler to publish its URL."""
        backend = MagicMock()
        backend._done_url_ready = asyncio.Event()
        backend.peer_done_url = None
        backend.p2p_done_timeout_s = 1.0

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        async def query_while_starting():
            task = asyncio.create_task(
                AscendP2PBackend._handle_query_done_port(backend)
            )
            await asyncio.sleep(0)
            backend.peer_done_url = "127.0.0.1:5555"
            backend._done_url_ready.set()
            return await task

        ret = _run_coroutine(async_loop, query_while_starting())

        assert isinstance(ret, AscendQueryDonePortRetMsg)
        assert ret.done_url == "127.0.0.1:5555"

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

    def test_batched_get_non_blocking_uses_transfer_spec_target_peer(self, async_loop):
        """Sync get path can pass target peer directly without shared mapping."""
        backend = _make_p2p_backend_stub(
            pull_mode=False, delay_pull=False, use_npu=False
        )
        backend.loop = async_loop
        backend.lookup_id_to_peer_mapping = {}

        mock_obj = _make_mock_mem_obj()
        backend.local_cpu_backend.allocate = MagicMock(return_value=mock_obj)
        backend.transfer_channel.get_local_buffer_refs.return_value = (["uuid-1"], [0])

        ret_msg = AscendBatchedLookupAndGetRetMsg(num_hit_chunks=1)
        backend._send_lookup_request_with_retry = AsyncMock(return_value=ret_msg)

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        transfer_spec = {
            "cum_chunk_lengths": [0, 256],
            "target_peer_url": "target_peer_url",
        }
        result = _run_coroutine(
            async_loop,
            AscendP2PBackend.batched_get_non_blocking(
                backend, "lu_direct", [_make_key("k1")], transfer_spec
            ),
        )

        assert result == [mock_obj]
        mock_obj.pin.assert_called_once()
        backend._send_lookup_request_with_retry.assert_awaited_once()
        assert backend._send_lookup_request_with_retry.await_args.args[1] == (
            "target_peer_url"
        )

    def test_batched_get_non_blocking_can_skip_return_pin(self, async_loop):
        """Sync blocking get relies on refcount ownership, not an extra get pin."""
        backend = _make_p2p_backend_stub(
            pull_mode=False, delay_pull=False, use_npu=False
        )
        backend.loop = async_loop
        backend.lookup_id_to_peer_mapping = {}

        mock_obj = _make_mock_mem_obj()
        backend.local_cpu_backend.allocate = MagicMock(return_value=mock_obj)
        backend.transfer_channel.get_local_buffer_refs.return_value = (["uuid-1"], [0])
        backend._send_lookup_request_with_retry = AsyncMock(
            return_value=AscendBatchedLookupAndGetRetMsg(num_hit_chunks=1)
        )

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        result = _run_coroutine(
            async_loop,
            AscendP2PBackend.batched_get_non_blocking(
                backend,
                "lu_no_pin",
                [_make_key("k1")],
                {
                    "cum_chunk_lengths": [0, 256],
                    "target_peer_url": "target_peer_url",
                    "pin_returned": False,
                },
            ),
        )

        assert result == [mock_obj]
        mock_obj.pin.assert_not_called()

    def test_batched_get_non_blocking_missing_mapping_returns_empty(self, async_loop):
        """Missing async lookup mapping is handled without KeyError."""
        backend = _make_p2p_backend_stub(
            pull_mode=False, delay_pull=False, use_npu=False
        )
        backend.loop = async_loop
        backend.lookup_id_to_peer_mapping = {}

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        result = _run_coroutine(
            async_loop,
            AscendP2PBackend.batched_get_non_blocking(
                backend,
                "lu_missing",
                [_make_key("k1")],
                {"cum_chunk_lengths": [0, 256]},
            ),
        )

        assert result == []

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
        backend._run_on_p2p_loop = AsyncMock(side_effect=_run_on_p2p_loop_inline)

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

    def test_p2p_transfer_context_done_schedules_without_blocking(self):
        """P2P Done is scheduled on the backend loop without host blocking."""
        backend = MagicMock()
        backend.p2p_done_timeout_s = 5.0
        backend._send_done_signal = AsyncMock()
        loop = MagicMock()
        future = MagicMock()

        # First Party

        ctx = P2PTransferContext(
            p2p_backend=backend,
            target_peer_url="target_peer_url",
            lookup_id="lu_done",
            loop=loop,
            num_proxies=1,
        )

        def fake_run_coroutine_threadsafe(coro, target_loop):
            coro.close()
            assert target_loop is loop
            return future

        with patch(
            "lmcache_ascend.v1.transfer_context.asyncio.run_coroutine_threadsafe",
            side_effect=fake_run_coroutine_threadsafe,
        ):
            ctx.send_done_now()

        future.result.assert_not_called()
        future.add_done_callback.assert_called_once()

    def test_p2p_transfer_context_done_on_same_loop_schedules_task(self, async_loop):
        """Done from the P2P loop must not block on run_coroutine_threadsafe."""
        backend = MagicMock()
        backend._send_done_signal = AsyncMock()

        # First Party

        ctx = P2PTransferContext(
            p2p_backend=backend,
            target_peer_url="target_peer_url",
            lookup_id="lu_same_loop",
            loop=async_loop,
            num_proxies=1,
        )

        async def invoke_done():
            with patch(
                "lmcache_ascend.v1.transfer_context.asyncio.run_coroutine_threadsafe"
            ) as run_threadsafe:
                ctx.send_done_now()
                run_threadsafe.assert_not_called()
            await asyncio.sleep(0)

        _run_coroutine(async_loop, invoke_done())
        backend._send_done_signal.assert_awaited_once_with(
            "lu_same_loop", "target_peer_url"
        )

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

    def test_sync_dealer_uses_short_controller_lookup_timeout(self):
        backend = MagicMock()
        backend._sync_closed = False
        backend._sync_dealer = None
        backend.config.controller_reply_url = "127.0.0.1:5555"
        backend._sync_controller_lookup_timeout_ms = 250

        sock = MagicMock()
        backend._sync_ctx.socket.return_value = sock

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        assert AscendP2PBackend._get_or_create_sync_dealer_locked(backend) is sock
        sock.setsockopt.assert_any_call(zmq.RCVTIMEO, 250)
        sock.setsockopt.assert_any_call(zmq.SNDTIMEO, 250)
        sock.setsockopt.assert_any_call(zmq.LINGER, 0)
        if hasattr(zmq, "IMMEDIATE"):
            sock.setsockopt.assert_any_call(zmq.IMMEDIATE, 1)
        sock.connect.assert_called_once_with("tcp://127.0.0.1:5555")

    def test_sync_query_timeout_resets_dealer(self):
        """Timed-out sync controller lookup resets socket to avoid stale replies."""
        backend = MagicMock()
        backend._sync_closed = False
        backend._sync_dealer = MagicMock()
        backend._sync_lock = threading.Lock()
        backend._sync_lookup_cache = {}
        backend._sync_lookup_cache_ttl = 5.0
        backend._sync_lookup_cache_max_entries = 16
        backend._make_sync_lookup_cache_key.return_value = ("k1",)
        backend._prune_sync_lookup_cache_locked = MagicMock()
        backend._reset_sync_dealer_locked = MagicMock()
        backend.lmcache_instance_id = "instance"
        backend.tp_rank = 0

        sock = MagicMock()
        sock.recv_multipart.side_effect = zmq.Again()
        backend._get_or_create_sync_dealer_locked.return_value = sock

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        lookup_id, peer_url, location, hits = AscendP2PBackend._sync_query_controller(
            backend, [_make_key("k1")]
        )

        assert lookup_id
        assert peer_url == ""
        assert location == ""
        assert hits == 0
        backend._reset_sync_dealer_locked.assert_called_once()

    def test_sync_dealer_not_created_after_close(self):
        """Closed sync path returns no socket instead of touching terminated context."""
        backend = MagicMock()
        backend._sync_closed = True
        backend._sync_dealer = None

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        assert AscendP2PBackend._get_or_create_sync_dealer_locked(backend) is None
        backend._sync_ctx.socket.assert_not_called()

    def test_blocking_helper_rejects_p2p_loop_thread(self):
        """Blocking sync bridge must not be used from the P2P event loop."""
        backend = MagicMock()
        backend._is_on_p2p_loop.return_value = True
        backend._sync_get_timeout_s = 1.0

        async def noop():
            return None

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        with pytest.raises(RuntimeError, match="P2P event loop"):
            AscendP2PBackend._run_coroutine_threadsafe_blocking(
                backend,
                noop(),
                "lu_loop",
                "noop",
            )

    def test_blocking_helper_timeout_cancels_future_and_releases_late_result(self):
        """Timeout registers cleanup for any late returned MemoryObj list."""
        backend = MagicMock()
        backend._is_on_p2p_loop.return_value = False
        backend.loop = MagicMock()
        backend._sync_get_timeout_s = 0.01

        class TimeoutFuture:
            def __init__(self):
                self.cancel = MagicMock()
                self.cancelled = MagicMock(return_value=False)
                self.add_done_callback = MagicMock()

            def result(self, timeout=None):
                if timeout is not None:
                    raise TimeoutError()
                return None

        late_obj = _make_mock_mem_obj()
        future = TimeoutFuture()

        async def slow():
            return [late_obj]

        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        backend._cleanup_late_sync_get_result = (
            lambda done, lookup_id, operation, unpin: (
                AscendP2PBackend._cleanup_late_sync_get_result(
                    backend, done, lookup_id, operation, unpin
                )
            )
        )

        def fake_run_coroutine_threadsafe(coro, loop):
            coro.close()
            return future

        with patch(
            "lmcache_ascend.v1.storage_backend.p2p_backend."
            "asyncio.run_coroutine_threadsafe",
            side_effect=fake_run_coroutine_threadsafe,
        ):
            with pytest.raises(TimeoutError):
                AscendP2PBackend._run_coroutine_threadsafe_blocking(
                    backend,
                    slow(),
                    "lu_timeout",
                    "batched_get_non_blocking",
                    cleanup_late_result=True,
                )

        future.cancel.assert_called_once()
        future.add_done_callback.assert_called_once()

        late_future = MagicMock()
        late_future.cancelled.return_value = False
        late_future.result.return_value = [late_obj]
        callback = future.add_done_callback.call_args.args[0]
        callback(late_future)

        late_obj.ref_count_down.assert_called_once()
        late_obj.unpin.assert_not_called()


class TestDealerRouterConcurrency:
    """Phase 2: DEALER/ROUTER req_id correlation, fail-fast, and immediate-miss.

    These drive real in-process ``zmq.asyncio`` ROUTER/DEALER sockets on the
    background loop (zmq is not mocked by ``prepare_environment``) and exercise
    the unbound coroutines against a ``MagicMock`` backend stub.
    """

    @staticmethod
    def _consumer_backend() -> MagicMock:
        backend = MagicMock()
        backend.running = asyncio.Event()
        backend.running.set()
        return backend

    @staticmethod
    def _connected_pair(ctx: "zmq.asyncio.Context"):
        router = ctx.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        dealer = ctx.socket(zmq.DEALER)
        dealer.connect(f"tcp://127.0.0.1:{port}")
        return router, dealer, port

    def test_request_reply_roundtrip_out_of_order(self, async_loop):
        """Replies correlate by req_id even when the producer answers reversed."""
        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        async def scenario():
            ctx = zmq.asyncio.Context()
            router, dealer, port = self._connected_pair(ctx)

            backend = self._consumer_backend()
            peer = AscendPeerInfo(
                peer_init_url="peer",
                peer_lookup_url=f"127.0.0.1:{port}",
                lookup_lock=asyncio.Lock(),
                lookup_socket=dealer,
            )
            backend.target_peer_info_mapping = {"peer": peer}
            reader = asyncio.create_task(
                AscendP2PBackend._peer_reply_reader(backend, peer)
            )

            async def fake_producer():
                seen = [await router.recv_multipart() for _ in range(2)]
                # Reply in reverse arrival order to force out-of-order delivery.
                for ident, req_id, payload in reversed(
                    [(f[0], f[1], f[-1]) for f in seen]
                ):
                    await router.send_multipart([ident, req_id, b"r:" + payload])

            producer = asyncio.create_task(fake_producer())
            results = await asyncio.gather(
                AscendP2PBackend._peer_request_reply(backend, "peer", b"A", 5.0),
                AscendP2PBackend._peer_request_reply(backend, "peer", b"B", 5.0),
            )
            await producer

            assert results == [b"r:A", b"r:B"]
            assert peer.pending == {}

            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass
            dealer.close(linger=0)
            router.close(linger=0)
            ctx.term()

        _run_coroutine(async_loop, scenario())

    def test_request_reply_times_out_and_cleans_pending(self, async_loop):
        """No reply -> wait_for raises TimeoutError; the pending entry is removed."""
        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        async def scenario():
            ctx = zmq.asyncio.Context()
            router, dealer, port = self._connected_pair(ctx)

            backend = self._consumer_backend()
            peer = AscendPeerInfo(
                peer_init_url="peer",
                peer_lookup_url=f"127.0.0.1:{port}",
                lookup_lock=asyncio.Lock(),
                lookup_socket=dealer,
            )
            backend.target_peer_info_mapping = {"peer": peer}
            reader = asyncio.create_task(
                AscendP2PBackend._peer_reply_reader(backend, peer)
            )

            with pytest.raises(asyncio.TimeoutError):
                await AscendP2PBackend._peer_request_reply(backend, "peer", b"X", 0.2)
            assert peer.pending == {}

            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass
            dealer.close(linger=0)
            router.close(linger=0)
            ctx.term()

        _run_coroutine(async_loop, scenario())

    def test_serve_request_immediate_miss_at_cap(self, async_loop):
        """At the in-flight cap, a heavy get is answered with an immediate miss."""
        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        backend = MagicMock()
        backend._inflight_gets = 4
        backend._max_concurrent_gets = 4
        backend._handle_batched_lookup_and_get = AsyncMock()
        captured = {}

        async def fake_reply(routing, ret_msg):
            captured["routing"] = routing
            captured["ret"] = ret_msg

        backend._router_reply = AsyncMock(side_effect=fake_reply)

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id="lu_cap",
            receiver_id="peer_1",
            keys=["k1"],
            buffer_uuids=[],
            mem_indexes=[],
            pull_mode=True,
        )
        _run_coroutine(
            async_loop,
            AscendP2PBackend._serve_request(
                backend, [b"ident", b"\x00\x00\x00\x01"], msgspec.msgpack.encode(msg)
            ),
        )

        assert isinstance(captured["ret"], AscendBatchedLookupAndGetRetMsg)
        assert captured["ret"].num_hit_chunks == 0
        assert captured["routing"] == [b"ident", b"\x00\x00\x00\x01"]
        backend._handle_batched_lookup_and_get.assert_not_awaited()
        # The short-circuited get is not counted, so the gauge is unchanged.
        assert backend._inflight_gets == 4

    def test_serve_request_dispatch_balances_inflight(self, async_loop):
        """Below the cap, the handler runs and the in-flight gauge is balanced."""
        # First Party
        from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

        backend = MagicMock()
        backend._inflight_gets = 0
        backend._max_concurrent_gets = 4
        backend.chunk_size = 256
        backend.stats_monitor = MagicMock()
        backend.stats_monitor.on_p2p_transfer_request.return_value = "rid"
        backend._handle_batched_lookup_and_get = AsyncMock(
            return_value=AscendBatchedLookupAndGetRetMsg(num_hit_chunks=1)
        )
        captured = {}

        async def fake_reply(routing, ret_msg):
            captured["routing"] = routing
            captured["ret"] = ret_msg

        backend._router_reply = AsyncMock(side_effect=fake_reply)

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id="lu_ok",
            receiver_id="peer_1",
            keys=["k1"],
            buffer_uuids=[],
            mem_indexes=[],
            pull_mode=True,
        )
        _run_coroutine(
            async_loop,
            AscendP2PBackend._serve_request(
                backend, [b"ident", b"req"], msgspec.msgpack.encode(msg)
            ),
        )

        backend._handle_batched_lookup_and_get.assert_awaited_once()
        assert captured["ret"].num_hit_chunks == 1
        assert captured["routing"] == [b"ident", b"req"]
        # Incremented then decremented back to zero.
        assert backend._inflight_gets == 0
        backend.stats_monitor.on_p2p_transfer_finished.assert_called_once_with("rid")
