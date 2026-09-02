# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Tests for XioBackend.

Unit tests use mocked connections (no real Xio/TCP endpoint required).
"""

# Standard
from unittest.mock import MagicMock, patch
import socket
import struct
import threading

# Third Party
import pytest

# First Party
# ---------------------------------------------------------------------------
# Conditional import: mock lmcache so the module can be imported without it.
# ---------------------------------------------------------------------------
_lmcache_mocked = False
import sys
if "lmcache" not in sys.modules:
    _mock_lmcache = MagicMock()
    _mock_lmcache.logging.init_logger.return_value = MagicMock()
    _mock_lmcache.utils.CacheEngineKey = type("CacheEngineKey", (), {})
    _mock_lmcache.v1.memory_management.MemoryObj = type("MemoryObj", (), {})

    class _StubSBI:
        def __init__(self, dst_device="cpu"):
            self.dst_device = dst_device

    _mock_lmcache.v1.storage_backend.abstract_backend.StorageBackendInterface = _StubSBI
    sys.modules["lmcache"] = _mock_lmcache
    sys.modules["lmcache.logging"] = _mock_lmcache.logging
    sys.modules["lmcache.utils"] = _mock_lmcache.utils
    sys.modules["lmcache.v1"] = _mock_lmcache.v1
    sys.modules["lmcache.v1.memory_management"] = _mock_lmcache.v1.memory_management
    sys.modules["lmcache.v1.storage_backend"] = _mock_lmcache.v1.storage_backend
    sys.modules["lmcache.v1.storage_backend.abstract_backend"] = _mock_lmcache.v1.storage_backend.abstract_backend

# Load xio_backend directly via importlib to avoid triggering the full
# lmcache_ascend package __init__.py (which requires torch_npu).
import importlib.util as _ilu
import os as _os

_xio_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "lmcache_ascend", "v1", "storage_backend", "xio_backend.py"))
_spec = _ilu.spec_from_file_location("xio_backend", _xio_path)
_xio_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_xio_mod)

XioBackend = _xio_mod.XioBackend
XioConnection = _xio_mod.XioConnection
_parse_xio_url = _xio_mod._parse_xio_url
HEADER_FMT = _xio_mod.HEADER_FMT
HEADER_SIZE = _xio_mod.HEADER_SIZE
MSG_CONTAINS = _xio_mod.MSG_CONTAINS
MSG_GET = _xio_mod.MSG_GET
MSG_PUT = _xio_mod.MSG_PUT
RESP_ERROR = _xio_mod.RESP_ERROR
RESP_HEADER_FMT = _xio_mod.RESP_HEADER_FMT
RESP_HEADER_SIZE = _xio_mod.RESP_HEADER_SIZE
RESP_NOT_FOUND = _xio_mod.RESP_NOT_FOUND
RESP_OK = _xio_mod.RESP_OK

MemoryObj = sys.modules["lmcache.v1.memory_management"].MemoryObj

# -- URL parsing tests --


class TestParseXioUrl:
    def test_basic_url(self):
        assert _parse_xio_url("xio://localhost:9876") == ("localhost", 9876)

    def test_tcp_scheme(self):
        assert _parse_xio_url("xio+tcp://10.0.0.1:5555") == ("10.0.0.1", 5555)

    def test_rdma_scheme(self):
        assert _parse_xio_url("xio+rdma://rdma-host:7777") == ("rdma-host", 7777)

    def test_no_scheme(self):
        assert _parse_xio_url("myhost:1234") == ("myhost", 1234)

    def test_missing_port_raises(self):
        with pytest.raises(ValueError, match="expected format"):
            _parse_xio_url("xio://localhost")

    def test_non_integer_port_raises(self):
        with pytest.raises(ValueError, match="port must be an integer"):
            _parse_xio_url("xio://localhost:abc")

    def test_ipv6_style_host(self):
        host, port = _parse_xio_url("xio://[::1]:9999")
        assert port == 9999


# -- XioConnection tests --


class TestXioConnection:
    def test_initial_state(self):
        conn = XioConnection("localhost", 9999)
        assert not conn.is_connected

    @patch("socket.socket")
    def test_connect_success(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        conn = XioConnection("localhost", 9999)
        assert conn.connect() is True
        assert conn.is_connected
        mock_sock.connect.assert_called_once_with(("localhost", 9999))
        mock_sock.setsockopt.assert_called_once()

    @patch("socket.socket")
    def test_connect_failure(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("Connection refused")
        mock_socket_cls.return_value = mock_sock

        conn = XioConnection("localhost", 9999, reconnect_interval=0.01)
        assert conn.connect() is False
        assert not conn.is_connected

    @patch("socket.socket")
    def test_reconnect_cooldown(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("Connection refused")
        mock_socket_cls.return_value = mock_sock

        conn = XioConnection("localhost", 9999, reconnect_interval=10.0)
        assert conn.connect() is False

        # Immediate retry should be blocked by cooldown
        mock_sock.connect.side_effect = None  # would succeed now
        mock_socket_cls.return_value = MagicMock()
        assert conn.connect() is False

    @patch("socket.socket")
    def test_close(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        conn = XioConnection("localhost", 9999)
        conn.connect()
        conn.close()
        assert not conn.is_connected
        mock_sock.close.assert_called_once()

    def test_close_when_not_connected(self):
        conn = XioConnection("localhost", 9999)
        conn.close()  # should not raise
        assert not conn.is_connected

    @patch("socket.socket")
    def test_put_success(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        resp_header = struct.pack(RESP_HEADER_FMT, RESP_OK, 0)
        mock_sock.recv.return_value = resp_header

        conn = XioConnection("localhost", 9999)
        conn.connect()
        result = conn.put(b"test_key", b"test_value")
        assert result is True
        mock_sock.sendall.assert_called_once()

    @patch("socket.socket")
    def test_put_when_disconnected(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("fail")
        mock_socket_cls.return_value = mock_sock

        conn = XioConnection("localhost", 9999, reconnect_interval=0.01)
        result = conn.put(b"key", b"val")
        assert result is False

    @patch("socket.socket")
    def test_get_success(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        payload = b"cached_data"
        resp_header = struct.pack(RESP_HEADER_FMT, RESP_OK, len(payload))
        mock_sock.recv.side_effect = [resp_header, payload]

        conn = XioConnection("localhost", 9999)
        conn.connect()
        result = conn.get(b"test_key")
        assert result == payload

    @patch("socket.socket")
    def test_get_not_found(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        resp_header = struct.pack(RESP_HEADER_FMT, RESP_NOT_FOUND, 0)
        mock_sock.recv.return_value = resp_header

        conn = XioConnection("localhost", 9999)
        conn.connect()
        result = conn.get(b"missing_key")
        assert result is None

    @patch("socket.socket")
    def test_get_when_disconnected(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("fail")
        mock_socket_cls.return_value = mock_sock

        conn = XioConnection("localhost", 9999, reconnect_interval=0.01)
        result = conn.get(b"key")
        assert result is None

    @patch("socket.socket")
    def test_exists_true(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        resp_header = struct.pack(RESP_HEADER_FMT, RESP_OK, 1)
        mock_sock.recv.side_effect = [resp_header, b"\x01"]

        conn = XioConnection("localhost", 9999)
        conn.connect()
        assert conn.exists(b"test_key") is True

    @patch("socket.socket")
    def test_exists_false(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        resp_header = struct.pack(RESP_HEADER_FMT, RESP_OK, 1)
        mock_sock.recv.side_effect = [resp_header, b"\x00"]

        conn = XioConnection("localhost", 9999)
        conn.connect()
        assert conn.exists(b"test_key") is False

    @patch("socket.socket")
    def test_send_failure_closes_connection(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.sendall.side_effect = OSError("Broken pipe")

        conn = XioConnection("localhost", 9999)
        conn.connect()
        assert conn.is_connected

        result = conn.put(b"key", b"val")
        assert result is False
        assert not conn.is_connected

    @patch("socket.socket")
    def test_recv_failure_closes_connection(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recv.side_effect = OSError("Connection reset")

        conn = XioConnection("localhost", 9999)
        conn.connect()

        result = conn.get(b"key")
        assert result is None
        assert not conn.is_connected

    @patch("socket.socket")
    def test_recv_eof_closes_connection(self, mock_socket_cls):
        """Empty recv (peer closed) should close the connection."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recv.return_value = b""  # EOF

        conn = XioConnection("localhost", 9999)
        conn.connect()

        result = conn.get(b"key")
        assert result is None
        assert not conn.is_connected


# -- XioBackend tests --


def _make_mock_key(key_id: str = "test_key"):
    """Create a mock CacheEngineKey."""
    mock = MagicMock()
    mock.to_string.return_value = f"model::2::0::{hash(key_id)}::bfloat16::None"
    mock.__eq__ = lambda self, other: (
        self.to_string() == other.to_string()
        if hasattr(other, "to_string")
        else NotImplemented
    )
    mock.__hash__ = lambda self: hash(self.to_string())
    return mock


def _make_mock_mem_obj():
    """Create a mock MemoryObj with a minimal tensor interface."""
    import torch

    mock = MagicMock(spec=MemoryObj)
    tensor = torch.randn(2, 2, 4, 8)
    mock.tensor = tensor
    mock.ref_count_down = MagicMock()
    mock.ref_count_up = MagicMock()
    return mock


class TestXioBackend:
    def test_str(self):
        with patch.object(XioConnection, "connect", return_value=False):
            backend = XioBackend("xio://localhost:9999")
        assert str(backend) == "XioBackend"

    def test_contains_delegates_to_connection(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.exists.return_value = True

        key = _make_mock_key("k1")
        assert backend.contains(key) is True
        backend._connection.exists.assert_called_once()

    def test_contains_returns_false_when_closed(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._closed = True

        key = _make_mock_key("k1")
        assert backend.contains(key) is False
        backend._connection.exists.assert_not_called()

    def test_contains_with_pin_still_works(self):
        """pin parameter is accepted but has no special behavior for Xio."""
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.exists.return_value = False

        key = _make_mock_key("k1")
        assert backend.contains(key, pin=True) is False

    def test_exists_in_put_tasks(self):
        backend = XioBackend("xio://localhost:9999")
        key = _make_mock_key("k1")

        assert backend.exists_in_put_tasks(key) is False

        with backend._put_tasks_lock:
            backend._put_tasks.add(key)

        assert backend.exists_in_put_tasks(key) is True

    def test_batched_submit_put_task(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.put.return_value = True

        key = _make_mock_key("k1")
        obj = _make_mock_mem_obj()

        result = backend.batched_submit_put_task([key], [obj])
        assert result is None
        backend._connection.put.assert_called_once()

        # put_tasks should be cleared after completion
        assert key not in backend._put_tasks

    def test_batched_submit_put_task_with_callback(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.put.return_value = True

        key = _make_mock_key("k1")
        obj = _make_mock_mem_obj()
        callback = MagicMock()

        backend.batched_submit_put_task(
            [key], [obj], on_complete_callback=callback
        )
        callback.assert_called_once_with(key)

    def test_batched_submit_put_task_callback_not_called_on_failure(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.put.return_value = False

        key = _make_mock_key("k1")
        obj = _make_mock_mem_obj()
        callback = MagicMock()

        backend.batched_submit_put_task(
            [key], [obj], on_complete_callback=callback
        )
        callback.assert_not_called()

    def test_batched_submit_put_task_when_closed(self):
        backend = XioBackend("xio://localhost:9999")
        backend._closed = True
        backend._connection = MagicMock()

        key = _make_mock_key("k1")
        obj = _make_mock_mem_obj()
        result = backend.batched_submit_put_task([key], [obj])
        assert result is None
        backend._connection.put.assert_not_called()

    def test_batched_submit_put_task_multiple_keys(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.put.return_value = True

        keys = [_make_mock_key(f"k{i}") for i in range(3)]
        objs = [_make_mock_mem_obj() for _ in range(3)]

        backend.batched_submit_put_task(keys, objs)
        assert backend._connection.put.call_count == 3

    def test_get_blocking_delegates_to_connection(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.get.return_value = b"some_data"

        key = _make_mock_key("k1")
        # _deserialize_memory_obj returns None in the stub
        result = backend.get_blocking(key)
        assert result is None
        backend._connection.get.assert_called_once()

    def test_get_blocking_returns_none_when_not_found(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.get.return_value = None

        key = _make_mock_key("k1")
        result = backend.get_blocking(key)
        assert result is None

    def test_get_blocking_returns_none_when_closed(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._closed = True

        key = _make_mock_key("k1")
        result = backend.get_blocking(key)
        assert result is None
        backend._connection.get.assert_not_called()

    def test_batched_get_blocking(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.get.return_value = None

        keys = [_make_mock_key(f"k{i}") for i in range(3)]
        results = backend.batched_get_blocking(keys)
        assert len(results) == 3
        assert all(r is None for r in results)

    def test_close(self):
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()

        backend.close()
        assert backend._closed is True
        backend._connection.close.assert_called_once()

    def test_touch_cache_is_noop(self):
        backend = XioBackend("xio://localhost:9999")
        backend.touch_cache()  # should not raise

    def test_serialize_memory_obj(self):
        import torch

        obj = MagicMock()
        tensor = torch.ones(2, 3, dtype=torch.float32)
        obj.tensor = tensor

        data = XioBackend._serialize_memory_obj(obj)
        assert isinstance(data, bytes)
        
        # Test full roundtrip serde preserves tensor
        restored_obj = XioBackend._deserialize_memory_obj(data)
        assert restored_obj is not None
        assert torch.equal(restored_obj.tensor, tensor)

    def test_serialize_non_contiguous_tensor(self):
        import torch

        obj = MagicMock()
        tensor = torch.ones(4, 4, dtype=torch.float32)
        # transpose makes it non-contiguous
        obj.tensor = tensor.t()

        data = XioBackend._serialize_memory_obj(obj)
        assert isinstance(data, bytes)
        
        # Test full roundtrip serde preserves tensor
        restored_obj = XioBackend._deserialize_memory_obj(data)
        assert restored_obj is not None
        assert torch.equal(restored_obj.tensor, obj.tensor.contiguous())

    def test_put_tasks_tracking_during_put(self):
        """put_tasks is populated during put and cleared after."""
        backend = XioBackend("xio://localhost:9999")

        put_was_in_tasks = []

        def mock_put(key_bytes, val_bytes):
            k = _make_mock_key("k1")
            put_was_in_tasks.append(backend.exists_in_put_tasks(k))
            return True

        backend._connection = MagicMock()
        backend._connection.put.side_effect = mock_put

        key = _make_mock_key("k1")
        obj = _make_mock_mem_obj()
        backend.batched_submit_put_task([key], [obj])

        assert put_was_in_tasks == [True]
        assert not backend.exists_in_put_tasks(key)

    def test_put_tasks_cleared_on_exception(self):
        """put_tasks entry is removed even when serialization raises."""
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()

        key = _make_mock_key("k1")
        obj = MagicMock(spec=MemoryObj)
        obj.tensor = MagicMock()
        obj.tensor.cpu.side_effect = RuntimeError("device error")

        backend.batched_submit_put_task([key], [obj])
        assert not backend.exists_in_put_tasks(key)


class TestXioBackendConcurrency:
    def test_concurrent_puts(self):
        """Multiple threads can put concurrently without data races."""
        backend = XioBackend("xio://localhost:9999")
        backend._connection = MagicMock()
        backend._connection.put.return_value = True

        errors = []

        def do_put(i):
            try:
                key = _make_mock_key(f"concurrent_{i}")
                obj = _make_mock_mem_obj()
                backend.batched_submit_put_task([key], [obj])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_put, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert backend._connection.put.call_count == 10
        assert len(backend._put_tasks) == 0


# -- Wire protocol tests --


class TestWireProtocol:
    def test_put_message_format(self):
        """Verify the PUT wire format: [type][key_len][val_len][key][val]."""
        key = b"test_key"
        value = b"test_value"
        header = struct.pack(HEADER_FMT, MSG_PUT, len(key), len(value))
        message = header + key + value

        msg_type, key_len, val_len = struct.unpack(
            HEADER_FMT, message[:HEADER_SIZE]
        )
        assert msg_type == MSG_PUT
        assert key_len == len(key)
        assert val_len == len(value)
        assert message[HEADER_SIZE : HEADER_SIZE + key_len] == key
        assert message[HEADER_SIZE + key_len :] == value

    def test_get_message_format(self):
        key = b"test_key"
        header = struct.pack(HEADER_FMT, MSG_GET, len(key), 0)
        message = header + key

        msg_type, key_len, val_len = struct.unpack(
            HEADER_FMT, message[:HEADER_SIZE]
        )
        assert msg_type == MSG_GET
        assert key_len == len(key)
        assert val_len == 0

    def test_contains_message_format(self):
        key = b"test_key"
        header = struct.pack(HEADER_FMT, MSG_CONTAINS, len(key), 0)
        message = header + key

        msg_type, key_len, val_len = struct.unpack(
            HEADER_FMT, message[:HEADER_SIZE]
        )
        assert msg_type == MSG_CONTAINS

    def test_response_ok_format(self):
        payload = b"response_data"
        header = struct.pack(RESP_HEADER_FMT, RESP_OK, len(payload))
        message = header + payload

        resp_type, payload_len = struct.unpack(
            RESP_HEADER_FMT, message[:RESP_HEADER_SIZE]
        )
        assert resp_type == RESP_OK
        assert payload_len == len(payload)
        assert message[RESP_HEADER_SIZE:] == payload

    def test_response_not_found_format(self):
        header = struct.pack(RESP_HEADER_FMT, RESP_NOT_FOUND, 0)
        resp_type, payload_len = struct.unpack(RESP_HEADER_FMT, header)
        assert resp_type == RESP_NOT_FOUND
        assert payload_len == 0

    def test_response_error_format(self):
        header = struct.pack(RESP_HEADER_FMT, RESP_ERROR, 0)
        resp_type, payload_len = struct.unpack(RESP_HEADER_FMT, header)
        assert resp_type == RESP_ERROR


# -- Integration-style test with real TCP sockets --


class TestXioConnectionIntegration:
    """End-to-end test with a real TCP echo-style server."""

    def test_put_get_roundtrip_with_real_sockets(self):
        """Full roundtrip using real TCP sockets and a mock Xio server."""
        server_ready = threading.Event()
        server_port = [0]

        def mock_xio_server():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            server_port[0] = srv.getsockname()[1]
            srv.listen(1)
            server_ready.set()

            conn, _ = srv.accept()
            store = {}

            try:
                while True:
                    hdr = b""
                    while len(hdr) < HEADER_SIZE:
                        chunk = conn.recv(HEADER_SIZE - len(hdr))
                        if not chunk:
                            return
                        hdr += chunk

                    msg_type, key_len, val_len = struct.unpack(HEADER_FMT, hdr)

                    key = b""
                    while len(key) < key_len:
                        key += conn.recv(key_len - len(key))

                    value = b""
                    while len(value) < val_len:
                        value += conn.recv(val_len - len(value))

                    if msg_type == MSG_PUT:
                        store[key] = value
                        resp = struct.pack(RESP_HEADER_FMT, RESP_OK, 0)
                        conn.sendall(resp)
                    elif msg_type == MSG_GET:
                        if key in store:
                            data = store[key]
                            resp = struct.pack(
                                RESP_HEADER_FMT, RESP_OK, len(data)
                            )
                            conn.sendall(resp + data)
                        else:
                            resp = struct.pack(
                                RESP_HEADER_FMT, RESP_NOT_FOUND, 0
                            )
                            conn.sendall(resp)
                    elif msg_type == MSG_CONTAINS:
                        exists = b"\x01" if key in store else b"\x00"
                        resp = struct.pack(RESP_HEADER_FMT, RESP_OK, 1)
                        conn.sendall(resp + exists)
            except OSError:
                pass
            finally:
                conn.close()
                srv.close()

        server_thread = threading.Thread(
            target=mock_xio_server, daemon=True
        )
        server_thread.start()
        assert server_ready.wait(timeout=5)

        conn = XioConnection(
            "127.0.0.1", server_port[0], connect_timeout=2.0
        )

        # PUT
        assert conn.put(b"key1", b"value1") is True

        # CONTAINS (exists)
        assert conn.exists(b"key1") is True

        # CONTAINS (not exists)
        assert conn.exists(b"key2") is False

        # GET (found)
        result = conn.get(b"key1")
        assert result == b"value1"

        # GET (not found)
        result = conn.get(b"key_missing")
        assert result is None

        # Multiple PUTs
        assert conn.put(b"key2", b"value2") is True
        assert conn.put(b"key3", b"value3") is True
        assert conn.get(b"key2") == b"value2"
        assert conn.get(b"key3") == b"value3"

        conn.close()


class TestXioBackendFactory:
    """Test that the backend factory correctly handles Xio config."""

    @pytest.mark.skip(reason="Cannot be tested in isolated environment without full LMCache/NPU framework")
    def test_xio_url_required_when_enabled(self):
        """enable_xio=True without xio_url should raise ValueError."""
        config = MagicMock()
        config.enable_xio = True
        config.xio_url = None
        config.enable_pd = False
        config.enable_p2p = False
        config.local_cpu = True
        config.local_disk = False
        config.max_local_cpu_size = 0
        config.remote_storage_plugins = None
        config.remote_url = None
        config.extra_config = None
        config.use_layerwise = False

        metadata = MagicMock()
        metadata.role = "scheduler"

        loop = MagicMock()

        with pytest.raises(ValueError, match="xio_url"):
            from lmcache_ascend.v1.storage_backend import CreateStorageBackends

            CreateStorageBackends(config, metadata, loop)
