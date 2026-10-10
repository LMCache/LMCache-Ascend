# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Tests for AscendPDBackend and its sender/receiver mixins.

Unit tests use mocks (no NPU required).  Integration tests require NPU
hardware and are gated behind ``@pytest.mark.skipif``.
"""

# Standard
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import MagicMock, patch
import ctypes
import threading
import time

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.pd_backend import AllocRequest
import msgspec
import numpy as np
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

# HCCL/HIXL require the KV transfer buffer VA to be 2MB aligned.
ALIGN_2MB = 2 * 1024 * 1024


def _make_controlled_raw_buf(size_bytes: int, aligned: bool) -> torch.Tensor:
    """Create a CPU tensor of *size_bytes* with a controllable 2MB alignment.

    The tensor shares memory with a ctypes buffer that it keeps alive, so
    the caller does not need to hold a separate reference.

    Args:
        size_bytes: Size of the returned tensor in bytes.
        aligned: If True, ``data_ptr`` lands exactly on a 2MB boundary; if
            False, it lands 1 byte past one (initially misaligned).

    Returns:
        A uint8 CPU tensor of *size_bytes* bytes.
    """
    base = ctypes.create_string_buffer(size_bytes + ALIGN_2MB)
    base_addr = ctypes.addressof(base)
    aligned_offset = (ALIGN_2MB - (base_addr % ALIGN_2MB)) % ALIGN_2MB
    offset = aligned_offset if aligned else aligned_offset + 1
    arr = np.frombuffer(base, dtype=np.uint8, count=size_bytes, offset=offset)
    tensor = torch.from_numpy(arr)
    assert tensor.data_ptr() % ALIGN_2MB == (0 if aligned else 1)
    return tensor


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
        shapes=[DEFAULT_SHAPE],
        dtypes=[DEFAULT_DTYPE],
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
    kv_shape: Tuple[int, ...] = DEFAULT_SHAPE,
    kv_dtype: torch.dtype = DEFAULT_DTYPE,
):
    """Create a mock object with the minimal attributes needed by PD backend methods."""
    # First Party
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
    backend._kv_shapes = [DEFAULT_SHAPE]
    backend._kv_dtypes = [kv_dtype]

    # Wire internal delegation methods to their real implementations so tests
    # that call e.g. AscendPDBackend.contains(backend, ...) actually exercise
    # the eviction / partition logic instead of hitting auto-mocked no-ops.
    backend._lookup = lambda key, pin=False: AscendPDBackend._lookup(
        backend, key, pin=pin
    )
    backend._contains_and_pin = lambda key: AscendPDBackend._contains_and_pin(
        backend, key
    )
    backend._partition_keys = lambda keys: AscendPDBackend._partition_keys(
        backend, keys
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

    @pytest.mark.parametrize(
        "aligned", [True, False], ids=["initially-aligned", "initially-unaligned"]
    )
    def test_initialize_allocator_npu_buffer_2mb_aligned(self, aligned):
        """PD NPU buffer is 2MB aligned; size rounds up to whole KV pages.

        Regression test for the 2MB-aligned PD NPU buffer allocation in
        ``AscendPDBackend.initialize_allocator`` (HCCL/HIXL require a 2MB
        aligned VA for buffer registration).  Runs on CPU only: the NPU
        allocation and device operations are mocked, while the real
        ``PagedTensorMemoryAllocator`` is exercised.
        """
        # First Party
        from lmcache_ascend.v1.storage_backend.pd import backend as pd_backend_module
        from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend

        kv_shape = DEFAULT_SHAPE  # [2, 2, 256, 512]
        kv_dtype = DEFAULT_DTYPE  # bfloat16
        page_size = get_size_bytes([kv_shape], [kv_dtype])
        assert page_size == ALIGN_2MB // 2  # 1 MiB per page

        # 1.5 MiB request -> rounded up to 2 whole pages (2 MiB usable)
        requested = 3 * page_size // 2
        expected_size = 2 * page_size

        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=kv_dtype,
            kv_shape=tuple(kv_shape),
        )
        config = SimpleNamespace(pd_buffer_size=requested, pd_cpu_buffer_size=0)

        backend = AscendPDBackend.__new__(AscendPDBackend)
        backend.pd_config = SimpleNamespace(buffer_device="npu:0")
        backend.use_cpu_offload = False

        total_alloc_size = expected_size + ALIGN_2MB
        raw_tensor = _make_controlled_raw_buf(total_alloc_size, aligned=aligned)
        raw_addr = raw_tensor.data_ptr()

        requested_sizes = []

        def fake_empty(size, **kwargs):
            requested_sizes.append(size)
            return raw_tensor

        with (
            patch.object(pd_backend_module.torch.npu, "set_device"),
            patch.object(pd_backend_module.torch, "empty", side_effect=fake_empty),
        ):
            alloc = backend.initialize_allocator(config, metadata)

        # 1. Base is 2MB aligned and usable size is rounded up to whole pages
        gpu = alloc.gpu_allocator
        aligned_addr = (raw_addr + ALIGN_2MB - 1) & ~(ALIGN_2MB - 1)
        assert requested_sizes == [total_alloc_size]
        assert gpu.buffer_ptr == aligned_addr
        assert gpu.buffer_ptr % ALIGN_2MB == 0
        assert gpu.buffer_size == expected_size
        assert alloc._npu_raw_buf is raw_tensor  # raw storage kept alive

        n_pages = gpu.buffer_size // page_size
        assert n_pages == 2
        assert len(gpu.free_blocks) == n_pages

        # 2. All pages allocate; addresses stay in the registered range and
        #    are unique (the transfer channel registers
        #    [gpu_allocator.buffer_ptr, buffer_ptr + buffer_size)).
        objs = []
        for _ in range(n_pages):
            obj = gpu.allocate([kv_shape], [kv_dtype], MemoryFormat.KV_2LTD)
            assert obj is not None
            objs.append(obj)
        addrs = [obj.raw_data.data_ptr() for obj in objs]
        assert len(set(addrs)) == n_pages
        for addr, obj in zip(addrs, objs, strict=True):
            assert gpu.buffer_ptr <= addr < gpu.buffer_ptr + gpu.buffer_size
            assert addr == gpu.buffer_ptr + obj.meta.address * page_size
        # Buffer is exhausted after all pages are allocated
        assert gpu.allocate([kv_shape], [kv_dtype], MemoryFormat.KV_2LTD) is None

        # 3. Freeing and re-allocating preserves the same properties
        for obj in objs:
            gpu.free(obj)
        assert len(gpu.free_blocks) == n_pages
        re_addrs = []
        for _ in range(n_pages):
            obj = gpu.allocate([kv_shape], [kv_dtype], MemoryFormat.KV_2LTD)
            assert obj is not None
            re_addrs.append(obj.raw_data.data_ptr())
        assert len(set(re_addrs)) == n_pages
        for addr in re_addrs:
            assert gpu.buffer_ptr <= addr < gpu.buffer_ptr + gpu.buffer_size

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

    def test_pull_eager_flow(self):
        """Pull-eager: allocates, reads from sender, returns ack + callback."""
        # First Party
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
        backend.put.assert_called_once()

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
        backend.put = MagicMock()
        backend._send_pull_done_to_sender = MagicMock()

        msg = PullReadyNotif(
            pull_id="pull_delay_done_once",
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
        assert backend.put.call_count == 1

        proxy_obj = backend.put.call_args.args[1]
        assert isinstance(proxy_obj, ProxyMemoryObj)

        transfer_ctx = proxy_obj.transfer_context
        transfer_ctx.send_done_now()
        transfer_ctx.send_done_now()
        backend._send_pull_done_to_sender.assert_called_once_with(
            "sender_1", "pull_delay_done_once"
        )

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
