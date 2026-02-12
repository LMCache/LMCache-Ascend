# SPDX-License-Identifier: Apache-2.0
"""
ProxyMemoryObj: A deferred-fetch memory object for P2P KV cache transfer.

Instead of fetching data from a remote peer during async_lookup_and_prefetch,
ProxyMemoryObj acts as a lightweight wrapper that carries P2P transfer metadata
through the cache engine unchanged. When the NPU connector encounters a
ProxyMemoryObj during batched_to_gpu, it resolves the data (fetches from remote)
and scatters it to the paged KV cache in a pipelined fashion.

This avoids the intermediate step of fetching all chunks to CPU/NPU memory
before scattering, enabling overlap of remote fetch and NPU scatter operations.
"""

# Standard
from typing import Any, List, Optional
import asyncio
import threading

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
import torch

logger = init_logger(__name__)


class P2PTransferContext:
    """Shared context for a batch of ProxyMemoryObjs from the same P2P lookup.

    Manages the lifecycle of a P2P pull-mode transfer, including sending the
    Done signal to the remote peer when all proxy objects have been consumed
    (either resolved and used, or released as unused).

    The Done signal tells the remote peer to release its pinned resources.
    It is sent exactly once, when all proxy objects' ref counts reach zero.
    """

    def __init__(
        self,
        p2p_backend: Any,
        target_peer_init_url: str,
        lookup_id: str,
        remote_buffer_uuids: List[str],
        remote_mem_indexes: List[int],
        loop: asyncio.AbstractEventLoop,
        num_proxies: int,
        memory_allocator: Any = None,
        shapes: Optional[List[torch.Size]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        use_npu: bool = False,
    ):
        self._p2p_backend = p2p_backend
        self._target_peer_init_url = target_peer_init_url
        self._lookup_id = lookup_id
        self._remote_buffer_uuids = remote_buffer_uuids
        self._remote_mem_indexes = remote_mem_indexes
        self._loop = loop
        self._ref_count = num_proxies
        self._done_sent = False
        self._lock = threading.Lock()

        # Buffer allocation metadata for ping-pong pipeline
        self._memory_allocator = memory_allocator
        self._shapes = shapes
        self._dtypes = dtypes
        self._fmt = fmt
        self._use_npu = use_npu
        logger.info(
            f"Initialized P2PTransferContext: lookup_id={lookup_id}, "
            f"target_peer={target_peer_init_url}, "
            f"num_buffer_refs={len(remote_buffer_uuids)}, "
            f"num_proxies={num_proxies}, use_npu={use_npu}, "
            f"shapes={shapes}, dtypes={dtypes}, fmt={fmt}"
        )

    @property
    def lookup_id(self) -> str:
        return self._lookup_id

    @property
    def target_peer_init_url(self) -> str:
        return self._target_peer_init_url

    @property
    def remote_buffer_uuids(self) -> List[str]:
        return self._remote_buffer_uuids

    @property
    def remote_mem_indexes(self) -> List[int]:
        return self._remote_mem_indexes

    @property
    def max_pipeline_depth(self) -> int:
        """Compute the max micro-batch size from the NPU buffer capacity.

        Returns ``npu_buffer_size // (chunk_size * 2)`` so that two
        ping-pong pools fit entirely inside the registered buffer,
        clamped to [1, ∞).  Falls back to a default of 4 when the
        allocator is unavailable or not using NPU memory.
        """
        _DEFAULT_PIPELINE_DEPTH = 4
        if self._memory_allocator is None:
            return _DEFAULT_PIPELINE_DEPTH

        allocator = (
            self._memory_allocator.gpu_allocator
            if self._use_npu
            else self._memory_allocator.cpu_allocator
        )
        chunk_bytes = allocator.align_bytes  # one chunk's physical size
        buffer_bytes = allocator.buffer_size
        # Two pools must fit in the buffer
        depth = buffer_bytes // (chunk_bytes * 2)
        return max(depth, 1)

    def allocate_buffers(self, count: int) -> List[MemoryObj]:
        """Allocate scratch buffers from the registered memory pool.

        Used by the NPU connector for ping-pong pipeline buffers.
        """
        assert self._memory_allocator is not None, (
            "memory_allocator not set on P2PTransferContext"
        )
        allocator_type = "gpu" if self._use_npu else "cpu"
        result = self._memory_allocator.batched_allocate(
            self._shapes, self._dtypes, count, self._fmt, allocator_type
        )
        if result is None:
            raise RuntimeError(
                f"Failed to allocate {count} ping-pong buffers "
                f"from {allocator_type} allocator"
            )
        return result

    def release_buffers(self, buffers: List[MemoryObj]) -> None:
        """Release scratch buffers back to the memory pool."""
        if not buffers:
            return
        assert self._memory_allocator is not None
        allocator_type = "gpu" if self._use_npu else "cpu"
        self._memory_allocator.batched_free(buffers, allocator_type)

    def decref(self) -> None:
        """Decrement reference count. Sends Done signal when count reaches 0."""
        send_done = False
        with self._lock:
            self._ref_count -= 1
            if self._ref_count <= 0 and not self._done_sent:
                self._done_sent = True
                send_done = True

        if send_done:
            self._send_done_async()

    def send_done_now(self) -> None:
        """Force send the Done signal immediately. Idempotent."""
        with self._lock:
            if self._done_sent:
                return
            self._done_sent = True
        self._send_done_async()

    def _send_done_async(self) -> None:
        """Send the Done signal to the remote peer via the event loop."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._p2p_backend._send_done_signal(
                    self._lookup_id,
                    self._target_peer_init_url,
                ),
                self._loop,
            )
            future.result(timeout=30)
        except Exception as e:
            logger.error(
                "Failed to send P2P Done signal for lookup_id %s: %s",
                self._lookup_id,
                e,
            )


class ProxyMemoryObj(MemoryObj):
    """A deferred-fetch memory object for P2P KV cache transfer.

    Can be created in two modes:
    - With a backing_obj: The backing MemoryObj is pre-allocated and will
      be filled upon resolve().
    - Without a backing_obj (lightweight): Only carries transfer metadata.
      A backing buffer is assigned later by the NPU connector's ping-pong
      pipeline via set_backing_obj() before data is fetched.

    The NPU connector detects ProxyMemoryObj instances and handles
    resolve + scatter in a pipelined fashion.
    """

    def __init__(
        self,
        backing_obj: Optional[MemoryObj],
        transfer_channel: BaseTransferChannel,
        target_peer_init_url: str,
        remote_buffer_uuid: str,
        remote_mem_index: int,
        transfer_context: P2PTransferContext,
        chunk_index: int,
        shapes: Optional[List[torch.Size]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
    ):
        """
        Args:
            backing_obj: Pre-allocated MemoryObj (CPU or NPU) where data
                will be read into upon resolve(). Can be None for
                lightweight mode (ping-pong pipeline).
            transfer_channel: The transfer channel (e.g., HCCL) to use
                for reading data from the remote peer.
            target_peer_init_url: The remote peer's init URL, used as the
                receiver_id for the transfer channel.
            remote_buffer_uuid: Opaque UUID identifying the remote buffer.
            remote_mem_index: Mem index within the remote buffer.
            transfer_context: Shared context managing the P2P transfer lifecycle.
            chunk_index: The index of this chunk within the batch.
            shapes: Tensor shapes (required when backing_obj is None).
            dtypes: Tensor dtypes (required when backing_obj is None).
            fmt: Memory format (required when backing_obj is None).
        """
        # Don't call super().__init__() since we manage meta ourselves
        self._backing_obj = backing_obj
        self._transfer_channel = transfer_channel
        self._target_peer_init_url = target_peer_init_url
        self._remote_buffer_uuid = remote_buffer_uuid
        self._remote_mem_index = remote_mem_index
        self._transfer_context = transfer_context
        self._chunk_index = chunk_index
        self._resolved = False

        # Store allocation metadata for deferred buffer operations
        if backing_obj is not None:
            self._shapes = backing_obj.get_shapes()
            self._dtypes = backing_obj.get_dtypes()
            self._fmt = backing_obj.get_memory_format()
        else:
            assert shapes is not None and dtypes is not None, (
                "shapes and dtypes are required when backing_obj is None"
            )
            self._shapes = shapes
            self._dtypes = dtypes
            self._fmt = fmt

        # Pre-compute size for get_size()
        self._size_bytes = sum(
            s.numel() * d.itemsize
            for s, d in zip(self._shapes, self._dtypes, strict=False)
        )

    @property
    def is_proxy(self) -> bool:
        """Identifying marker for ProxyMemoryObj."""
        return True

    @property
    def resolved(self) -> bool:
        """Whether the data has been fetched from remote."""
        return self._resolved

    @property
    def backing_obj(self) -> Optional[MemoryObj]:
        """The underlying allocated MemoryObj, or None if not yet assigned."""
        return self._backing_obj

    @property
    def transfer_context(self) -> P2PTransferContext:
        """The shared P2P transfer context."""
        return self._transfer_context

    def set_backing_obj(self, obj: MemoryObj) -> None:
        """Assign (or reassign) a backing buffer for this proxy.

        Called by the NPU connector's ping-pong pipeline before submitting
        a read. Resets the resolved flag so data will be fetched into
        the new buffer.
        """
        self._backing_obj = obj
        self._resolved = False

    def clear_backing_obj(self) -> None:
        """Remove the backing buffer reference.

        Called after scatter completes to release the buffer back to the
        ping-pong pool. Does NOT reset resolved.
        """
        self._backing_obj = None

    def resolve(self) -> None:
        """Fetch data from the remote peer into the backing memory object.

        Uses the transfer channel's async_batched_read via the event loop.
        This is a blocking call that waits for the read to complete.
        """
        if self._resolved:
            return

        assert self._backing_obj is not None, (
            "Cannot resolve: no backing buffer assigned. Call set_backing_obj() first."
        )

        channel_transfer_spec = {
            "receiver_id": self._target_peer_init_url,
            "remote_buffer_uuids": [self._remote_buffer_uuid],
            "remote_mem_indexes": [self._remote_mem_index],
        }

        future = asyncio.run_coroutine_threadsafe(
            self._transfer_channel.async_batched_read(
                buffers=[self._backing_obj],
                transfer_spec=channel_transfer_spec,
            ),
            self._transfer_context._loop,
        )
        future.result()
        self._resolved = True

    @staticmethod
    def resolve_batch(
        proxies: List["ProxyMemoryObj"],
    ) -> None:
        """Resolve a batch of ProxyMemoryObjs in a single transfer call.

        More efficient than resolving one at a time since it batches the
        RDMA/HCCL read operations.

        Args:
            proxies: List of ProxyMemoryObjs to resolve together. All must
                share the same transfer channel and target peer, and have
                backing buffers assigned.
        """
        unresolved = [p for p in proxies if not p._resolved]
        if not unresolved:
            return

        first = unresolved[0]
        buffers = []
        remote_buffer_uuids = []
        remote_mem_indexes = []
        for p in unresolved:
            assert p._backing_obj is not None, (
                "Cannot resolve: no backing buffer assigned"
            )
            buffers.append(p._backing_obj)
            remote_buffer_uuids.append(p._remote_buffer_uuid)
            remote_mem_indexes.append(p._remote_mem_index)

        channel_transfer_spec = {
            "receiver_id": first._target_peer_init_url,
            "remote_buffer_uuids": remote_buffer_uuids,
            "remote_mem_indexes": remote_mem_indexes,
        }

        future = asyncio.run_coroutine_threadsafe(
            first._transfer_channel.async_batched_read(
                buffers=buffers,
                transfer_spec=channel_transfer_spec,
            ),
            first._transfer_context._loop,
        )
        future.result()

        for p in unresolved:
            p._resolved = True

    @staticmethod
    def submit_resolve_batch(
        proxies: List["ProxyMemoryObj"],
    ) -> Optional[torch.npu.Event]:
        """Submit a batched read for proxies without waiting for completion.

        Unlike resolve_batch which blocks until data is fetched, this method
        submits read operations to the transport stream and returns an NPU
        event. The caller must wait on this event before accessing the
        backing objects' data, e.g. via load_stream.wait_event(event).

        Falls back to synchronous resolve_batch if the transfer channel
        does not support submit_batched_read.

        Args:
            proxies: List of ProxyMemoryObjs to resolve. All must share
                the same transfer channel and target peer, and have
                backing buffers assigned.

        Returns:
            An NPU event if submission was asynchronous, None if resolved
            synchronously (fallback) or all proxies were already resolved.
        """
        unresolved = [p for p in proxies if not p._resolved]
        if not unresolved:
            return None

        first = unresolved[0]
        channel = first._transfer_channel

        if not hasattr(channel, "submit_batched_read"):
            # Channel doesn't support non-blocking submission; fall back
            ProxyMemoryObj.resolve_batch(proxies)
            return None

        buffers = []
        remote_buffer_uuids = []
        remote_mem_indexes = []
        for p in unresolved:
            assert p._backing_obj is not None, (
                "Cannot resolve: no backing buffer assigned"
            )
            buffers.append(p._backing_obj)
            remote_buffer_uuids.append(p._remote_buffer_uuid)
            remote_mem_indexes.append(p._remote_mem_index)

        channel_transfer_spec = {
            "receiver_id": first._target_peer_init_url,
            "remote_buffer_uuids": remote_buffer_uuids,
            "remote_mem_indexes": remote_mem_indexes,
        }

        event = channel.submit_batched_read(
            buffers=buffers,
            transfer_spec=channel_transfer_spec,
        )

        for p in unresolved:
            p._resolved = True

        return event

    @property
    def raw_data(self) -> Any:
        assert self._backing_obj is not None, "No backing buffer assigned"
        return self._backing_obj.raw_data

    @property
    def meta(self) -> MemoryObjMetadata:
        if self._backing_obj is not None:
            return self._backing_obj.meta
        # Stub metadata for lightweight mode
        return MemoryObjMetadata(
            shape=self._shapes[0] if len(self._shapes) == 1 else torch.Size([0]),
            dtype=self._dtypes[0] if len(self._dtypes) == 1 else None,
            address=0,
            phy_size=self._size_bytes,
            ref_count=1,
            fmt=self._fmt,
            shapes=self._shapes,
            dtypes=self._dtypes,
        )

    @meta.setter
    def meta(self, value: MemoryObjMetadata) -> None:
        if self._backing_obj is not None:
            self._backing_obj.meta = value

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        assert self._backing_obj is not None, "No backing buffer assigned"
        return self._backing_obj.tensor

    @property
    def byte_array(self) -> bytes:
        assert self._backing_obj is not None, "No backing buffer assigned"
        return self._backing_obj.byte_array

    @property
    def data_ptr(self) -> int:
        assert self._backing_obj is not None, "No backing buffer assigned"
        return self._backing_obj.data_ptr

    @property
    def is_pinned(self) -> bool:
        if self._backing_obj is not None:
            return self._backing_obj.is_pinned
        return False

    @property
    def can_evict(self) -> bool:
        return False

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        if self._backing_obj is not None:
            return self._backing_obj.raw_tensor
        return None

    def invalidate(self) -> None:
        if self._backing_obj is not None:
            self._backing_obj.invalidate()

    def is_valid(self) -> bool:
        if self._backing_obj is not None:
            return self._backing_obj.is_valid()
        return True  # lightweight proxy is always "valid" as metadata carrier

    def get_size(self) -> int:
        return self._size_bytes

    def get_shape(self) -> torch.Size:
        if self._backing_obj is not None:
            return self._backing_obj.get_shape()
        return self._shapes[0] if len(self._shapes) == 1 else torch.Size([0])

    def get_dtype(self) -> Optional[torch.dtype]:
        if self._backing_obj is not None:
            return self._backing_obj.get_dtype()
        return self._dtypes[0] if len(self._dtypes) == 1 else None

    def get_shapes(self) -> list[torch.Size]:
        return self._shapes

    def get_dtypes(self) -> list[torch.dtype]:
        return self._dtypes

    def get_memory_format(self) -> MemoryFormat:
        return self._fmt

    def get_physical_size(self) -> int:
        if self._backing_obj is not None:
            return self._backing_obj.get_physical_size()
        return self._size_bytes

    def pin(self) -> bool:
        if self._backing_obj is not None:
            return self._backing_obj.pin()
        return False

    def unpin(self) -> bool:
        if self._backing_obj is not None:
            return self._backing_obj.unpin()
        return False

    def ref_count_up(self) -> None:
        if self._backing_obj is not None:
            self._backing_obj.ref_count_up()

    def ref_count_down(self) -> None:
        if self._backing_obj is not None:
            self._backing_obj.ref_count_down()
        self._transfer_context.decref()

    def get_ref_count(self) -> int:
        if self._backing_obj is not None:
            return self._backing_obj.get_ref_count()
        return 0

    def get_num_tokens(self) -> int:
        if self._backing_obj is not None:
            return self._backing_obj.get_num_tokens()
        # TODO (gingfung): depends on format
        if self._fmt == MemoryFormat.KV_2LTD:
            return self._shapes[0][2] if self._shapes else 0
        elif self._fmt == MemoryFormat.KV_2TD:
            return self._shapes[0][1] if self._shapes else 0
        elif self._fmt == MemoryFormat.KV_T2D:
            return self._shapes[0][0] if self._shapes else 0
        else:
            return 0

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        assert self._backing_obj is not None, "No backing buffer assigned"
        return self._backing_obj.get_tensor(index)
