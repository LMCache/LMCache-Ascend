# SPDX-License-Identifier: Apache-2.0
"""
Transfer contexts for Ascend KV cache transfers.

Contains ``AscendBaseTransferContext`` (shared base) and its two concrete
subclasses:

* ``P2PTransferContext`` — async, event-loop-driven (P2P pull mode).
* ``PDTransferContext`` — synchronous, callback-driven (PD pull mode).

All three manage the same lifecycle:

1. Allocate / release ping-pong buffers from a registered memory pool.
2. Track a reference count across proxy memory objects.
3. Send a *Done* signal exactly once when all proxies are consumed.

Subclasses only need to implement :meth:`_send_done`.
"""

# Standard
from typing import Any, List, Optional
import asyncio
import threading

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
import torch

logger = init_logger(__name__)

_DEFAULT_PIPELINE_DEPTH = 4


class AscendBaseTransferContext:
    """Common base for P2P and PD transfer contexts.

    Manages:
    * Ping-pong buffer allocation / release from a
      :class:`PagedCpuGpuMemoryAllocator`.
    * Reference-counted Done-signal lifecycle — the signal is sent
      exactly once, either when the reference count reaches zero
      (via :meth:`decref`) or explicitly (via :meth:`send_done_now`).

    Subclasses **must** override :meth:`_send_done` to deliver the
    signal through the appropriate transport (asyncio / ZMQ callback /
    etc.).

    Parameters
    ----------
    num_proxies : int
        Number of proxy memory objects sharing this context.
    memory_allocator : PagedCpuGpuMemoryAllocator | None
        Memory allocator for ping-pong buffer management.
    shapes : list[torch.Size] | None
        Tensor shapes for buffer allocation.
    dtypes : list[torch.dtype] | None
        Tensor dtypes for buffer allocation.
    fmt : MemoryFormat
        Memory format for buffer allocation.
    """

    def __init__(
        self,
        num_proxies: int,
        memory_allocator: Any = None,
        shapes: Optional[List[torch.Size]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
    ):
        self._ref_count = num_proxies
        self._done_sent = False
        self._lock = threading.Lock()

        self._memory_allocator = memory_allocator
        self._shapes = shapes
        self._dtypes = dtypes
        self._fmt = fmt

    @property
    def _allocator_type(self) -> str:
        """Return ``"gpu"`` or ``"cpu"`` to select the backing allocator.

        Defaults to ``"gpu"``.  ``P2PTransferContext`` overrides this to
        honour its ``use_npu`` flag.
        """
        return "gpu"

    @property
    def max_pipeline_depth(self) -> int:
        """Max micro-batch size so two ping-pong pools fit in the
        registered buffer."""
        if self._memory_allocator is None:
            return _DEFAULT_PIPELINE_DEPTH

        allocator = (
            self._memory_allocator.gpu_allocator
            if self._allocator_type == "gpu"
            else self._memory_allocator.cpu_allocator
        )
        if allocator is None:
            return _DEFAULT_PIPELINE_DEPTH

        chunk_bytes = allocator.align_bytes
        buffer_bytes = allocator.buffer_size
        depth = buffer_bytes // (chunk_bytes * 2)
        return max(depth, 1)

    def allocate_buffers(self, count: int) -> List[MemoryObj]:
        """Allocate *count* scratch buffers from the registered memory pool."""
        assert self._memory_allocator is not None, (
            "memory_allocator not set on transfer context"
        )
        result = self._memory_allocator.batched_allocate(
            self._shapes, self._dtypes, count, self._fmt, self._allocator_type
        )
        if result is None:
            raise RuntimeError(
                f"Failed to allocate {count} ping-pong buffers "
                f"from {self._allocator_type} allocator"
            )
        return result

    def release_buffers(self, buffers: List[MemoryObj]) -> None:
        """Release scratch buffers back to the memory pool."""
        if not buffers:
            return
        assert self._memory_allocator is not None
        self._memory_allocator.batched_free(buffers, self._allocator_type)

    def decref(self) -> None:
        """Decrement reference count; sends Done when count reaches 0."""
        send_done = False
        with self._lock:
            self._ref_count -= 1
            if self._ref_count <= 0 and not self._done_sent:
                self._done_sent = True
                send_done = True
        if send_done:
            self._send_done()

    def send_done_now(self) -> None:
        """Force-send the Done signal immediately.  Idempotent."""
        with self._lock:
            if self._done_sent:
                return
            self._done_sent = True
        self._send_done()

    def _send_done(self) -> None:
        """Deliver the Done signal.  **Must be overridden by subclasses.**"""
        raise NotImplementedError("_send_done() must be implemented by subclasses")


class P2PTransferContext(AscendBaseTransferContext):
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
        target_peer_url: str,
        lookup_id: str,
        loop: asyncio.AbstractEventLoop,
        num_proxies: int,
        memory_allocator: Any = None,
        shapes: Optional[List[torch.Size]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        use_npu: bool = False,
    ):
        super().__init__(
            num_proxies=num_proxies,
            memory_allocator=memory_allocator,
            shapes=shapes,
            dtypes=dtypes,
            fmt=fmt,
        )
        self._p2p_backend = p2p_backend
        self._target_peer_url = target_peer_url
        self._lookup_id = lookup_id
        self._loop = loop
        self._use_npu = use_npu
        logger.info(
            f"Initialized P2PTransferContext: lookup_id={lookup_id}, "
            f"target_peer={target_peer_url}, "
            f"num_proxies={num_proxies}, use_npu={use_npu}, "
            f"shapes={shapes}, dtypes={dtypes}, fmt={fmt}"
        )

    @property
    def _allocator_type(self) -> str:
        return "gpu" if self._use_npu else "cpu"

    @property
    def lookup_id(self) -> str:
        return self._lookup_id

    @property
    def target_peer_url(self) -> str:
        return self._target_peer_url

    def _send_done(self) -> None:
        """Send the Done signal to the remote peer via the event loop."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._p2p_backend._send_done_signal(
                    self._lookup_id,
                    self._target_peer_url,
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


class PDTransferContext(AscendBaseTransferContext):
    """Shared context for a batch of ProxyMemoryObjs from the same PD
    pull-mode transfer.

    Manages the lifecycle of a pull-mode transfer for PD, including:

    * Allocating/releasing ping-pong buffers from the *receiver's*
      registered HCCL memory pool.
    * Sending a ``DoneSignal`` ZMQ message to the *sender* so it can
      release its pinned resources. The signal is sent exactly once,
      triggered by :meth:`send_done_now` (called from the NPU
      connector after all proxies have been scattered).

    Parameters
    ----------
    sender_id : str
        The sender's HCCL peer ID (used as ``receiver_id`` in
        ``transfer_spec`` for read operations).
    done_callback : callable
        A zero-argument callable that sends the Done signal to the sender.
        The PD backend supplies this so the context need not know about
        ZMQ sockets directly.
    num_proxies : int
        Total number of :class:`ProxyMemoryObj` instances that share
        this context.
    memory_allocator : PagedCpuGpuMemoryAllocator
        The receiver's memory allocator for ping-pong buffer allocation.
    shapes : list[torch.Size]
        Tensor shapes for buffer allocation.
    dtypes : list[torch.dtype]
        Tensor dtypes for buffer allocation.
    fmt : MemoryFormat
        Memory format for buffer allocation.
    """

    def __init__(
        self,
        sender_id: str,
        done_callback: Any,
        num_proxies: int,
        memory_allocator: Any,
        shapes: List[torch.Size],
        dtypes: List[torch.dtype],
        fmt: MemoryFormat,
    ):
        super().__init__(
            num_proxies=num_proxies,
            memory_allocator=memory_allocator,
            shapes=shapes,
            dtypes=dtypes,
            fmt=fmt,
        )
        self._sender_id = sender_id
        self._done_callback = done_callback
        self._loop = None

        logger.debug(
            "PDTransferContext: sender_id=%s, num_proxies=%d, "
            "shapes=%s, dtypes=%s, fmt=%s",
            sender_id,
            num_proxies,
            shapes,
            dtypes,
            fmt,
        )

    def _send_done(self) -> None:
        """Invoke the done callback to signal the sender."""
        try:
            self._done_callback()
        except Exception as e:
            logger.error(
                "Failed to send PD Done signal to sender %s: %s",
                self._sender_id,
                e,
            )
