# SPDX-License-Identifier: Apache-2.0
"""
AscendBaseTransferContext: shared base for P2P and PD transfer contexts.

Both ``P2PTransferContext`` (async, event-loop-driven) and
``PDTransferContext`` (synchronous, callback-driven) manage the same
lifecycle:

1. Allocate / release ping-pong buffers from a registered memory pool.
2. Track a reference count across proxy memory objects.
3. Send a *Done* signal exactly once when all proxies are consumed.

This module extracts that common logic so subclasses only need to
implement :meth:`_send_done`.
"""

# Standard
from typing import Any, List, Optional
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
