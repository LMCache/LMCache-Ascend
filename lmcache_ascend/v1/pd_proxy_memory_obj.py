# SPDX-License-Identifier: Apache-2.0
"""
PDProxyMemoryObj & PDTransferContext: Deferred-fetch memory objects for PD
disaggregated KV cache transfer using pull mode.

In pull mode the prefiller (sender) pins its KV MemObjs and advertises
their HCCL buffer references.  The decoder (receiver) creates lightweight
PDProxyMemoryObj wrappers instead of pre-allocating NPU memory.  When the
NPU connector encounters these proxies during ``batched_to_gpu``, it
resolves the data (reads from the remote prefiller) and scatters it into
the paged KV cache in a pipelined fashion — identical to
:class:`ProxyMemoryObj` used in P2P.

This module mirrors the P2P variant in ``proxy_memory_obj.py`` but is
adapted for the synchronous, thread-based PD backend:

* ``PDTransferContext`` manages a pool of ping-pong buffers and sends
  a ZMQ Done signal (synchronously) to release the sender's pinned
  resources.
* ``PDProxyMemoryObj`` reuses :class:`ProxyMemoryObj` by composition,
  but the transfer-context lifecycle is driven by the PD protocol.
"""

# Standard
from typing import Any, List, Optional

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
import torch

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.transfer_context import AscendBaseTransferContext

logger = init_logger(__name__)


class PDTransferContext(AscendBaseTransferContext):
    """Shared context for a batch of PDProxyMemoryObjs from the same PD
    pull-mode transfer.

    Manages the lifecycle of a pull-mode transfer for PD, including:

    * Allocating/releasing ping-pong buffers from the *receiver's*
      registered HCCL memory pool.
    * Sending a ``DoneSignal`` ZMQ message to the *sender* so it can
      release its pinned resources. The signal is sent exactly once,
      triggered either by :meth:`send_done_now` (called from the NPU
      connector after all proxies have been scattered) or when the last
      proxy's ``ref_count_down()`` fires.

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
        Total number of :class:`PDProxyMemoryObj` instances that share
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

        logger.info(
            "PDTransferContext: sender_id=%s, num_proxies=%d, "
            "shapes=%s, dtypes=%s, fmt=%s",
            sender_id,
            num_proxies,
            shapes,
            dtypes,
            fmt,
        )

    # ──────────────────────────────────────────────────────────
    # Done-signal delivery
    # ──────────────────────────────────────────────────────────

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


class PDProxyMemoryObj(ProxyMemoryObj):
    """A pull-mode proxy for PD disaggregated KV cache transfer.

    Thin subclass of :class:`ProxyMemoryObj` that uses
    :class:`PDTransferContext` instead of ``P2PTransferContext``.

    The NPU connector detects these (via ``isinstance(obj, ProxyMemoryObj)``)
    and handles them identically to P2P proxies — the pipeline logic is
    shared.
    """

    def __init__(
        self,
        backing_obj: Optional[MemoryObj],
        transfer_channel: BaseTransferChannel,
        sender_id: str,
        remote_buffer_uuid: str,
        remote_mem_index: int,
        transfer_context: PDTransferContext,
        chunk_index: int,
        shapes: Optional[List[torch.Size]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
    ):
        # ProxyMemoryObj expects target_peer_init_url as the receiver_id
        # for the transfer_spec.  In PD pull mode the "peer" is the sender.
        super().__init__(
            backing_obj=backing_obj,
            transfer_channel=transfer_channel,
            target_peer_init_url=sender_id,
            remote_buffer_uuid=remote_buffer_uuid,
            remote_mem_index=remote_mem_index,
            transfer_context=transfer_context,
            chunk_index=chunk_index,
            shapes=shapes,
            dtypes=dtypes,
            fmt=fmt,
        )
