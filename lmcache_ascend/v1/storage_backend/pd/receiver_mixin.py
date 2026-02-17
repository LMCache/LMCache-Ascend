# SPDX-License-Identifier: Apache-2.0
"""Receiver (decoder) mixin for the Ascend PD backend.

Contains all receiver-side methods: push-mode allocation, pull-mode
eager/delay handlers, done-signal sending, and the alloc message loop.
"""

# Standard
from typing import Callable, Optional
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    CacheEngineKey,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.rpc_utils import get_zmq_socket
from lmcache.v1.storage_backend.pd_backend import AllocRequest
import msgspec
import torch
import torch_npu  # noqa: F401
import zmq

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.storage_backend.pd.messages import (
    AscendAllocResponse,
    AscendPDMsg,
    PullDoneSignal,
    PullReadyDoneAck,
    PullReadyNotif,
)
from lmcache_ascend.v1.storage_backend.utils import (
    adjust_last_chunk_shape,
    allocate_with_retry,
    build_channel_transfer_spec,
    release_memory_objects,
)
from lmcache_ascend.v1.transfer_context import PDTransferContext

logger = init_logger(__name__)


class AscendPDReceiverMixin:
    """Mixin providing receiver/decoder methods for :class:`AscendPDBackend`.

    This mixin is not intended to be instantiated directly.  It relies on
    attributes initialised by ``AscendPDBackend.__init__`` (e.g.
    ``self.transfer_channel``, ``self.memory_allocator``, ``self.data``).
    """

    # ──────────────────────────────────────────────────────────
    # Receiver initialisation
    # ──────────────────────────────────────────────────────────

    def _init_receiver(self):
        """Extend receiver init with done-socket URL tracking for pull mode."""
        super()._init_receiver()

        if self.pull_mode:
            # Mapping from sender_id -> done URL so we can send
            # PullDoneSignal back to the correct sender.
            self._sender_done_urls: dict[str, str] = {}
            self._pull_done_sockets: dict[str, zmq.Socket] = {}

    # ──────────────────────────────────────────────────────────
    # Push-mode allocation handler
    # ──────────────────────────────────────────────────────────

    def _allocate_and_put(self, alloc_request: AllocRequest) -> AscendAllocResponse:
        """Allocate memory for incoming chunks and return UUID-based refs.

        Used in **push mode** only.  The receiver pre-allocates NPU pages
        and returns their HCCL buffer references so the sender can write.
        """
        total_allocs = len(alloc_request.keys)
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = list(alloc_request.shape)

        already_sent_indexes, already_sent_objs, new_indexes = self._partition_keys(
            alloc_request.keys
        )

        remote_buffer_uuids: list[str] = []
        remote_mem_indexes: list[int] = []
        allocated_keys: list[CacheEngineKey] = []
        allocated_objs: list[MemoryObj] = []
        for idx in new_indexes:
            key = CacheEngineKey.from_string(alloc_request.keys[idx])

            alloc_shape = adjust_last_chunk_shape(
                shape,
                idx,
                total_allocs,
                fmt,
                alloc_request.last_chunk_toks,
            )

            mem_obj = allocate_with_retry(
                self.allocate,
                torch.Size(alloc_shape),
                dtype,
                fmt,
            )

            if mem_obj is None:
                # Allocation timed out — undo already-stored chunks and
                # report failure to the sender.
                logger.error(
                    "Push-mode: allocation failed at chunk %d/%d. "
                    "Releasing %d already-allocated objects.",
                    idx,
                    total_allocs,
                    len(allocated_objs),
                )
                with self.data_lock:
                    for k in allocated_keys:
                        self.data.pop(k, None)
                release_memory_objects(allocated_objs + already_sent_objs)
                return AscendAllocResponse(
                    already_sent_indexes=already_sent_indexes,
                    remote_buffer_uuids=[],
                    remote_indexes=[],
                    alloc_failed=True,
                )

            buf_uuid, mem_idx = self.transfer_channel.get_local_buffer_refs([mem_obj])
            remote_buffer_uuids.append(buf_uuid[0])
            remote_mem_indexes.append(mem_idx[0])

            self.put(key, mem_obj)
            allocated_keys.append(key)
            allocated_objs.append(mem_obj)

        release_memory_objects(already_sent_objs)

        return AscendAllocResponse(
            already_sent_indexes=already_sent_indexes,
            remote_buffer_uuids=remote_buffer_uuids,
            remote_indexes=remote_mem_indexes,
        )

    # ──────────────────────────────────────────────────────────
    # Pull-mode receiver handlers
    # ──────────────────────────────────────────────────────────

    def _handle_pull_ready(
        self, msg: PullReadyNotif, sender_id: str
    ) -> tuple[PullReadyDoneAck, Optional[Callable]]:
        """Handle a ``PullReadyNotif`` from the sender in **pull mode**.

        Returns ``(ack, post_ack_callback_or_None)``.  The caller must
        send the ack on the REP socket first, then invoke the callback
        (if not ``None``) to send the ``PullDoneSignal``.
        """
        if not self.delay_pull:
            return self._handle_pull_eager(msg, sender_id)
        else:
            return self._handle_pull_delay(msg, sender_id)

    def _handle_pull_eager(
        self, msg: PullReadyNotif, sender_id: str
    ) -> tuple[PullReadyDoneAck, Optional[Callable]]:
        """Handle a ``PullReadyNotif`` from the sender in **pull mode** with eager.
        Allocate NPU actual mem_obj and stores them in ``self.data``.
        The NPU connector will pull data during ``batched_to_gpu``
        in one batch.

        Returns ``(ack, post_ack_callback)``.  The caller
        (``_mem_alloc_loop``) must send the ack on the REP socket
        **before** invoking the callback.  This ensures the sender's
        main thread processes the ack and registers ``_pull_pending``
        before the ``PullDoneSignal`` arrives on the listener thread,
        eliminating the race between the two sender threads.
        """
        total_allocs = len(msg.keys)
        fmt = MemoryFormat(msg.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[msg.dtype]
        shape = list(msg.shape)

        already_sent_indexes, already_sent_objs, new_indexes = self._partition_keys(
            msg.keys
        )

        remote_buffer_uuids: list[str] = []
        remote_mem_indexes: list[int] = []
        mem_objs: list[MemoryObj] = []
        mem_keys: list[CacheEngineKey] = []
        for idx in new_indexes:
            key = CacheEngineKey.from_string(msg.keys[idx])

            alloc_shape = adjust_last_chunk_shape(
                shape,
                idx,
                total_allocs,
                fmt,
                msg.last_chunk_toks,
            )

            mem_obj = allocate_with_retry(
                self.allocate,
                torch.Size(alloc_shape),
                dtype,
                fmt,
            )

            if mem_obj is None:
                # Allocation timed out — clean up already-allocated
                # objects and report failure to the sender.
                logger.error(
                    "Pull-eager: allocation failed at chunk %d/%d. "
                    "Releasing %d already-allocated objects.",
                    idx,
                    total_allocs,
                    len(mem_objs),
                )
                # release the mem objs + sent
                release_memory_objects(mem_objs + already_sent_objs)
                return (
                    PullReadyDoneAck(
                        already_sent_indexes=already_sent_indexes,
                        alloc_failed=True,
                    ),
                    None,
                )

            mem_objs.append(mem_obj)
            remote_buffer_uuids.append(msg.sender_buffer_uuids[idx])
            remote_mem_indexes.append(msg.sender_mem_indexes[idx])
            mem_keys.append(key)

        channel_transfer_spec = build_channel_transfer_spec(
            sender_id,
            remote_buffer_uuids,
            remote_mem_indexes,
        )
        self.transfer_channel.batched_read(
            buffers=mem_objs,
            transfer_spec=channel_transfer_spec,
        )

        # batched_read() synchronizes the transport stream, so all RDMA
        # reads are complete at this point.  Store the received data.
        for mem_obj, key in zip(mem_objs, mem_keys, strict=False):
            self.put(key, mem_obj)

        release_memory_objects(already_sent_objs)

        # Build a callback that sends PullDoneSignal AFTER the ack reply
        # has been sent on the REP socket.  This prevents the sender's
        # listener thread from processing the Done signal before the
        # sender's main thread has finished processing the ack.
        pull_id = msg.pull_id

        def _post_ack_send_done():
            self._send_pull_done_to_sender(sender_id, pull_id)

        ack = PullReadyDoneAck(already_sent_indexes=already_sent_indexes)
        return ack, _post_ack_send_done

    def _handle_pull_delay(
        self, msg: PullReadyNotif, sender_id: str
    ) -> tuple[PullReadyDoneAck, Optional[Callable]]:
        """Handle a ``PullReadyNotif`` from the sender in **pull mode** with delay.
        Instead of allocating NPU pages, creates lightweight
        :class:`ProxyMemoryObj` wrappers and stores them in ``self.data``.
        The NPU connector will pull data on-the-fly during ``batched_to_gpu``
        using a pipelined ping-pong approach.

        Returns ``(ack, None)`` — no post-ack callback is needed because
        the ``PullDoneSignal`` is sent later by the NPU connector via
        ``PDTransferContext.send_done_now()``.

        Done-signal flow:

        The alloc socket (REQ/REP) cannot be used for asynchronous
        notifications because ZMQ REQ/REP strictly alternates send/recv.
        Instead, the sender binds a **separate** ZMQ PULL socket on a
        dedicated ``pull_done_port``.  We hand each
        :class:`PDTransferContext` a ``done_callback`` closure that
        PUSHes a :class:`PullDoneSignal` to that port.  The NPU
        connector calls ``transfer_context.send_done_now()`` after all
        proxy chunks have been pulled and scattered, which invokes the
        callback exactly once.  The sender's ``_pull_done_listener_loop``
        receives it and releases the pinned MemObjs.
        """
        already_sent_indexes, already_sent_objs, new_indexes = self._partition_keys(
            msg.keys
        )

        num_proxies = len(new_indexes)

        if num_proxies > 0:
            pull_id = msg.pull_id

            def done_callback():
                self._send_pull_done_to_sender(sender_id, pull_id)

            total_allocs = len(msg.keys)
            fmt = MemoryFormat(msg.fmt)
            shape = list(msg.shape)
            dtype = STR_DTYPE_TO_TORCH_DTYPE[msg.dtype]

            # Use the sender's shape/dtype/fmt for the transfer context
            # so that ping-pong backing buffers are allocated with the
            # sender's tensor layout.  The RDMA read copies raw bytes in
            # the sender's layout; the scatter kernel
            # (multi_layer_kv_transfer) derives num_layers and hidden_dims
            # from the tensor shape, so a mismatch would corrupt the KV
            # cache scatter.
            sender_shapes = [torch.Size(shape)]
            sender_dtypes = [dtype]

            transfer_context = PDTransferContext(
                sender_id=sender_id,
                done_callback=done_callback,
                num_proxies=num_proxies,
                memory_allocator=self.memory_allocator,
                shapes=sender_shapes,
                dtypes=sender_dtypes,
                fmt=fmt,
            )

            for proxy_seq, msg_idx in enumerate(new_indexes):
                key = CacheEngineKey.from_string(msg.keys[msg_idx])

                alloc_shape = adjust_last_chunk_shape(
                    shape,
                    msg_idx,
                    total_allocs,
                    fmt,
                    msg.last_chunk_toks,
                )

                proxy = ProxyMemoryObj(
                    backing_obj=None,
                    transfer_channel=self.transfer_channel,
                    target_peer_url=sender_id,
                    remote_buffer_uuid=msg.sender_buffer_uuids[msg_idx],
                    remote_mem_index=msg.sender_mem_indexes[msg_idx],
                    transfer_context=transfer_context,
                    chunk_index=proxy_seq,
                    shapes=[torch.Size(alloc_shape)],
                    dtypes=self._kv_dtypes,
                    fmt=self._fmt,
                )
                self.put(key, proxy)

            logger.debug(
                "Pull mode: created %d proxies for pull_id %s from sender %s.",
                num_proxies,
                msg.pull_id,
                sender_id,
            )

        release_memory_objects(already_sent_objs)

        return PullReadyDoneAck(already_sent_indexes=already_sent_indexes), None

    # ──────────────────────────────────────────────────────────
    # Done-signal sender (receiver → sender)
    # ──────────────────────────────────────────────────────────

    def _send_pull_done_to_sender(self, sender_id: str, pull_id: str) -> None:
        """Send a ``PullDoneSignal`` to the sender on its done-listener socket.

        This is called from the NPU connector thread (via
        ``PDTransferContext.send_done_now``) after all proxy objects in a
        pull batch have been consumed and scattered into the KV cache.
        """
        try:
            done_signal = PullDoneSignal(pull_id=pull_id)
            assert hasattr(self, "_pull_done_sockets"), (
                "pull_done_sockets must be initialized"
            )

            if sender_id not in self._pull_done_sockets:
                done_url = self._sender_done_urls.get(sender_id)
                if done_url is None:
                    logger.error(
                        "No done URL for sender %s. Cannot send Done signal.",
                        sender_id,
                    )
                    return
                sock = get_zmq_socket(
                    self.zmq_context, done_url, "tcp", zmq.PUSH, "connect"
                )
                self._pull_done_sockets[sender_id] = sock

            self._pull_done_sockets[sender_id].send(msgspec.msgpack.encode(done_signal))
            logger.debug(
                "Sent PullDoneSignal for pull_id %s to sender %s.",
                pull_id,
                sender_id,
            )
        except Exception as e:
            logger.error(
                "Failed to send PullDoneSignal for pull_id %s: %s",
                pull_id,
                e,
            )

    # ──────────────────────────────────────────────────────────
    # Receiver message loop
    # ──────────────────────────────────────────────────────────

    def _mem_alloc_loop(self):
        """Message loop for the receiver side.

        Handles both push-mode ``AllocRequest`` and pull-mode
        ``PullReadyNotif`` messages on the same REP socket.
        """
        # Set the NPU device context for this thread so that HCCL RDMA
        # operations (used by pull-eager mode's batched_read) work
        # correctly on non-default devices (e.g. TP1 on npu:1).
        torch.npu.set_device(self.transfer_channel.handle_device)

        self.alloc_side_channel.setsockopt(zmq.RCVTIMEO, 1000)

        while self.running:
            try:
                msg_bytes = self.alloc_side_channel.recv()
            except zmq.Again:
                continue
            except Exception as e:
                logger.error("Failed to receive in mem alloc loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)
                continue

            try:
                msg = msgspec.msgpack.decode(msg_bytes, type=AscendPDMsg)

                if isinstance(msg, AllocRequest):
                    # Push mode: allocate NPU memory and return refs
                    resp = self._allocate_and_put(msg)
                    self.alloc_side_channel.send(msgspec.msgpack.encode(resp))

                elif isinstance(msg, PullReadyNotif):
                    # Pull mode: create proxies and ack.
                    # The sender_id and done_url come from the message.
                    sender_id = msg.sender_id
                    # Register the done URL so we can send PullDoneSignal
                    if sender_id not in self._sender_done_urls:
                        self._sender_done_urls[sender_id] = msg.sender_done_url
                        logger.debug(
                            "Pull mode: registered done URL %s for sender %s",
                            msg.sender_done_url,
                            sender_id,
                        )
                    ack, post_ack_fn = self._handle_pull_ready(msg, sender_id)
                    # Send the ack FIRST so the sender's main thread can
                    # process it and register _pull_pending before the
                    # PullDoneSignal arrives on the listener thread.
                    self.alloc_side_channel.send(msgspec.msgpack.encode(ack))
                    if post_ack_fn is not None:
                        post_ack_fn()

                else:
                    logger.error(
                        "Unexpected message type in alloc loop: %s",
                        type(msg),
                    )
                    # Must reply to keep REQ/REP in sync
                    self.alloc_side_channel.send(b"")

            except Exception as e:
                logger.error("Failed to process mem alloc loop: %s", str(e))
                # Must send *something* to keep the REP socket in sync,
                # otherwise it enters the "must send" state permanently
                # and every subsequent recv() fails with
                # "Operation cannot be accomplished in current state".
                try:
                    self.alloc_side_channel.send(b"")
                except Exception:
                    pass
                if self.running:
                    time.sleep(0.01)
