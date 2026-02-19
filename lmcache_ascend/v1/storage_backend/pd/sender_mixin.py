# SPDX-License-Identifier: Apache-2.0
"""Sender (prefiller) mixin for the Ascend PD backend.

Contains all sender-side methods: push/pull transfer initiation,
backpressure management, pull-done listener, and circuit breaker logic.
"""

# Standard
from typing import Any, List, Sequence
import threading
import time
import uuid as _uuid

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import TORCH_DTYPE_TO_STR_DTYPE, CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_zmq_socket
from lmcache.v1.storage_backend.pd_backend import (
    AllocRequest,
    ProxyNotif,
)
import msgspec
import zmq

# First Party
from lmcache_ascend.v1.storage_backend.pd.messages import (
    AscendAllocResponse,
    AscendPDMsg,
    PullDoneSignal,
    PullReadyDoneAck,
    PullReadyNotif,
)
from lmcache_ascend.v1.storage_backend.utils import (
    build_channel_transfer_spec,
    release_memory_objects,
)

logger = init_logger(__name__)


class AscendPDSenderMixin:
    """Mixin providing sender/prefiller methods for :class:`AscendPDBackend`.

    This mixin is not intended to be instantiated directly.  It relies on
    attributes initialised by ``AscendPDBackend.__init__`` (e.g.
    ``self.transfer_channel``, ``self.pd_config``, ``self.memory_allocator``).
    """

    # ──────────────────────────────────────────────────────────
    # Sender initialisation
    # ──────────────────────────────────────────────────────────

    def _init_sender(self):
        """Extend sender init with a Done-listener for pull mode."""
        super()._init_sender()

        if self.pull_mode:
            # The sender binds a ZMQ PULL socket for receiving
            # PullDoneSignal from receivers.  The port is configured
            # via ``pd_pull_done_port`` (list[int], one per TP rank)
            pd_pull_done_ports = getattr(self._config, "pd_pull_done_port", None)
            if pd_pull_done_ports is not None:
                self._pull_done_port = pd_pull_done_ports[self.tp_rank]
            else:
                raise ValueError(
                    "Pull mode requires pd_pull_done_port or "
                    "pd_peer_alloc_port to derive a done-listener port."
                )

            # Pull-mode: pinned resources waiting for Done signal.
            # Each entry is (pinned_at_timestamp, list[MemoryObj]).
            self._pull_pending: dict[str, tuple[float, list[MemoryObj]]] = {}
            self._pull_pending_lock = threading.Lock()
            # Safety net: if a PullDoneSignal arrives before the main
            # thread has registered _pull_pending (extremely unlikely
            # after the ack-before-done reordering, but defensive),
            # buffer the pull_id here so the main thread can release
            # immediately after registration instead of waiting for
            # the TTL sweep.
            self._early_pull_done: set[str] = set()
            # TTL in seconds for pull_pending entries.  If a receiver
            # crashes and never sends PullDoneSignal, pinned MemObjs are
            # released after this timeout to prevent memory leaks.
            self._pull_pending_ttl: float = getattr(
                self._config, "pd_pull_pending_ttl", 360.0
            )

            # The sender's bind host — same host used for peer_host
            self._sender_host = self.pd_config.peer_host
            assert self._sender_host is not None, (
                "pd_peer_host must be set on the sender for pull mode "
                "(needed to bind the done-listener socket)."
            )

            done_url = f"{self._sender_host}:{self._pull_done_port}"
            self.local_id = done_url
            logger.info("Pull-mode sender local_id: %s", done_url)
            self._sender_done_url = done_url
            self._pull_done_socket = get_zmq_socket(
                self.zmq_context, done_url, "tcp", zmq.PULL, "bind"
            )
            self.side_channels.append(self._pull_done_socket)

            self._pull_done_thread = threading.Thread(
                target=self._pull_done_listener_loop, daemon=True
            )
            self._pull_done_thread.start()
            self.running_threads.append(self._pull_done_thread)
            logger.info("Pull-mode sender: Done listener started on %s", done_url)

            # Backpressure: track pinned page count and enforce a
            # high-water mark so slow receivers don't exhaust the
            # sender's buffer pool.
            self._pull_pending_pinned_count: int = 0

            # Reserve this percentage of the sender's buffer pool as
            # free headroom.  When pinned pages exceed
            # (1 - reserve_pct/100) * total_pages, new put tasks
            # block until the daemon listener thread frees entries.
            self._pull_bp_reserve_pct: float = getattr(
                self._config, "pd_pull_backpressure_reserve_pct", 2.0
            )

            sender_alloc = (
                self.memory_allocator.cpu_allocator
                if self.use_cpu_offload or self.pd_config.buffer_device == "cpu"
                else self.memory_allocator.gpu_allocator
            )
            total_pages = sender_alloc.buffer_size // sender_alloc.align_bytes
            self._pull_pending_hwm: int = int(
                total_pages * (1.0 - self._pull_bp_reserve_pct / 100.0)
            )
            logger.info(
                "Pull mode backpressure: total_pages=%d, reserve=%.1f%%, hwm=%d pages",
                total_pages,
                self._pull_bp_reserve_pct,
                self._pull_pending_hwm,
            )

    # ──────────────────────────────────────────────────────────
    # Pull-done listener (daemon thread)
    # ──────────────────────────────────────────────────────────

    def _pull_done_listener_loop(self):
        """Listen for PullDoneSignal from receivers and release pinned
        resources.  Also sweeps expired entries on every poll cycle."""
        while self.running:
            try:
                # Use a poll timeout so we can check self.running
                if self._pull_done_socket.poll(timeout=1000):
                    msg_bytes = self._pull_done_socket.recv(zmq.NOBLOCK)
                    msg = msgspec.msgpack.decode(msg_bytes, type=AscendPDMsg)
                    if isinstance(msg, PullDoneSignal):
                        self._handle_pull_done(msg.pull_id)
                    else:
                        logger.warning("Unexpected msg in done listener: %s", type(msg))
                # Sweep expired entries every poll cycle (~1 s)
                self._sweep_expired_pull_pending()
            except zmq.ZMQError as e:
                if self.running:
                    logger.error("ZMQ error in done listener: %s", e)
                    time.sleep(0.01)
            except Exception as e:
                logger.error("Error in done listener: %s", e)
                if self.running:
                    time.sleep(0.01)

    def _sweep_expired_pull_pending(self):
        """Release pinned MemObjs whose TTL has expired.

        This handles the case where a receiver crashes or becomes
        unreachable and never sends a PullDoneSignal.  Without this,
        the sender's pinned buffers would leak indefinitely.
        """
        now = time.monotonic()
        expired_ids: list[str] = []
        with self._pull_pending_lock:
            for pull_id, (pinned_at, _objs) in self._pull_pending.items():
                if now - pinned_at > self._pull_pending_ttl:
                    expired_ids.append(pull_id)
        # Release outside the scan loop to keep the critical section small
        for pull_id in expired_ids:
            with self._pull_pending_lock:
                entry = self._pull_pending.pop(pull_id, None)
                if entry is not None:
                    self._pull_pending_pinned_count -= len(entry[1])
            if entry is not None:
                _pinned_at, pinned_objs = entry
                release_memory_objects(pinned_objs)
                logger.warning(
                    "Pull mode: TTL expired for pull_id %s — released "
                    "%d pinned MemObjs (receiver may have crashed).",
                    pull_id,
                    len(pinned_objs),
                )

    def _wait_for_backpressure(self, num_new_pages: int) -> None:
        """Block until pinned pages drop below the high-water mark.

        Called before pinning new MemObjs to prevent the sender's
        buffer pool from being exhausted by slow-draining receivers.

        The daemon listener thread (:meth:`_pull_done_listener_loop`)
        concurrently processes ``PullDoneSignal`` messages and releases
        entries from ``_pull_pending``, eventually unblocking this method.

        Parameters
        ----------
        num_new_pages:
            Number of pages about to be pinned by the upcoming put task.
        """
        logged = False
        while True:
            with self._pull_pending_lock:
                if (
                    self._pull_pending_pinned_count + num_new_pages
                    <= self._pull_pending_hwm
                ):
                    return
                current_pinned = self._pull_pending_pinned_count
            if not logged:
                logger.warning(
                    "Pull mode backpressure: %d pinned + %d new > "
                    "hwm %d. Waiting for receivers to drain...",
                    current_pinned,
                    num_new_pages,
                    self._pull_pending_hwm,
                )
                logged = True
            time.sleep(0.005)

    def _handle_pull_done(self, pull_id: str) -> None:
        """Release pinned MemObjs when the receiver has finished pulling.

        If the pull_id is not yet in ``_pull_pending`` (the main thread
        hasn't finished processing the ack), buffer it in
        ``_early_pull_done`` so the main thread releases immediately
        after registration.
        """
        with self._pull_pending_lock:
            entry = self._pull_pending.pop(pull_id, None)
            if entry is None:
                # Main thread hasn't registered yet — buffer for later.
                self._early_pull_done.add(pull_id)
                logger.debug(
                    "Pull mode: buffered early PullDoneSignal for "
                    "pull_id %s (main thread not yet registered).",
                    pull_id,
                )
                return
            self._pull_pending_pinned_count -= len(entry[1])
        _pinned_at, pinned_objs = entry
        release_memory_objects(pinned_objs)
        logger.debug(
            "Pull mode: released %d pinned MemObjs for pull_id %s.",
            len(pinned_objs),
            pull_id,
        )

    # ──────────────────────────────────────────────────────────
    # Peer connection & remote allocation
    # ──────────────────────────────────────────────────────────

    def _ensure_peer_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_init_port: int,
        receiver_alloc_port: int,
    ) -> None:
        """Override to call parent and handle any Ascend-specific setup."""
        super()._ensure_peer_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_init_port=receiver_init_port,
            receiver_alloc_port=receiver_alloc_port,
        )

    def _remote_allocate(
        self, receiver_id: str, alloc_request: AllocRequest
    ) -> AscendAllocResponse:
        """Send an ``AllocRequest`` and decode the response as
        ``AscendAllocResponse`` (with UUID-based buffer refs)."""
        side_channel = self.mem_alloc_sockets[receiver_id]
        side_channel.send(msgspec.msgpack.encode(alloc_request))
        msg = side_channel.recv()
        alloc_response = msgspec.msgpack.decode(msg, type=AscendPDMsg)
        assert isinstance(alloc_response, AscendAllocResponse), (
            f"Expected AscendAllocResponse, got {type(alloc_response)}"
        )
        return alloc_response

    # ──────────────────────────────────────────────────────────
    # Main put-task entry point
    # ──────────────────────────────────────────────────────────

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        """Send KV chunks to the remote decoder.

        In **push mode** (default): HCCL-writes data into pre-allocated
        remote NPU memory.

        In **pull mode** (``pd_pull_mode=True``): advertises the sender's
        buffer references so the receiver can read on-demand.  The sender
        pins the MemObjs and waits for a Done signal from the receiver
        before releasing them.

        If the target peer is currently backed off (circuit breaker),
        the transfer is skipped entirely to avoid wasted network I/O.
        """
        # Per-peer circuit breaker: skip transfer if the receiver
        # recently reported an allocation failure.  The lock prevents a
        # concurrent call from slipping past the check before a failing
        # call has set the backoff timestamp.
        receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
        receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
        with self._peer_alloc_backoff_lock:
            now = time.monotonic()
            backoff_until = self._peer_alloc_backoff.get(receiver_id, 0)
            if now < backoff_until:
                logger.warning(
                    "Peer %s is backed off (%.1fs remaining). "
                    "Skipping KV transfer for %d chunks.",
                    receiver_id,
                    backoff_until - now,
                    len(memory_objs),
                )
                # NOTE: Do NOT call release_memory_objects here.
                # The caller (storage_manager.batched_put) always
                # calls ref_count_down on every memory_obj after
                # batched_submit_put_task returns (line 420 in
                # storage_manager.py).  Since we never called
                # ref_count_up (that happens inside the pull/push
                # sub-methods), calling release_memory_objects here
                # would double-decrement the ref count, causing a
                # premature free and PagedTensorMemoryAllocator
                # double-free corruption.
                if transfer_spec.is_last_prefill:
                    notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
                    self.proxy_side_channel.send(msgspec.msgpack.encode(notif_msg))
                return

        if self.pull_mode:
            self._batched_submit_put_task_pull(keys, memory_objs, transfer_spec)
        else:
            self._batched_submit_put_task_push(keys, memory_objs, transfer_spec)

    # ──────────────────────────────────────────────────────────
    # Push mode
    # ──────────────────────────────────────────────────────────

    def _batched_submit_put_task_push(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        """Push mode: HCCL-write from sender NPU → receiver NPU."""
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

        receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
        receiver_alloc_port = transfer_spec.receiver_alloc_port[self.tp_rank]
        receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
        receiver_host = transfer_spec.receiver_host

        self._ensure_peer_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_init_port=receiver_init_port,
            receiver_alloc_port=receiver_alloc_port,
        )

        # Remote allocation — returns UUID-based refs
        alloc_request = self._get_remote_alloc_request(keys, memory_objs)
        alloc_response = self._remote_allocate(receiver_id, alloc_request)

        if alloc_response.alloc_failed:
            # Receiver could not allocate — release all pinned
            # MemObjs, set per-peer backoff, and skip transfer.
            logger.warning(
                "Push mode: receiver %s reported alloc_failed. "
                "Releasing %d pinned MemObjs.",
                receiver_id,
                len(memory_objs),
            )
            release_memory_objects(memory_objs)
            with self._peer_alloc_backoff_lock:
                self._peer_alloc_backoff[receiver_id] = (
                    time.monotonic() + self._peer_alloc_backoff_ttl
                )
            if transfer_spec.is_last_prefill:
                notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
                self.proxy_side_channel.send(msgspec.msgpack.encode(notif_msg))
            return

        already_sent_indexes = set(alloc_response.already_sent_indexes)
        remote_buffer_uuids = alloc_response.remote_buffer_uuids
        remote_mem_indexes = alloc_response.remote_indexes

        # Filter out already-sent memory objects
        mem_objs_to_send = []
        send_buffer_uuids = []
        send_mem_indexes = []
        to_send_idx = 0
        for idx, mem_obj in enumerate(memory_objs):
            if idx in already_sent_indexes:
                mem_obj.ref_count_down()
            else:
                mem_objs_to_send.append(mem_obj)
                send_buffer_uuids.append(remote_buffer_uuids[to_send_idx])
                send_mem_indexes.append(remote_mem_indexes[to_send_idx])
                to_send_idx += 1

        if mem_objs_to_send:
            # Build transfer spec with UUID-based remote refs
            channel_transfer_spec = build_channel_transfer_spec(
                receiver_id,
                send_buffer_uuids,
                send_mem_indexes,
            )

            self.transfer_channel.batched_write(
                objects=mem_objs_to_send,
                transfer_spec=channel_transfer_spec,
            )

            release_memory_objects(mem_objs_to_send)
        else:
            logger.debug(
                "All memory objects already sent to remote peer. Skipping transfer."
            )

        if transfer_spec.is_last_prefill:
            notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
            self.proxy_side_channel.send(msgspec.msgpack.encode(notif_msg))

    # ──────────────────────────────────────────────────────────
    # Pull mode
    # ──────────────────────────────────────────────────────────

    def _batched_submit_put_task_pull(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        """Pull mode: advertise sender buffer refs, let receiver read.

        The sender pins the MemObjs and sends a ``PullReadyNotif`` to the
        receiver.  The receiver acks with already-sent indexes.  The sender
        keeps un-acked MemObjs pinned until a ``PullDoneSignal`` arrives
        (handled in ``_pull_done_listener_loop``).
        """
        # Backpressure: block if too many pages are already pinned.
        # The daemon thread (_pull_done_listener_loop) drains entries
        # concurrently, so this will eventually unblock.
        self._wait_for_backpressure(len(memory_objs))

        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

        receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
        receiver_alloc_port = transfer_spec.receiver_alloc_port[self.tp_rank]
        receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
        receiver_host = transfer_spec.receiver_host

        self._ensure_peer_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_init_port=receiver_init_port,
            receiver_alloc_port=receiver_alloc_port,
        )

        # Resolve local buffer references for the sender's MemObjs
        sender_buffer_uuids, sender_mem_indexes = (
            self.transfer_channel.get_local_buffer_refs(memory_objs)
        )

        # Build PullReadyNotif with sender's buffer refs
        fmt = memory_objs[0].meta.fmt
        shape = memory_objs[0].meta.shape
        dtype = TORCH_DTYPE_TO_STR_DTYPE[memory_objs[0].meta.dtype]
        token_dim = fmt.token_dim()
        last_chunk_toks = memory_objs[-1].meta.shape[token_dim]

        pull_id = _uuid.uuid4().hex

        # The done URL was computed during _init_sender and tells the
        # receiver where to PUSH the PullDoneSignal.
        sender_done_url = self._sender_done_url

        pull_notif = PullReadyNotif(
            pull_id=pull_id,
            keys=[k.to_string() for k in keys],
            sender_buffer_uuids=sender_buffer_uuids,
            sender_mem_indexes=sender_mem_indexes,
            sender_id=self.local_id,
            sender_done_url=sender_done_url,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
        )

        # Send PullReadyNotif and receive ack.
        # NOTE: _pull_pending is NOT registered yet — this avoids a race
        # where the listener thread processes a PullDoneSignal (from the
        # receiver) before we have narrowed the entry to pinned_objs only.
        side_channel = self.mem_alloc_sockets[receiver_id]
        side_channel.send(msgspec.msgpack.encode(pull_notif))
        ack_bytes = side_channel.recv()
        ack = msgspec.msgpack.decode(ack_bytes, type=AscendPDMsg)
        assert isinstance(ack, PullReadyDoneAck), (
            f"Expected PullReadyDoneAck, got {type(ack)}"
        )

        if ack.alloc_failed:
            # Receiver could not allocate — release all pinned
            # MemObjs, set per-peer backoff, and skip pending.
            logger.warning(
                "Pull mode: receiver %s reported alloc_failed. "
                "Releasing %d pinned MemObjs.",
                receiver_id,
                len(memory_objs),
            )
            release_memory_objects(memory_objs)
            with self._peer_alloc_backoff_lock:
                self._peer_alloc_backoff[receiver_id] = (
                    time.monotonic() + self._peer_alloc_backoff_ttl
                )
            if transfer_spec.is_last_prefill:
                notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
                self.proxy_side_channel.send(msgspec.msgpack.encode(notif_msg))
            return

        # Release already-sent objects, pin the rest
        already_sent = set(ack.already_sent_indexes)
        pinned_objs = []
        for idx, mem_obj in enumerate(memory_objs):
            if idx in already_sent:
                mem_obj.ref_count_down()
            else:
                pinned_objs.append(mem_obj)

        if pinned_objs:
            # Register _pull_pending with ONLY the pinned objects.
            # Then check if the PullDoneSignal already arrived (early)
            # while we were processing the ack.
            early_done = False
            with self._pull_pending_lock:
                if pull_id in self._early_pull_done:
                    # Done signal arrived before we registered —
                    # release immediately, don't register.
                    self._early_pull_done.discard(pull_id)
                    early_done = True
                else:
                    self._pull_pending[pull_id] = (
                        time.monotonic(),
                        pinned_objs,
                    )
                    self._pull_pending_pinned_count += len(pinned_objs)

            if early_done:
                release_memory_objects(pinned_objs)
                logger.debug(
                    "Pull mode: early PullDoneSignal for pull_id %s — "
                    "released %d pinned MemObjs immediately.",
                    pull_id,
                    len(pinned_objs),
                )
            else:
                logger.debug(
                    "Pull mode: pinned %d MemObjs for pull_id %s, "
                    "awaiting Done signal from receiver (TTL=%.0fs).",
                    len(pinned_objs),
                    pull_id,
                    self._pull_pending_ttl,
                )
        else:
            # All objects were already sent — nothing left to pin.
            # _pull_pending[pull_id] cannot exist here because this
            # pull_id was never registered (only the pinned_objs branch
            # above registers it).  Just discard any early-done signal
            # if there were any.
            with self._pull_pending_lock:
                self._early_pull_done.discard(pull_id)
            logger.debug(
                "Pull mode: all objects already sent for pull_id %s.",
                pull_id,
            )

        if transfer_spec.is_last_prefill:
            notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
            self.proxy_side_channel.send(msgspec.msgpack.encode(notif_msg))
