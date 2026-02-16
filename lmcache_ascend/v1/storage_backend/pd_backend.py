# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Callable, List, Optional, Sequence, Union
import threading
import time
import uuid as _uuid

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.pd_backend import (
    AllocRequest,
    AllocResponse,
    PDBackend,
    PDConfig,
    ProxyNotif,
)
import msgspec
import torch
import torch_npu  # noqa: F401
import zmq

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.transfer_context import PDTransferContext
from lmcache_ascend.v1.storage_backend.utils import (
    adjust_last_chunk_shape,
    allocate_with_retry,
    build_channel_transfer_spec,
    release_memory_objects,
    resolve_memory_format,
)
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel, get_correct_device

logger = init_logger(__name__)


class AscendAllocResponse(AllocResponse):
    """Allocation response carrying UUID-based buffer references.

    Instead of just raw page addresses (``remote_indexes``), the receiver
    returns ``(remote_buffer_uuids, remote_indexes)`` pairs so
    the sender can resolve remote memory via the HCCL channel's
    ``PeerMemHandleList.resolve_addr(uuid, page_index)`` on write.
    """

    remote_buffer_uuids: list[str]
    alloc_failed: bool = False


# ──────────────────────────────────────────────────────────
# Pull-mode message types
# ──────────────────────────────────────────────────────────


class PullReadyNotif(msgspec.Struct, tag=True):
    """Sent by the sender (prefiller) to the receiver (decoder) to advertise
    that KV chunks are ready to be pulled.

    Contains the sender's HCCL buffer references so the receiver can
    construct RDMA read operations.
    """

    pull_id: str  # Unique ID for this pull batch
    keys: list[str]
    sender_buffer_uuids: list[str]
    sender_mem_indexes: list[int]
    sender_id: str  # Sender's HCCL peer ID (for transfer_spec receiver_id)
    sender_done_url: str  # URL where receiver PUSHes PullDoneSignal
    fmt: int
    shape: list[int]
    dtype: str
    last_chunk_toks: int


class PullReadyDoneAck(msgspec.Struct, tag=True):
    """Sent by the receiver back to the sender to acknowledge the
    PullReadyNotif.  Contains indexes of keys already received.

    When ``alloc_failed`` is ``True``, the receiver could not allocate
    memory for the requested chunks.  The sender should release its
    pinned resources and skip the transfer.
    """

    already_sent_indexes: list[int]
    alloc_failed: bool = False


class PullDoneSignal(msgspec.Struct, tag=True):
    """Sent by the receiver to the sender after all chunks in a pull batch
    have been read.  The sender releases its pinned resources."""

    pull_id: str


AscendPDMsg = Union[
    AllocRequest,
    AscendAllocResponse,
    ProxyNotif,
    PullReadyNotif,
    PullReadyDoneAck,
    PullDoneSignal,
]


class AscendPDBackend(PDBackend):
    """PD backend for Ascend (NPU) using HCCL transfer channel.

    Overrides the base :class:`PDBackend` to:

    * initialize **both** CPU and NPU allocators so that the sender can
      offload KV caches to CPU first and transfer via RDMA from host
      memory, while the receiver allocates directly on NPU,
    * create an HCCL transfer channel via
      :func:`lmcache_ascend.v1.transfer_channel.CreateTransferChannel`
      with both CPU and NPU buffers registered (multi-buffer pattern),
    * use UUID-based buffer references in alloc responses and transfer specs
      (required by the HCCL channel's ``_resolve_remote_addrs``).
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ):
        self.running = True
        self.tp_rank = metadata.worker_id

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )

        # CPU offload: sender offloads KV to CPU first, then RDMA from CPU.
        # Read from LMCacheEngineConfig (not PDConfig, which is upstream).
        self.use_cpu_offload: bool = getattr(config, "pd_use_cpu_offload", False)

        # Receiver-side KV store
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self.data_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

        # Pull mode: the receiver reads from the sender instead of the
        # sender writing to the receiver.
        self.pull_mode: bool = getattr(config, "pd_pull_mode", False)
        if self.pull_mode:
            logger.info("PD pull mode enabled.")

        self.delay_pull: bool = getattr(config, "pd_delay_pull", False)
        if self.delay_pull:
            assert self.pull_mode, "Delay pull only works when pull mode is enabled"
            assert self.pd_config.buffer_device.startswith("npu"), "Delay pull only works when buffer device is NPU"

        # Keep config ref for extra_config access (e.g., pull_done_port)
        self._config = config

        # Per-peer circuit breaker: when a receiver fails to allocate,
        # skip all transfers to that peer until the TTL expires.
        # Protected by _peer_alloc_backoff_lock to prevent a race
        # where a concurrent call slips past the check before the
        # failing call has set the backoff timestamp.
        self._peer_alloc_backoff: dict[str, float] = {}
        self._peer_alloc_backoff_lock = threading.Lock()
        self._peer_alloc_backoff_ttl: float = getattr(
            config, "pd_alloc_fail_backoff_ttl", 5.0
        )

        # Peer init URL / local id
        peer_init_url = None
        self.local_id = ""
        if self.pd_config.peer_init_port is not None:
            peer_init_url = (
                f"{self.pd_config.peer_host}:{self.pd_config.peer_init_port}"
            )
            self.local_id = self.pd_config.peer_host + str(
                self.pd_config.peer_init_port
            )

        # Register both CPU and NPU buffers with the transfer channel
        # so that RDMA can operate on either memory region.
        # (Mirrors the multi-buffer pattern used by AscendP2PBackend.)
        buffer_ptr = []
        buffer_size = []
        buffer_type = []
        align_bytes = []
        if self.pd_config.buffer_device.startswith("npu"):
            buffer_ptr.append(self.memory_allocator.gpu_allocator.buffer_ptr)
            buffer_size.append(self.memory_allocator.gpu_allocator.buffer_size)
            buffer_type.append("npu")
            align_bytes.append(self.memory_allocator.gpu_allocator.align_bytes)

        if self.pd_config.buffer_device == "cpu" or self.use_cpu_offload:
            buffer_ptr.append(self.memory_allocator.cpu_allocator.buffer_ptr)
            buffer_size.append(self.memory_allocator.cpu_allocator.buffer_size)
            buffer_type.append("cpu")
            align_bytes.append(self.memory_allocator.cpu_allocator.align_bytes)
        
        assert len(buffer_ptr) > 0, "No buffer pointers found"
        assert len(buffer_size) > 0, "No buffer sizes found"
        assert len(buffer_type) > 0, "No buffer types found"
        assert len(align_bytes) > 0, "No align bytes found"

        self.transfer_channel = CreateTransferChannel(
            channel_type=config.transfer_channel,
            async_mode=False,
            role=self.pd_config.role,
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            buffer_type=buffer_type,
            align_bytes=align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=peer_init_url,
        )

        # Role-specific initialization
        if self.pd_config.role == "sender":
            self._init_sender()
            self.initialized_peers: set[str] = set()
            self.mem_alloc_sockets: dict[str, zmq.Socket] = {}
        elif self.pd_config.role == "receiver":
            self._init_receiver()
        else:
            raise ValueError("Invalid PD role.")

        self.full_chunk_size = config.chunk_size

        # Cache metadata for proxy creation on receiver side
        self._metadata = metadata
        self._fmt = resolve_memory_format(metadata.use_mla)
        self._kv_shapes = [torch.Size(metadata.kv_shape)]
        self._kv_dtypes = [metadata.kv_dtype]

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        npu_corrected_device = get_correct_device("npu", metadata.worker_id)
        logger.info("Setting NPU device to %s", npu_corrected_device)
        torch.npu.set_device(npu_corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()
        fmt = resolve_memory_format(metadata.use_mla)
        sizes = [torch.Size(metadata.kv_shape)]
        dtypes = [metadata.kv_dtype]
        total_size = get_size_bytes(sizes, dtypes)

        if self.pd_config.buffer_device.startswith("npu"):
            # NPU allocator — needed for RDMA buffer registration and
            # receiver-side allocation (incoming KV lands directly on NPU).
            npu_aligned_byte = (
                (config.pd_buffer_size + total_size - 1) // total_size * total_size
            )
            paged_mem_allocator.init_gpu_memory_allocator(
                npu_aligned_byte, sizes, dtypes, fmt, npu_corrected_device
            )
            logger.info(
                "Initialized NPU allocator: %.2f MB",
                npu_aligned_byte / (1024 * 1024),
            )

        if self.pd_config.buffer_device == "cpu" or self.use_cpu_offload:
            # CPU allocator — for sender-side KV offload (NPU → CPU → RDMA).
            # or configured to use CPU as the buffer device.
            # Falls back to pd_buffer_size when pd_cpu_buffer_size is not set.
            cpu_buffer_size = getattr(
                config, "pd_cpu_buffer_size", config.pd_buffer_size
            )
            cpu_aligned_byte = (
                (cpu_buffer_size + total_size - 1) // total_size * total_size
            )
            paged_mem_allocator.init_cpu_memory_allocator(
                cpu_aligned_byte, sizes, dtypes, fmt
            )

            logger.info(
                "Initialized CPU allocator: %.2f MB",
                cpu_aligned_byte / (1024 * 1024),
            )

        return paged_mem_allocator

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate memory with role-aware placement.

        * **Sender** (prefiller): allocates on **CPU** so that
          ``gpu_connector.batched_from_gpu()`` performs an NPU → CPU
          offload.  The CPU buffer is registered for RDMA, enabling the
          receiver to pull (or the sender to push) directly from host
          memory.
        * **Receiver** (decoder): allocates on **NPU** so that incoming
          KV data lands directly on the accelerator.
        """
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        # Sender + cpu_offload: offload to CPU first  →  RDMA from CPU
        # Otherwise (receiver, or sender without offload): allocate on NPU
        use_cpu = (self.pd_config.buffer_device == "cpu" or 
            (self.pd_config.role == "sender" and self.use_cpu_offload))
        alloc_type = "cpu" if use_cpu else "gpu"
        return self.memory_allocator.allocate(
            shapes, dtypes, fmt=fmt, allocator_type=alloc_type
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if *key* exists in the receiver's data store.

        Overrides the base :meth:`PDBackend.contains` to evict consumed
        :class:`ProxyMemoryObj` instances whose remote buffer
        references are stale.

        Pinning is safe for both regular MemoryObj and proxies because
        ``ProxyMemoryObj.ref_count_up/down`` are no-ops — the proxy
        lifecycle is managed by its transfer context, not by ref counts.
        """
        assert isinstance(key, CacheEngineKey)
        with self.data_lock:
            mem_obj = self.data.get(key, None)
            if mem_obj is None:
                return False

            # Consumed proxies hold stale remote refs — evict and
            # report the key as absent so the caller re-fetches.
            if isinstance(mem_obj, ProxyMemoryObj) and mem_obj.consumed:
                del self.data[key]
                return False

            if pin:
                mem_obj.ref_count_up()
            return True

    def _contains_and_pin(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Check if *key* exists, pin it, and return the object.

        Combines the existence check with an atomic ``ref_count_up()``
        under ``data_lock``, and returns the **object reference** so the
        caller can later call ``ref_count_down()`` to release the pin.

        Returns ``None`` when the key is absent or is a consumed proxy.

        .. note::

           ``ref_count_up/down`` are no-ops on proxy objects, so
           pinning them is safe and callers can use the same API
           uniformly.  The caller **must** call ``ref_count_down()``
           on every returned object once the pin is no longer needed.
        """
        with self.data_lock:
            mem_obj = self.data.get(key, None)
            if mem_obj is None:
                return None

            if isinstance(mem_obj, ProxyMemoryObj) and mem_obj.consumed:
                del self.data[key]
                return None

            mem_obj.ref_count_up()
            return mem_obj

    def _partition_keys(
        self,
        keys: list[str],
    ) -> tuple[list[int], list[MemoryObj], list[int]]:
        """Partition message keys into already-sent (pinned) and new indexes.

        Iterates over *keys*, calling :meth:`_contains_and_pin` for each.
        Keys that already exist in ``self.data`` are pinned and collected
        as "already sent"; the rest are collected as "new".

        Returns
        -------
        already_sent_indexes : list[int]
            Indexes (into *keys*) of chunks that were already present.
        already_sent_objs : list[MemoryObj]
            The pinned MemoryObj for each already-sent key.  The caller
            **must** call :meth:`_release_pinned` when done.
        new_indexes : list[int]
            Indexes (into *keys*) of chunks that need to be fetched.
        """
        already_sent_indexes: list[int] = []
        already_sent_objs: list[MemoryObj] = []
        new_indexes: list[int] = []
        for idx, key_str in enumerate(keys):
            key = CacheEngineKey.from_string(key_str)
            pinned = self._contains_and_pin(key)
            if pinned is not None:
                already_sent_indexes.append(idx)
                already_sent_objs.append(pinned)
            else:
                new_indexes.append(idx)
        return already_sent_indexes, already_sent_objs, new_indexes


    # ──────────────────────────────────────────────────────────
    # Sender / prefiller overrides
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
            logger.info(f"Pull-mode sender local_id: {done_url}")
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

    def _init_receiver(self):
        """Extend receiver init with done-socket URL tracking for pull mode."""
        super()._init_receiver()

        if self.pull_mode:
            # Mapping from sender_id -> done URL so we can send
            # PullDoneSignal back to the correct sender.
            self._sender_done_urls: dict[str, str] = {}
            self._pull_done_sockets: dict[str, zmq.Socket] = {}

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
                    self.proxy_side_channel.send(
                        msgspec.msgpack.encode(notif_msg)
                    )
                return

        if self.pull_mode:
            self._batched_submit_put_task_pull(keys, memory_objs, transfer_spec)
        else:
            self._batched_submit_put_task_push(keys, memory_objs, transfer_spec)

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
                self.proxy_side_channel.send(
                    msgspec.msgpack.encode(notif_msg)
                )
            return

        already_sent_indexes = alloc_response.already_sent_indexes
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
                self.proxy_side_channel.send(
                    msgspec.msgpack.encode(notif_msg)
                )
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
                logger.info(
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
    # Decoder / receiver overrides
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

        already_sent_indexes, already_sent_objs, new_indexes = (
            self._partition_keys(alloc_request.keys)
        )

        remote_buffer_uuids: list[str] = []
        remote_mem_indexes: list[int] = []
        allocated_keys: list[CacheEngineKey] = []
        allocated_objs: list[MemoryObj] = []
        for idx in new_indexes:
            key = CacheEngineKey.from_string(alloc_request.keys[idx])

            alloc_shape = adjust_last_chunk_shape(
                shape, idx, total_allocs, fmt, alloc_request.last_chunk_toks,
            )

            mem_obj = allocate_with_retry(
                self.allocate, torch.Size(alloc_shape), dtype, fmt,
            )

            if mem_obj is None:
                # Allocation timed out — undo already-stored chunks and
                # report failure to the sender.
                logger.error(
                    "Push-mode: allocation failed at chunk %d/%d. "
                    "Releasing %d already-allocated objects.",
                    idx, total_allocs, len(allocated_objs),
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

    # ── Pull-mode receiver handler ────────────────────────────

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

        already_sent_indexes, already_sent_objs, new_indexes = (
            self._partition_keys(msg.keys)
        )

        remote_buffer_uuids: list[str] = []
        remote_mem_indexes: list[int] = []
        mem_objs: list[MemoryObj] = []
        mem_keys: list[CacheEngineKey] = []
        for idx in new_indexes:
            key = CacheEngineKey.from_string(msg.keys[idx])

            alloc_shape = adjust_last_chunk_shape(
                shape, idx, total_allocs, fmt, msg.last_chunk_toks,
            )

            mem_obj = allocate_with_retry(
                self.allocate, torch.Size(alloc_shape), dtype, fmt,
            )

            if mem_obj is None:
                # Allocation timed out — clean up already-allocated
                # objects and report failure to the sender.
                logger.error(
                    "Pull-eager: allocation failed at chunk %d/%d. "
                    "Releasing %d already-allocated objects.",
                    idx, total_allocs, len(mem_objs),
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
            sender_id, remote_buffer_uuids, remote_mem_indexes,
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
        already_sent_indexes, already_sent_objs, new_indexes = (
            self._partition_keys(msg.keys)
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
                    shape, msg_idx, total_allocs, fmt, msg.last_chunk_toks,
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

    def _send_pull_done_to_sender(self, sender_id: str, pull_id: str) -> None:
        """Send a ``PullDoneSignal`` to the sender on its done-listener socket.

        This is called from the NPU connector thread (via
        ``PDTransferContext.send_done_now``) after all proxy objects in a
        pull batch have been consumed and scattered into the KV cache.
        """
        try:
            done_signal = PullDoneSignal(pull_id=pull_id)
            # Use a fresh PUSH socket to the sender's done-listener port.
            # The port is derived from sender_id (same host, done_port)
            assert hasattr(self, "_pull_done_sockets"), (
                "pull_done_sockets must be initialized"
            )

            if sender_id not in self._pull_done_sockets:
                # Build the done URL from sender_id.
                # sender_id format: "<host><init_port>"
                # The done port is the pull_done_port stored during init.
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
