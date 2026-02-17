# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union
import asyncio

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.rpc_utils import (
    DEFAULT_SOCKET_RECV_TIMEOUT_MS,
    DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    get_zmq_context,
    get_zmq_socket_with_timeout,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.p2p_backend import (
    BatchedLookupAndGetMsg,
    BatchedLookupAndGetRetMsg,
    BatchedLookupAndPutMsg,
    BatchedLookupAndPutRetMsg,
    P2PBackend,
    P2PErrorCode,
    P2PErrorMsg,
    PeerInfo,
)
import msgspec
import zmq.asyncio

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.storage_backend.utils import (
    build_channel_transfer_spec,
    release_memory_objects,
    resolve_memory_format,
)
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel, get_correct_device
from lmcache_ascend.v1.transfer_context import P2PTransferContext

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.cache_controller import LMCacheWorker

logger = init_logger(__name__)


class AscendBatchedLookupAndGetMsg(BatchedLookupAndGetMsg):
    # buffer uuids to lookup indiviudal mem handles
    # as we extended the transfer channel to support multiple buffers
    # (e.g., CPU and NPU), we need a way to identify
    # which buffer the remote mem handle belongs to,
    # so we use buffer uuid as the identifier.
    buffer_uuids: list[str] = []
    pull_mode: bool = False


class AscendBatchedLookupAndGetRetMsg(BatchedLookupAndGetRetMsg):
    remote_buffer_uuids: list[str] = []
    # remote mem indexes to accompany the pull mode,
    # so that the server can identify which mem handle and buffer
    # the client is referring to in the subsequent pull request.
    remote_mem_indexes: list[int] = []


class AscendBatchedLookupAndGetDoneMsg(msgspec.Struct, tag=True):
    lookup_id: str


class AscendBatchedLookupAndGetDoneRetMsg(msgspec.Struct, tag=True):
    pass


class AscendBatchedLookupAndPutMsg(BatchedLookupAndPutMsg):
    # Opaque buffer references (UUID + mem_index) replacing raw mem_addrs
    buffer_uuids: list[str] = []


AscendP2PMsg = Union[
    AscendBatchedLookupAndGetMsg,
    AscendBatchedLookupAndGetRetMsg,
    AscendBatchedLookupAndPutMsg,
    BatchedLookupAndPutRetMsg,
    P2PErrorMsg,
    AscendBatchedLookupAndGetDoneMsg,
    AscendBatchedLookupAndGetDoneRetMsg,
]


class AscendP2PBackend(P2PBackend):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        lmcache_worker: "LMCacheWorker",
    ):
        self.config = config
        self.loop = loop
        self.lmcache_worker = lmcache_worker
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        assert config.p2p_host is not None, "p2p_host must be specified"
        assert config.p2p_init_ports is not None, "p2p_init_ports must be specified"
        assert config.p2p_lookup_ports is not None, "p2p_lookup_ports must be specified"

        # Load timeout configurations from extra_config (in milliseconds)
        self.socket_recv_timeout_ms = config.get_extra_config_value(
            "p2p_socket_recv_timeout_ms", DEFAULT_SOCKET_RECV_TIMEOUT_MS
        )
        self.socket_send_timeout_ms = config.get_extra_config_value(
            "p2p_socket_send_timeout_ms", DEFAULT_SOCKET_SEND_TIMEOUT_MS
        )

        # Load max retry count from extra_config
        self.max_retry_count = config.get_extra_config_value("p2p_max_retry_count", 3)

        # tp rank is worker id for now
        self.tp_rank = metadata.worker_id

        self.peer_host = config.p2p_host
        self.peer_init_port = config.p2p_init_ports[self.tp_rank]
        self.peer_init_url = f"{self.peer_host}:{self.peer_init_port}"

        self.peer_lookup_port = config.p2p_lookup_ports[self.tp_rank]
        self.peer_lookup_url = f"{self.peer_host}:{self.peer_lookup_port}"

        self.lmcache_instance_id = config.lmcache_instance_id

        # A CacheEngineKey (in int form) -> a list of
        # (peer_init_url, peer_lookup_url, location)
        self.local_lookup_cache: dict[int, tuple[str, str, str]] = {}
        # the target peer info mapping
        self.target_peer_info_mapping: dict[str, PeerInfo] = {}
        # the lock for updating target peer info mapping
        self.update_peer_lock = asyncio.Lock()

        # A lookup_id -> (peer_init_url, location)
        # TODO(chunxiaozheng): location is not used for now
        self.lookup_id_to_peer_mapping: dict[str, tuple[str, str]] = {}

        # Dictionary to store memory objects during pull mode operations.
        # Key: lookup_id, Value: (pinned_at_timestamp, list of MemoryObj).
        # The timestamp is used by the TTL sweep to release entries whose
        # peer crashed and never sent the Done signal.
        self.pending_pull_resources: dict[str, tuple[float, list[MemoryObj]]] = {}

        # TTL in seconds for pending_pull_resources entries.  If a peer
        # crashes mid-pull and never sends the Done signal, pinned MemObjs
        # are released after this timeout to prevent memory leaks.
        self._pull_pending_ttl: float = config.get_extra_config_value(
            "p2p_pull_pending_ttl", 360.0
        )

        # NOTE (gingfung): adding support using npu memory.
        self.use_npu = config.p2p_use_npu

        self.local_cpu_backend = local_cpu_backend
        self.memory_allocator = local_cpu_backend.get_memory_allocator()
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        # NOTE (gingfung): enable using pull mode
        self.pull_mode = config.p2p_pull_mode
        if self.pull_mode:
            logger.info("P2P pull mode enabled. ")

        self.delay_pull = config.p2p_delay_pull
        if self.delay_pull:
            assert self.pull_mode, "Delay pull only works when pull mode is enabled"
            if not self.use_npu:
                logger.warning(
                    "P2P delay pull is enabled "
                    "but NPU buffer is not initialized. "
                    "Setting delay_pull to False."
                )
                self.delay_pull = False
            else:
                logger.info(
                    "P2P delay pull enabled. "
                    "The npu connector will pull the data on-the-fly."
                )

        self.full_size_shapes = self.memory_allocator.cpu_allocator.shapes
        self.dtypes = self.memory_allocator.cpu_allocator.dtypes
        self.fmt: MemoryFormat = resolve_memory_format(metadata.use_mla)

        buffer_ptrs = [self.memory_allocator.cpu_allocator.buffer_ptr]
        buffer_sizes = [self.memory_allocator.cpu_allocator.buffer_size]
        buffer_types = ["cpu"]
        align_bytes = [self.memory_allocator.cpu_allocator.align_bytes]

        if self.use_npu:
            if (
                hasattr(self.memory_allocator, "gpu_buffer")
                and self.memory_allocator.gpu_buffer is not None
            ):
                logger.warning("NPU buffer already initialize.")
            else:
                logger.info(
                    "Initializing NPU memory allocator with "
                    f"size {config.p2p_npu_buffer_size} bytes"
                )
                # get align bytes
                _align_bytes = self.memory_allocator.cpu_allocator.align_bytes
                align_allocator_bytes = (
                    (config.p2p_npu_buffer_size + _align_bytes - 1)
                    // _align_bytes
                    * _align_bytes
                )

                self.memory_allocator.init_gpu_memory_allocator(
                    align_allocator_bytes,
                    self.full_size_shapes,
                    self.dtypes,
                    self.fmt,
                    get_correct_device("npu", metadata.worker_id),
                )

            gpu_alloc = self.memory_allocator.gpu_allocator
            num_npu_pages = len(gpu_alloc.free_blocks)
            page_size_mb = gpu_alloc.align_bytes / (1024 * 1024)
            total_npu_mb = gpu_alloc.buffer_size / (1024 * 1024)
            logger.info(
                "NPU buffer initialized: %.2f MB total, "
                "%d pages available (%.2f MB per page)",
                total_npu_mb,
                num_npu_pages,
                page_size_mb,
            )
            if total_npu_mb <= 1024 and not self.delay_pull:
                logger.warning(
                    f"NPU buffer size ({total_npu_mb} MB) "
                    f"is quite small with "
                    f"(page size: {page_size_mb} MB). "
                )

            buffer_ptrs.append(gpu_alloc.buffer_ptr)
            buffer_sizes.append(gpu_alloc.buffer_size)
            buffer_types.append("npu")
            align_bytes.append(gpu_alloc.align_bytes)

        self.chunk_size = config.chunk_size

        self.transfer_channel = CreateTransferChannel(
            channel_type=config.transfer_channel,
            async_mode=True,
            role="both",
            buffer_ptr=buffer_ptrs,
            buffer_size=buffer_sizes,
            buffer_type=buffer_types,
            align_bytes=align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=self.peer_init_url,
            peer_lookup_url=self.peer_lookup_url,
            event_loop=loop,
        )

        self.running = asyncio.Event()
        self.running.set()
        self.async_context: Optional[zmq.asyncio.Context] = None
        self.async_peer_socket: Optional[zmq.asyncio.Socket] = None
        asyncio.run_coroutine_threadsafe(
            self._run_peer_request_handler_with_recovery(), loop
        )
        asyncio.run_coroutine_threadsafe(
            self._sweep_expired_pending_pull_resources(), loop
        )

    async def _handle_peer_requests(self):
        """
        Handle `BatchedLookupAndGetMsg` issued by peers in `batched_get_non_blocking`.
        """

        logger.info(
            "Starting P2P backend batched get handler at %s", self.peer_lookup_url
        )
        self.async_context = get_zmq_context()
        self.async_peer_socket = get_zmq_socket_with_timeout(
            self.async_context,
            self.peer_lookup_url,
            "tcp",
            zmq.REP,
            "bind",
            self.socket_recv_timeout_ms,
            self.socket_send_timeout_ms,
        )

        while self.running.is_set():
            msg_bytes = await self.async_peer_socket.recv()
            msg = msgspec.msgpack.decode(msg_bytes, type=AscendP2PMsg)

            # Done messages are control signals with no data transfer;
            # Get/Put messages carry `keys` inherited from their base classes.
            num_keys = len(getattr(msg, "keys", []))
            num_tokens = num_keys * self.chunk_size
            monitor_req_id = None
            if num_tokens > 0:
                monitor_req_id = self.stats_monitor.on_p2p_transfer_request(num_tokens)

            if isinstance(msg, AscendBatchedLookupAndGetMsg):
                ret_msg = await self._handle_batched_lookup_and_get(msg)
            elif isinstance(msg, AscendBatchedLookupAndPutMsg):
                ret_msg = await self._handle_batched_lookup_and_put(msg)
            elif isinstance(msg, AscendBatchedLookupAndGetDoneMsg):
                ret_msg = await self._handle_batched_lookup_and_get_done(msg)
            else:
                logger.error("Unknown message type: %s", type(msg))
                ret_msg = P2PErrorMsg(error_code=P2PErrorCode.UNKNOWN_MSG_TYPE)

            if monitor_req_id is not None:
                logger.info(f"P2P transfer finished for request {monitor_req_id}")
                self.stats_monitor.on_p2p_transfer_finished(monitor_req_id)
            else:
                logger.info("P2P transfer finished for control signal with no tokens.")

            await self.async_peer_socket.send(msgspec.msgpack.encode(ret_msg))

    async def _handle_batched_lookup_and_get(
        self, msg: AscendBatchedLookupAndGetMsg
    ) -> Union[AscendBatchedLookupAndGetRetMsg, P2PErrorMsg]:
        lookup_id = msg.lookup_id
        mem_objs = None
        should_release = True
        try:
            logger.info(
                "Received P2P batched lookup and get msg, lookup_id: %s", lookup_id
            )
            receiver_id = msg.receiver_id
            if not self.transfer_channel.remote_xfer_handler_exists(receiver_id):
                logger.error(
                    "Receiver %s does not exist in transfer channel",
                    receiver_id,
                )
                return P2PErrorMsg(
                    error_code=P2PErrorCode.REMOTE_XFER_HANDLER_NOT_INITIALIZED
                )

            keys = [CacheEngineKey.from_string(key) for key in msg.keys]

            # TODO(Jiayi): Optimally, there's no need to use async call
            # for some backends (e.g., local cpu) as there's overhead for
            # async function call.
            num_hit_chunks = await self.local_cpu_backend.batched_async_contains(
                lookup_id=lookup_id,
                keys=keys,
                pin=True,
            )

            mem_objs = await self.local_cpu_backend.batched_get_non_blocking(
                lookup_id=lookup_id,
                keys=keys[:num_hit_chunks],
            )

            if msg.pull_mode:
                # actual data transfer will be triggered
                # by the receiver's pull request.
                remote_buffer_uuids = []
                remote_mem_indexes = []
                if num_hit_chunks > 0 and mem_objs:
                    remote_buffer_uuids, remote_mem_indexes = (
                        self.transfer_channel.get_local_buffer_refs(mem_objs)
                    )

                    # Store mem_objs to prevent premature release.
                    # Record the timestamp so the TTL sweep can detect
                    # stale entries if the peer never sends Done.
                    self.pending_pull_resources[lookup_id] = (
                        self.loop.time(),
                        mem_objs,
                    )
                    should_release = False
                else:
                    logger.warning(
                        "Pull mode enabled but no hit chunks "
                        "for lookup_id %s, receiver_id %s",
                        lookup_id,
                        receiver_id,
                    )
                return AscendBatchedLookupAndGetRetMsg(
                    num_hit_chunks=num_hit_chunks,
                    remote_buffer_uuids=remote_buffer_uuids,
                    remote_mem_indexes=remote_mem_indexes,
                )
            else:
                remote_buffer_uuids = msg.buffer_uuids
                remote_mem_indexes = msg.mem_indexes

            channel_transfer_spec = build_channel_transfer_spec(
                receiver_id,
                remote_buffer_uuids[:num_hit_chunks],
                remote_mem_indexes[:num_hit_chunks],
            )
            await self.transfer_channel.async_batched_write(
                objects=mem_objs,
                transfer_spec=channel_transfer_spec,
            )

            return AscendBatchedLookupAndGetRetMsg(num_hit_chunks=num_hit_chunks)
        except Exception as e:
            logger.error(
                "Error during P2P batched lookup and get operation "
                "for lookup_id %s: %s",
                lookup_id,
                e,
                exc_info=True,
            )
            return P2PErrorMsg(error_code=P2PErrorCode.P2P_SERVER_ERROR)
        finally:
            if should_release and mem_objs is not None:
                release_memory_objects(mem_objs, unpin=True)

    async def _handle_batched_lookup_and_get_done(
        self, msg: AscendBatchedLookupAndGetDoneMsg
    ) -> AscendBatchedLookupAndGetDoneRetMsg:
        lookup_id = msg.lookup_id
        logger.info("Received Done signal for lookup_id %s", lookup_id)

        if lookup_id in self.pending_pull_resources:
            _, mem_objs = self.pending_pull_resources.pop(lookup_id)
            release_memory_objects(mem_objs, unpin=True)
            logger.info("Released resources for lookup_id %s", lookup_id)
        else:
            logger.warning("No pending resources found for lookup_id %s", lookup_id)

        return AscendBatchedLookupAndGetDoneRetMsg()

    async def _sweep_expired_pending_pull_resources(self):
        """Periodically release pinned MemObjs whose TTL has expired.

        This handles the case where a peer crashes or becomes unreachable
        mid-pull and never sends the Done signal.  Without this, the
        server's pinned buffers would leak indefinitely.

        Since this coroutine runs on the same event loop as the request
        handler, no locking is required for ``pending_pull_resources``.
        """
        while self.running.is_set():
            try:
                now = self.loop.time()
                expired_ids = [
                    pid
                    for pid, (ts, _) in self.pending_pull_resources.items()
                    if now - ts > self._pull_pending_ttl
                ]
                for pid in expired_ids:
                    entry = self.pending_pull_resources.pop(pid, None)
                    if entry is not None:
                        _, mem_objs = entry
                        release_memory_objects(mem_objs, unpin=True)
                        logger.warning(
                            "P2P pull mode: TTL expired for lookup_id %s "
                            "— released %d pinned MemObjs "
                            "(peer may have crashed).",
                            pid,
                            len(mem_objs),
                        )
            except Exception as e:
                logger.error(
                    "Error in pending pull resources sweep: %s",
                    e,
                    exc_info=True,
                )
            await asyncio.sleep(10)

    async def _handle_batched_lookup_and_put(
        self, msg: AscendBatchedLookupAndPutMsg
    ) -> P2PErrorMsg:
        logger.error(
            "_handle_batched_lookup_and_put is not implemented "
            "for AscendP2PBackend"
        )
        return P2PErrorMsg(error_code=P2PErrorCode.P2P_SERVER_ERROR)

    def _allocate_memory_for_keys(
        self,
        keys: list[CacheEngineKey],
        cum_chunk_lengths: list[int],
    ) -> tuple[list[MemoryObj], list[str]]:
        """Allocate memory objects for each key and return (mem_objs, str_keys)."""
        mem_objs = []
        str_keys = []
        keys_len = len(keys)
        allocator_label = "NPU" if self.use_npu else "CPU"
        for idx, key in enumerate(keys):
            if not self.config.save_unfull_chunk or idx < keys_len - 1:
                shapes = self.full_size_shapes
            else:
                shapes = self._get_unfull_chunk_shapes(
                    cum_chunk_lengths[idx + 1] - cum_chunk_lengths[idx]
                )

            if self.use_npu:
                mem_obj = self.memory_allocator.gpu_allocator.allocate(
                    shapes, self.dtypes, self.fmt
                )
            else:
                mem_obj = self.local_cpu_backend.allocate(shapes, self.dtypes, self.fmt)

            if mem_obj is None:
                logger.error(
                    "%s allocator out of memory when allocating "
                    "chunk %d/%d for key %s. "
                    "Consider increasing the buffer size.",
                    allocator_label,
                    idx + 1,
                    keys_len,
                    key.to_string(),
                )
                # Release already-allocated objects before returning
                release_memory_objects(mem_objs)
                return [], []

            mem_objs.append(mem_obj)
            str_keys.append(key.to_string())
        return mem_objs, str_keys

    async def _send_lookup_request_with_retry(
        self,
        lookup_id: str,
        target_peer_url: str,
        msg: AscendBatchedLookupAndGetMsg,
    ) -> Optional[AscendP2PMsg]:
        """Send lookup request to peer with retry logic.

        Returns the decoded response message, or None on unrecoverable failure.
        """
        retry_count = 0
        while retry_count < self.max_retry_count:
            peer_info = self.target_peer_info_mapping[target_peer_url]
            async with peer_info.lookup_lock:
                try:
                    retry_count += 1
                    await peer_info.lookup_socket.send(msgspec.msgpack.encode(msg))
                    ret_msg_bytes = await peer_info.lookup_socket.recv()
                    ret_msg = msgspec.msgpack.decode(ret_msg_bytes, type=AscendP2PMsg)
                    if (
                        isinstance(ret_msg, P2PErrorMsg)
                        and ret_msg.error_code
                        == P2PErrorCode.REMOTE_XFER_HANDLER_NOT_INITIALIZED
                    ):
                        logger.warning(
                            "Peer connection not initialized for lookup_id %s, "
                            "ensure peer connection first, retry count: %s",
                            lookup_id,
                            retry_count,
                        )
                        await self._ensure_peer_connection(target_peer_url, True)
                    else:
                        return ret_msg
                except zmq.ZMQError as e:
                    logger.error(
                        "ZMQ error occurred for lookup_id %s. Error: %s",
                        lookup_id,
                        e,
                    )
                    await self._ensure_peer_connection(target_peer_url, True)
                    if retry_count == self.max_retry_count:
                        logger.error(
                            "Max retry count reached for lookup_id %s",
                            lookup_id,
                        )
                        return None
                except Exception as e:
                    logger.error(
                        "Error during P2P get operation for lookup_id %s: %s",
                        lookup_id,
                        e,
                        exc_info=True,
                    )
                    return None
        return None

    async def _send_done_signal(
        self,
        lookup_id: str,
        target_peer_url: str,
    ) -> None:
        """Send Done signal to peer so it can release pinned resources.

        This is critical to prevent memory leaks on the server side.
        """
        try:
            done_msg = AscendBatchedLookupAndGetDoneMsg(
                lookup_id=lookup_id,
            )
            peer_info = self.target_peer_info_mapping[target_peer_url]
            async with peer_info.lookup_lock:
                await peer_info.lookup_socket.send(msgspec.msgpack.encode(done_msg))
                # Wait for Ack (required for ZMQ REQ/REP state machine)
                await peer_info.lookup_socket.recv()
        except Exception as e:
            logger.error(
                "Error sending P2P Done signal for lookup_id %s: %s",
                lookup_id,
                e,
                exc_info=True,
            )

    async def _handle_pull_mode_transfer(
        self,
        lookup_id: str,
        target_peer_url: str,
        hit_mem_objs: list[MemoryObj],
        remote_buffer_uuids: list[str],
        remote_mem_indexes: list[int],
    ) -> bool:
        """Execute pull-mode: read data from remote, then send Done signal.

        Returns True on success, False on failure.
        The Done signal is always sent regardless of read outcome.
        """
        if not remote_buffer_uuids:
            logger.error(
                "Pull mode enabled but remote_buffer_uuids is empty for lookup_id %s",
                lookup_id,
            )
            return False

        read_success = False
        try:
            channel_transfer_spec = build_channel_transfer_spec(
                target_peer_url,
                remote_buffer_uuids,
                remote_mem_indexes,
            )
            await self.transfer_channel.async_batched_read(
                buffers=hit_mem_objs,
                transfer_spec=channel_transfer_spec,
            )
            read_success = True
        except Exception as e:
            logger.error(
                "Error during P2P batched read operation for lookup_id %s: %s",
                lookup_id,
                e,
                exc_info=True,
            )
            # Do not return yet — must send Done signal to server

        await self._send_done_signal(lookup_id, target_peer_url)
        return read_success

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        target_peer_url, _ = self.lookup_id_to_peer_mapping.pop(lookup_id)

        assert isinstance(transfer_spec, dict)
        cum_chunk_lengths = transfer_spec.get("cum_chunk_lengths", None)
        assert cum_chunk_lengths is not None, "cum_chunk_lengths must be provided"
        assert isinstance(cum_chunk_lengths, list), "cum_chunk_lengths must be a list"

        # For delay_pull, skip pre-allocation of backing memory.
        # Buffers will be allocated on-the-fly in the NPU connector's
        # ping-pong pipeline.
        if self.pull_mode and self.delay_pull:
            str_keys = [key.to_string() for key in keys]
            mem_objs = None
            local_buffer_uuids = []
            local_mem_indexes = []
        else:
            mem_objs, str_keys = self._allocate_memory_for_keys(keys, cum_chunk_lengths)
            if not mem_objs:
                logger.warning(
                    "Failed to allocate memory for lookup_id %s, "
                    "returning empty result.",
                    lookup_id,
                )
                return []
            local_buffer_uuids, local_mem_indexes = (
                self.transfer_channel.get_local_buffer_refs(mem_objs)
            )

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id=lookup_id,
            receiver_id=self.peer_init_url,
            keys=str_keys,
            buffer_uuids=local_buffer_uuids,
            mem_indexes=local_mem_indexes,
            pull_mode=self.pull_mode,
        )

        ret_msg = await self._send_lookup_request_with_retry(
            lookup_id, target_peer_url, msg
        )
        if ret_msg is None or isinstance(ret_msg, P2PErrorMsg):
            if isinstance(ret_msg, P2PErrorMsg):
                logger.error(
                    "P2P error for lookup_id %s, error code: %s",
                    lookup_id,
                    ret_msg.error_code,
                )
            if mem_objs is not None:
                self._cleanup_memory_objects(mem_objs)
            return []

        num_hit_chunks = ret_msg.num_hit_chunks

        if num_hit_chunks > 0 and self.pull_mode and self.delay_pull:
            # Create lightweight ProxyMemoryObjs (no backing memory).
            # The NPU connector will allocate ping-pong buffers on-the-fly.
            remote_buffer_uuids = ret_msg.remote_buffer_uuids
            remote_mem_indexes = ret_msg.remote_mem_indexes
            transfer_context = P2PTransferContext(
                p2p_backend=self,
                target_peer_url=target_peer_url,
                lookup_id=lookup_id,
                loop=self.loop,
                num_proxies=num_hit_chunks,
                memory_allocator=self.memory_allocator,
                shapes=self.full_size_shapes,
                dtypes=self.dtypes,
                fmt=self.fmt,
                use_npu=self.use_npu,
            )

            proxy_objs: list[MemoryObj] = []
            for idx in range(num_hit_chunks):
                proxy = ProxyMemoryObj(
                    backing_obj=None,
                    transfer_channel=self.transfer_channel,
                    target_peer_url=target_peer_url,
                    remote_buffer_uuid=remote_buffer_uuids[idx],
                    remote_mem_index=remote_mem_indexes[idx],
                    transfer_context=transfer_context,
                    chunk_index=idx,
                    shapes=self.full_size_shapes,
                    dtypes=self.dtypes,
                    fmt=self.fmt,
                )
                proxy_objs.append(proxy)

            return proxy_objs

        # NOTE (gingfung): mem_objs is only none
        # iff 1) num_hit_chunks is 0, or
        # 2) delay_pull is enabled (so we skip pre-allocation).
        # therefore if mem_objs is None, we can directly return without cleanup.
        if mem_objs is None:
            return []

        hit_mem_objs = mem_objs[:num_hit_chunks]

        if num_hit_chunks > 0 and self.pull_mode:
            if not self.delay_pull:
                success = await self._handle_pull_mode_transfer(
                    lookup_id,
                    target_peer_url,
                    hit_mem_objs,
                    ret_msg.remote_buffer_uuids,
                    ret_msg.remote_mem_indexes,
                )
                if not success:
                    self._cleanup_memory_objects(mem_objs)
                    return []

        release_memory_objects(mem_objs[num_hit_chunks:])
        return hit_mem_objs

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        raise NotImplementedError("Batched put is not implemented for AscendP2PBackend")
