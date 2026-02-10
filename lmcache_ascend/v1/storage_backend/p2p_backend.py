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
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel, get_correct_device

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.cache_controller import LMCacheWorker

logger = init_logger(__name__)


class AscendBatchedLookupAndGetMsg(BatchedLookupAndGetMsg):
    mem_addrs: list[int] = []
    pull_mode: bool = False


class AscendBatchedLookupAndGetRetMsg(BatchedLookupAndGetRetMsg):
    remote_mem_addrs: list[int] = []


class AscendBatchedLookupAndGetDoneMsg(msgspec.Struct, tag=True):
    lookup_id: str
    mem_addrs: list[int] = []


class AscendBatchedLookupAndGetDoneRetMsg(msgspec.Struct, tag=True):
    pass


class AscendBatchedLookupAndPutMsg(BatchedLookupAndPutMsg):
    mem_addrs: list[int] = []


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

        # Dictionary to store memory objects during pull mode operations
        # Key: lookup_id, Value: list of MemoryObj
        self.pending_pull_resources: dict[str, list[MemoryObj]] = {}

        # NOTE (gingfung): adding support using npu memory.
        self.local_cpu_backend = local_cpu_backend
        self.memory_allocator = local_cpu_backend.get_memory_allocator()
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        # NOTE (gingfung): enable using pull mode
        if config.p2p_pull_mode:
            logger.info("P2P pull mode enabled. ")
            self.pull_mode = True
        else:
            self.pull_mode = False

        self.full_size_shapes = self.memory_allocator.cpu_allocator.shapes
        self.dtypes = self.memory_allocator.cpu_allocator.dtypes
        self.fmt: MemoryFormat = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )

        buffer_ptrs = [self.memory_allocator.cpu_allocator.buffer_ptr]
        buffer_sizes = [self.memory_allocator.cpu_allocator.buffer_size]
        buffer_types = ["cpu"]
        align_bytes = [self.memory_allocator.cpu_allocator.align_bytes]
        self.use_npu = config.p2p_use_npu

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
                self.memory_allocator.init_gpu_memory_allocator(
                    config.p2p_npu_buffer_size,
                    self.full_size_shapes,
                    self.dtypes,
                    self.fmt,
                    get_correct_device("npu", metadata.worker_id),
                )
            buffer_ptrs.append(self.memory_allocator.gpu_allocator.buffer_ptr)
            buffer_sizes.append(self.memory_allocator.gpu_allocator.buffer_size)
            buffer_types.append("npu")
            align_bytes.append(self.memory_allocator.gpu_allocator.align_bytes)

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

            num_tokens = len(msg.mem_addrs) * self.chunk_size
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

            logger.info(f"P2P transfer finished for request {monitor_req_id}")
            self.stats_monitor.on_p2p_transfer_finished(monitor_req_id)

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
                remote_mem_addrs = []
                if num_hit_chunks > 0 and mem_objs:
                    remote_mem_addrs = self.transfer_channel.get_local_mem_addrs(
                        mem_objs
                    )

                    # Store mem_objs to prevent premature release
                    self.pending_pull_resources[lookup_id] = mem_objs
                    should_release = False
                else:
                    logger.warning(
                        "Pull mode enabled but no hit chunks for lookup_id %s, receiver_id %s",
                        lookup_id,
                        receiver_id,
                    )
                return AscendBatchedLookupAndGetRetMsg(
                    num_hit_chunks=num_hit_chunks,
                    remote_mem_addrs=remote_mem_addrs,
                )
            else:
                remote_mem_addrs = msg.mem_addrs

            channel_transfer_spec = {
                "receiver_id": receiver_id,
                "remote_addrs": remote_mem_addrs[:num_hit_chunks],
            }
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
                for mem_obj in mem_objs:
                    mem_obj.ref_count_down()
                    mem_obj.unpin()

    async def _handle_batched_lookup_and_get_done(
        self, msg: AscendBatchedLookupAndGetDoneMsg
    ) -> AscendBatchedLookupAndGetDoneRetMsg:
        lookup_id = msg.lookup_id
        logger.info("Received Done signal for lookup_id %s", lookup_id)

        if lookup_id in self.pending_pull_resources:
            mem_objs = self.pending_pull_resources.pop(lookup_id)
            for mem_obj in mem_objs:
                mem_obj.ref_count_down()
                mem_obj.unpin()
            logger.info("Released resources for lookup_id %s", lookup_id)
        else:
            logger.warning("No pending resources found for lookup_id %s", lookup_id)

        return AscendBatchedLookupAndGetDoneRetMsg()

    def _allocate_memory_for_keys(
        self,
        keys: list[CacheEngineKey],
        cum_chunk_lengths: list[int],
    ) -> tuple[list[MemoryObj], list[str]]:
        """Allocate memory objects for each key and return (mem_objs, str_keys)."""
        mem_objs = []
        str_keys = []
        keys_len = len(keys)
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

            mem_objs.append(mem_obj)
            str_keys.append(key.to_string())
        return mem_objs, str_keys

    async def _send_lookup_request_with_retry(
        self,
        lookup_id: str,
        target_peer_init_url: str,
        msg: AscendBatchedLookupAndGetMsg,
    ) -> Optional[AscendP2PMsg]:
        """Send lookup request to peer with retry logic.

        Returns the decoded response message, or None on unrecoverable failure.
        """
        retry_count = 0
        while retry_count < self.max_retry_count:
            peer_info = self.target_peer_info_mapping[target_peer_init_url]
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
                        await self._ensure_peer_connection(target_peer_init_url, True)
                    else:
                        return ret_msg
                except zmq.ZMQError as e:
                    logger.error(
                        "ZMQ error occurred for lookup_id %s. Error: %s",
                        lookup_id,
                        e,
                    )
                    await self._ensure_peer_connection(target_peer_init_url, True)
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
        target_peer_init_url: str,
        remote_mem_addrs: list[int],
    ) -> None:
        """Send Done signal to peer so it can release pinned resources.

        This is critical to prevent memory leaks on the server side.
        """
        try:
            done_msg = AscendBatchedLookupAndGetDoneMsg(
                lookup_id=lookup_id, mem_addrs=remote_mem_addrs
            )
            peer_info = self.target_peer_info_mapping[target_peer_init_url]
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
        target_peer_init_url: str,
        hit_mem_objs: list[MemoryObj],
        remote_mem_addrs: list[int],
    ) -> bool:
        """Execute pull-mode: read data from remote, then send Done signal.

        Returns True on success, False on failure.
        The Done signal is always sent regardless of read outcome.
        """
        if not remote_mem_addrs:
            logger.error(
                "Pull mode enabled but remote_mem_addrs is empty for lookup_id %s",
                lookup_id,
            )
            return False

        read_success = False
        try:
            channel_transfer_spec = {
                "receiver_id": target_peer_init_url,
                "remote_addrs": remote_mem_addrs,
            }
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

        await self._send_done_signal(lookup_id, target_peer_init_url, remote_mem_addrs)
        return read_success

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        target_peer_init_url, _ = self.lookup_id_to_peer_mapping.pop(lookup_id)

        assert isinstance(transfer_spec, dict)
        cum_chunk_lengths = transfer_spec.get("cum_chunk_lengths", None)
        assert cum_chunk_lengths is not None, "cum_chunk_lengths must be provided"
        assert isinstance(cum_chunk_lengths, list), "cum_chunk_lengths must be a list"

        mem_objs, str_keys = self._allocate_memory_for_keys(keys, cum_chunk_lengths)
        local_addrs = self.transfer_channel.get_local_mem_addrs(mem_objs)

        msg = AscendBatchedLookupAndGetMsg(
            lookup_id=lookup_id,
            receiver_id=self.peer_init_url,
            keys=str_keys,
            mem_indexes=[],  # we don't use mem_indexes
            mem_addrs=local_addrs,
            pull_mode=self.pull_mode,
        )

        ret_msg = await self._send_lookup_request_with_retry(
            lookup_id, target_peer_init_url, msg
        )
        if ret_msg is None or isinstance(ret_msg, P2PErrorMsg):
            if isinstance(ret_msg, P2PErrorMsg):
                logger.error(
                    "P2P error for lookup_id %s, error code: %s",
                    lookup_id,
                    ret_msg.error_code,
                )
            self._cleanup_memory_objects(mem_objs)
            return []

        num_hit_chunks = ret_msg.num_hit_chunks
        hit_mem_objs = mem_objs[:num_hit_chunks]

        if num_hit_chunks > 0 and self.pull_mode:
            success = await self._handle_pull_mode_transfer(
                lookup_id,
                target_peer_init_url,
                hit_mem_objs,
                ret_msg.remote_mem_addrs,
            )
            if not success:
                self._cleanup_memory_objects(mem_objs)
                return []

        for missed_mem_obj in mem_objs[num_hit_chunks:]:
            missed_mem_obj.ref_count_down()
        return hit_mem_objs

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        raise NotImplementedError("Batched put is not implemented for AscendP2PBackend")
