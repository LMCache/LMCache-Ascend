# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional
import asyncio

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.rpc_utils import (
    DEFAULT_SOCKET_RECV_TIMEOUT_MS,
    DEFAULT_SOCKET_SEND_TIMEOUT_MS,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.p2p_backend import P2PBackend, PeerInfo
import zmq.asyncio

# First Party
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel, get_correct_device

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.cache_controller import LMCacheWorker

logger = init_logger(__name__)


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

        # NOTE (gingfung): adding support using npu memory.
        self.local_cpu_backend = local_cpu_backend
        self.memory_allocator = local_cpu_backend.get_memory_allocator()
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.full_size_shapes = self.memory_allocator.cpu_allocator.shapes
        self.dtypes = self.memory_allocator.cpu_allocator.dtypes
        self.fmt: MemoryFormat = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )

        buffer_ptrs = [self.memory_allocator.cpu_allocator.buffer_ptr]
        buffer_sizes = [self.memory_allocator.cpu_allocator.buffer_size]
        buffer_types = ["cpu"]
        align_bytes = [self.memory_allocator.cpu_allocator.align_bytes]

        if config.p2p_use_npu:
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
