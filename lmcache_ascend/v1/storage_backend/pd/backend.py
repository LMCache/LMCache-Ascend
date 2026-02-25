# SPDX-License-Identifier: Apache-2.0
"""Ascend PD backend — shared core logic.

Defines :class:`AscendPDBackend` which composes the sender and receiver
mixins with the upstream :class:`PDBackend` base class.  Only shared /
role-neutral code lives here: initialisation, allocator setup, memory
allocation, and key lookup/partitioning.
"""

# Standard
from typing import Optional, Union
import threading

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.rpc_utils import get_zmq_context
from lmcache.v1.storage_backend.pd_backend import PDBackend, PDConfig
import torch
import torch_npu  # noqa: F401
import zmq

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import AscendPDReceiverMixin
from lmcache_ascend.v1.storage_backend.pd.sender_mixin import AscendPDSenderMixin
from lmcache_ascend.v1.storage_backend.utils import resolve_memory_format
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel, get_correct_device

logger = init_logger(__name__)


class AscendPDBackend(AscendPDSenderMixin, AscendPDReceiverMixin, PDBackend):
    """PD backend for Ascend (NPU) using HCCL transfer channel.

    Overrides the base :class:`PDBackend` to:

    * initialize **both** CPU and NPU allocators so that the sender can
      offload KV caches to CPU first (pd_use_cpu_offload) and
      transfer via RDMA from host memory,
      while the receiver allocates directly on NPU (pd_buffer_device),
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
            assert self.pd_config.buffer_device.startswith("npu"), (
                "Delay pull only works when buffer device is NPU"
            )

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
            config, "pd_alloc_fail_backoff_ttl", 2.0
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

        assert buffer_ptr, (
            "No buffers registered — at least one of NPU or CPU must be configured"
        )

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

    # ──────────────────────────────────────────────────────────
    # Allocator setup
    # ──────────────────────────────────────────────────────────

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
            # CPU allocator — for sender-side KV offload (NPU -> CPU -> RDMA).
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

    # ──────────────────────────────────────────────────────────
    # Memory allocation
    # ──────────────────────────────────────────────────────────

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
          ``gpu_connector.batched_from_gpu()`` performs an NPU -> CPU
          offload.  The CPU buffer is registered for RDMA, enabling the
          receiver to pull (or the sender to push) directly from host
          memory.
        * **Receiver** (decoder): allocates on **NPU** so that incoming
          KV data lands directly on the accelerator.
        """
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        # Sender + cpu_offload: offload to CPU first  ->  RDMA from CPU
        # Otherwise (receiver, or sender without offload): allocate on NPU
        use_cpu = self.pd_config.buffer_device == "cpu" or (
            self.pd_config.role == "sender" and self.use_cpu_offload
        )
        alloc_type = "cpu" if use_cpu else "gpu"
        return self.memory_allocator.allocate(
            shapes, dtypes, fmt=fmt, allocator_type=alloc_type
        )

    # ──────────────────────────────────────────────────────────
    # Key lookup / partitioning
    # ──────────────────────────────────────────────────────────

    def _lookup(self, key: CacheEngineKey, pin: bool = False) -> Optional[MemoryObj]:
        """Look up *key*, optionally pin it, and return the :class:`MemoryObj`.

        Consumed :class:`ProxyMemoryObj` instances are evicted from the
        store and treated as absent.

        Pinning is safe for both regular ``MemoryObj`` and proxies because
        ``ProxyMemoryObj.ref_count_up/down`` are no-ops — the proxy
        lifecycle is managed by its transfer context, not by ref counts.

        The caller **must** call ``ref_count_down()`` on every returned
        object when *pin* is ``True`` once the pin is no longer needed.
        """
        with self.data_lock:
            mem_obj = self.data.get(key, None)
            if mem_obj is None:
                return None

            if isinstance(mem_obj, ProxyMemoryObj) and mem_obj.consumed:
                del self.data[key]
                return None

            if pin:
                mem_obj.ref_count_up()
            return mem_obj

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if *key* exists in the receiver's data store.

        Overrides the base :meth:`PDBackend.contains` to evict consumed
        :class:`ProxyMemoryObj` instances whose remote buffer
        references are stale.
        """
        assert isinstance(key, CacheEngineKey)
        return self._lookup(key, pin=pin) is not None

    def _contains_and_pin(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Check if *key* exists, pin it, and return the object.

        Combines the existence check with an atomic ``ref_count_up()``
        under ``data_lock``, and returns the **object reference** so the
        caller can later call ``ref_count_down()`` to release the pin.

        Returns ``None`` when the key is absent or is a consumed proxy.
        """
        return self._lookup(key, pin=True)

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
