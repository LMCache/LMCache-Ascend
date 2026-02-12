# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List, Optional, Sequence, Union
import threading
import time

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.integration.vllm.utils import get_size_bytes
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
    ProxyNotif,
    PDConfig
)
import msgspec
import torch
import torch_npu  # noqa: F401
import zmq

# First Party
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


AscendPDMsg = Union[AllocRequest, AscendAllocResponse, ProxyNotif]


class AscendPDBackend(PDBackend):
    """PD backend for Ascend (NPU) using HCCL transfer channel.

    Overrides the base :class:`PDBackend` to:

    * initialize allocator on NPU instead of CUDA,
    * create an HCCL transfer channel via
      :func:`lmcache_ascend.v1.transfer_channel.CreateTransferChannel`,
    * use UUID-based buffer references in alloc responses and transfer specs
      (required by the HCCL channel's ``_resolve_remote_addrs``).
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ):
        # Skip PDBackend.__init__ and do our own setup to avoid
        # importing CUDA-specific transfer channel / device utilities.
        self.running = True
        self.tp_rank = metadata.worker_id

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank)

        # Receiver-side KV store
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self.data_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

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

        gpu_alloc = self.memory_allocator.gpu_allocator
        self.transfer_channel = CreateTransferChannel(
            channel_type=config.transfer_channel,
            async_mode=False,
            role=self.pd_config.role,
            buffer_ptr=gpu_alloc.buffer_ptr,
            buffer_size=gpu_alloc.buffer_size,
            buffer_type="npu",
            align_bytes=gpu_alloc.align_bytes,
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


    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        corrected_device = get_correct_device(
            config.pd_buffer_device, metadata.worker_id
        )
        logger.info("Setting NPU device to %s", corrected_device)
        torch.npu.set_device(corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()
        fmt = MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        sizes = [torch.Size(metadata.kv_shape)]
        dtypes = [metadata.kv_dtype]
        total_size = get_size_bytes(sizes, dtypes)
        aligned_byte = (config.pd_buffer_size + total_size - 1) // total_size * total_size
        paged_mem_allocator.init_gpu_memory_allocator(
            aligned_byte,
            sizes,
            dtypes,
            fmt,
            corrected_device,
        )
        return paged_mem_allocator

    # ──────────────────────────────────────────────────────────
    # Sender / prefiller overrides
    # ──────────────────────────────────────────────────────────

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
        """Send KV chunks to the remote decoder via HCCL write.

        Builds ``channel_transfer_spec`` with
        ``remote_buffer_uuids`` / ``remote_mem_indexes``
        so the HCCL channel can resolve remote memory addresses.
        """
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
            channel_transfer_spec = {
                "receiver_id": receiver_id,
                "remote_buffer_uuids": send_buffer_uuids,
                "remote_mem_indexes": send_mem_indexes,
            }

            self.transfer_channel.batched_write(
                objects=mem_objs_to_send,
                transfer_spec=channel_transfer_spec,
            )

            for mem_obj in mem_objs_to_send:
                mem_obj.ref_count_down()
        else:
            logger.debug(
                "All memory objects already sent to remote peer. "
                "Skipping transfer."
            )

        if transfer_spec.is_last_prefill:
            notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
            self.proxy_side_channel.send(msgspec.msgpack.encode(notif_msg))

    # ──────────────────────────────────────────────────────────
    # Decoder / receiver overrides
    # ──────────────────────────────────────────────────────────

    def _allocate_and_put(self, alloc_request: AllocRequest) -> AscendAllocResponse:
        """Allocate memory for incoming chunks and return UUID-based refs."""
        total_allocs = len(alloc_request.keys)
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = list(alloc_request.shape)

        already_sent_indexes: list[int] = []
        remote_buffer_uuids: list[str] = []
        remote_mem_indexes: list[int] = []

        for idx, key_str in enumerate(alloc_request.keys):
            key = CacheEngineKey.from_string(key_str)
            if self.contains(key, pin=True):
                already_sent_indexes.append(idx)
                continue

            # Adjust shape for last (possibly partial) chunk
            alloc_shape = list(shape)
            if idx == total_allocs - 1:
                token_dim = fmt.token_dim()
                alloc_shape[token_dim] = alloc_request.last_chunk_toks

            mem_obj = self.allocate(torch.Size(alloc_shape), dtype, fmt)

            # Busy-loop until allocation succeeds
            wait_time = 0.01
            while mem_obj is None:
                logger.warning("Failed to allocate memory object, retrying...")
                time.sleep(wait_time)
                mem_obj = self.allocate(torch.Size(alloc_shape), dtype, fmt)

            # Resolve UUID + page index for this allocation
            buf_uuid, mem_idx = self.transfer_channel.get_local_buffer_refs([mem_obj])
            remote_buffer_uuids.append(buf_uuid[0])
            remote_mem_indexes.append(mem_idx[0])

            self.put(key, mem_obj)

        return AscendAllocResponse(
            already_sent_indexes=already_sent_indexes,
            remote_buffer_uuids=remote_buffer_uuids,
            remote_indexes=remote_mem_indexes,
        )

    def _mem_alloc_loop(self):
        """Memory allocation loop that speaks the Ascend PD protocol."""
        while self.running:
            try:
                alloc_req_bytes = self.alloc_side_channel.recv()
                alloc_req = msgspec.msgpack.decode(alloc_req_bytes, type=AscendPDMsg)
                assert isinstance(alloc_req, AllocRequest), (
                    f"Expected AllocRequest, got {type(alloc_req)}"
                )

                alloc_resp = self._allocate_and_put(alloc_req)
                self.alloc_side_channel.send(msgspec.msgpack.encode(alloc_resp))

            except Exception as e:
                logger.error("Failed to process mem alloc loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)