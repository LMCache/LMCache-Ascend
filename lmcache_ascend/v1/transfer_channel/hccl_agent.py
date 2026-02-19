# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional
import uuid as _uuid

# Third Party
from lmcache.logging import init_logger
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops
import lmcache_ascend.hccl_npu_comms as hcomm

logger = init_logger(__name__)


class BufferType(Enum):
    CPU = auto()
    NPU = auto()


@dataclass
class BufferConfig:
    ptr: int
    size: int
    device_id: int
    device_type: BufferType
    # ????
    align_bytes: int


@dataclass
class HcclMemHandleMeta:
    mem_handle: hcomm.RmaMemDesc
    buffer_ptr: int
    buffer_size: int
    page_size: int
    local_buffer_addrs: List[int] = None
    buffer_type: BufferType = BufferType.CPU
    uuid: str = field(default_factory=lambda: str(_uuid.uuid4()))


@dataclass
class HcclAgentWrapper:
    agent: "hcomm.HcclAgent"
    mem_handles: List[HcclMemHandleMeta]

    def __init__(
        self,
        buffers: List[BufferConfig],
    ):
        """
        Initialize the hccl agent.

        Args:
            buffers (List[BufferConfig]): List of buffer configurations.
        """

        device_id = torch.npu.current_device()
        hccl_agent = hcomm.HcclAgent.get_instance(device_id)
        hccl_agent.init()

        self.agent = hccl_agent
        self.mem_handles = []
        self.local_index_addr = {}
        self._uuid_to_handle: Dict[str, HcclMemHandleMeta] = {}

        for buf in buffers:
            buffer_ptr = buf.ptr
            buffer_size = buf.size
            page_size = buf.align_bytes

            already_registered = lmc_ops.get_device_ptr(buffer_ptr) is not None
            # HCCL Doesn't allow the host registering a host pointer twice. We could
            # register the corresponding device pointer, however this limits the
            # registereable memory to the device capacity.
            if already_registered:
                lmc_ops.unregister_ptr(buffer_ptr)

            mem_handle = hccl_agent.register_mem(buffer_ptr, buffer_size)
            logger.info(
                "Registered memory with HCCL: "
                "buffer_ptr=%s, buffer_size=%s, mem_handle=%s",
                buffer_ptr,
                buffer_size,
                mem_handle,
            )
            device_ptr = hccl_agent.get_registered_dev_addr(buffer_ptr)
            logger.info(
                "Got device pointer from HCCL: buffer_ptr=%s, device_ptr=%s",
                buffer_ptr,
                device_ptr,
            )

            if (
                already_registered
            ):  # Re register memory to make it accessible by lmc_ops kernels
                lmc_ops.register_mapping(buffer_ptr, device_ptr, buffer_size)

            buffer_addrs = []
            for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
                buffer_addrs.append(base_addr)

            # Register the memory
            mem_handle_meta = HcclMemHandleMeta(
                mem_handle=mem_handle,
                buffer_ptr=buffer_ptr,
                buffer_size=buffer_size,
                page_size=page_size,
                local_buffer_addrs=buffer_addrs,
                buffer_type=buf.device_type,
            )
            self.mem_handles.append(mem_handle_meta)
            self._uuid_to_handle[mem_handle_meta.uuid] = mem_handle_meta

    def get_handle_by_uuid(self, buffer_uuid: str) -> Optional[HcclMemHandleMeta]:
        """Look up a local mem handle by its UUID."""
        return self._uuid_to_handle.get(buffer_uuid)

    def resolve_local_addr(self, buffer_uuid: str, page_index: int) -> int:
        """Resolve a (buffer_uuid, page_index) to an actual local memory address.

        Raises ValueError if the UUID is unknown or page_index is out of bounds.
        """
        meta = self._uuid_to_handle.get(buffer_uuid)
        if meta is None:
            raise ValueError(
                f"Buffer UUID {buffer_uuid} not found in registered handles"
            )
        if meta.local_buffer_addrs is None:
            raise ValueError(f"Buffer UUID {buffer_uuid} has no local_buffer_addrs")
        num_pages = len(meta.local_buffer_addrs)
        if not (0 <= page_index < num_pages):
            raise IndexError(
                f"page_index {page_index} out of range [0, {num_pages}) "
                f"for buffer {buffer_uuid}"
            )
        return meta.local_buffer_addrs[page_index]

    def get_buffer_ref(self, data_ptr: int, page_index: int) -> tuple:
        """Find the buffer UUID for a given data pointer and return (uuid, page_index).

        The page_index is passed through (it comes from the memory allocator's
        meta.address) and validated against the buffer's local_buffer_addrs.
        """
        for meta in self.mem_handles:
            if meta.buffer_ptr <= data_ptr < meta.buffer_ptr + meta.buffer_size:
                if meta.local_buffer_addrs is not None:
                    num_pages = len(meta.local_buffer_addrs)
                    if not (0 <= page_index < num_pages):
                        raise IndexError(
                            f"page_index {page_index} out of range [0, {num_pages}) "
                            f"for buffer {meta.uuid}"
                        )
                return (meta.uuid, page_index)
        raise ValueError(
            f"Pointer {data_ptr} not found in any registered memory handle."
        )

    def get_local_addr(self, ptr: int, idx: int) -> int:
        # we need to first check whether this is in our mem_handles
        # if it is, then we obtain the local address
        for meta in self.mem_handles:
            if meta.buffer_ptr <= ptr < meta.buffer_ptr + meta.buffer_size:
                return meta.local_buffer_addrs[idx]
        raise ValueError(f"Pointer {ptr} not found in any registered memory handle.")

    def close(self):
        for meta in self.mem_handles:
            self.agent.deregister_mem(meta.buffer_ptr)
