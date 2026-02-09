# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

# Third Party
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops
import lmcache_ascend.hccl_npu_comms as hcomm


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
    buffer_type: BufferType = BufferType.CPU


@dataclass
class HcclAgentWrapper:
    agent: "hcomm.HcclAgent"
    mem_handles: List[HcclMemHandleMeta]

    # (start, end)
    _mem_regions: List[Tuple[int, int]] = None

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
        self.buffer_ptrs = []
        self.local_index_addr = []

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
            device_ptr = hccl_agent.get_registered_dev_addr(buffer_ptr)

            if (
                already_registered
            ):  # Re register memory to make it accessible by lmc_ops kernels
                lmc_ops.register_mapping(buffer_ptr, device_ptr, buffer_size)

            # Register the memory
            mem_handle_meta = HcclMemHandleMeta(
                mem_handle=mem_handle,
                buffer_ptr=buffer_ptr,
                buffer_size=buffer_size,
                page_size=page_size,
            )

            for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
                self.local_index_addr.append(base_addr)

            self.mem_handles.append(mem_handle_meta)
            self.buffer_ptrs.append(buffer_ptr)

    def close(self):
        for buffer_ptr in self.buffer_ptrs:
            self.agent.deregister_mem(buffer_ptr)
