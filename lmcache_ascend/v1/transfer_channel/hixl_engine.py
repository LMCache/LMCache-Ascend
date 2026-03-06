# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, List

import lmcache_ascend.c_ops as lmc_ops
import lmcache_ascend.hixl_npu_comms as hixl_comms
import torch
# Third Party
from lmcache.logging import init_logger
from lmcache.v1.rpc_utils import get_ip
# First Party
from lmcache_ascend.v1.rpc_utils import _find_free_port

# Local
from .buffer_config import (BufferConfig, BufferType, MemHandleMeta,
                            resolve_buffer_ref, resolve_local_addr)

logger = init_logger(__name__)


class HixlEngineWrapper:
    """Manages a single HIXL engine with multiple registered memory buffers."""

    def __init__(
        self,
        buffers: List[BufferConfig],
        buffer_pool: str = "0:0",
    ):
        self.engine = hixl_comms.Hixl()

        ip = get_ip()
        port = _find_free_port()
        self.engine_id = f"{ip}:{port}"

        options = {"BufferPool": buffer_pool}
        self.engine.initialize(self.engine_id, options)

        self.mem_handles: List[MemHandleMeta] = []
        self._uuid_to_handle: Dict[str, MemHandleMeta] = {}

        for buf in buffers:
            meta = self._register_buffer(buf)
            self.mem_handles.append(meta)
            self._uuid_to_handle[meta.uuid] = meta

    def _register_buffer(self, buf: BufferConfig) -> MemHandleMeta:
        """Register a single buffer with the HIXL engine."""
        buffer_ptr = buf.ptr
        buffer_size = buf.size
        page_size = buf.align_bytes
        device_id = torch.npu.current_device()
        is_device = buf.device_type == BufferType.NPU

        already_registered = lmc_ops.get_device_ptr(buffer_ptr) is not None
        if already_registered:
            lmc_ops.unregister_ptr(buffer_ptr)

        mem_type = hixl_comms.MEM_DEVICE if is_device else hixl_comms.MEM_HOST

        logger.debug(
            "Registering HIXL memory with type: %s, ptr in hex: %s, and the ptr: %s",
            mem_type,
            hex(buffer_ptr),
            buffer_ptr,
        )

        mem_handle = self.engine.register_mem(buffer_ptr, buffer_size, mem_type)

        logger.info(
            "HIXL memory registered mem_handle=0x%x",
            mem_handle,
        )

        dev_ptr = hixl_comms.get_dev_va(device_id, buffer_ptr, buffer_size)
        if dev_ptr is not None:
            lmc_ops.register_mapping(buffer_ptr, dev_ptr, buffer_size)
            logger.info(
                "Re-registered lmc_ops mapping via MemMappingManager (devVA=0x%x)",
                dev_ptr,
            )

        buffer_addrs = _build_addr_list(buffer_ptr, buffer_size, page_size)

        return MemHandleMeta(
            mem_handle=mem_handle,
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            page_size=page_size,
            local_buffer_addrs=buffer_addrs,
            buffer_type=buf.device_type,
        )

    def get_buffer_ref(self, data_ptr: int, page_index: int) -> tuple:
        return resolve_buffer_ref(self.mem_handles, data_ptr, page_index)

    def get_local_addr(self, ptr: int, idx: int) -> int:
        return resolve_local_addr(self.mem_handles, ptr, idx)

    def close(self):
        for meta in self.mem_handles:
            if meta.mem_handle is not None:
                try:
                    self.engine.deregister_mem(meta.mem_handle)
                except Exception as e:
                    logger.warning("Failed to deregister HIXL memory: %s", e)
        self.engine.finalize()


def _build_addr_list(buffer_ptr: int, buffer_size: int, page_size: int) -> list[int]:
    return list(range(buffer_ptr, buffer_ptr + buffer_size, page_size))


def _is_device_memory(ptr: int) -> bool:
    """Check if the pointer refers to device or host memory via ACL runtime."""
    return hixl_comms.is_device_memory(ptr)
