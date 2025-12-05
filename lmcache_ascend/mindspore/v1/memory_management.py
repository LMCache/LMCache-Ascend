# Copyright 2024-2025 LMCache Authors.
# Copyright 2025 Ilya Yanok, Serapheim Dimitropoulos.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
import abc
import ctypes
import threading
import numpy as np
import mindspore as ms

# Third Party
import sortedcontainers
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from ._tensor import (
    get_data_ptr,
    get_numel,
    get_element_size,
    get_itemsize,
    get_dtype_compat,
    view_and_shape
)
from lmcache.v1.memory_management import (
    MemoryObjMetadata,
    TensorMemoryAllocator,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
    FreeBlock,
    BufferAllocator,
    PinMemoryAllocator
)
from lmcache.v1.system_detection import NUMAMapping
import lmcache_ascend.mindspore.c_ops as lmc_ops

logger = init_logger(__name__)


def MemoryObjMetadata_get_size(self):
    """
    Calculate the size of the memory object in bytes
    """
    num_elements = get_numel(self.shape)
    if num_elements == 0:
        return 0
    if self.dtype is None:
        return 0
    element_size = get_itemsize(self.dtype)
    size_in_bytes = num_elements * element_size
    return size_in_bytes

def _allocate_cpu_memory(
    size: int,
    numa_mapping: Optional[NUMAMapping] = None,
) -> torch.Tensor:
    return lmc_ops.create_pinned_tensor(size)

def TensorMemoryObj___init__(
    self,
    raw_data: torch.Tensor,
    metadata: MemoryObjMetadata,
    parent_allocator: Optional["MemoryAllocatorInterface"] = None,
    framework: str = None
):
    self.raw_data = raw_data
    self.meta = metadata
    self.valid = True
    self.lock = threading.Lock()
    self.parent_allocator = parent_allocator
    self._framework = framework

def TensorMemoryObj_get_size(self) -> int:
    num_elements = get_numel(self.raw_data)
    element_size = get_element_size(self.raw_data)
    size_in_bytes = num_elements * element_size
    return size_in_bytes

@property
def TensorMemoryObj_tensor(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
    if not self.valid:
        logger.warning("Trying to access an invalidated MemoryObj")
        return None
    assert self.meta.dtype is not None
    return view_and_shape(self.raw_data, self.meta.dtype, self.meta.shape)

@property
def TensorMemoryObj_byte_array(self) -> bytes:
    kv_chunk = self.tensor
    assert kv_chunk is not None
    num_bytes = get_numel(kv_chunk) * get_element_size(kv_chunk)
    ptr = get_data_ptr(kv_chunk)
    ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    byte_array = (ctypes.c_ubyte * num_bytes).from_address(
        ctypes.addressof(ubyte_ptr.contents)
    )
    return memoryview(byte_array)

def TensorMemoryAllocator___init__(self, tensor: Union[torch.Tensor, np.ndarray], align_bytes: int = TensorMemoryAllocator.ALIGN_BYTES):
    # NOTE (Gingfung:) changed to reshape to enable 
    assert self._is_uint8_type(tensor)
    self.buffer = tensor.reshape(-1)

    self.align_bytes = align_bytes

    self.explicit_list = sortedcontainers.SortedList(key=lambda x: x.start)

    el_size = get_numel(self.buffer)

    self.explicit_list.add(FreeBlock(start=0, size=el_size))

    # For debugging purposes
    self.num_active_allocations = 0
    self.total_allocated_size = 0

    self.stats_monitor = LMCStatsMonitor.GetOrCreate()

@staticmethod
@_lmcache_nvtx_annotate
def TensorMemoryAllocator__Compute_raw_size(shape: torch.Size, dtype: torch.dtype) -> int:
    return shape.numel() * get_itemsize(dtype)

def TensorMemoryAllocator__is_uint8_type(self, tensor: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor, np.ndarray):
        return tensor.dtype == np.uint8
    elif isinstance(tensor, torch.Tensor):
        return tensor.dtype == torch.uint8
    else:
        raise ValueError(f"tensor of type: {type(tensor)} not supported.")

@_lmcache_nvtx_annotate
def TensorMemoryAllocator_allocate(
    self,
    shape: Union[torch.Size, Tuple[int, ...]],
    dtype: Optional[torch.dtype],
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    parent_allocator: Optional["MemoryAllocatorInterface"] = None,
) -> Optional[TensorMemoryObj]:
    if not isinstance(shape, torch.Size):
        shape = torch.Size(shape)

    assert dtype is not None, "dtype must be specified"
    dtype = get_dtype_compat(dtype)

    # Calculate the size of the tensor
    raw_size = TensorMemoryAllocator._Compute_raw_size(shape, dtype)
    if raw_size % self.align_bytes != 0:
        aligned_size = TensorMemoryAllocator._Compute_aligned_size(
            raw_size, self.align_bytes
        )
    else:
        aligned_size = raw_size

    # Find the first block that fits the shape
    for block in self.explicit_list:
        if block.size >= aligned_size:
            break
    else:
        logger.debug(
            f"Failed to allocate memory for "
            f"tensor({shape}, {dtype}) because "
            "no memory is available"
        )
        return None

    # Do not add the block back if `block.size == aligned_size`
    self.explicit_list.remove(block)
    # Update the explicit list
    if block.size > aligned_size:
        self.explicit_list.add(
            FreeBlock(
                start=block.start + aligned_size,
                size=block.size - aligned_size,
            )
        )

    # TODO (Jiayi): need a flag to drop these debug ops
    # Update debug status
    self.total_allocated_size += aligned_size
    self.num_active_allocations += 1
    self.stats_monitor.update_local_cache_usage(self.total_allocated_size)

    # Allocate the block
    return TensorMemoryObj(
        raw_data=self.buffer[block.start : block.start + raw_size],
        metadata=MemoryObjMetadata(
            shape, dtype, block.start, aligned_size, 1, False, fmt
        ),
        parent_allocator=parent_allocator
    )

@_lmcache_nvtx_annotate
def TensorMemoryAllocator_batched_allocate(
    self,
    shape: Union[torch.Size, Tuple[int, ...]],
    dtype: Optional[torch.dtype],
    batch_size: int,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    parent_allocator: Optional["MemoryAllocatorInterface"] = None,
) -> Optional[List[TensorMemoryObj]]:
    """
    Batched allocate tensor memory objs with equal sizes.
    """
    if not isinstance(shape, torch.Size):
        shape = torch.Size(shape)

    assert dtype is not None, "dtype must be specified"
    dtype = get_dtype_compat(dtype)

    # Calculate the size of the tensor
    unit_raw_size = TensorMemoryAllocator._Compute_raw_size(shape, dtype)

    if unit_raw_size % self.align_bytes != 0:
        unit_aligned_size = TensorMemoryAllocator._Compute_aligned_size(
            unit_raw_size, self.align_bytes
        )
    else:
        unit_aligned_size = unit_raw_size

    total_aligned_size = unit_aligned_size * batch_size

    # Find the first block that fits the shape
    for block in self.explicit_list:
        if block.size >= total_aligned_size:
            break
    else:
        logger.debug(
            f"Failed to batched allocate memory for "
            f"{batch_size} tensor({shape}, {dtype}) because "
            "no memory is available"
        )
        return None

    # Do not add the block back if `block.size == aligned_size`
    self.explicit_list.remove(block)
    # Update the explicit list
    if block.size > total_aligned_size:
        self.explicit_list.add(
            FreeBlock(
                start=block.start + total_aligned_size,
                size=block.size - total_aligned_size,
            )
        )

    # TODO (Jiayi): need a flag to drop these debug ops
    # Update debug status
    self.total_allocated_size += total_aligned_size
    self.num_active_allocations += batch_size
    self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
    buffer_slice = self.buffer[block.start : block.start + total_aligned_size]
    raw_datas = np.array_split(buffer_slice, batch_size)
    # raw_datas = torch.chunk(
    #     self.buffer[block.start : block.start + total_aligned_size],
    #     batch_size,
    # )
    tensor_mem_objs = []
    temp_start = block.start
    for raw_data in raw_datas:
        tensor_mem_objs.append(
            TensorMemoryObj(
                raw_data=raw_data,
                metadata=MemoryObjMetadata(
                    shape, dtype, temp_start, unit_aligned_size, 1, False, fmt
                ),
                parent_allocator=parent_allocator
            )
        )
        temp_start += unit_aligned_size

    return tensor_mem_objs

def TensorMemoryAllocator_memcheck(self):
    """For debug purposes.
    Returns True is everything is fine, otherwise False.
    """
    clear = True
    logger.info("Checking memory allocator consistency")
    logger.info(f" - Total active allocations: {self.num_active_allocations}")
    logger.info(
        f" - Total allocated size: {self.total_allocated_size / 1048576} MB"
    )

    # Check the real total free size
    total_free_size = sum([block.size for block in self.explicit_list])
    logger.info(f" - Total free size: {total_free_size / 1048576} MB")

    # Check if the numbers are consistent
    if total_free_size + self.total_allocated_size != get_numel(self.buffer):
        logger.error("Memory allocator size is inconsistent")
        logger.error("This implies a bug in the memory allocator")
        clear = False

    # Check if the blocks are coalesced
    for prev, succ in zip(
        self.explicit_list[:-1], self.explicit_list[1:], strict=False
    ):
        if prev.can_be_coalesced(succ):
            logger.error("Memory allocator has non-coalesced blocks")
            logger.error("This implies a bug in the memory allocator")
            clear = False
    return clear


def MixedMemoryAllocator___init__(self, size: int, use_paging: bool = False, **kwargs):
    """
    :param int size: The size of the pinned memory in bytes.
    """

    self.buffer = lmc_ops.create_pinned_tensor(size)

    if use_paging:
        assert "shape" in kwargs, (
            "shape must be specified for paged memory allocator"
        )
        assert "dtype" in kwargs, (
            "dtype must be specified for paged memory allocator"
        )
        assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
        self.pin_allocator = PagedTensorMemoryAllocator(
            tensor=self.buffer,
            shape=kwargs["shape"],
            dtype=kwargs["dtype"],
            fmt=kwargs["fmt"],
        )
    else:
        self.pin_allocator = TensorMemoryAllocator(self.buffer)

    self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    self.buffer_allocator = BufferAllocator("cpu")


@_lmcache_nvtx_annotate
def AdHocMemoryAllocator_allocate(
    self,
    shape: Union[torch.Size, Tuple[int, ...]],
    dtype: Optional[torch.dtype],
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
) -> Optional[MemoryObj]:
    """
    Returns a dummy MemoryObj for testing purposes.
    """
    if not isinstance(shape, torch.Size):
        shape = torch.Size(shape)
    
    assert dtype is not None, "dtype must be specified"
    dtype = get_dtype_compat(dtype)

    # Return a dummy object with no actual memory allocation
    return TensorMemoryObj(
        raw_data = np.empty(shape, dtype=dtype),
        metadata=MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=0,
            phy_size=0,
            ref_count=1,
            is_pin=False,
            fmt=fmt,
        ),
        parent_allocator=self,
    )

@_lmcache_nvtx_annotate
def HostMemoryAllocator___init__(self, size: int):
    buffer = np.zeros(size, dtype=np.uint8)
    self.allocator = TensorMemoryAllocator(buffer)
    self.host_mem_lock = threading.Lock()

@_lmcache_nvtx_annotate
def GPUMemoryAllocator___init__(self, size: int, device="cuda", align_bytes: Optional[int] = None):
    self.tensor = np.zeros(size, dtype=np.uint8)
    kwargs = {}
    if align_bytes is not None:
        kwargs["align_bytes"] = align_bytes
    self.allocator = TensorMemoryAllocator(self.tensor, **kwargs)

    self.device_mem_lock = threading.Lock()

@_lmcache_nvtx_annotate
def PinMemoryAllocator___init__(self, size: int):
    self.buffer = np.zeros(size, dtype=np.uint8)
    self.allocator = TensorMemoryAllocator(self.buffer)
    self.host_mem_lock = threading.Lock()