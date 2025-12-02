# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from enum import Enum, auto
from typing import List, Union, Tuple
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryAllocator,
    BufferAllocator,
    PinMemoryAllocator,
)
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

_IS_310P = None

def is_310p():
    global _IS_310P
    if _IS_310P is None:
        from lmcache_ascend import _build_info
        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P

class KVCacheFormat(Enum):
    """
    The storage format enumeration of KV cache is used to distinguish 
    the KV cache data structures of different versions of vLLM.
    
    The order of enum values MUST match the KVCacheFormat 
    definition in kernels/types.h to ensure correct interoperability 
    between Python and C++ code.
    """

    UNDEFINED = 0

    MERGED_KV = auto()
    """merge format (eg: vLLM 0.9.2 ...)
    layer: [num_kv, num_blocks, block_size, num_heads, head_dim]
    """

    SEPARATE_KV = auto()
    """Separation format (eg: vLLM 0.11.0+ ...)
    layer: tuple: (K_tensor, V_tensor)
    - K_tensor.shape = [num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [num_blocks, block_size, num_heads, head_dim]

    eg: kvcaches[0] = (K, V)
    """
    
    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV
    
    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV
    
    @staticmethod
    def detect(
        kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
    ) -> 'KVCacheFormat':
        if not kvcaches:
            return KVCacheFormat.UNDEFINED
        
        first_cache = kvcaches[0]
        
        if isinstance(first_cache, tuple):
            return KVCacheFormat.SEPARATE_KV
        elif isinstance(first_cache, torch.Tensor):
            if first_cache.shape[0] == 2:
                return KVCacheFormat.MERGED_KV
        
        return KVCacheFormat.UNDEFINED


# NOTE (Gingfung): it is not really used in v1, mainly for testing.
class AscendPinMemoryAllocator(PinMemoryAllocator):
    """Allocates memory in the pre-allocated pinned memory."""

    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """

        self.buffer = torch.empty(
            size, dtype=torch.uint8, device="cpu", pin_memory=True
        )

        if not is_310p():
            lmc_ops.host_register(self.buffer)

        if use_paging:
            assert "shape" in kwargs, (
                "shape must be specified for paged memory allocator"
            )
            assert "dtype" in kwargs, (
                "dtype must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shape=kwargs["shape"],
                dtype=kwargs["dtype"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    def close(self):
        pass


class AscendMixedMemoryAllocator(MixedMemoryAllocator):
    def __init__(self, size: int, use_paging: bool = False, **kwargs) -> None:
        """
        :param int size: The size of the pinned memory in bytes.
        """

        self.buffer = torch.empty(
            size, dtype=torch.uint8, device="cpu", pin_memory=True
        )

        if not is_310p():
            lmc_ops.host_register(self.buffer)

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

    def close(self):
        pass
