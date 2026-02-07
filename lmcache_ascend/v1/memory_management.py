# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from typing import Optional
import threading

# Third Party
import torch

from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    PagedTensorMemoryAllocator,
    TensorMemoryAllocator,
)

logger = init_logger(__name__)


def GPUMemoryAllocator__init__(
    self,
    size: int,
    device="cuda",
    align_bytes: Optional[int] = None,
    use_paging: bool = False,
    **kwargs,
):
    """
    :param int size: The size of the GPU memory in bytes.
    :param Optional[int] align_bytes: The byte alignment for allocations.
    """
    # NOTE(niming): In sglang's 'transfer_to_npu', torch.cuda.is_available
    # is manually mocked to False to prevent CUDA-specific logic.
    # This implementation keeps the original device behavior.
    # torch.cuda.is_available = lambda: False
    if not torch.npu.is_available():
        device = "cpu"

    self.tensor = torch.empty(size, dtype=torch.uint8, device=device)

    self.allocator: MemoryAllocatorInterface
    if use_paging:
        assert "shapes" in kwargs, "shapes must be specified for paged memory allocator"
        assert "dtypes" in kwargs, "dtypes must be specified for paged memory allocator"
        assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
        self.allocator = PagedTensorMemoryAllocator(
            tensor=self.tensor,
            shapes=kwargs["shapes"],
            dtypes=kwargs["dtypes"],
            fmt=kwargs["fmt"],
        )
    else:
        kwargs = {}
        if align_bytes is not None:
            kwargs["align_bytes"] = align_bytes
        self.allocator = TensorMemoryAllocator(self.tensor, **kwargs)

    self.device_mem_lock = threading.Lock() if not use_paging else nullcontext()
