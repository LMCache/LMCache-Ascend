# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
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
