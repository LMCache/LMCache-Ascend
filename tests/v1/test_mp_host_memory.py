# SPDX-License-Identifier: Apache-2.0
"""Check native allocation integration with the normal Mixed allocator."""

# Standard
import ctypes

# Third Party
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
import lmcache
import pytest
import torch


@pytest.mark.parametrize("fail_release", [False, True])
def test_native_pinned_pair_works_with_normal_mixed(monkeypatch, fail_release):
    buffers, events = {}, []

    def allocate(size, flags):
        assert flags == 0
        buffer = (ctypes.c_uint8 * size)()
        ptr = ctypes.addressof(buffer)
        buffers[ptr] = buffer
        events.append("allocate")
        return ptr

    def free(ptr):
        nonlocal fail_release
        if fail_release:
            fail_release = False
            raise RuntimeError("ACL release failed")
        del buffers[ptr]
        events.append("free")

    # The native test exercises these C++ functions; only their ACL allocation
    # boundary is substituted here, leaving Mixed's construction/close intact.
    monkeypatch.setattr(lmcache.device_ops, "alloc_pinned_ptr", allocate)
    monkeypatch.setattr(lmcache.device_ops, "free_pinned_ptr", free)
    pool = MixedMemoryAllocator(4096, align_bytes=64)
    obj = pool.allocate(torch.Size([32]), torch.uint8, MemoryFormat.KV_2LTD)
    assert obj is not None
    obj.tensor.fill_(17)
    assert list(buffers[pool.buffer.data_ptr()])[:32] == [17] * 32
    pool.free(obj)
    assert events == ["allocate"]
    if fail_release:
        with pytest.raises(RuntimeError, match="ACL release failed"):
            pool.close()
        assert len(buffers) == 1
    pool.close()
    pool.close()
    assert events == ["allocate", "free"]
    assert buffers == {}
