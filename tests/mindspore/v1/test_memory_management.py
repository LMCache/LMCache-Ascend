import pytest
import numpy as np

from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    GPUMemoryAllocator,
    HostMemoryAllocator,
    MemoryFormat,
    TensorMemoryAllocator,
)

from lmcache.v1.memory_management import (
    MixedMemoryAllocator
)

def check_allocator(allocator, max_size):
    # 512 * 512 * 4 = 1MB
    data1 = allocator.allocate([512, 512], dtype=np.dtype('float32'))
    assert data1 is not None
    assert data1.tensor.dtype == np.dtype('float32')
    assert data1.tensor.shape == (512, 512)

    # 1024 * 1024 * 2 = 2MB
    data2 = allocator.allocate([1024, 1024], dtype=np.dtype('bfloat16'))
    assert data2 is not None
    assert data2.tensor.dtype == np.dtype('bfloat16')
    assert data2.tensor.shape == (1024, 1024)

    # 2048 * 2048 * 1 = 4MB
    data3 = allocator.allocate([2048, 2048], dtype=np.dtype('int8'))
    assert data3 is not None
    assert data3.tensor.dtype == np.dtype('int8')
    assert data3.tensor.shape == (2048, 2048)

    allocator.free(data2)
    assert data2.tensor is None
    assert allocator.memcheck()

    allocator.free(data1)
    assert data1.tensor is None
    assert allocator.memcheck()

    allocator.free(data2)  # This should not crash

    data4 = allocator.allocate([3, 5, 7], dtype=np.dtype('float16'))
    assert data4 is not None
    assert data4.tensor.dtype == np.dtype('float16')
    assert data4.tensor.shape == (3, 5, 7)

    data_fail = allocator.allocate([max_size], dtype=np.dtype('float32'))  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.free(data1)
    allocator.free(data2)
    allocator.free(data3)
    allocator.free(data4)

    assert allocator.memcheck()

    # allocator.close()


@pytest.mark.parametrize(
    "use_paging",
    [
        False,
        # True
    ],
)
def test_tensor_allocator(use_paging):
    total_size = 1024 * 1024 * 128  # 128MB
    tensor_buffer = np.zeros(total_size, dtype=np.uint8)
    allocator = TensorMemoryAllocator(tensor_buffer)
    check_allocator(allocator, total_size)
    # allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_inplace_modification(alloc_cls):
    total_size = 1024
    allocator = alloc_cls(total_size)

    data = allocator.allocate([10], np.dtype('float32'))
    assert data is not None
    assert data.tensor.dtype == np.dtype('float32')
    assert data.tensor.shape == (10,)

    data.tensor.fill(1.0)
    assert np.all(data.tensor == 1.0)

    data.tensor[1] = 2.0
    assert data.tensor[1] == 2.0

    # allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_boundary_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    data1 = allocator.allocate([512, 10], np.dtype('float32'))
    allocator.allocate([512, 10], np.dtype('float32'))
    allocator.free(data1)

    # `FreeBlock` with size 0 shouldn't exist in the allocator
    allocator.allocate([512, 10], np.dtype('float32'))

    if isinstance(allocator, MixedMemoryAllocator):
        assert len(allocator.pin_allocator.explicit_list) == 1
    else:
        assert len(allocator.allocator.explicit_list) == 1

    # allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_batched_alloc(alloc_cls):
    total_size = 32 * 100 * 2 * 1024 * 2
    batch_size = 32
    allocator = alloc_cls(total_size)
    objs = allocator.batched_allocate(
        [100, 2, 1024], np.dtype('bfloat16'), batch_size, MemoryFormat.KV_T2D
    )

    assert len(objs) == batch_size
    for obj in objs:
        assert obj is not None
        assert obj.tensor is not None
        assert obj.tensor.dtype == np.dtype('bfloat16')
        assert obj.tensor.shape == (100, 2, 1024)
    allocator.batched_free(objs)

    if isinstance(allocator, MixedMemoryAllocator):
        assert len(allocator.pin_allocator.explicit_list) == 1
    else:
        assert len(allocator.allocator.explicit_list) == 1

    # allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        MixedMemoryAllocator,
    ],
)
def test_mixed_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    data1 = allocator.allocate([512, 0], None, MemoryFormat.BINARY_BUFFER)
    allocator.allocate([512, 10], np.dtype('float32'))
    allocator.free(data1)

    assert len(allocator.pin_allocator.explicit_list) == 1

    assert isinstance(data1, BytesBufferMemoryObj)

    assert len(data1.byte_array) == 512

    # allocator.close()
