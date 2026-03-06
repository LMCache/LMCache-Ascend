# SPDX-License-Identifier: Apache-2.0
# Standard
import random

# Third Party
from lmcache.v1.memory_management import MemoryFormat, MixedMemoryAllocator
from utils import check_paged_kv_cache_equal, generate_kv_cache_paged_list_tuple_tensors
import mindspore as ms
import numpy as np
import pytest
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.skip("WIP")
def test_extract_and_load_back(num_tokens):
    pass


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("head_size", [256])
def test_multi_layer_kernel(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    device = "Ascend"

    num_blocks = 1000
    dtype = torch.bfloat16
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16
    dtype_ms = ms.bfloat16
    dtype_np = np.float16
    page_buffer_size = num_blocks * block_size

    kv_cache = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, "cpu", num_layers, num_heads, head_size, block_size, dtype
    )
    kv_cache = [
        tuple(
            ms.Tensor(kv.to(torch.float32).numpy(), dtype=dtype_ms).move_to(device)
            for kv in layer
        )
        for layer in kv_cache
    ]

    kv_cache_new = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, "cpu", num_layers, num_heads, head_size, block_size, dtype
    )
    kv_cache_new = [
        tuple(
            ms.Tensor(kv.to(torch.float32).numpy(), dtype=dtype_ms).move_to(device)
            for kv in layer
        )
        for layer in kv_cache_new
    ]

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device="cpu", dtype=int)
    slot_mapping = ms.Tensor(slot_mapping.numpy(), dtype=ms.int32).move_to(device)

    allocator = MixedMemoryAllocator(1024 * 1024 * 1024)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(num_layers * 2, dtype=torch.int64, device="cpu")

    for i in range(num_layers):
        kv_cache_pointers[i * 2 + 0] = kv_cache[i][0].data_ptr()  # Key pointer
        kv_cache_pointers[i * 2 + 1] = kv_cache[i][1].data_ptr()  # Value pointer

    # on ascend kv_cache_pointers need to be on device
    kv_cache_pointers = ms.Tensor(kv_cache_pointers.numpy(), dtype=ms.int64).move_to(
        device
    )

    kv_cache_pointers_new = torch.empty(num_layers * 2, dtype=torch.int64, device="cpu")

    for i in range(num_layers):
        kv_cache_pointers_new[i * 2 + 0] = kv_cache_new[i][0].data_ptr()
        kv_cache_pointers_new[i * 2 + 1] = kv_cache_new[i][1].data_ptr()

    kv_cache_pointers_new = ms.Tensor(
        kv_cache_pointers_new.numpy(), dtype=ms.int64
    ).move_to(device)

    gpu_buffer_shape = (2, num_tokens, hidden_dim_size)
    mem_format = MemoryFormat.KV_T2D
    memory_obj_new = allocator.allocate(
        gpu_buffer_shape, dtype=np.dtype(dtype_np), fmt=mem_format
    )

    lmc_ops.multi_layer_kv_transfer(
        memory_obj_new.tensor,
        kv_cache_pointers,
        slot_mapping,
        page_buffer_size,
        True,
        False,
        2,  # SEPARATE_KV
    )

    # wait for all the operations to finish
    ms.runtime.synchronize()

    lmc_ops.multi_layer_kv_transfer(
        memory_obj_new.tensor,
        kv_cache_pointers_new,
        slot_mapping,
        page_buffer_size,
        False,  # to gpu
        False,
        2,
    )

    # wait for all the operations to finish
    ms.runtime.synchronize()

    check_paged_kv_cache_equal(
        kv_cache, kv_cache_new, slot_mapping, num_heads=num_heads, head_size=head_size
    )


@pytest.mark.skip("WIP")
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
def test_multi_layer_kernel_use_mla(num_tokens):
    pass


@pytest.mark.skip("WIP")
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("token_major", [True, False])
def test_single_layer_kernel(num_tokens, token_major):
    pass
