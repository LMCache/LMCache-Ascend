# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List
import random

# Third Party
from utils import (
    check_mem_obj_equal,
    check_paged_kv_cache_equal,
    generate_kv_cache_paged,
    generate_kv_cache_paged_list_tensors,
    generate_mla_kv_cache_paged_list_tensors,
)
import pytest
import torch
import numpy as np
import mindspore as ms

# First Party
from lmcache.v1.memory_management import (
    PinMemoryAllocator,
    MixedMemoryAllocator,
    MemoryFormat,
)
import lmcache_ascend.mindspore.c_ops as lmc_ops


def _tuple_kv_to_blob(
    kv_tensors,
) -> torch.Tensor:
    k_temp = []
    v_temp = []
    for kv_layer in kv_tensors:
        k_temp.append(kv_layer[0])
        v_temp.append(kv_layer[1])
    k_tensor_blob = torch.stack(k_temp)
    v_tensor_blob = torch.stack(v_temp)

    # kv_tensors: [num_layer, 2, num_tok, num_kv_head, head_size]
    kv_tensors_flatten = torch.stack((k_tensor_blob, v_tensor_blob))
    kv_tensors_flatten = kv_tensors_flatten.permute([1, 0, 2, 3, 4])

    return kv_tensors_flatten


def _slice_kv_at(
    start_idx: int,
    kv_tensors: torch.Tensor,
    chunk_size: int,
) -> List[torch.Tensor]:
    return [
        x.contiguous()
        for x in list(
            torch.split(
                kv_tensors[:, :, start_idx:, ...],
                chunk_size,
                dim=2,
            )
        )
    ]


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.skip("WIP")
def test_extract_and_load_back(num_tokens):
    pass


@pytest.mark.skip("WIP")
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
def test_multi_layer_kernel(num_tokens):
    pass


@pytest.mark.skip("WIP")
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
def test_multi_layer_kernel_use_mla(num_tokens):
    pass


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("token_major", [True, False])
def test_single_layer_kernel(num_tokens, token_major):
    device = "Ascend"

    num_layers = 32 # Must match generate_kv_cache_paged_list_tensors
    num_blocks = 1000
    block_size = 16
    num_heads = 8
    head_size = 128
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16
    dtype_ms = ms.bfloat16
    dtype_np = np.float16

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, "cpu", block_size, dtype
    )
    kv_cache = [ms.Tensor(kv.to(torch.float32).numpy(), dtype=dtype_ms).move_to(device) for kv in kv_cache]

    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks, "cpu", block_size, dtype
    )
    kv_cache_new = [ms.Tensor(kv.to(torch.float32).numpy(), dtype=dtype_ms).move_to(device) for kv in kv_cache_new]

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device="cpu", dtype=int)
    slot_mapping = ms.Tensor(slot_mapping.numpy(), dtype=ms.int32).move_to(device)

    allocator = MixedMemoryAllocator(1024 * 1024 * 1024)
    if token_major:
        gpu_buffer_shape = (num_tokens, 2, hidden_dim_size)
        mem_format = MemoryFormat.KV_2LTD
    else:
        gpu_buffer_shape = (2, num_tokens, hidden_dim_size)
        mem_format = MemoryFormat.KV_T2D
    tmp_gpu_buffer = allocator.allocate(
        gpu_buffer_shape, dtype=np.dtype(dtype_np), fmt=mem_format
    ).tensor

    k_0 = kv_cache[0][0][slot_mapping[0]//block_size][slot_mapping[0]%block_size][0][0]
    k_new_0 = kv_cache_new[0][0][slot_mapping[0]//block_size][slot_mapping[0]%block_size][0][0]
    via = tmp_gpu_buffer[0][0]
    print(f"Sample key is {k_0} aiming to re-write (via lmc) {k_new_0} (should NOT yet match) transfer via {via}")
    for layer_id in range(num_layers):
        # PagedBuffer (kv_cache) to LMCache (tmp_gpu_buffer)
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache[layer_id][0],
            kv_cache[layer_id][1],
            slot_mapping,
            True,
            token_major,
        )

        # PagedBuffer (tmp_gpu_buffer) to LMCache' (kv_cache_new)
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_new[layer_id][0],
            kv_cache_new[layer_id][1],
            slot_mapping,
            False,
            token_major,
        )
    k_0 = kv_cache[0][0][slot_mapping[0]//block_size][slot_mapping[0]%block_size][0][0]
    k_new_0 = kv_cache_new[0][0][slot_mapping[0]//block_size][slot_mapping[0]%block_size][0][0]
    via = tmp_gpu_buffer[0][0]
    print(f"New sample key is now {k_new_0} should match original {k_0} (transfered via {via})")

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        num_heads,
        head_size,
    )
