# SPDX-License-Identifier: Apache-2.0
# Standard
import random

# Third Party
from lmcache.v1.memory_management import PinMemoryAllocator
from lmcache_tests.v1.utils import check_mem_obj_equal
import pytest
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops

# Local
from .utils import (
    check_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
    generate_kv_cache_paged_list_tuple_tensors,
)


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("chunk_size", [256, 512])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("head_size", [256])
def test_multi_layer_kernel_kvcache_merged_fmt(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    device = "npu"

    num_blocks = 256
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # lmc_ops.multi_layer_kv_transfer(memory_obj_new.tensor,
    #                                kv_cache_pointers, # TODO: initialize this
    #                                slot_mapping_temp,
    #                                kv_cache[0].device,
    #                                len(slot_mapping_temp), True)

    # layer by layer extract
    memory_obj_old_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = [2, num_layers, len(slot_mapping_temp), num_heads * head_size]

        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)
        for layer_id in range(num_layers):
            lmc_ops.load_and_reshape_flash(
                memory_obj_old.tensor,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        memory_obj_old_list.append(memory_obj_old)
    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Old extract time: ", elapsed_time_ms / 1000)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()

    # on ascend kv_cache_pointers need to be on device
    kv_cache_pointers = kv_cache_pointers.npu()

    memory_obj_new_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = [2, num_layers, len(slot_mapping_temp), num_heads * head_size]

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0].device,
            page_buffer_size,
            True,
            False,
            1,  # MERGED_KV
        )
        memory_obj_new_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)

    check_mem_obj_equal(
        memory_obj_old_list,
        memory_obj_new_list,
    )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    kv_cache_pointers_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()

    kv_cache_pointers_new = kv_cache_pointers_new.npu()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0].device,
            page_buffer_size,
            False,
            False,
            1,  # MERGED_KV
        )

    check_paged_kv_cache_equal(
        kv_cache, kv_cache_new, slot_mapping, num_heads=num_heads, head_size=head_size
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("chunk_size", [256, 512])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("head_size", [128])
def test_multi_layer_kernel_kvcache_separate_fmt(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    device = "npu"

    num_blocks = 1000
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )

    # for i in range(num_layers):
    #     kv_cache_pointers[i] = kv_cache[i].data_ptr()

    for i in range(num_layers):
        kv_cache_pointers[i * 2 + 0] = kv_cache[i][0].data_ptr()  # Key pointer
        kv_cache_pointers[i * 2 + 1] = kv_cache[i][1].data_ptr()  # Value pointer

    # on ascend kv_cache_pointers need to be on device
    kv_cache_pointers = kv_cache_pointers.npu()

    memory_obj_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = [2, num_layers, len(slot_mapping_temp), num_heads * head_size]

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0][0].device,
            page_buffer_size,
            True,
            False,
            2,  # SEPARATE_KV
        )
        memory_obj_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"New extract time: {elapsed_time_ms / 1000:.4f}s")

    # check_mem_obj_equal(
    #     memory_obj_old_list,
    #     memory_obj_new_list,
    # )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    kv_cache_pointers_new = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )

    for i in range(num_layers):
        kv_cache_pointers_new[i * 2 + 0] = kv_cache_new[i][0].data_ptr()
        kv_cache_pointers_new[i * 2 + 1] = kv_cache_new[i][1].data_ptr()

    kv_cache_pointers_new = kv_cache_pointers_new.npu()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_list[chunk_id]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0][0].device,
            page_buffer_size,
            False,  # to gpu
            False,
            2,
        )

    check_paged_kv_cache_equal(
        kv_cache, kv_cache_new, slot_mapping, num_heads=num_heads, head_size=head_size
    )

    mem_allocator.close()


# TODO: MLA is not supported for layerwise yet
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_heads", [8, 1])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize("vllm_two_major", [True, False])
def test_single_layer_kernel(
    num_tokens,
    num_layers,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    token_major,
    vllm_two_major,
):
    device = "npu"
    kvs = 2
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
        vllm_two_major=vllm_two_major,
    )
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
        vllm_two_major=vllm_two_major,
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    if token_major:
        tmp_gpu_buffer = torch.empty(
            (num_tokens, kvs, hidden_dim_size), dtype=dtype, device=device
        )
    else:
        tmp_gpu_buffer = torch.empty(
            (kvs, num_tokens, hidden_dim_size), dtype=dtype, device=device
        )

    for layer_id in range(num_layers):
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache[layer_id],
            slot_mapping,
            True,
            token_major,
            vllm_two_major,
        )
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_new[layer_id],
            slot_mapping,
            False,
            token_major,
            vllm_two_major,
        )
    torch.npu.synchronize()
    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )
