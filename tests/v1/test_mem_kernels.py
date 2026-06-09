# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

import random

# Third Party
from lmcache.v1.memory_management import PinMemoryAllocator
from lmcache_tests.v1.utils import check_mem_obj_equal
import pytest
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops
from lmcache_ascend.v1.kv_format import KVCacheFormat

# Local
from .utils import (
    check_paged_kv_cache_equal,
    generate_dsa_c8_kv_cache,
    generate_dsa_kv_cache,
    generate_kv_cache_paged_list_tensors,
    generate_kv_cache_paged_list_tuple_tensors,
    generate_mla_kv_cache,
)



def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


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
    hidden_dim_size = num_heads * head_size
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # layer by layer extract
    memory_obj_old_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

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
    torch.npu.synchronize()
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
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

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
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_new_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)

    check_mem_obj_equal(
        memory_obj_old_list,
        memory_obj_new_list,
    )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
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
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    check_paged_kv_cache_equal(
        kv_cache, kv_cache_new, slot_mapping, num_heads=num_heads, head_size=head_size
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
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
    hidden_dim_size = num_heads * head_size
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
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

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
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.npu.synchronize()
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
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    check_paged_kv_cache_equal(
        kv_cache, kv_cache_new, slot_mapping, num_heads=num_heads, head_size=head_size
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("chunk_size", [256, 512])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("head_size", [256])
def test_fused_multi_layer_kvcache_merged_fmt(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    """
    - from_gpu: Verify that the output of the fused method is
      consistent with the baseline method.
    - to_gpu: Verify that the KV cache content is correct
      after data is written back.
    """
    device = "npu"
    num_blocks = 256
    dtype = torch.bfloat16
    kvs = 2
    hidden_dim_size = num_heads * head_size

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # kv_cache_pointers
    kv_cache_pointers = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()
    kv_cache_pointers = kv_cache_pointers.npu()

    # Staging buffer for fused method
    staging_buffer = torch.empty(
        [kvs, num_layers, chunk_size, hidden_dim_size], dtype=dtype, device=device
    )

    # from_gpu: Baseline method for data extraction
    memory_obj_baseline_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([kvs, num_layers, len(slot_temp), hidden_dim_size])

        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            1,  # MERGED_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_baseline_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"\nBaseline from_gpu time: {elapsed_time_ms:.6f} ms")

    # from_gpu: Fused method for data extraction
    memory_obj_fused_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([kvs, num_layers, len(slot_temp), hidden_dim_size])
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        staging_cache = staging_buffer[:, :, : len(slot_temp), :]
        lmc_ops.fused_multi_layer_kv_transfer(
            memory_obj.tensor,
            staging_cache,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            1,  # MERGED_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_fused_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Fused from_gpu time: {elapsed_time_ms:.6f} ms")

    check_mem_obj_equal(memory_obj_baseline_list, memory_obj_fused_list)

    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )

    kv_cache_pointers_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()
    kv_cache_pointers_new = kv_cache_pointers_new.npu()

    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_fused_list[chunk_id].tensor,
            kv_cache_pointers_new,
            slot_temp,
            kv_cache_new[0].device,
            page_buffer_size,
            False,  # to_gpu
            False,
            1,  # MERGED_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("chunk_size", [256, 512])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("head_size", [128])
def test_fused_multi_layer_kvcache_separate_fmt(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    """
    - from_gpu: Verify that the output of the fused method is
      consistent with the baseline method.
    - to_gpu: Verify that the KV cache content is correct
      after data is written back.
    """
    device = "npu"
    num_blocks = 1000
    dtype = torch.bfloat16
    kvs = 2
    hidden_dim_size = num_heads * head_size

    kv_cache = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    kv_cache_pointers = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )

    for i in range(num_layers):
        kv_cache_pointers[i * 2 + 0] = kv_cache[i][0].data_ptr()  # Key pointer
        kv_cache_pointers[i * 2 + 1] = kv_cache[i][1].data_ptr()  # Value pointer
    kv_cache_pointers = kv_cache_pointers.npu()

    # Staging buffer for fused method
    staging_buffer = torch.empty(
        [kvs, num_layers, chunk_size, hidden_dim_size], dtype=dtype, device=device
    )

    # from_gpu: Baseline method for data extraction
    memory_obj_baseline_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([kvs, num_layers, len(slot_temp), hidden_dim_size])
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,  # use_mla
            2,  # SEPARATE_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_baseline_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"\nBaseline from_gpu time: {elapsed_time_ms:.6f} ms")

    # from_gpu: Fused method for data extraction
    memory_obj_fused_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([kvs, num_layers, len(slot_temp), hidden_dim_size])

        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        staging_cache = staging_buffer[:, :, : len(slot_temp), :]
        lmc_ops.fused_multi_layer_kv_transfer(
            memory_obj.tensor,
            staging_cache,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,  # use_mla
            2,  # SEPARATE_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_fused_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Fused from_gpu time: {elapsed_time_ms:.6f} ms")

    check_mem_obj_equal(memory_obj_baseline_list, memory_obj_fused_list)

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

    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_fused_list[chunk_id].tensor,
            kv_cache_pointers_new,
            slot_temp,
            kv_cache_new[0][0].device,
            page_buffer_size,
            False,  # to_gpu
            False,  # use_mla
            2,  # SEPARATE_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
    )

    mem_allocator.close()


# TODO: MLA is not supported for layerwise yet
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
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
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
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
            1,  # MERGED KV
            token_major,
            vllm_two_major,
        )
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_new[layer_id],
            slot_mapping,
            False,
            1,  # MERGED KV
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


# TODO: MLA is not supported for layerwise yet
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_heads", [8, 1])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("token_major", [True, False])
def test_single_layer_kernel_separate_kv(
    num_tokens,
    num_layers,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    token_major,
):
    """
    Test single_layer_kv_transfer with SEPARATE_KV format
    - from_gpu: GPU KV cache -> staging buffer
    - to_gpu: staging buffer -> GPU KV cache
    - Verify: src == dst
    """
    device = "npu"
    kvs = 2
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16

    # Generate SEPARATE_KV format caches: List[Tuple[Tensor, Tensor]]
    kv_cache_src = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks,
        device,
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
    )

    kv_cache_dst = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks,
        device,
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    # Staging buffer
    if token_major:
        tmp_gpu_buffer = torch.empty(
            (num_tokens, kvs, hidden_dim_size), dtype=dtype, device=device
        )
    else:
        tmp_gpu_buffer = torch.empty(
            (kvs, num_tokens, hidden_dim_size), dtype=dtype, device=device
        )

    # Transfer test: GPU -> staging -> GPU
    for layer_id in range(num_layers):
        # Convert tuple to list for C++ interface

        # from_gpu: GPU cache -> staging buffer
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_src[layer_id],
            slot_mapping,
            True,  # from_gpu
            2,  # SEPARATE KV
            token_major,
            False,  # vllm_two_major (not used for SEPARATE KV)
        )

        # to_gpu: staging buffer -> GPU cache
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_dst[layer_id],
            slot_mapping,
            False,  # to_gpu
            2,  # SEPARATE KV
            token_major,
            False,
        )

    torch.npu.synchronize()

    # Verify correctness
    check_paged_kv_cache_equal(
        kv_cache_src,
        kv_cache_dst,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=False,  # Not applicable for SEPARATE KV
        kv_format=2,  # SEPARATE KV
    )


# TODO: MLA is not supported for layerwise yet
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("num_chunks", [1, 3, 5])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize("vllm_two_major", [True, False])
def test_batched_fused_single_layer_kernel(
    num_tokens,
    num_layers,
    num_chunks,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    token_major,
    vllm_two_major,
):
    """
    - Baseline: 1 kernel call + N torch.copy_ calls
    - Batched fusion: batched_fused_single_layer_kv_transfer
    """
    device = "npu"
    kvs = 2
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16

    # Compute chunk partitioning
    base_chunk_size = num_tokens // num_chunks
    remainder = num_tokens % num_chunks

    chunk_sizes = []
    chunk_offsets = []
    current_offset = 0

    for i in range(num_chunks):
        size = base_chunk_size + (1 if i < remainder else 0)
        chunk_sizes.append(size)
        chunk_offsets.append(current_offset)
        current_offset += size

    kv_cache_src = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )

    kv_cache_dst_baseline = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )

    kv_cache_dst_fused = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )

    slot_mapping_list = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping_full = torch.tensor(slot_mapping_list, device=device)

    # Buffer shapes
    def get_buffer_shape(n_tokens):
        if token_major:
            return (n_tokens, kvs, hidden_dim_size)
        else:
            return (kvs, n_tokens, hidden_dim_size)

    full_buffer_shape = get_buffer_shape(num_tokens)
    chunk_buffer_shapes = [get_buffer_shape(chunk_sizes[i]) for i in range(num_chunks)]

    # Staging buffers
    staging_cache_baseline = torch.empty(full_buffer_shape, dtype=dtype, device=device)
    staging_cache_fused = torch.empty(full_buffer_shape, dtype=dtype, device=device)

    total_cpu_bytes = 0
    for chunk_size in chunk_sizes:
        chunk_bytes = chunk_size * kvs * hidden_dim_size * torch.finfo(dtype).bits // 8
        total_cpu_bytes += chunk_bytes * num_layers
    total_cpu_bytes *= 2
    total_cpu_bytes = int(total_cpu_bytes * 1.2)

    mem_allocator = PinMemoryAllocator(total_cpu_bytes)

    try:

        def create_chunked_cpu_buffers():
            cpu_memory_objs = []
            cpu_tensors = []
            for layer_id in range(num_layers):
                layer_memory_objs = []
                layer_tensors = []
                for chunk_id in range(num_chunks):
                    chunk_shape = torch.Size(chunk_buffer_shapes[chunk_id])
                    memory_obj = mem_allocator.allocate(chunk_shape, dtype)
                    layer_memory_objs.append(memory_obj)
                    layer_tensors.append(memory_obj.tensor)
                cpu_memory_objs.append(layer_memory_objs)
                cpu_tensors.append(layer_tensors)
            return cpu_memory_objs, cpu_tensors

        cpu_mem_objs_baseline, cpu_baseline = create_chunked_cpu_buffers()
        cpu_mem_objs_fused, cpu_fused = create_chunked_cpu_buffers()

        # from_gpu: Baseline
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.single_layer_kv_transfer(
                staging_cache_baseline,
                kv_cache_src[layer_id],
                slot_mapping_full,
                True,
                1,  # MERGED KV
                token_major,
                vllm_two_major,
            )

            for chunk_id in range(num_chunks):
                offset = chunk_offsets[chunk_id]
                chunk_size = chunk_sizes[chunk_id]

                if token_major:
                    staging_chunk = staging_cache_baseline[offset : offset + chunk_size]
                else:
                    staging_chunk = staging_cache_baseline[
                        :, offset : offset + chunk_size
                    ]

                cpu_baseline[layer_id][chunk_id].copy_(staging_chunk, non_blocking=True)

        end_event.record()
        torch.npu.synchronize()
        baseline_from_time = start_event.elapsed_time(end_event)

        # from_gpu: Batched Fused
        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.batched_fused_single_layer_kv_transfer(
                cpu_fused[layer_id],
                staging_cache_fused,
                kv_cache_src[layer_id],
                slot_mapping_full,
                chunk_offsets,
                chunk_sizes,
                True,
                1,  # MERGED KV
                token_major,
                vllm_two_major,
            )

        end_event.record()
        torch.npu.synchronize()
        fused_from_time = start_event.elapsed_time(end_event)

        print(
            f"\n[from_gpu] Baseline: {baseline_from_time:.3f} ms, "
            f"Fused: {fused_from_time:.3f} ms"
        )

        # Verify from_gpu correctness
        for layer_id in range(num_layers):
            for chunk_id in range(num_chunks):
                assert torch.equal(
                    cpu_baseline[layer_id][chunk_id], cpu_fused[layer_id][chunk_id]
                ), f"from_gpu mismatch at layer={layer_id}, chunk={chunk_id}"

        # to_gpu: Baseline
        start_event.record()

        for layer_id in range(num_layers):
            for chunk_id in range(num_chunks):
                offset = chunk_offsets[chunk_id]
                chunk_size = chunk_sizes[chunk_id]

                if token_major:
                    staging_chunk = staging_cache_baseline[offset : offset + chunk_size]
                else:
                    staging_chunk = staging_cache_baseline[
                        :, offset : offset + chunk_size
                    ]

                staging_chunk.copy_(cpu_baseline[layer_id][chunk_id], non_blocking=True)

            lmc_ops.single_layer_kv_transfer(
                staging_cache_baseline,
                kv_cache_dst_baseline[layer_id],
                slot_mapping_full,
                False,
                1,  # MERGED KV
                token_major,
                vllm_two_major,
            )

        end_event.record()
        torch.npu.synchronize()
        baseline_to_time = start_event.elapsed_time(end_event)

        # to_gpu: Batched Fused
        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.batched_fused_single_layer_kv_transfer(
                cpu_baseline[layer_id],
                staging_cache_fused,
                kv_cache_dst_fused[layer_id],
                slot_mapping_full,
                chunk_offsets,
                chunk_sizes,
                False,
                1,  # MERGED KV
                token_major,
                vllm_two_major,
            )

        end_event.record()
        torch.npu.synchronize()
        fused_to_time = start_event.elapsed_time(end_event)

        print(
            f"[to_gpu] Baseline: {baseline_to_time:.3f} ms, "
            f"Fused: {fused_to_time:.3f} ms"
        )

        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst_baseline,
            slot_mapping_full,
            num_heads=num_heads,
            head_size=head_size,
            vllm_two_major=vllm_two_major,
        )

        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst_fused,
            slot_mapping_full,
            num_heads=num_heads,
            head_size=head_size,
            vllm_two_major=vllm_two_major,
        )

    finally:
        mem_allocator.close()
        del kv_cache_src
        del kv_cache_dst_baseline
        del kv_cache_dst_fused
        del staging_cache_baseline
        del staging_cache_fused
        del cpu_baseline
        del cpu_fused
        torch.npu.empty_cache()


@pytest.mark.parametrize("num_tokens", [256, 1024, 2048])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("num_chunks", [1, 3, 5])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("token_major", [True, False])
def test_batched_fused_single_layer_kernel_separate_kv(
    num_tokens,
    num_layers,
    num_chunks,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    token_major,
):
    """
    Test for SEPARATE_KV format (K and V stored separately)
    """
    device = "npu"
    kvs = 2
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16

    # Compute chunk partitioning
    base_chunk_size = num_tokens // num_chunks
    remainder = num_tokens % num_chunks

    chunk_sizes = []
    chunk_offsets = []
    current_offset = 0

    for i in range(num_chunks):
        size = base_chunk_size + (1 if i < remainder else 0)
        chunk_sizes.append(size)
        chunk_offsets.append(current_offset)
        current_offset += size

    # Generate KV caches with SEPARATE_KV format
    kv_cache_src = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    kv_cache_dst = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    slot_mapping_list = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping_full = torch.tensor(slot_mapping_list, device=device)

    def get_buffer_shape(n_tokens):
        if token_major:
            return (n_tokens, kvs, hidden_dim_size)
        else:
            return (kvs, n_tokens, hidden_dim_size)

    full_buffer_shape = get_buffer_shape(num_tokens)

    staging_cache = torch.empty(full_buffer_shape, dtype=dtype, device=device)

    total_cpu_bytes = 0
    for chunk_size in chunk_sizes:
        chunk_bytes = chunk_size * kvs * hidden_dim_size * torch.finfo(dtype).bits // 8
        total_cpu_bytes += chunk_bytes * num_layers
    total_cpu_bytes = int(total_cpu_bytes * 1.2)

    mem_allocator = PinMemoryAllocator(total_cpu_bytes)

    try:
        cpu_memory_objs = []
        cpu_tensors = []
        for layer_id in range(num_layers):
            layer_memory_objs = []
            layer_tensors = []
            for chunk_id in range(num_chunks):
                chunk_shape = torch.Size(get_buffer_shape(chunk_sizes[chunk_id]))
                memory_obj = mem_allocator.allocate(chunk_shape, dtype)
                layer_memory_objs.append(memory_obj)
                layer_tensors.append(memory_obj.tensor)
            cpu_memory_objs.append(layer_memory_objs)
            cpu_tensors.append(layer_tensors)

        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.batched_fused_single_layer_kv_transfer(
                cpu_tensors[layer_id],
                staging_cache,
                kv_cache_src[layer_id],
                slot_mapping_full,
                chunk_offsets,
                chunk_sizes,
                True,  # from_gpu
                2,  # SEPARATE_KV
                token_major,
                False,
            )

        end_event.record()
        torch.npu.synchronize()
        from_gpu_time = start_event.elapsed_time(end_event)

        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.batched_fused_single_layer_kv_transfer(
                cpu_tensors[layer_id],
                staging_cache,
                kv_cache_dst[layer_id],
                slot_mapping_full,
                chunk_offsets,
                chunk_sizes,
                False,  # to_gpu
                2,  # SEPARATE_KV
                token_major,
                False,
            )

        end_event.record()
        torch.npu.synchronize()
        to_gpu_time = start_event.elapsed_time(end_event)

        print(
            f"\n[SEPARATE_KV] from_gpu: {from_gpu_time:.3f} ms, "
            f"to_gpu: {to_gpu_time:.3f} ms"
        )

        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_full,
            num_heads=num_heads,
            head_size=head_size,
            vllm_two_major=False,
            kv_format=2,  # SEPARATE_KV
        )

    finally:
        mem_allocator.close()
        del kv_cache_src
        del kv_cache_dst
        del staging_cache
        del cpu_memory_objs
        del cpu_tensors
        torch.npu.empty_cache()


@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [128])
def test_multi_layer_kv_transfer_mla_format(
    num_tokens,
    num_kv_heads,
    chunk_size,
    num_layers,
    block_size,
    kv_lora_rank,
    qk_rope_head_dim,
):
    """
    Test MLA (Multilayer Attention) format KV cache transfer.
    - from_gpu: Extract KV cache from paged memory to CPU buffer
    - to_gpu: Write KV cache from CPU buffer back to paged memory
    - Verify correctness
    """
    device = "npu"
    num_blocks = 1000
    dtype = torch.bfloat16

    # Generate MLA format KV cache
    kv_cache_src = generate_mla_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, page_buffer_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # Prepare kv cache pointers
    kv_cache_pointers = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache = kv_cache_src[i]
        kv_cache_pointers[i * 2 + 0] = k_cache.data_ptr()
        kv_cache_pointers[i * 2 + 1] = v_cache.data_ptr()
    kv_cache_pointers = kv_cache_pointers.npu()

    # from_gpu: Extract KV cache
    memory_obj_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    total_hidden_dims = kv_lora_rank + qk_rope_head_dim
    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([1, num_layers, len(slot_temp), total_hidden_dims])
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache_src[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            3,  # MLA_KV
            kv_lora_rank,  # k_hidden_dims
            qk_rope_head_dim,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"\nMLA from_gpu time: {elapsed_time_ms:.6f} ms")

    # Generate destination KV cache
    kv_cache_dst = generate_mla_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        dtype=dtype,
    )

    # Prepare destination pointers
    kv_cache_pointers_dst = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache = kv_cache_dst[i]
        kv_cache_pointers_dst[i * 2 + 0] = k_cache.data_ptr()
        kv_cache_pointers_dst[i * 2 + 1] = v_cache.data_ptr()
    kv_cache_pointers_dst = kv_cache_pointers_dst.npu()

    # to_gpu: Write back
    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_list[chunk_id].tensor,
            kv_cache_pointers_dst,
            slot_temp,
            kv_cache_dst[0][0].device,
            page_buffer_size,
            False,  # to_gpu
            False,
            3,  # MLA_KV
            kv_lora_rank,  # k_hidden_dims
            qk_rope_head_dim,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    # Verify correctness
    check_paged_kv_cache_equal(
        kv_cache_src,
        kv_cache_dst,
        slot_mapping,
        num_heads=num_kv_heads,
        head_size=128,
        kv_format=3,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [128])
@pytest.mark.parametrize("dsa_head_dim", [128])
def test_multi_layer_kv_transfer_dsa_format(
    num_tokens,
    num_kv_heads,
    chunk_size,
    num_layers,
    block_size,
    kv_lora_rank,
    qk_rope_head_dim,
    dsa_head_dim,
):
    """
    Test DSA (Deep Sparse Attention) format KV cache transfer.
    - from_gpu: Extract KV cache from paged memory to CPU buffer
    - to_gpu: Write KV cache from CPU buffer back to paged memory
    - Verify correctness
    """
    device = "npu"
    num_blocks = 1000
    dtype = torch.bfloat16

    # Generate DSA format KV cache
    kv_cache_src = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, page_buffer_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # Prepare kv cache pointers
    kv_cache_pointers = torch.empty(
        num_layers * 3, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache, dsa_k_cache = kv_cache_src[i]
        kv_cache_pointers[i * 3 + 0] = k_cache.data_ptr()
        kv_cache_pointers[i * 3 + 1] = v_cache.data_ptr()
        kv_cache_pointers[i * 3 + 2] = dsa_k_cache.data_ptr()
    kv_cache_pointers = kv_cache_pointers.npu()

    # from_gpu: Extract KV cache
    memory_obj_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    total_hidden_dims = kv_lora_rank + qk_rope_head_dim + dsa_head_dim
    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([1, num_layers, len(slot_temp), total_hidden_dims])
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache_src[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            4,  # DSA_KV
            kv_lora_rank,  # k_hidden_dims
            qk_rope_head_dim,  # v_hidden_dims
            dsa_head_dim,  # dsa_hidden_dims
        )
        memory_obj_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"\nDSA from_gpu time: {elapsed_time_ms:.6f} ms")

    # Generate destination KV cache
    kv_cache_dst = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )

    # Prepare destination pointers
    kv_cache_pointers_dst = torch.empty(
        num_layers * 3, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache, dsa_k_cache = kv_cache_dst[i]
        kv_cache_pointers_dst[i * 3 + 0] = k_cache.data_ptr()
        kv_cache_pointers_dst[i * 3 + 1] = v_cache.data_ptr()
        kv_cache_pointers_dst[i * 3 + 2] = dsa_k_cache.data_ptr()
    kv_cache_pointers_dst = kv_cache_pointers_dst.npu()

    # to_gpu: Write back
    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_list[chunk_id].tensor,
            kv_cache_pointers_dst,
            slot_temp,
            kv_cache_dst[0][0].device,
            page_buffer_size,
            False,  # to_gpu
            False,
            4,  # DSA_KV
            kv_lora_rank,  # k_hidden_dims
            qk_rope_head_dim,  # v_hidden_dims
            dsa_head_dim,  # dsa_hidden_dims
        )

    # Verify correctness
    check_paged_kv_cache_equal(
        kv_cache_src,
        kv_cache_dst,
        slot_mapping,
        num_heads=num_kv_heads,
        head_size=128,
        kv_format=4,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
    )

    mem_allocator.close()

@pytest.mark.parametrize(
    "num_tokens,chunk_size",
    [
        (64, 512),
        (127, 512),
        (128, 512),
        (129, 512),
        (255, 512),
        (256, 512),
        (257, 512),
        (511, 512),
        (512, 512),
        (513, 512),
        (1024, 256),
        (1025, 512),
        (40959, 512),
        (40960, 512),
        (40961, 512),
    ],
)
@pytest.mark.parametrize("num_kv_heads", [8, 1])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [128])
@pytest.mark.parametrize("dsa_head_dim", [128])
def test_multi_layer_kv_transfer_dsa_c8_format(
    num_tokens,
    num_kv_heads,
    chunk_size,
    num_layers,
    block_size,
    kv_lora_rank,
    qk_rope_head_dim,
    dsa_head_dim,
):
    """DSA+C8: byte-oriented LMCache chunk (uint8) and four paged pointers per layer."""
    # Unique scope: this is the only direct c_ops multi-plane round-trip (store+load+parity).
    # Other multi-plane tests route through connector wrappers or only verify store-side layout.
    from lmcache_ascend.v1.kv_layer_groups import _multi_plane_lmc_row_bytes

    device = "npu"
    num_blocks = max((num_tokens // block_size) * 2, 256)
    kv_dtype = torch.bfloat16

    kv_cache_src = generate_dsa_c8_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        kv_dtype=kv_dtype,
    )
    page_buffer_size = num_blocks * block_size
    k0, v0, d0, s0 = kv_cache_src[0]
    kb = int(k0.numel() * k0.element_size() // page_buffer_size)
    vb = int(v0.numel() * v0.element_size() // page_buffer_size)
    db = int(d0.numel() * d0.element_size() // page_buffer_size)
    sb = int(s0.numel() * s0.element_size() // page_buffer_size)
    plane_bytes = (kb, vb, db, sb)

    slot_mapping = torch.arange(0, num_tokens, device=device, dtype=torch.long)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    # Regression guard: _multi_plane_layer_block_bytes previously asserted
    # plane_stride % 32 == 0, which fires for the scale plane (hd=2, sb=2) when
    # a partial last chunk has a token count not divisible by 16.  Verify every
    # chunk size — including partial ones — is handled without error, and that
    # the returned value matches the kernel's AlignUp32Bytes accumulation.
    for chunk_toks in {int(c.shape[0]) for c in slot_mapping_chunked}:
        block = _multi_plane_layer_block_bytes(list(plane_bytes), chunk_toks)
        expected = sum((hd * chunk_toks + 31) & ~31 for hd in plane_bytes)
        assert block == expected, (
            f"_multi_plane_layer_block_bytes mismatch for chunk_toks={chunk_toks}: "
            f"got {block}, expected {expected}"
        )

    row_bytes = _multi_plane_lmc_row_bytes(list(plane_bytes), chunk_size)
    pinned_cpu_size = max(
        256 * 1024 * 1024,
        row_bytes * num_layers * num_tokens * 2,
    )
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    kv_cache_pointers = torch.empty(
        num_layers * 4, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache, dsa_k_cache, dsa_k_scale = kv_cache_src[i]
        kv_cache_pointers[i * 4 + 0] = k_cache.data_ptr()
        kv_cache_pointers[i * 4 + 1] = v_cache.data_ptr()
        kv_cache_pointers[i * 4 + 2] = dsa_k_cache.data_ptr()
        kv_cache_pointers[i * 4 + 3] = dsa_k_scale.data_ptr()
    kv_cache_pointers = kv_cache_pointers.npu()

    def _dsa_c8_multi_plane_xfer(
        mem_tensor: torch.Tensor,
        ptrs: torch.Tensor,
        slots: torch.Tensor,
        *,
        is_store: bool,
    ) -> None:
        n_tok = int(slots.shape[0])
        chunk_tensor = mem_tensor
        if chunk_tensor.dtype != torch.uint8:
            chunk_tensor = chunk_tensor.view(torch.uint8)
        dev = kv_cache_src[0][0].device
        slot_ptr = slots.data_ptr()
        slot_ptrs = torch.full((4,), slot_ptr, dtype=torch.int64, device=dev)
        slot_starts = torch.zeros(4, dtype=torch.int32, device=dev)
        slot_counts = torch.full((4,), n_tok, dtype=torch.int32, device=dev)
        pbs = torch.tensor([page_buffer_size] * 4, dtype=torch.int32, device=dev)
        bss = torch.tensor([block_size] * 4, dtype=torch.int32, device=dev)
        hds = torch.tensor(list(plane_bytes), dtype=torch.int32, device=dev)
        lmc_row_offsets = torch.zeros(4, dtype=torch.int32, device=dev)
        lmc_ops.multi_layer_kv_transfer_multi_plane(
            chunk_tensor,
            ptrs,
            slot_ptrs,
            slot_starts,
            slot_counts,
            pbs,
            bss,
            hds,
            max(plane_bytes),
            dev,
            is_store,
            4,
            lmc_row_offsets,
        )

    memory_obj_list = []
    for slot_temp in slot_mapping_chunked:
        chunk_row_bytes = _multi_plane_lmc_row_bytes(list(plane_bytes), chunk_size)
        mem_obj_shape = torch.Size([1, num_layers, chunk_size, chunk_row_bytes])
        memory_obj = mem_allocator.allocate(mem_obj_shape, torch.uint8)
        _dsa_c8_multi_plane_xfer(
            memory_obj.tensor, kv_cache_pointers, slot_temp, is_store=True
        )
        memory_obj_list.append(memory_obj)

    kv_cache_dst = generate_dsa_c8_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        kv_dtype=kv_dtype,
    )

    kv_cache_pointers_dst = torch.empty(
        num_layers * 4, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache, dsa_k_cache, dsa_k_scale = kv_cache_dst[i]
        kv_cache_pointers_dst[i * 4 + 0] = k_cache.data_ptr()
        kv_cache_pointers_dst[i * 4 + 1] = v_cache.data_ptr()
        kv_cache_pointers_dst[i * 4 + 2] = dsa_k_cache.data_ptr()
        kv_cache_pointers_dst[i * 4 + 3] = dsa_k_scale.data_ptr()
    kv_cache_pointers_dst = kv_cache_pointers_dst.npu()

    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        _dsa_c8_multi_plane_xfer(
            memory_obj_list[chunk_id].tensor,
            kv_cache_pointers_dst,
            slot_temp,
            is_store=False,
        )

    check_paged_kv_cache_equal(
        kv_cache_src,
        kv_cache_dst,
        slot_mapping,
        num_heads=num_kv_heads,
        head_size=128,
        kv_format=5,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
    )

    mem_allocator.close()


@pytest.mark.parametrize(
    "num_tokens,chunk_size",
    [
        (256, 512),
        (255, 512),
        (513, 512),
    ],
)
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [128])
@pytest.mark.parametrize("dsa_head_dim", [128])
def test_fused_multi_layer_kv_transfer_dsa_c8_format(
    num_tokens,
    num_kv_heads,
    chunk_size,
    num_layers,
    block_size,
    kv_lora_rank,
    qk_rope_head_dim,
    dsa_head_dim,
):
    """DSA-C8 fused offload matches direct multi-plane extract (from_gpu path)."""
    from lmcache_ascend.v1.kv_layer_groups import _multi_plane_lmc_row_bytes

    device = "npu"
    num_blocks = max((num_tokens // block_size) * 2, 256)
    kv_dtype = torch.bfloat16

    kv_cache = generate_dsa_c8_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        kv_dtype=kv_dtype,
    )
    page_buffer_size = num_blocks * block_size
    k0, v0, d0, s0 = kv_cache[0]
    kb = int(k0.numel() * k0.element_size() // page_buffer_size)
    vb = int(v0.numel() * v0.element_size() // page_buffer_size)
    db = int(d0.numel() * d0.element_size() // page_buffer_size)
    sb = int(s0.numel() * s0.element_size() // page_buffer_size)
    plane_bytes = (kb, vb, db, sb)
    chunk_row_bytes = _multi_plane_lmc_row_bytes(list(plane_bytes), chunk_size)

    slot_mapping = torch.arange(0, num_tokens, device=device, dtype=torch.long)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = max(
        256 * 1024 * 1024,
        chunk_row_bytes * num_layers * chunk_size * 4,
    )
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    kv_cache_pointers = torch.empty(
        num_layers * 4, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache, dsa_k_cache, dsa_k_scale = kv_cache[i]
        kv_cache_pointers[i * 4 + 0] = k_cache.data_ptr()
        kv_cache_pointers[i * 4 + 1] = v_cache.data_ptr()
        kv_cache_pointers[i * 4 + 2] = dsa_k_cache.data_ptr()
        kv_cache_pointers[i * 4 + 3] = dsa_k_scale.data_ptr()
    kv_cache_pointers = kv_cache_pointers.npu()

    staging_buffer = torch.empty(
        (1, num_layers, chunk_size, chunk_row_bytes),
        dtype=torch.uint8,
        device=device,
    )

    def _dsa_c8_multi_plane_from_gpu(
        mem_tensor: torch.Tensor, slots: torch.Tensor
    ) -> None:
        n_tok = int(slots.shape[0])
        chunk_tensor = mem_tensor
        if chunk_tensor.dtype != torch.uint8:
            chunk_tensor = chunk_tensor.view(torch.uint8)
        dev = kv_cache[0][0].device
        slot_ptr = slots.data_ptr()
        slot_ptrs = torch.full((4,), slot_ptr, dtype=torch.int64, device=dev)
        slot_starts = torch.zeros(4, dtype=torch.int32, device=dev)
        slot_counts = torch.full((4,), n_tok, dtype=torch.int32, device=dev)
        pbs = torch.tensor([page_buffer_size] * 4, dtype=torch.int32, device=dev)
        bss = torch.tensor([block_size] * 4, dtype=torch.int32, device=dev)
        hds = torch.tensor(list(plane_bytes), dtype=torch.int32, device=dev)
        lmc_row_offsets = torch.zeros(4, dtype=torch.int32, device=dev)
        lmc_ops.multi_layer_kv_transfer_multi_plane(
            chunk_tensor,
            kv_cache_pointers,
            slot_ptrs,
            slot_starts,
            slot_counts,
            pbs,
            bss,
            hds,
            max(plane_bytes),
            dev,
            True,
            4,
            lmc_row_offsets,
        )

    memory_obj_baseline_list = []
    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([1, num_layers, chunk_size, chunk_row_bytes])
        memory_obj = mem_allocator.allocate(mem_obj_shape, torch.uint8)
        memory_obj.tensor.zero_()
        _dsa_c8_multi_plane_from_gpu(memory_obj.tensor, slot_temp)
        memory_obj_baseline_list.append(memory_obj)

    memory_obj_fused_list = []
    for slot_temp in slot_mapping_chunked:
        staging_buffer.zero_()
        mem_obj_shape = torch.Size([1, num_layers, chunk_size, chunk_row_bytes])
        memory_obj = mem_allocator.allocate(mem_obj_shape, torch.uint8)
        memory_obj.tensor.zero_()
        lmc_ops.fused_multi_layer_kv_transfer(
            memory_obj.tensor,
            staging_buffer,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            KVCacheFormat.DSA_C8_KV.value,
            kb,
            vb,
            db,
            sb,
            block_size,
        )
        memory_obj_fused_list.append(memory_obj)

    torch.npu.synchronize()
    for baseline_obj, fused_obj, slot_temp in zip(
        memory_obj_baseline_list, memory_obj_fused_list, slot_mapping_chunked, strict=True
    ):
        n_tok = int(slot_temp.shape[0])
        left = baseline_obj.tensor
        right = fused_obj.tensor
        assert left.shape == right.shape
        assert torch.equal(left[:, :, :n_tok, :], right[:, :, :n_tok, :])

    mem_allocator.close()


# --- Generic multi-plane kernel tests (real NPU; see also test_ds4_kvcache_roundtrip) ---

from .conftest_kvcache import (
    device as _kvcache_device,
    fill_multi_plane_pattern,
    npu_available as _kvcache_npu_available,
    pinned_lmc_chunk,
)
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
    _build_slot_mapping_for_group,
)
from lmcache_ascend.v1.kv_layer_groups import _multi_plane_layer_block_bytes
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemNPUConnectorV2,
    _derive_group_params,
)
from lmcache_ascend.v1.slot_mapping_utils import (
    build_filtered_slot_mappings,
    compact_slot_mapping_chunk,
    multi_plane_slot_slice_bounds,
)


_GENERIC_BLOCK_SIZES = (128, 128, 128, 1024, 32, 32, 128, 128, 128, 128, 64, 64)


def _generic_compress_ratios_12() -> tuple[int, ...]:
    """Unit ratios for 12 scheduler groups (1024 // block_size)."""
    return tuple(1024 // bs for bs in _GENERIC_BLOCK_SIZES)


def test_compact_slot_mapping_chunk_filters_dead_rows_preserving_order() -> None:
    sm = torch.tensor([-1, -1, 10, 11, -1, 12], dtype=torch.long)
    compact = compact_slot_mapping_chunk(sm, 0, 6, 0, (1,))
    assert compact.tolist() == [10, 11, 12]


def _generic_slot_mappings(
    num_tokens: int, dev: torch.device
) -> tuple[torch.Tensor, ...]:
    block_ids = list(range(1, 17))
    ratios = _generic_compress_ratios_12()
    mappings: list[torch.Tensor] = []
    for ratio, bs in zip(ratios, _GENERIC_BLOCK_SIZES):
        mappings.append(
            _build_slot_mapping_for_group(
                block_ids,
                int(bs),
                num_tokens,
                is_store=False,
                compress_ratio=ratio,
            ).to(dev)
        )
    return tuple(mappings)


@pytest.mark.skipif(
    not _kvcache_npu_available(),
    reason="NPU required for per-plane contiguous LMCache layout check",
)
def test_multi_plane_chunk_uses_per_plane_layout() -> None:
    """Store lays out each plane as a contiguous block inside the layer slice."""
    # Unique scope: validates packed LMCache byte layout (plane contiguity and boundaries) post-store.
    # Other tests focus on round-trip parity or slot-window corner cases, not raw packed offsets.
    dev = _kvcache_device()
    num_blocks = 4
    chunk = 64
    ratios = _generic_compress_ratios_12()
    planes = (
        torch.zeros(num_blocks, 128, 1, 512, dtype=torch.bfloat16, device=dev),
        torch.zeros(num_blocks, 128, 1, 64, dtype=torch.bfloat16, device=dev),
        torch.zeros(num_blocks, 1024, 1, 32, dtype=torch.int8, device=dev),
        torch.zeros(num_blocks, 32, 1, 64, dtype=torch.float32, device=dev),
    )
    sched_groups = [0, 2, 3, 4]
    entry_format = KVCacheFormat.detect([planes])
    assert entry_format == KVCacheFormat.MULTI_PLANE_KV
    layout_hints = {
        "block_sizes_by_group": _GENERIC_BLOCK_SIZES,
        "inference_engine_logical_block_size": 1024,
        "scheduler_groups_per_plane": sched_groups,
    }
    shape_desc = __import__(
        "lmcache_ascend.c_ops", fromlist=["PageBufferShapeDesc"]
    ).PageBufferShapeDesc()
    shape_desc.nb = num_blocks
    shape_desc.bs = 128
    group_params = _derive_group_params(
        planes, entry_format, shape_desc, layout_hints=layout_hints, num_tokens=chunk
    )
    plane_bytes = group_params["per_plane_hidden_dim_bytes"]
    # AlignUp32 per plane to match the kernel's AlignUp32Bytes accumulation.
    plane_block_sizes = [(b * chunk + 31) & ~31 for b in plane_bytes]
    lmc_chunk_row_bytes = int(group_params["k_extra"])
    layer_block_bytes = _multi_plane_layer_block_bytes(plane_bytes, chunk)
    ptrs = torch.tensor(
        [p.data_ptr() for p in planes], dtype=torch.int64, device=dev
    )
    slot_mappings = _generic_slot_mappings(max(chunk, 128), dev)
    connector = VLLMPagedMemNPUConnectorV2.__new__(VLLMPagedMemNPUConnectorV2)
    connector.kvcaches_device = dev
    connector.layout_hints = layout_hints

    fill_multi_plane_pattern(list(planes), sched_groups, slot_mappings, chunk, ratios)
    filtered, prefixes = build_filtered_slot_mappings(
        tuple(sm.cpu() for sm in slot_mappings),
        compress_ratios=ratios,
    )
    filtered_npu = tuple(f.to(dev) for f in filtered)
    with pinned_lmc_chunk((1, 1, chunk, lmc_chunk_row_bytes), torch.uint8) as (
        _mem_obj,
        lmc_chunk,
    ):
        connector._invoke_multi_plane_kv_transfer(
            mem_tensor=lmc_chunk,
            group_ptrs=ptrs,
            group_params=group_params,
            slot_mappings_by_group=slot_mappings,
            filtered_slot_mappings_npu=filtered_npu,
            slot_valid_prefix_by_group=prefixes,
            compress_ratios=ratios,
            g_start=0,
            g_end=chunk,
            is_store=True,
            npu_group_idx=0,
        )
        torch.npu.synchronize()

        sm_parts: list[torch.Tensor] = []
        for sched_g in sched_groups:
            sm = slot_mappings[sched_g]
            sm_parts.append(
                compact_slot_mapping_chunk(
                    sm,
                    0,
                    chunk,
                    sched_g,
                    ratios,
                )
            )
        n_toks_per_plane = [int(p.shape[0]) for p in sm_parts]

        layer_base = 0
        flat = lmc_chunk.reshape(-1)
        for pi, (plane_block, hd, n_tok) in enumerate(
            zip(plane_block_sizes, plane_bytes, n_toks_per_plane)
        ):
            plane_block_base = layer_base + sum(plane_block_sizes[q] for q in range(pi))
            plane_block_end = plane_block_base + plane_block
            assert plane_block_end <= flat.numel()
            plane_block_payload = flat[plane_block_base:plane_block_end]
            assert plane_block_payload.any(), (
                f"plane {pi} block has no written bytes after store"
            )
            if pi + 1 < len(plane_block_sizes):
                next_plane_base = layer_base + sum(
                    plane_block_sizes[q] for q in range(pi + 1)
                )
                assert next_plane_base == plane_block_end
        assert layer_block_bytes == sum(plane_block_sizes)
        assert layer_block_bytes <= lmc_chunk_row_bytes * chunk


@pytest.mark.skipif(
    not _kvcache_npu_available(),
    reason="NPU required for windowed cross-block bulk regression",
)
def test_multi_plane_windowed_cross_block_boundary_bulk() -> None:
    # Unique scope: stresses windowed copy where mapped slots cross a paged block boundary.
    # Other tests do not isolate this boundary-transition bulk path with hand-crafted slot slices.
    dev = _kvcache_device()
    num_blocks = 8
    chunk = 64
    ratios = _generic_compress_ratios_12()
    planes = (torch.zeros(num_blocks, 128, 1, 512, dtype=torch.bfloat16, device=dev),)
    sched_groups = [0]
    layout_hints = {
        "block_sizes_by_group": _GENERIC_BLOCK_SIZES,
        "inference_engine_logical_block_size": 1024,
        "scheduler_groups_per_plane": sched_groups,
    }
    shape_desc = __import__(
        "lmcache_ascend.c_ops", fromlist=["PageBufferShapeDesc"]
    ).PageBufferShapeDesc()
    shape_desc.nb = num_blocks
    shape_desc.bs = 128
    group_params = _derive_group_params(
        planes, KVCacheFormat.MULTI_PLANE_KV, shape_desc, layout_hints=layout_hints, num_tokens=chunk
    )
    hd = group_params["per_plane_hidden_dim_bytes"][0]
    lmc_chunk_row_bytes = int(group_params["k_extra"])
    ptrs = torch.tensor([planes[0].data_ptr()], dtype=torch.int64, device=dev)
    sched_g = 0
    block_size = 128
    s0, s1 = multi_plane_slot_slice_bounds(0, chunk, sched_g, ratios, chunk)
    n_plane = s1 - s0
    block_ids = list(range(1, num_blocks + 1))
    sm = _build_slot_mapping_for_group(
        block_ids, block_size, 132, is_store=False
    )[124:132].to(dev)
    assert int(sm.shape[0]) == n_plane
    slot_mappings = list(_generic_slot_mappings(max(chunk, 128), dev))
    slot_mappings[sched_g] = sm

    fill_multi_plane_pattern(list(planes), sched_groups, tuple(slot_mappings), chunk, ratios)
    filtered, prefixes = build_filtered_slot_mappings(
        tuple(sm.cpu() for sm in slot_mappings),
        compress_ratios=ratios,
    )
    filtered_npu = tuple(f.to(dev) for f in filtered)
    connector = VLLMPagedMemNPUConnectorV2.__new__(VLLMPagedMemNPUConnectorV2)
    connector.kvcaches_device = dev
    connector.layout_hints = layout_hints
    with pinned_lmc_chunk((1, 1, chunk, lmc_chunk_row_bytes), torch.uint8) as (
        _mem_obj,
        lmc_chunk,
    ):
        connector._invoke_multi_plane_kv_transfer(
            mem_tensor=lmc_chunk,
            group_ptrs=ptrs,
            group_params=group_params,
            slot_mappings_by_group=tuple(slot_mappings),
            filtered_slot_mappings_npu=filtered_npu,
            slot_valid_prefix_by_group=prefixes,
            compress_ratios=ratios,
            g_start=0,
            g_end=chunk,
            is_store=True,
            npu_group_idx=0,
        )
        torch.npu.synchronize()

        flat = lmc_chunk.reshape(-1)
        for t in range(n_plane):
            row = flat[t * hd : (t + 1) * hd]
            assert row.any(), f"token {t} empty after cross-block store"
