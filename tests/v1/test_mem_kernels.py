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
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
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
        mem_obj_shape = [kvs, num_layers, len(slot_temp), hidden_dim_size]
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
        mem_obj_shape = [kvs, num_layers, len(slot_temp), hidden_dim_size]
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
        )
        memory_obj_fused_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Fused from_gpu time: {elapsed_time_ms:.6f} ms")

    check_mem_obj_equal(memory_obj_baseline_list, memory_obj_fused_list)

    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
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
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
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
        mem_obj_shape = [kvs, num_layers, len(slot_temp), hidden_dim_size]
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            2,  # SEPARATE_KV
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
        mem_obj_shape = [kvs, num_layers, len(slot_temp), hidden_dim_size]
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
            False,
            2,  # SEPARATE_KV
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
            False,
            2,  # SEPARATE_KV
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
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
        vllm_two_major=vllm_two_major,
    )

    kv_cache_dst_baseline = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
        vllm_two_major=vllm_two_major,
    )

    kv_cache_dst_fused = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
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
                    chunk_shape = list(chunk_buffer_shapes[chunk_id])
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
                token_major,
                vllm_two_major,
            )

        end_event.record()
        torch.npu.synchronize()
        fused_to_time = start_event.elapsed_time(end_event)

        print(
            f"\n[to_gpu] Baseline: {baseline_to_time:.3f} ms, "
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
