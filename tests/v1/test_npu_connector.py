# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Standard
from unittest.mock import patch
import random

# Third Party
# TODO (gingfung): once we have sglang kernel, re-enable test_sglang_connector_with_gpu_and_mla
from lmcache_tests.v1.test_gpu_connector import (
    test_batched_layerwise_vllm_paged_connector_with_gpu as original_test_batched_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_layerwise_vllm_paged_connector_with_gpu as original_test_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_vllm_paged_connector_v2_to_gpu_bench as original_test_vllm_paged_connector_v2_to_gpu_bench,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_vllm_paged_connector_v2_with_gpu_and_mla as original_test_vllm_paged_connector_v2_with_gpu_and_mla,
)
import pytest
import torch


from lmcache.v1.memory_management import PinMemoryAllocator, MemoryFormat


# First Party
from lmcache_ascend.v1.npu_connector import (
    VLLMPagedMemLayerwiseNPUConnector,
    VLLMPagedMemNPUConnectorV2,
    SGLangLayerwiseNPUConnector,
)

from lmcache_ascend.tests.v1.utils import(
    generate_sglang_npu_kv_cache,
    check_sglang_npu_kv_cache_equal,
)

@pytest.mark.parametrize("use_npu", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
def test_vllm_paged_connector_v2_with_npu_and_mla(use_npu, use_mla):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"

    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_with_gpu_and_mla(use_npu, use_mla)


@pytest.mark.parametrize("use_npu", [True])
def test_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_layerwise_vllm_paged_connector_with_gpu(use_npu)


@pytest.mark.parametrize("use_npu", [True])
def test_batched_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_batched_layerwise_vllm_paged_connector_with_gpu(use_npu)


def test_vllm_paged_connector_v2_to_npu_bench(benchmark):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"

    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_to_gpu_bench(benchmark)


@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize("use_mla", [True, False])
def test_sglang_layerwise_connector_with_npu(use_gpu, use_mla):
    """
    Test SGLang NPU integration with LMCache-Ascend.
    
    This test verifies the complete workflow of SGLang NPU with LMCache-Ascend:
    1. Generate SGLang NPU Layer-Concatenated format KV cache
    2. Test KV cache transfer from NPU to CPU (store)
    3. Test KV cache transfer from CPU to NPU (load)
    4. Verify the data integrity after round-trip transfer
    
    KV cache format: [2, layer_nums, num_blocks, block_size, num_heads, head_dim]
    """
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "npu"
    dtype = torch.bfloat16
    hidden_dim = num_heads * head_size
    
    num_tokens = num_blocks * block_size // 2
    chunk_size = 256
    
    allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    
    gpu_kv_src = generate_sglang_npu_kv_cache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        device=device,
        dtype=dtype,
    )
    gpu_kv_dst = generate_sglang_npu_kv_cache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        device=device,
        dtype=dtype,
    )
    
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)
    
    # Check the gpu_kv_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_sglang_npu_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
        )
    
    connector = SGLangLayerwiseNPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = SGLangLayerwiseNPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla
    
    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []
    
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []
        
        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)
        
        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)
    
    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
    
    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        sync=True,
    )
    
    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)
    
    # from cpu to gpu
    mem_obj_consumer = connector2.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
        sync=True,
    )

    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    
    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()
    
    assert allocator.memcheck()
    
    assert connector.gpu_buffer_allocator.memcheck()
    
    check_sglang_npu_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
    )
    
    allocator.close()

