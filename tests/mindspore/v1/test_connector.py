# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import asyncio
import tempfile
import numpy as np
import random

# Third Party
from utils import (
    check_mem_obj_equal,
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
    check_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
)

import pytest
import torch
import mindspore as ms

# First Party
 
from lmcache.v1.memory_management import (
    PinMemoryAllocator,
    MixedMemoryAllocator,
    MemoryFormat,
)
from lmcache.v1.storage_backend.connector import CreateConnector

from lmcache.v1.gpu_connector import (
    VLLMPagedMemLayerwiseGPUConnector,
)

@pytest.mark.parametrize("use_gpu", [True, False])
def test_layerwise_vllm_paged_connector_with_gpu(use_gpu):
    #####
    # Arrange
    #####
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "Ascend"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = MixedMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, "cpu", block_size)
    gpu_kv_src = [ms.Tensor(kv.to(torch.float32).numpy(), dtype = ms.bfloat16).move_to(device) for kv in gpu_kv_src]

    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, "cpu", block_size)
    gpu_kv_dst = [ms.Tensor(kv.to(torch.float32).numpy(), dtype = ms.bfloat16).move_to(device) for kv in gpu_kv_dst]

    dtype = gpu_kv_src[0][0].dtype

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device="cpu", dtype=torch.int64)
    slot_mapping = ms.Tensor(slot_mapping.numpy(), dtype=ms.int64).move_to(device)

    connector = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    )

    starts = []
    ends = []
    memory_objs = []

    for chunk_start in range(0, num_tokens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_tokens)
        tokens_in_chunk = chunk_end - chunk_start
        shape_single_layer_chunked = connector.get_shape(tokens_in_chunk)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer_chunked, dtype=np.dtype(np.float16), fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(chunk_start)
        ends.append(chunk_end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    #####
    # Act
    #####
    # from npu to cpu
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

    with pytest.raises(StopIteration):
        next(mem_obj_generator)

    # from cpu to npu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
        sync=True,
    )

    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    with pytest.raises(StopIteration):
        next(mem_obj_consumer)

    #####
    # Assert
    #####
    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
    )

    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

# @pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
# @pytest.mark.parametrize(
#     "url",
#     [
#         "lm://localhost:65000",
#     ],
# )
# def test_lm_connector(url, autorelease_v1, lmserver_v1_process):
#     if url.startswith("lm"):
#         url = lmserver_v1_process.server_url

#     async_loop, async_thread = init_asyncio_loop()
#     memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
#     connector = autorelease_v1(CreateConnector(url, async_loop, memory_allocator))

#     random_key = dumb_cache_engine_key()
#     future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
#     assert not future.result()

#     num_tokens = 1000
#     mem_obj_shape = [2, 32, num_tokens, 1024]
#     dtype = torch.bfloat16
#     memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
#     memory_obj.ref_count_up()

#     torch.manual_seed(42)
#     test_tensor = torch.randint(0, 100, [2 * 32 * num_tokens * 1024], dtype=torch.int64)
#     memory_obj.raw_data = test_tensor.astype(np.float32).astype(dtype).copy()

#     future = asyncio.run_coroutine_threadsafe(
#         connector.put(random_key, memory_obj), async_loop
#     )
#     future.result()

#     future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
#     assert future.result()
#     assert memory_obj.get_ref_count() == 1

#     future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
#     retrieved_memory_obj = future.result()

#     check_mem_obj_equal(
#         [retrieved_memory_obj],
#         [memory_obj],
#     )

#     close_asyncio_loop(async_loop, async_thread)

#     memory_allocator.close()


# @pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
# def test_fs_connector(lmserver_v1_process, autorelease_v1):
#     """Test filesystem connector: exists, put, get, list, and file store."""

#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Setup
#         url = f"fs://host:0/{temp_dir}/"
#         async_loop, async_thread = init_asyncio_loop()
#         memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
#         connector = autorelease_v1(CreateConnector(url, async_loop, memory_allocator))
#         random_key = dumb_cache_engine_key()

#         # Test 1: Verify key doesn't exist initially
#         future = asyncio.run_coroutine_threadsafe(
#             connector.exists(random_key), async_loop
#         )
#         assert not future.result()

#         # Test 2: Create and store test data
#         dtype = torch.bfloat16
#         memory_obj = memory_allocator.allocate([2, 32, 1000, 1024], dtype)
#         memory_obj.ref_count_up()
#         # Fill with deterministic test data
#         torch.manual_seed(42)
#         test_tensor = torch.randint(
#             0, 100, memory_obj.raw_data.shape, dtype=torch.int64
#         )
#         memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

#         future = asyncio.run_coroutine_threadsafe(
#             connector.put(random_key, memory_obj), async_loop
#         )
#         future.result()

#         # Test 3: Verify key exists after putting data
#         future = asyncio.run_coroutine_threadsafe(
#             connector.exists(random_key), async_loop
#         )
#         assert future.result()
#         assert memory_obj.get_ref_count() == 1

#         # Test 4: Retrieve and verify data
#         future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
#         check_mem_obj_equal([future.result()], [memory_obj])

#         # Test 5: List the keys
#         future = asyncio.run_coroutine_threadsafe(connector.list(), async_loop)
#         assert future.result() == [random_key.to_string()]

#         # Test 6: Verify file existence and format
#         files = list(Path(temp_dir).glob("*.data"))
#         assert len(files) == 1
#         assert files[0].name == f"{random_key.to_string()}.data"

#         close_asyncio_loop(async_loop, async_thread)

#         memory_allocator.close()


# @pytest.mark.parametrize(
#     "url",
#     [
#         "redis://localhost:6379",
#         "redis://user:password@localhost:6379/0",
#         "redis://:password@localhost:6379/1",
#         "rediss://user:password@localhost:6380?ssl_cert_reqs=CERT_REQUIRED",
#         "unix:///tmp/redis.sock",
#     ],
# )
# def test_redis_connector(url, autorelease_v1):
#     """Test Redis connector: exists, put, get operations.

#     This test uses the MockRedis from conftest.py to simulate
#     Redis behavior without requiring an actual Redis server.
#     """

#     async_loop, async_thread = init_asyncio_loop()
#     memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
#     connector = autorelease_v1(CreateConnector(url, async_loop, memory_allocator))

#     random_key = dumb_cache_engine_key()

#     # Test 1: Verify key doesn't exist initially
#     future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
#     assert not future.result()

#     # Test 2: Create and store test data
#     num_tokens = 1000
#     mem_obj_shape = [2, 32, num_tokens, 1024]
#     dtype = torch.bfloat16
#     memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
#     memory_obj.ref_count_up()

#     torch.manual_seed(42)
#     test_tensor = torch.randint(0, 100, memory_obj.raw_data.shape, dtype=torch.int64)
#     memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

#     # Test 3: Put data
#     future = asyncio.run_coroutine_threadsafe(
#         connector.put(random_key, memory_obj), async_loop
#     )
#     future.result()

#     # Test 4: Verify key exists after putting data
#     future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
#     assert future.result()
#     assert memory_obj.get_ref_count() == 1

#     # Test 5: Retrieve and verify data
#     future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
#     retrieved_memory_obj = future.result()

#     check_mem_obj_equal(
#         [retrieved_memory_obj],
#         [memory_obj],
#     )

#     close_asyncio_loop(async_loop, async_thread)

#     memory_allocator.close()


# @pytest.mark.parametrize(
#     "url",
#     [
#         "redis-sentinel://localhost:26379,localhost:26380,localhost:26381",
#         "redis-sentinel://user:password@localhost:26379,localhost:26380",
#         "redis-sentinel://localhost:26379",
#     ],
# )
# def test_redis_sentinel_connector(url, autorelease_v1):
#     """Test Redis Sentinel connector: exists, put, get operations.

#     This test uses the MockRedisSentinel from conftest.py to simulate
#     Redis Sentinel behavior without requiring an actual Redis Sentinel setup.
#     """
#     # Standard
#     import os

#     # Set required environment variables for Redis Sentinel
#     os.environ["REDIS_SERVICE_NAME"] = "mymaster"
#     os.environ["REDIS_TIMEOUT"] = "5"

#     async_loop, async_thread = init_asyncio_loop()
#     memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
#     connector = autorelease_v1(CreateConnector(url, async_loop, memory_allocator))

#     random_key = dumb_cache_engine_key()

#     # Test 1: Verify key doesn't exist initially
#     future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
#     assert not future.result()

#     # Test 2: Create and store test data
#     num_tokens = 1000
#     mem_obj_shape = [2, 32, num_tokens, 1024]
#     dtype = torch.bfloat16
#     memory_obj = memory_allocator.allocate(mem_obj_shape, dtype)
#     memory_obj.ref_count_up()

#     # Fill with deterministic test data for Redis Sentinel test
#     torch.manual_seed(123)
#     test_tensor = torch.randint(0, 100, memory_obj.raw_data.shape, dtype=torch.int64)
#     memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

#     # Test 3: Put data
#     future = asyncio.run_coroutine_threadsafe(
#         connector.put(random_key, memory_obj), async_loop
#     )
#     future.result()

#     # Test 4: Verify key exists after putting data
#     future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
#     assert future.result()

#     # Test 5: Retrieve and verify data
#     future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
#     future.result()

#     close_asyncio_loop(async_loop, async_thread)

#     memory_allocator.close()
