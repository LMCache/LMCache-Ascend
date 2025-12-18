import os
import sys

import torch
import torch_npu

import multiprocessing as mp
import pickle
import time
import traceback
from typing import List, Tuple

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)

from lmcache_ascend.v1.transfer_channel import CreateTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

def wait_for_key(shared_dict, key, timeout=30):
    """
    Helper function to replicate the blocking behavior of store.get().
    It polls the shared dictionary until the key appears.
    """
    start_time = time.time()
    while key not in shared_dict:
        time.sleep(0.1)
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timed out waiting for key '{key}' in shared dictionary.")
    return shared_dict[key]

num_layer = 31
chunk_size = 256
num_kv_head = 8
head_size = 128
kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)

def calculate_tensor_byte_size(kv_shape, dtype):
    num_elements = 1
    for dim_size in kv_shape:
        num_elements *= dim_size
    
    item_size = torch.tensor([], dtype=dtype).itemsize
    total_byte_size = num_elements * item_size
    return total_byte_size

def generate_test_data(
    allocator, num_objs: int, shape: torch.Size, dtype: torch.dtype, empty: bool = False
) -> Tuple[List[CacheEngineKey], List[MemoryObj]]:
    keys = []
    objs = []
    for i in range(num_objs):
        keys.append(
            CacheEngineKey(
                fmt="test",
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=dtype
            )
        )
        obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD, allocator_type="gpu")
        if not empty:
            obj.tensor.fill_(i + 1)  # Fill with some test data, e.g., the index
        objs.append(obj)
    return keys, objs

def get_allocator(device_id, kv_shape, dtype: torch.dtype = torch.bfloat16):
    paged_mem_allocator = PagedCpuGpuMemoryAllocator()

    paged_mem_allocator.init_gpu_memory_allocator(
        calculate_tensor_byte_size(kv_shape, dtype) * 128, # Can fit 128 KVs
        torch.Size(kv_shape),
        dtype,
        MemoryFormat.KV_2LTD,
        device_id,
    )

    return paged_mem_allocator

def sender_process(send_devid, recv_devid, shared_dict, logger):
    num_objs = 2

    torch.npu.set_device(int(send_devid))

    allocator = get_allocator(send_devid, kv_shape)

    keys, objs = generate_test_data(allocator, num_objs, torch.Size(kv_shape), torch.bfloat16)
    total_size = sum(obj.get_size() for obj in objs)
    logger.info(
        f"Generated {len(objs)} objects with total size "
        f"{total_size / (1024 * 1024):.2f} MB"
    )

    local_url = "0.0.0.0:377" + str(send_devid)
    remote_url = "0.0.0.0:377" + str(recv_devid)

    channel = CreateTransferChannel(
        channel_type="hccl",
        async_mode=False,
        role="sender",
        buffer_ptr=allocator.gpu_allocator.buffer_ptr,
        buffer_size=allocator.gpu_allocator.buffer_size,
        align_bytes=calculate_tensor_byte_size(kv_shape, torch.bfloat16),
        tp_rank=0,
        peer_init_url=local_url,
    )

    channel.lazy_init_peer_connection(
        local_id=str(send_devid), peer_id=str(recv_devid), peer_init_url=remote_url
    )

    transfer_spec = {
        "receiver_id": str(recv_devid),
        "remote_indexes": [i for i in range(len(objs))]
    }

    channel.batched_write(
        objects=objs,
        transfer_spec=transfer_spec,
    )

    shared_dict["write_complete"] = True

    channel.close()

def receiver_process(recv_devid, shared_dict, logger):
    num_objs = 2
    torch.npu.set_device(int(recv_devid))

    allocator = get_allocator(recv_devid, kv_shape)

    keys, objs = generate_test_data(allocator, num_objs, torch.Size(kv_shape), torch.bfloat16, empty=True)
    total_size = sum(obj.get_size() for obj in objs)
    logger.info(
        f"Generated {len(objs)} empty objects with total size "
        f"{total_size / (1024 * 1024):.2f} MB"
    )

    local_url = "0.0.0.0:377" + str(recv_devid)

    channel = CreateTransferChannel(
        channel_type="hccl",
        async_mode=False,
        role="receiver",
        buffer_ptr=allocator.gpu_allocator.buffer_ptr,
        buffer_size=allocator.gpu_allocator.buffer_size,
        align_bytes=calculate_tensor_byte_size(kv_shape, torch.bfloat16),
        tp_rank=0,
        peer_init_url=local_url,
    )

    wait_for_key(shared_dict, "write_complete", timeout=6000)
    print("Write completion signal received.", flush=True)

    for obj in objs:
        print(obj.tensor)

    channel.close()

if __name__ == "__main__":
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("FATAL ERROR: torch_npu is not available.")
        exit()

    send_devid = 0
    recv_devid = 1

    mp.set_start_method("fork", force=True)

    with mp.Manager() as manager:
        shared_dict = manager.dict()
        logger = init_logger(__name__)

        p1 = mp.Process(target=receiver_process, args=(recv_devid, shared_dict, logger,))
        p2 = mp.Process(target=sender_process, args=(send_devid, recv_devid, shared_dict, logger,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()