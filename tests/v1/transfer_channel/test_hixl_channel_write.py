# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Standard
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import multiprocessing as mp
import sys
import time
import warnings

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, PagedCpuGpuMemoryAllocator
import pytest
import torch

# First Party
from lmcache_ascend import _build_info
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

pytestmark = pytest.mark.skipif(
    _build_info.cann_version_tuple() < (8, 5),
    reason="HIXL channel requires CANN 8.5.0 or later",
)


@dataclass
class HixlTestConfig:
    num_objs: int
    kv_shape: Tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16
    send_device_id: int = 0
    recv_device_id: int = 1
    timeout: int = 60
    use_host_memory: bool = False
    sender_use_host: bool | None = None
    receiver_use_host: bool | None = None

    def __post_init__(self):
        if self.sender_use_host is None:
            self.sender_use_host = self.use_host_memory
        if self.receiver_use_host is None:
            self.receiver_use_host = self.use_host_memory


def calculate_tensor_byte_size(kv_shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    num_elements = 1
    for dim_size in kv_shape:
        num_elements *= dim_size
    item_size = torch.tensor([], dtype=dtype).itemsize
    return num_elements * item_size


def get_allocator(
    device_id: int, kv_shape: Tuple[int, ...], dtype: torch.dtype, use_host: bool
) -> PagedCpuGpuMemoryAllocator:
    allocator = PagedCpuGpuMemoryAllocator()
    buffer_size = calculate_tensor_byte_size(kv_shape, dtype) * 200

    allocator.init_gpu_memory_allocator(
        buffer_size,
        [torch.Size(kv_shape)],
        [dtype],
        MemoryFormat.KV_2LTD,
        device_id,
    )

    if use_host:
        allocator.init_cpu_memory_allocator(
            buffer_size,
            [torch.Size(kv_shape)],
            [dtype],
            MemoryFormat.KV_2LTD,
        )
    return allocator


def sender_process(config: HixlTestConfig, shared_dict: Dict[str, Any]) -> None:
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.send_device_id)
        logger.info(f"Sender: Using device {config.send_device_id}")
        allocator = get_allocator(
            config.send_device_id, config.kv_shape, config.dtype, config.sender_use_host
        )
        alloc_type = "cpu" if config.sender_use_host else "gpu"

        if config.sender_use_host:
            buffer_ptr = allocator.cpu_allocator.buffer_ptr
            buffer_size = allocator.cpu_allocator.buffer_size
        else:
            buffer_ptr = allocator.gpu_allocator.buffer_ptr
            buffer_size = allocator.gpu_allocator.buffer_size

        objs = []
        expected_sums = []
        for i in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type=alloc_type,
            )
            fill_val = float(i) + 0.5
            obj.tensor.fill_(fill_val)
            objs.append(obj)
            expected_sums.append(fill_val)

        local_url = f"0.0.0.0:378{config.send_device_id}"
        remote_url = f"0.0.0.0:378{config.recv_device_id}"

        channel = CreateTransferChannel(
            channel_type="hixl",
            async_mode=False,
            role="sender",
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            align_bytes=calculate_tensor_byte_size(config.kv_shape, config.dtype),
            tp_rank=0,
            peer_init_url=local_url,
        )

        channel.lazy_init_peer_connection(
            local_id=str(config.send_device_id),
            peer_id=str(config.recv_device_id),
            peer_init_url=remote_url,
        )

        shared_dict["sender_init_done"] = True

        wait_start = time.time()
        while "receiver_init_done" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > 30:
                raise TimeoutError(
                    "Sender timed out waiting for Receiver initialization"
                )

        time.sleep(0.5)

        transfer_spec = {
            "receiver_id": str(config.recv_device_id),
            "remote_indexes": list(range(len(objs))),
        }

        logger.info(f"Sender ({alloc_type}): Starting HIXL transfer...")
        start_time = time.time()

        channel.batched_write(
            objects=objs,
            transfer_spec=transfer_spec,
        )

        duration = time.time() - start_time
        logger.info(f"Sender: HIXL Transfer finished in {duration:.4f}s")

        shared_dict["expected_values"] = expected_sums
        shared_dict["write_complete"] = True

        channel.close()

    except Exception as e:
        logger.error(f"Sender Process Failed: {e}")
        sys.exit(1)


def receiver_process(config: HixlTestConfig, shared_dict: Dict[str, Any]) -> None:
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.recv_device_id)
        logger.info(f"Receiver: Using device {config.recv_device_id}")
        allocator = get_allocator(
            config.recv_device_id, config.kv_shape, config.dtype, config.receiver_use_host
        )
        alloc_type = "cpu" if config.receiver_use_host else "gpu"

        if config.receiver_use_host:
            buffer_ptr = allocator.cpu_allocator.buffer_ptr
            buffer_size = allocator.cpu_allocator.buffer_size
        else:
            buffer_ptr = allocator.gpu_allocator.buffer_ptr
            buffer_size = allocator.gpu_allocator.buffer_size

        objs = []
        for _ in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type=alloc_type,
            )
            obj.tensor.zero_()
            objs.append(obj)

        local_url = f"0.0.0.0:378{config.recv_device_id}"

        channel = CreateTransferChannel(
            channel_type="hixl",
            async_mode=False,
            role="receiver",
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            align_bytes=calculate_tensor_byte_size(config.kv_shape, config.dtype),
            tp_rank=0,
            peer_init_url=local_url,
        )

        shared_dict["receiver_init_done"] = True

        wait_start = time.time()
        while "sender_init_done" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > 30:
                raise TimeoutError(
                    "Receiver timed out waiting for Sender initialization"
                )

        wait_start = time.time()
        while "write_complete" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > config.timeout:
                raise TimeoutError("Timed out waiting for write completion.")

        expected_values = shared_dict["expected_values"]
        logger.info(f"Receiver ({alloc_type}): Verifying data integrity...")

        for i, obj in enumerate(objs):
            expected_val = expected_values[i]
            tensor_data = obj.tensor if config.receiver_use_host else obj.tensor.cpu()

            is_equal = (tensor_data == expected_val).all()

            if not is_equal:
                sample = tensor_data.flatten()[:5].float().numpy()
                logger.error(
                    f"Mismatch in object {i}. Expected {expected_val}, got: {sample}"
                )
                raise AssertionError(f"Data verification failed for object {i}")

        logger.info(f"Receiver: Successfully verified {config.num_objs} objects.")
        channel.close()

    except Exception as e:
        logger.error(f"Receiver Process Failed: {e}")
        sys.exit(1)


def run_hixl_test(config: HixlTestConfig):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with mp.Manager() as manager:
        shared_dict = manager.dict()

        p_recv = mp.Process(
            target=receiver_process,
            args=(config, shared_dict),
            name="ReceiverProcess",
        )
        p_send = mp.Process(
            target=sender_process,
            args=(config, shared_dict),
            name="SenderProcess",
        )

        p_recv.start()
        p_send.start()

        p_send.join(timeout=config.timeout)
        p_recv.join(timeout=config.timeout)

        errors = []
        if p_send.is_alive():
            p_send.terminate()
            errors.append("Sender process timed out")
        elif p_send.exitcode != 0:
            errors.append(f"Sender process failed with exitcode {p_send.exitcode}")

        if p_recv.is_alive():
            p_recv.terminate()
            errors.append("Receiver process timed out")
        elif p_recv.exitcode != 0:
            errors.append(f"Receiver process failed with exitcode {p_recv.exitcode}")

        if errors:
            pytest.fail("\n".join(errors))


@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPU devices",
)
@pytest.mark.parametrize(
    "num_objs, num_layer, chunk_size, num_kv_head, head_size",
    [
        (2, 31, 256, 8, 128),
        (10, 31, 256, 8, 128),
    ],
)
def test_hixl_write_device(num_objs, num_layer, chunk_size, num_kv_head, head_size):
    config = HixlTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120 if num_objs >= 10 else 60,
        use_host_memory=False,
    )
    run_hixl_test(config)


@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPU devices",
)
@pytest.mark.parametrize(
    "num_objs, num_layer, chunk_size, num_kv_head, head_size",
    [
        (2, 31, 256, 8, 128),
        (10, 31, 256, 8, 128),
    ],
)
def test_hixl_write_host(num_objs, num_layer, chunk_size, num_kv_head, head_size):
    config = HixlTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120 if num_objs >= 10 else 60,
        use_host_memory=True,
    )
    run_hixl_test(config)


@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPU devices",
)
@pytest.mark.parametrize(
    "num_objs, num_layer, chunk_size, num_kv_head, head_size",
    [
        (2, 31, 256, 8, 128),
        (10, 31, 256, 8, 128),
    ],
)
def test_hixl_write_h2d(num_objs, num_layer, chunk_size, num_kv_head, head_size):
    config = HixlTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120 if num_objs >= 10 else 60,
        sender_use_host=True,
        receiver_use_host=False,
    )
    run_hixl_test(config)


@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPU devices",
)
@pytest.mark.parametrize(
    "num_objs, num_layer, chunk_size, num_kv_head, head_size",
    [
        (2, 31, 256, 8, 128),
        (10, 31, 256, 8, 128),
    ],
)
def test_hixl_write_d2h(num_objs, num_layer, chunk_size, num_kv_head, head_size):
    config = HixlTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120 if num_objs >= 10 else 60,
        sender_use_host=False,
        receiver_use_host=True,
    )
    run_hixl_test(config)
