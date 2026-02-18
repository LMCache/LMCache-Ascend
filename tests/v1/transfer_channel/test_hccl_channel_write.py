# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
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
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel


@dataclass
class HcclTestConfig:
    num_objs: int
    kv_shape: Tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16
    send_device_id: int = 0
    recv_device_id: int = 1
    timeout: int = 60
    use_host_memory: bool = False
    use_multi_buffer: bool = False


def calculate_tensor_byte_size(kv_shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    num_elements = 1
    for dim_size in kv_shape:
        num_elements *= dim_size
    item_size = torch.tensor([], dtype=dtype).itemsize
    return num_elements * item_size


def get_allocator(
    device_id: int,
    kv_shape: Tuple[int, ...],
    dtype: torch.dtype,
    use_host: bool,
    use_multi_buffer: bool = False,
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

    if use_host or use_multi_buffer:
        allocator.init_cpu_memory_allocator(
            buffer_size,
            [torch.Size(kv_shape)],
            [dtype],
            MemoryFormat.KV_2LTD,
        )
    return allocator


def _build_channel_buffers(
    allocator: PagedCpuGpuMemoryAllocator,
    kv_shape: Tuple[int, ...],
    dtype: torch.dtype,
    use_host: bool,
    use_multi_buffer: bool,
) -> Tuple[List[int], List[int], List[str], List[int]]:
    """Build multi-buffer channel args from allocator.

    Returns (buffer_ptrs, buffer_sizes, buffer_types, align_bytes_list).
    """
    page_size = calculate_tensor_byte_size(kv_shape, dtype)
    buffer_ptrs: List[int] = []
    buffer_sizes: List[int] = []
    buffer_types: List[str] = []
    align_bytes_list: List[int] = []

    if use_multi_buffer:
        # Register both NPU and CPU buffers
        buffer_ptrs.append(allocator.gpu_allocator.buffer_ptr)
        buffer_sizes.append(allocator.gpu_allocator.buffer_size)
        buffer_types.append("npu")
        align_bytes_list.append(page_size)

        buffer_ptrs.append(allocator.cpu_allocator.buffer_ptr)
        buffer_sizes.append(allocator.cpu_allocator.buffer_size)
        buffer_types.append("cpu")
        align_bytes_list.append(page_size)
    elif use_host:
        buffer_ptrs.append(allocator.cpu_allocator.buffer_ptr)
        buffer_sizes.append(allocator.cpu_allocator.buffer_size)
        buffer_types.append("cpu")
        align_bytes_list.append(page_size)
    else:
        buffer_ptrs.append(allocator.gpu_allocator.buffer_ptr)
        buffer_sizes.append(allocator.gpu_allocator.buffer_size)
        buffer_types.append("npu")
        align_bytes_list.append(page_size)

    return buffer_ptrs, buffer_sizes, buffer_types, align_bytes_list


# ──────────────────────────────────────────────────────────
# Write test processes (sender writes to receiver's memory)
# ──────────────────────────────────────────────────────────


def write_sender_process(config: HcclTestConfig, shared_dict: Dict[str, Any]) -> None:
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.send_device_id)

        allocator = get_allocator(
            config.send_device_id,
            config.kv_shape,
            config.dtype,
            config.use_host_memory,
            config.use_multi_buffer,
        )
        alloc_type = "cpu" if config.use_host_memory else "gpu"

        # Generate data objects
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

        local_url = f"0.0.0.0:377{config.send_device_id}"
        remote_url = f"0.0.0.0:377{config.recv_device_id}"

        buf_ptrs, buf_sizes, buf_types, align_list = _build_channel_buffers(
            allocator,
            config.kv_shape,
            config.dtype,
            config.use_host_memory,
            config.use_multi_buffer,
        )

        channel = CreateTransferChannel(
            channel_type="hccl",
            async_mode=False,
            role="sender",
            buffer_ptr=buf_ptrs,
            buffer_size=buf_sizes,
            buffer_type=buf_types,
            align_bytes=align_list,
            tp_rank=0,
            peer_init_url=local_url,
        )

        channel.lazy_init_peer_connection(
            local_id=str(config.send_device_id),
            peer_id=str(config.recv_device_id),
            peer_init_url=remote_url,
        )

        # wait for receiver to be initialized
        wait_start = time.time()
        while "receiver_init_done" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > 30:
                raise TimeoutError("Sender timed out waiting for receiver buffer refs")
            logger.info(
                "Sender: Waiting for receiver initialization, shared_dict keys: %s",
                list(shared_dict.keys()),
            )

        shared_dict["sender_init_done"] = True
        logger.info("Sender: Sender initialization complete")

        recv_buffer_uuids = list(shared_dict["receiver_buffer_refs_uuids"])
        recv_mem_indexes = list(shared_dict["receiver_buffer_refs_indexes"])

        time.sleep(0.5)

        transfer_spec = {
            "receiver_id": str(config.recv_device_id),
            "remote_buffer_uuids": recv_buffer_uuids,
            "remote_mem_indexes": recv_mem_indexes,
        }

        logger.info(f"Sender ({alloc_type}): Starting batched_write...")
        start_time = time.time()

        channel.batched_write(
            objects=objs,
            transfer_spec=transfer_spec,
        )

        duration = time.time() - start_time
        logger.info(f"Sender: Transfer finished in {duration:.4f}s")

        shared_dict["expected_values"] = expected_sums
        shared_dict["write_complete"] = True

        channel.close()

    except Exception as e:
        logger.error(f"Sender Process Failed: {e}")
        sys.exit(1)


def write_receiver_process(config: HcclTestConfig, shared_dict: Dict[str, Any]) -> None:
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.recv_device_id)

        allocator = get_allocator(
            config.recv_device_id,
            config.kv_shape,
            config.dtype,
            config.use_host_memory,
            config.use_multi_buffer,
        )
        alloc_type = "cpu" if config.use_host_memory else "gpu"

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

        local_url = f"0.0.0.0:377{config.recv_device_id}"

        buf_ptrs, buf_sizes, buf_types, align_list = _build_channel_buffers(
            allocator,
            config.kv_shape,
            config.dtype,
            config.use_host_memory,
            config.use_multi_buffer,
        )

        channel = CreateTransferChannel(
            channel_type="hccl",
            async_mode=False,
            role="receiver",
            buffer_ptr=buf_ptrs,
            buffer_size=buf_sizes,
            buffer_type=buf_types,
            align_bytes=align_list,
            tp_rank=0,
            peer_init_url=local_url,
        )

        # Get buffer refs for the receiver's objects (sender needs these
        # to know where to write)
        buffer_uuids, mem_indexes = channel.get_local_buffer_refs(objs)
        shared_dict["receiver_buffer_refs_uuids"] = buffer_uuids
        shared_dict["receiver_buffer_refs_indexes"] = mem_indexes
        shared_dict["receiver_init_done"] = True

        # Wait for sender to be initialized
        wait_start = time.time()
        while "sender_init_done" not in shared_dict:
            time.sleep(0.1)
            logger.info(
                "Receiver: Waiting for sender initialization, shared_dict keys: %s",
                list(shared_dict.keys()),
            )
            if time.time() - wait_start > 30:
                raise TimeoutError(
                    "Receiver timed out waiting for Sender initialization"
                )

        # Wait for write to complete
        wait_start = time.time()
        while "write_complete" not in shared_dict:
            time.sleep(0.1)
            logger.info(
                "Receiver: Waiting for write completion, shared_dict keys: %s",
                list(shared_dict.keys()),
            )
            if time.time() - wait_start > config.timeout:
                raise TimeoutError("Timed out waiting for write completion.")

        expected_values = shared_dict["expected_values"]
        logger.info(f"Receiver ({alloc_type}): Verifying data integrity...")

        for i, obj in enumerate(objs):
            expected_val = expected_values[i]
            tensor_data = obj.tensor if config.use_host_memory else obj.tensor.cpu()

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


# ──────────────────────────────────────────────────────────
# Read test processes (receiver reads from sender's memory)
# ──────────────────────────────────────────────────────────


def read_data_provider_process(
    config: HcclTestConfig, shared_dict: Dict[str, Any]
) -> None:
    """Sender-side process: fills data and exposes buffer refs for reader."""
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.send_device_id)

        allocator = get_allocator(
            config.send_device_id,
            config.kv_shape,
            config.dtype,
            config.use_host_memory,
        )
        alloc_type = "cpu" if config.use_host_memory else "gpu"

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

        local_url = f"0.0.0.0:377{config.send_device_id}"

        buf_ptrs, buf_sizes, buf_types, align_list = _build_channel_buffers(
            allocator,
            config.kv_shape,
            config.dtype,
            config.use_host_memory,
            use_multi_buffer=False,
        )

        channel = CreateTransferChannel(
            channel_type="hccl",
            async_mode=False,
            role="sender",
            buffer_ptr=buf_ptrs,
            buffer_size=buf_sizes,
            buffer_type=buf_types,
            align_bytes=align_list,
            tp_rank=0,
            peer_init_url=local_url,
        )

        # Wait for reader to be ready (it has the REP socket)
        wait_start = time.time()
        while "reader_init_done" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > 30:
                raise TimeoutError("Data provider timed out waiting for reader init")

        remote_url = f"0.0.0.0:377{config.recv_device_id}"
        channel.lazy_init_peer_connection(
            local_id=str(config.send_device_id),
            peer_id=str(config.recv_device_id),
            peer_init_url=remote_url,
        )

        # Share our buffer refs so the reader can read from our memory
        buffer_uuids, mem_indexes = channel.get_local_buffer_refs(objs)
        shared_dict["provider_buffer_refs_uuids"] = buffer_uuids
        shared_dict["provider_buffer_refs_indexes"] = mem_indexes
        shared_dict["expected_values"] = expected_sums
        shared_dict["provider_init_done"] = True

        logger.info(f"Data provider ({alloc_type}): Shared buffer refs, waiting...")

        # Keep alive until reader is done
        wait_start = time.time()
        while "read_complete" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > config.timeout:
                raise TimeoutError("Data provider timed out waiting for read.")

        logger.info("Data provider: Reader finished. Closing.")
        channel.close()

    except Exception as e:
        logger.error(f"Data provider process failed: {e}")
        sys.exit(1)


def read_reader_process(
    config: HcclTestConfig,
    shared_dict: Dict[str, Any],
    use_submit: bool = False,
) -> None:
    """Receiver-side process: reads from sender's memory via batched_read."""
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.recv_device_id)

        allocator = get_allocator(
            config.recv_device_id,
            config.kv_shape,
            config.dtype,
            config.use_host_memory,
        )
        alloc_type = "cpu" if config.use_host_memory else "gpu"

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

        local_url = f"0.0.0.0:377{config.recv_device_id}"

        buf_ptrs, buf_sizes, buf_types, align_list = _build_channel_buffers(
            allocator,
            config.kv_shape,
            config.dtype,
            config.use_host_memory,
            use_multi_buffer=False,
        )

        channel = CreateTransferChannel(
            channel_type="hccl",
            async_mode=False,
            role="receiver",
            buffer_ptr=buf_ptrs,
            buffer_size=buf_sizes,
            buffer_type=buf_types,
            align_bytes=align_list,
            tp_rank=0,
            peer_init_url=local_url,
        )

        shared_dict["reader_init_done"] = True

        # Wait for data provider to share buffer refs
        wait_start = time.time()
        while "provider_init_done" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > 30:
                raise TimeoutError("Reader timed out waiting for provider init")

        time.sleep(0.5)

        provider_buffer_uuids = list(shared_dict["provider_buffer_refs_uuids"])
        provider_mem_indexes = list(shared_dict["provider_buffer_refs_indexes"])
        expected_values = shared_dict["expected_values"]

        # The "receiver_id" in the spec refers to the peer whose memory
        # we are reading from (the data provider / sender)
        transfer_spec = {
            "receiver_id": str(config.send_device_id),
            "remote_buffer_uuids": provider_buffer_uuids,
            "remote_mem_indexes": provider_mem_indexes,
        }

        logger.info(f"Reader ({alloc_type}): Starting read (submit={use_submit})...")
        start_time = time.time()

        if use_submit:
            event = channel.submit_batched_read(
                buffers=objs,
                transfer_spec=transfer_spec,
            )
            event.synchronize()
        else:
            channel.batched_read(
                buffers=objs,
                transfer_spec=transfer_spec,
            )

        duration = time.time() - start_time
        logger.info(f"Reader: Read finished in {duration:.4f}s")

        logger.info(f"Reader ({alloc_type}): Verifying data integrity...")
        for i, obj in enumerate(objs):
            expected_val = expected_values[i]
            tensor_data = obj.tensor if config.use_host_memory else obj.tensor.cpu()

            is_equal = (tensor_data == expected_val).all()
            if not is_equal:
                sample = tensor_data.flatten()[:5].float().numpy()
                logger.error(
                    f"Mismatch in object {i}. Expected {expected_val}, got: {sample}"
                )
                raise AssertionError(f"Data verification failed for object {i}")

        logger.info(f"Reader: Successfully verified {config.num_objs} objects.")
        shared_dict["read_complete"] = True
        channel.close()

    except Exception as e:
        logger.error(f"Reader process failed: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# Multi-buffer write test processes (sender CPU → receiver NPU)
# ──────────────────────────────────────────────────────────


def multi_buffer_sender_process(
    config: HcclTestConfig, shared_dict: Dict[str, Any]
) -> None:
    """Sender allocates on CPU buffer; receiver allocates on NPU buffer."""
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.send_device_id)

        allocator = get_allocator(
            config.send_device_id,
            config.kv_shape,
            config.dtype,
            use_host=False,
            use_multi_buffer=True,
        )

        # Allocate sender data on CPU buffer
        objs = []
        expected_sums = []
        for i in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="cpu",
            )
            fill_val = float(i) + 0.5
            obj.tensor.fill_(fill_val)
            objs.append(obj)
            expected_sums.append(fill_val)

        local_url = f"0.0.0.0:377{config.send_device_id}"
        remote_url = f"0.0.0.0:377{config.recv_device_id}"

        buf_ptrs, buf_sizes, buf_types, align_list = _build_channel_buffers(
            allocator,
            config.kv_shape,
            config.dtype,
            use_host=False,
            use_multi_buffer=True,
        )

        channel = CreateTransferChannel(
            channel_type="hccl",
            async_mode=False,
            role="sender",
            buffer_ptr=buf_ptrs,
            buffer_size=buf_sizes,
            buffer_type=buf_types,
            align_bytes=align_list,
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
        while "receiver_buffer_refs" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > 30:
                raise TimeoutError(
                    "Multi-buffer sender timed out waiting for receiver refs"
                )

        recv_buffer_uuids = list(shared_dict["receiver_buffer_refs_uuids"])
        recv_mem_indexes = list(shared_dict["receiver_buffer_refs_indexes"])

        time.sleep(0.5)

        transfer_spec = {
            "receiver_id": str(config.recv_device_id),
            "remote_buffer_uuids": recv_buffer_uuids,
            "remote_mem_indexes": recv_mem_indexes,
        }

        logger.info("Multi-buffer sender (CPU): Starting batched_write...")
        start_time = time.time()

        channel.batched_write(
            objects=objs,
            transfer_spec=transfer_spec,
        )

        duration = time.time() - start_time
        logger.info(f"Multi-buffer sender: Transfer finished in {duration:.4f}s")

        shared_dict["expected_values"] = expected_sums
        shared_dict["write_complete"] = True

        channel.close()

    except Exception as e:
        logger.error(f"Multi-buffer sender failed: {e}")
        sys.exit(1)


def multi_buffer_receiver_process(
    config: HcclTestConfig, shared_dict: Dict[str, Any]
) -> None:
    """Receiver allocates on NPU buffer in a multi-buffer channel."""
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.recv_device_id)

        allocator = get_allocator(
            config.recv_device_id,
            config.kv_shape,
            config.dtype,
            use_host=False,
            use_multi_buffer=True,
        )

        # Allocate on NPU buffer
        objs = []
        for _ in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="gpu",
            )
            obj.tensor.zero_()
            objs.append(obj)

        local_url = f"0.0.0.0:377{config.recv_device_id}"

        buf_ptrs, buf_sizes, buf_types, align_list = _build_channel_buffers(
            allocator,
            config.kv_shape,
            config.dtype,
            use_host=False,
            use_multi_buffer=True,
        )

        channel = CreateTransferChannel(
            channel_type="hccl",
            async_mode=False,
            role="receiver",
            buffer_ptr=buf_ptrs,
            buffer_size=buf_sizes,
            buffer_type=buf_types,
            align_bytes=align_list,
            tp_rank=0,
            peer_init_url=local_url,
        )

        buffer_uuids, mem_indexes = channel.get_local_buffer_refs(objs)
        shared_dict["receiver_buffer_refs_uuids"] = buffer_uuids
        shared_dict["receiver_buffer_refs_indexes"] = mem_indexes
        shared_dict["receiver_buffer_refs"] = True
        shared_dict["receiver_init_done"] = True

        wait_start = time.time()
        while "sender_init_done" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > 30:
                raise TimeoutError(
                    "Multi-buffer receiver timed out waiting for sender init"
                )

        wait_start = time.time()
        while "write_complete" not in shared_dict:
            time.sleep(0.1)
            if time.time() - wait_start > config.timeout:
                raise TimeoutError("Timed out waiting for write completion.")

        expected_values = shared_dict["expected_values"]
        logger.info("Multi-buffer receiver (NPU): Verifying data integrity...")

        for i, obj in enumerate(objs):
            expected_val = expected_values[i]
            tensor_data = obj.tensor.cpu()

            is_equal = (tensor_data == expected_val).all()
            if not is_equal:
                sample = tensor_data.flatten()[:5].float().numpy()
                logger.error(
                    f"Mismatch in object {i}. Expected {expected_val}, got: {sample}"
                )
                raise AssertionError(f"Data verification failed for object {i}")

        logger.info(f"Multi-buffer receiver: Verified {config.num_objs} objects.")
        channel.close()

    except Exception as e:
        logger.error(f"Multi-buffer receiver failed: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# Test runners
# ──────────────────────────────────────────────────────────


def _run_two_process_test(
    config: HcclTestConfig,
    sender_fn,
    receiver_fn,
    sender_args: Tuple = (),
    receiver_args: Tuple = (),
):
    """Generic runner: spawns a sender and receiver process."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with mp.Manager() as manager:
        shared_dict = manager.dict()

        p_recv = mp.Process(
            target=receiver_fn,
            args=(config, shared_dict, *receiver_args),
            name="ReceiverProcess",
        )
        p_send = mp.Process(
            target=sender_fn,
            args=(config, shared_dict, *sender_args),
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
def test_hccl_write_device(num_objs, num_layer, chunk_size, num_kv_head, head_size):
    """NPU-to-NPU transfer via batched_write with UUID-based transfer specs."""
    config = HcclTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120 if num_objs > 10 else 60,
        use_host_memory=False,
    )
    _run_two_process_test(config, write_sender_process, write_receiver_process)


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
def test_hccl_write_host(num_objs, num_layer, chunk_size, num_kv_head, head_size):
    """CPU-to-CPU transfer via batched_write with UUID-based transfer specs."""
    config = HcclTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=60,
        use_host_memory=True,
    )
    _run_two_process_test(config, write_sender_process, write_receiver_process)


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
def test_hccl_multi_buffer(num_objs, num_layer, chunk_size, num_kv_head, head_size):
    """Both CPU and NPU buffers registered; sender writes from CPU, receiver on NPU."""
    config = HcclTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120 if num_objs > 10 else 60,
        use_host_memory=False,
        use_multi_buffer=True,
    )
    _run_two_process_test(
        config, multi_buffer_sender_process, multi_buffer_receiver_process
    )


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
def test_hccl_batched_read(num_objs, num_layer, chunk_size, num_kv_head, head_size):
    """Receiver uses batched_read() to pull data from sender's memory."""
    config = HcclTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120 if num_objs > 10 else 60,
        use_host_memory=False,
    )
    _run_two_process_test(
        config,
        read_data_provider_process,
        read_reader_process,
        receiver_args=(False,),
    )


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
def test_hccl_submit_batched_read(
    num_objs, num_layer, chunk_size, num_kv_head, head_size
):
    """Receiver uses submit_batched_read() + event.synchronize()."""
    config = HcclTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120 if num_objs > 10 else 60,
        use_host_memory=False,
    )
    _run_two_process_test(
        config,
        read_data_provider_process,
        read_reader_process,
        receiver_args=(True,),
    )
