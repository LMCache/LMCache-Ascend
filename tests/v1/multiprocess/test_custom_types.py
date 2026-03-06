# SPDX-License-Identifier: Apache-2.0
# Standard
from multiprocessing import Queue
import multiprocessing as mp

# Third Party
import pytest
import torch
import torch_npu  # noqa: F401

# First Party
# NOTE (gingfung): we have to import the bootstrap and prepare here because
# multiprocessing will run from the top of the file here and if not bootstrapped,
# 'lmcache_tests' is not recognized, and the relevant patches won't be applied.
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
# NOTE (gingfung): at this point,
# the CudaIPCWrapper should be patched already.
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper  # noqa: E402
from lmcache_tests.v1.multiprocess.test_custom_types import (  # noqa: F401, E402
    get_customized_decoder,
    get_customized_encoder,
    test_cudaipc_wrapper_list_serialization,
    test_cudaipc_wrapper_serialization,
    test_ipc_cache_engine_key_serialization,
)


def _worker_process_deserialize_and_reconstruct(
    encoded_data: bytes, result_queue: Queue
):
    """
    Worker function that runs in a separate process.
    Deserializes CudaIPCWrapper list and reconstructs tensors.
    """
    try:
        # Decode the list of wrappers
        torch.npu.init()
        decoder = get_customized_decoder(type=list[CudaIPCWrapper])
        decoded_wrappers = decoder.decode(encoded_data)

        # Convert each wrapper back to tensor and compute checksum
        checksums = []
        shapes = []
        for wrapper in decoded_wrappers:
            tensor = wrapper.to_tensor()
            # Compute checksum as sum of all elements
            checksum = float(tensor.sum().cpu().item())
            checksums.append(checksum)
            shapes.append(list(tensor.shape))

            # Do add 1 on the tensor to ensure it's writable
            tensor.add_(1)

        result_queue.put(("success", checksums, shapes))
    except Exception as e:
        result_queue.put(("error", str(e), None))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="NPU is required for IPCWrapper multiprocessing tests",
)
def test_cudaipc_wrapper_multiprocess_serialization():
    """
    Test CudaIPCWrapper serialization across processes using spawn method.
    This verifies that CUDA IPC handles can be properly shared between processes.
    """
    # Set multiprocessing start method to spawn
    ctx = mp.get_context("spawn")

    # Create test tensors and wrappers in the main process
    num_tensors = 3
    tensors = []
    test_data = []
    wrappers = []

    for i in range(num_tensors):
        # Create a tensor with known values
        tensor = torch.full(
            (2, 3), fill_value=float(i + 1), dtype=torch.float32, device="cuda"
        )
        tensors.append(tensor)
        wrapper = CudaIPCWrapper(tensor)
        wrappers.append(wrapper)

        # Store expected checksum and shape
        expected_checksum = float(tensor.sum().cpu().item())
        expected_shape = list(tensor.shape)
        test_data.append((expected_checksum, expected_shape))

    # Serialize the wrappers
    encoder = get_customized_encoder(type=list[CudaIPCWrapper])
    encoded_data = encoder.encode(wrappers)

    # Create a queue for results
    result_queue = ctx.Queue()

    # Start worker process
    process = ctx.Process(
        target=_worker_process_deserialize_and_reconstruct,
        args=(encoded_data, result_queue),
    )
    process.start()

    # NOTE (gingfung): we increased from 10 to 30 because of additional
    # torch_npu setup, and lmcache_tests import
    process.join(timeout=30)

    # Check if process completed successfully
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Worker process timed out")

    assert process.exitcode == 0, (
        f"Worker process failed with exit code {process.exitcode}"
    )

    # Get result from queue
    assert not result_queue.empty(), "No result received from worker process"
    status, checksums, shapes = result_queue.get()

    assert status == "success", f"Worker process encountered error: {checksums}"
    assert len(checksums) == num_tensors, "Number of checksums does not match"
    assert len(shapes) == num_tensors, "Number of shapes does not match"

    # Verify checksums and shapes match
    for i, (
        (expected_checksum, expected_shape),
        actual_checksum,
        actual_shape,
    ) in enumerate(zip(test_data, checksums, shapes, strict=False)):
        assert actual_shape == expected_shape, (
            f"Tensor {i}: shape mismatch. Expected {expected_shape}, got {actual_shape}"
        )
        assert abs(actual_checksum - expected_checksum) < 1e-5, (
            f"Tensor {i}: checksum mismatch. Expected {expected_checksum}, "
            f"got {actual_checksum}"
        )

    # Verify that the tensors are being modified in the worker process
    for i, (tensor, (expected_checksum, _)) in enumerate(
        zip(tensors, test_data, strict=False)
    ):
        # After adding 1 to each element, the new checksum should be:
        num_elements = tensor.numel()
        new_expected_checksum = expected_checksum + float(num_elements)
        actual_checksum = float(tensor.sum().cpu().item())
        assert abs(actual_checksum - new_expected_checksum) < 1e-5, (
            f"Tensor {i}: post-modification checksum mismatch. "
            f"Expected {new_expected_checksum}, got {actual_checksum}"
        )
