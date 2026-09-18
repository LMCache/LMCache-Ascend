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
from lmcache_tests.v1.multiprocess.test_custom_types import (  # noqa: F401, E402
    get_customized_decoder,
    get_customized_encoder,
    test_cudaipc_wrapper_list_serialization,
    test_cudaipc_wrapper_serialization,
    test_ipc_cache_engine_key_serialization,
)

# LMC-A: upstream moved wrapper dispatch to DeviceSpec.ipc_wrapper_cls and
# the plane-aggregating NPU wrapper itself upstream (npu/ipc_wrapper.py); the
# plugin no longer ships its own wrapper class.
from lmcache.v1.platform.npu.ipc_wrapper import NpuIPCWrapper  # noqa: E402


def _worker_process_deserialize_and_reconstruct(
    encoded_data: bytes, result_queue: Queue
):
    """
    Worker function that runs in a separate process.
    Deserializes NpuIPCWrapper list and reconstructs layers (tensor or
    plane tuples), computing a checksum over every reconstructed plane.
    """
    try:
        # Decode the list of wrappers
        torch.npu.init()
        decoder = get_customized_decoder(type=list[NpuIPCWrapper])
        decoded_wrappers = decoder.decode(encoded_data)

        # Convert each wrapper back to tensor and compute checksum
        checksums = []
        shapes = []
        for wrapper in decoded_wrappers:
            value = wrapper.to_tensor()
            planes = (value,) if isinstance(value, torch.Tensor) else value
            # Checksum over every plane of the layer (in plane order).
            checksum = float(sum(p.sum().cpu().item() for p in planes))
            checksums.append(checksum)
            shapes.append([list(p.shape) for p in planes])

            # Do add 1 on every plane to ensure they are writable
            for plane in planes:
                plane.add_(1)

        result_queue.put(("success", checksums, shapes))
    except Exception as e:
        result_queue.put(("error", str(e), None))


@pytest.mark.skipif(
    not torch.npu.is_available(),  # LMC-A: run on npu, not cuda
    reason="NPU is required for IPCWrapper multiprocessing tests",
)
def test_cudaipc_wrapper_multiprocess_serialization():
    """
    Test NpuIPCWrapper serialization across processes using spawn method.
    This verifies that NPU IPC handles can be properly shared between processes,
    for single-tensor layers as well as per-layer plane tuples.
    """
    # Set multiprocessing start method to spawn
    ctx = mp.get_context("spawn")

    # Create test tensors and wrappers in the main process. Registration
    # mirrors what vLLM-Ascend hands wrap_kv_caches: one value per layer,
    # either a bare tensor (layer 0) or a plane tuple (layers 1 and 2).
    num_layers = 3
    tensors = []
    test_data = []
    wrappers = []

    for i in range(num_layers):
        if i == 0:
            # A bare single-tensor layer.
            tensor = torch.full(
                (2, 3),
                fill_value=float(i + 1),
                dtype=torch.float32,
                device="npu",  # LMC-A: run on npu, not cuda
            )
            tensors.append(tensor)
            wrapper = NpuIPCWrapper.wrap(tensor)
            wrappers.append(wrapper)

            expected_checksum = float(tensor.sum().cpu().item())
            expected_shape = [list(tensor.shape)]
        else:
            # A per-layer plane tuple with unequal plane widths (MLA-style).
            latent = torch.full(
                (2, 3, 1, 4),
                fill_value=float(i + 1),
                dtype=torch.float32,
                device="npu",
            )
            rope = torch.full(
                (2, 3, 1, 2),
                fill_value=float(i + 2),
                dtype=torch.float32,
                device="npu",
            )
            tensors.extend([latent, rope])
            wrapper = NpuIPCWrapper.wrap((latent, rope))
            wrappers.append(wrapper)

            expected_checksum = float(
                latent.sum().cpu().item() + rope.sum().cpu().item()
            )
            expected_shape = [list(latent.shape), list(rope.shape)]
        test_data.append((expected_checksum, expected_shape))

    # Serialize the wrappers
    encoder = get_customized_encoder(type=list[NpuIPCWrapper])
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
    assert len(checksums) == num_layers, "Number of layers does not match"
    assert len(shapes) == num_layers, "Number of layers does not match"

    # Verify checksums and per-plane shapes match, layer by layer.
    for i, (
        (expected_checksum, expected_shapes),
        actual_checksum,
        actual_shapes,
    ) in enumerate(zip(test_data, checksums, shapes, strict=False)):
        assert actual_shapes == expected_shapes, (
            f"Layer {i}: plane shape mismatch. Expected {expected_shapes}, "
            f"got {actual_shapes}"
        )
        assert abs(actual_checksum - expected_checksum) < 1e-5, (
            f"Layer {i}: checksum mismatch. Expected {expected_checksum}, "
            f"got {actual_checksum}"
        )

    # Verify that the tensors are being modified in the worker process.
    # After adding 1 to every element of every plane of the layer, the new
    # checksum should grow by the layer's total element count.
    layer_tensors: list[list[torch.Tensor]] = []
    tensor_iter = iter(tensors)
    for i in range(num_layers):
        num_planes = len(test_data[i][1])
        layer_tensors.append([next(tensor_iter) for _ in range(num_planes)])
    for i, (planes, (expected_checksum, _)) in enumerate(
        zip(layer_tensors, test_data, strict=False)
    ):
        num_elements = sum(p.numel() for p in planes)
        new_expected_checksum = expected_checksum + float(num_elements)
        actual_checksum = float(sum(p.sum().cpu().item() for p in planes))
        assert abs(actual_checksum - new_expected_checksum) < 1e-5, (
            f"Layer {i}: post-modification checksum mismatch. "
            f"Expected {new_expected_checksum}, got {actual_checksum}"
        )
