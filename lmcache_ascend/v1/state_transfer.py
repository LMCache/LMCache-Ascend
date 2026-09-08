# SPDX-License-Identifier: Apache-2.0
"""Synchronous physical checkpoint copies; no scheduling or publication."""

# Third Party
import torch

# First Party
from lmcache_ascend import c_ops as lmc_ops
from lmcache_ascend.v1.state_checkpoint import StateOperation


def transfer_state(operation: StateOperation) -> None:
    """Copy one GDN checkpoint and wait for its runtime device's current stream.

    The caller must establish dependencies on any other producer stream and
    keep the buffer owned and the runtime block stable throughout this call.
    This function only borrows their tensors; it does not publish availability.
    After failure, a store's buffer must not be published and a load's runtime
    block must not be used for inference, since either may be partially copied.
    """
    memory_tensors = list(operation.buffer.planes)
    operation.runtime.validate(operation.buffer.layout)
    state_tensors = [
        entry[plane_index]
        for plane_index in range(len(memory_tensors))
        for entry in operation.runtime.tensors
    ]
    device = state_tensors[0].device
    if device.type != "npu":
        raise ValueError("State transfer requires NPU runtime tensors")

    with torch.npu.device(device):
        stream = torch.npu.current_stream()
        try:
            # The native boundary validates every plane before its first launch.
            lmc_ops.multi_layer_gdn_state_transfer(
                memory_tensors,
                state_tensors,
                operation.runtime.block_id,
                operation.direction == "store",
            )
        except BaseException as error:
            # A failed submission may already have queued work using these views.
            try:
                stream.synchronize()
            except BaseException as drain_error:
                raise error from drain_error
            raise
        stream.synchronize()
