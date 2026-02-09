# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Union

# Third Party
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
import torch
import torch_npu  # noqa: F401

# First Party
from lmcache_ascend.v1.transfer_channel.hccl_channel import HcclChannel

# Local
from .hccl_agent import BufferConfig, BufferType


def get_correct_device(device: str, worker_id: int) -> str:
    """
    Get the correct device based on the given device string.

    Args:
        device (str): The device string, could be cpu or npu.
        worker_id (int): The worker id to determine the npu device.

    Returns:
        str: The correct device string with device id.
    """
    if device == "cpu":
        return "cpu"
    elif device.startswith("npu"):
        return f"npu:{worker_id}"
    else:
        raise ValueError(f"Invalid device: {device}")


def get_device_buffer_type(device: str) -> BufferType:
    if device == "cpu":
        return BufferType.CPU
    elif device.startswith("npu"):
        return BufferType.NPU
    else:
        raise ValueError(f"Invalid device: {device}")


def CreateTransferChannel(
    channel_type: str,
    async_mode: bool,
    role: str,
    buffer_ptr: Union[
        int, List[int]
    ],  # accept both single buffer and multiple buffers for hccl
    buffer_size: Union[
        int, List[int]
    ],  # accept both single buffer and multiple buffers for hccl
    align_bytes: Union[int, List[int]],
    tp_rank: int,
    peer_init_url: str,
    **kwargs,
) -> BaseTransferChannel:
    """
    Create a transfer channel based on the specified channel type.

    :param channel_type: Type of the transfer channel (e.g., "hccl").
    :param async_mode: Whether to operate in asynchronous mode.
    :param role: Role of the channel (e.g., "both", "sender" or "receiver").
    :param buffer_ptr: Pointer to the pre-allocated buffer.
    :param buffer_size: Size of the pre-allocated buffer in bytes.
    :param align_bytes: Alignment requirement in bytes.
    :param tp_rank: Tensor parallel rank of the current process.
    :param peer_init_url: Initialization URL for the peer.
    :kwargs: Additional keyword arguments specific to the channel type.

    :return: An instance of the specified transfer channel.
    """

    assert channel_type in ["hccl"], f"Unsupported channel type: {channel_type}"

    # construct the buffer config here
    buffer_type = kwargs.get("buffer_type", [])
    buffer_configs = []
    if isinstance(buffer_ptr, int):
        buffer_ptr = [buffer_ptr]
        # since int, then we assert buffer size is also an int
        assert isinstance(buffer_size, int), (
            "buffer_size must be int when buffer_ptr is int"
        )
        buffer_size = [buffer_size]
        # TODO (gingfung): We should not assume this.
        buffer_type = [buffer_type] if buffer_type else ["cpu"]
    else:
        assert isinstance(buffer_ptr, list), "buffer_ptr must be int or list of int"
        assert isinstance(buffer_size, list), "buffer_size must be int or list of int"
        assert isinstance(align_bytes, list), "align_bytes must be int or list of int"
        assert len(buffer_ptr) == len(buffer_size), (
            "buffer_ptr and buffer_size must have the same length"
        )
        if not buffer_type:
            raise ValueError("buffer_type must be provided when buffer_ptr is a list")
        assert isinstance(buffer_type, list), (
            "buffer_type must be list when buffer_ptr is list"
        )
        assert len(buffer_type) == len(buffer_ptr), (
            "buffer_type must have the same length as buffer_ptr"
        )

    for ptr, size, b_type, align in zip(
        buffer_ptr, buffer_size, buffer_type, align_bytes, strict=False
    ):
        device_type = get_device_buffer_type(b_type)
        device_id = -1 if device_type == BufferType.CPU else torch.npu.current_device()
        buffer_configs.append(
            BufferConfig(
                ptr=ptr,
                size=size,
                device_id=device_id,
                device_type=device_type,
                align_bytes=align,
            )
        )

    transfer_channel = HcclChannel(
        async_mode=async_mode,
        role=role,
        buffers=buffer_configs,
        tp_rank=tp_rank,
        peer_init_url=peer_init_url,
        **kwargs,
    )
    return transfer_channel
