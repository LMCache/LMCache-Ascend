# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Union

import torch
import torch_npu  # noqa: F401
# Third Party
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel

# Local
from .buffer_config import BufferConfig, BufferType, get_device_buffer_type


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


def _build_buffer_configs(
    buffer_ptr: Union[int, List[int]],
    buffer_size: Union[int, List[int]],
    align_bytes: Union[int, List[int]],
    buffer_type: Union[str, List[str]],
) -> List[BufferConfig]:
    """Normalize scalar-or-list arguments into a list of BufferConfig."""
    if isinstance(buffer_ptr, int):
        buffer_ptr = [buffer_ptr]
        assert isinstance(buffer_size, int), (
            "buffer_size must be int when buffer_ptr is int"
        )
        assert isinstance(align_bytes, int), (
            "align_bytes must be int when buffer_ptr is int"
        )

        buffer_size = [buffer_size]

        if isinstance(buffer_type, str):
            buffer_type = [buffer_type] if buffer_type else ["cpu"]
        if not buffer_type:
            buffer_type = ["cpu"]

        if isinstance(align_bytes, int):
            align_bytes = [align_bytes]
    else:
        assert isinstance(buffer_ptr, list), "buffer_ptr must be int or list of int"
        assert isinstance(buffer_size, list), "buffer_size must be int or list of int"
        assert isinstance(align_bytes, list), (
            f"align_bytes must be int or list of int, but got {align_bytes}"
        )
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

    buffer_configs: List[BufferConfig] = []
    for ptr, size, b_type, align in zip(
        buffer_ptr, buffer_size, buffer_type, align_bytes, strict=True
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
    return buffer_configs


def CreateTransferChannel(
    channel_type: str,
    async_mode: bool,
    role: str,
    buffer_ptr: Union[int, List[int]],
    buffer_size: Union[int, List[int]],
    align_bytes: Union[int, List[int]],
    tp_rank: int,
    peer_init_url: str,
    **kwargs,
) -> BaseTransferChannel:
    """
    Create a transfer channel based on the specified channel type.

    :param channel_type: Type of the transfer channel
        (e.g., "hccl", "hixl", "hcomm_onesided").
    :param async_mode: Whether to operate in asynchronous mode.
    :param role: Role of the channel (e.g., "both", "sender" or "receiver").
    :param buffer_ptr: Pointer(s) to the pre-allocated buffer(s).
    :param buffer_size: Size(s) of the pre-allocated buffer(s) in bytes.
    :param align_bytes: Alignment requirement(s) in bytes.
    :param tp_rank: Tensor parallel rank of the current process.
    :param peer_init_url: Initialization URL for the peer.
    :kwargs: Additional keyword arguments specific to the channel type.

    :return: An instance of the specified transfer channel.
    """

    assert channel_type in [
        "hccl",
        "hixl",
        "hcomm_onesided",
    ], f"Unsupported channel type: {channel_type}"

    buffer_type = kwargs.pop("buffer_type", [])
    buffer_configs = _build_buffer_configs(
        buffer_ptr, buffer_size, align_bytes, buffer_type
    )

    if channel_type == "hixl":
        # First Party
        from lmcache_ascend.v1.transfer_channel.hixl_channel import HixlChannel

        return HixlChannel(
            async_mode=async_mode,
            buffers=buffer_configs,
            role=role,
            tp_rank=tp_rank,
            peer_init_url=peer_init_url,
            **kwargs,
        )
    elif channel_type == "hcomm_onesided":
        # First Party
        from lmcache_ascend.v1.transfer_channel.hcomm_onesided_channel import \
            HcommOneSidedChannel

        return HcommOneSidedChannel(
            async_mode=async_mode,
            buffers=buffer_configs,
            role=role,
            tp_rank=tp_rank,
            peer_init_url=peer_init_url,
            **kwargs,
        )
    else:
        # First Party
        from lmcache_ascend.v1.transfer_channel.hccl_channel import HcclChannel

        return HcclChannel(
            async_mode=async_mode,
            buffers=buffer_configs,
            role=role,
            tp_rank=tp_rank,
            peer_init_url=peer_init_url,
            **kwargs,
        )
