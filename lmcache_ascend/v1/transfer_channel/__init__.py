# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel

# todo：Deleted after adaptation to HCCL in 8.5.0
try:
    # First Party
    from lmcache_ascend.v1.transfer_channel.hccl_channel import HcclChannel

    HCCL_AVAILABLE = True
except ImportError:
    HCCL_AVAILABLE = False
    HcclChannel = None


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


def CreateTransferChannel(
    channel_type: str,
    async_mode: bool,
    role: str,
    buffer_ptr: int,
    buffer_size: int,
    align_bytes: int,
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

    if not HCCL_AVAILABLE:
        raise RuntimeError(
            "HcclChannel is not available. This is likely due to CANN 8.5 "
            "adaptation in progress. Please check your CANN version and "
            "lmcache_ascend installation."
        )

    transfer_channel = HcclChannel(
        async_mode=async_mode,
        role=role,
        buffer_ptr=buffer_ptr,
        buffer_size=buffer_size,
        align_bytes=align_bytes,
        tp_rank=tp_rank,
        peer_init_url=peer_init_url,
        **kwargs,
    )
    return transfer_channel
