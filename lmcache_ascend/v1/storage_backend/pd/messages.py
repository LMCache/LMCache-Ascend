# SPDX-License-Identifier: Apache-2.0
"""Message types for the Ascend PD backend protocol.

Defines the msgspec Struct types exchanged between sender (prefiller)
and receiver (decoder) over ZMQ side-channels, as well as the
``AscendPDMsg`` discriminated union used for decoding.
"""

# Standard
from typing import Union

# Third Party
from lmcache.v1.storage_backend.pd_backend import (
    AllocRequest,
    AllocResponse,
    ProxyNotif,
)
import msgspec


class AscendAllocResponse(AllocResponse):
    """Allocation response carrying UUID-based buffer references.

    Instead of just raw page addresses (``remote_indexes``), the receiver
    returns ``(remote_buffer_uuids, remote_indexes)`` pairs so
    the sender can resolve remote memory via the HCCL channel's
    ``PeerMemHandleList.resolve_addr(uuid, page_index)`` on write.
    """

    remote_buffer_uuids: list[str]
    alloc_failed: bool = False


# ──────────────────────────────────────────────────────────
# Pull-mode message types
# ──────────────────────────────────────────────────────────


class PullReadyNotif(msgspec.Struct, tag=True):
    """Sent by the sender (prefiller) to the receiver (decoder) to advertise
    that KV chunks are ready to be pulled.

    Contains the sender's HCCL buffer references so the receiver can
    construct RDMA read operations.
    """

    pull_id: str  # Unique ID for this pull batch
    keys: list[str]
    sender_buffer_uuids: list[str]
    sender_mem_indexes: list[int]
    sender_id: str  # Sender's HCCL peer ID (for transfer_spec receiver_id)
    sender_done_url: str  # URL where receiver PUSHes PullDoneSignal
    fmt: int
    shape: list[int]
    dtype: str
    last_chunk_toks: int


class PullReadyDoneAck(msgspec.Struct, tag=True):
    """Sent by the receiver back to the sender to acknowledge the
    PullReadyNotif.  Contains indexes of keys already received.

    When ``alloc_failed`` is ``True``, the receiver could not allocate
    memory for the requested chunks.  The sender should release its
    pinned resources and skip the transfer.
    """

    already_sent_indexes: list[int]
    alloc_failed: bool = False


class PullDoneSignal(msgspec.Struct, tag=True):
    """Sent by the receiver to the sender after all chunks in a pull batch
    have been read.  The sender releases its pinned resources."""

    pull_id: str


AscendPDMsg = Union[
    AllocRequest,
    AscendAllocResponse,
    ProxyNotif,
    PullReadyNotif,
    PullReadyDoneAck,
    PullDoneSignal,
]
