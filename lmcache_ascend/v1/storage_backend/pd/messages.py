# SPDX-License-Identifier: Apache-2.0
"""Message types for the Ascend PD backend protocol.

Defines the msgspec Struct types exchanged between sender (prefiller)
and receiver (decoder) over ZMQ side-channels, as well as the
``AscendPDMsg`` discriminated union used for decoding.
"""

# Standard
from typing import Union

# Third Party
from lmcache.v1.storage_backend.pd_backend import AllocRequest, AllocResponse
import msgspec


class AscendAllocRequest(AllocRequest):
    """Allocation request extended with the token context the receiver
    needs for its L1 residency check.

    The receiver-side alloc handler dedups incoming chunks against the
    receive buffer by PD chunk key (string equality). Extending the
    dedup to the decoder's LMCache server cannot reuse those keys: the
    server identifies chunks by rolling content hashes (computed from
    token 0) + model/rank, so the token sequence itself must ride on
    the request.

    ``token_ids`` carries the FULL prompt token ids (hashing a chunk
    requires every token before it) and ``token_start`` the absolute
    token position of the first key's chunk. Both default so older
    senders keep decoding; a request without tokens simply skips the
    L1 check (fail-open, same behavior as before).
    """

    token_ids: list[int] = []
    token_start: int = 0


class AscendAllocResponse(AllocResponse):
    """Allocation response carrying UUID-based buffer references.

    Instead of just raw page addresses (``remote_indexes``), the receiver
    returns ``(remote_buffer_uuids, remote_indexes)`` pairs so the
    sender can resolve remote memory via the HCCL channel's
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

    ``token_ids`` / ``token_start`` mirror :class:`AscendAllocRequest`:
    the token context for the receiver's L1 residency check.
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
    token_ids: list[int] = []
    token_start: int = 0


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
    AscendAllocRequest,
    AscendAllocResponse,
    PullReadyNotif,
    PullReadyDoneAck,
    PullDoneSignal,
]
