# SPDX-License-Identifier: Apache-2.0
"""Handshake message types for the Mooncake TE transfer channel.

Unlike HIXL, Mooncake TE needs no explicit ``connect()`` phase: the peer
is addressed by its engine name (``ip:rpc_port``) directly in
``batch_transfer_sync``. The handshake therefore only exchanges engine
names and buffer layout tables over the ZMQ init side channel.
"""

# Standard
from typing import List, Union

# Third Party
import msgspec

# Local
from .buffer_config import PeerBufferInfo


class MooncakeMsgBase(msgspec.Struct, tag=True):
    """Base class for all Mooncake-related handshake messages."""

    pass


class MooncakeInitRequest(MooncakeMsgBase):
    local_id: str
    engine_name: str  # ip:rpc_port of the requesting side


class MooncakeInitResponse(MooncakeMsgBase):
    engine_name: str  # ip:rpc_port of the responding side


class MooncakeMemInfoRequest(MooncakeMsgBase):
    local_id: str
    buffers: List[PeerBufferInfo]


class MooncakeMemInfoResponse(MooncakeMsgBase):
    buffers: List[PeerBufferInfo]


class MooncakeErrorResponse(MooncakeMsgBase):
    ok: bool = False


MooncakeMsg = Union[
    MooncakeInitRequest,
    MooncakeInitResponse,
    MooncakeMemInfoRequest,
    MooncakeMemInfoResponse,
    MooncakeErrorResponse,
]
