# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Union

# Third Party
import msgspec

# Local
from .buffer_config import PeerBufferInfo


class HixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all HIXL-related handshake messages."""

    pass


class HixlInitRequest(HixlMsgBase):
    local_id: str
    engine_id: str  # ip:port of the requesting side


class HixlInitResponse(HixlMsgBase):
    engine_id: str  # ip:port of the responding side


class HixlReadyRequest(HixlMsgBase):
    local_id: str


class HixlReadyResponse(HixlMsgBase):
    ok: bool


class HixlMemInfoRequest(HixlMsgBase):
    local_id: str
    buffers: List[PeerBufferInfo]


class HixlMemInfoResponse(HixlMsgBase):
    buffers: List[PeerBufferInfo]


HixlMsg = Union[
    HixlInitRequest,
    HixlInitResponse,
    HixlReadyRequest,
    HixlReadyResponse,
    HixlMemInfoRequest,
    HixlMemInfoResponse,
]
