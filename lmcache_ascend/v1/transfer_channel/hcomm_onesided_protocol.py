# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Union

# Third Party
import msgspec

# Local
from .buffer_config import PeerBufferInfo, RemotePeerBufferList


class HcommOsMsgBase(msgspec.Struct, tag=True):
    pass


class HcommDeviceInfo(msgspec.Struct):
    """Device info exchanged during handshake to build the rank table."""

    server_id: str
    phy_device_id: str
    device_ip: str
    super_device_id: str = "0"
    super_pod_id: str = "0"
    use_v2: bool = False


class HcommOsInitRequest(HcommOsMsgBase):
    local_id: str
    buffers: List[PeerBufferInfo]
    device_info: HcommDeviceInfo


class HcommOsInitResponse(HcommOsMsgBase):
    cluster_json: str
    comm_name: str
    server_rank: int
    client_rank: int
    buffers: List[PeerBufferInfo]


class HcommOsReadyRequest(HcommOsMsgBase):
    local_id: str


class HcommOsReadyResponse(HcommOsMsgBase):
    ok: bool


HcommOsMsg = Union[
    HcommOsInitRequest,
    HcommOsInitResponse,
    HcommOsReadyRequest,
    HcommOsReadyResponse,
]


class _PeerState:
    __slots__ = ("comm", "my_rank", "remote_rank", "remote_buffers")

    def __init__(
        self,
        comm: int,
        my_rank: int,
        remote_rank: int,
        remote_buffers: RemotePeerBufferList,
    ):
        self.comm = comm
        self.my_rank = my_rank
        self.remote_rank = remote_rank
        self.remote_buffers = remote_buffers
