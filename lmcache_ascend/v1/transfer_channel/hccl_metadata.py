# SPDX-License-Identifier: Apache-2.0
"""Data-only codec for HCCL handshake metadata.

The handshake exchanges ``ClientMeta``, ``ServerMeta`` and registered memory
handle metadata over a control socket. These payloads used to be pickled; they
are now encoded as typed msgspec structs so no executable serialization format
crosses the socket.

Wire format::

    b"LMC-A-HCCL-META\\x01" + msgpack(payload)

Both peers must run the same LMCache-Ascend version. A payload without the
magic header is rejected with a clear error instead of being decoded.
"""

# Standard
from typing import Any, Optional

# Third Party
from lmcache.logging import init_logger
import msgspec

# First Party
import lmcache_ascend.hccl_npu_comms as hcomm

# Local
from .buffer_config import BufferType
from .hccl_agent import HcclMemHandleMeta

logger = init_logger(__name__)

_MAGIC = b"LMC-A-HCCL-META\x01"
_TAG_SIZE = 32  # csrc/hccl/hccl_agent.h: TAG_SIZE
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


class _ClientMetaPayload(msgspec.Struct):
    dev_id: int
    ipv4_addr: int


class _ServerMetaPayload(msgspec.Struct):
    dev_id: int
    ipv4_addr: int
    listen_port: int
    tag_ctrl: bytes
    tag_data: bytes


class _RmaMemDescPayload(msgspec.Struct):
    local_rank_id: int
    remote_rank_id: int
    mem_desc: bytes


class _MemHandlePayload(msgspec.Struct):
    mem_handle: _RmaMemDescPayload
    buffer_ptr: int
    buffer_size: int
    page_size: int
    local_buffer_addrs: Optional[list[int]]
    buffer_type: int
    uuid: str


def _encode(payload: Any) -> bytes:
    return _MAGIC + msgspec.msgpack.encode(payload)


def _decode(data: bytes, payload_type: Any) -> Any:
    if not data.startswith(_MAGIC):
        raise ValueError(
            "Unsupported HCCL metadata encoding; both peers must run the same "
            "LMCache-Ascend version"
        )
    try:
        return msgspec.msgpack.decode(data[len(_MAGIC) :], type=payload_type)
    except msgspec.DecodeError as exc:
        logger.exception("Failed to decode HCCL metadata payload")
        raise ValueError("Malformed HCCL metadata payload") from exc


def _check_int(value: int, name: str, minimum: int, maximum: int) -> None:
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} out of range: {value}")


def _check_tag(value: bytes, name: str) -> None:
    if len(value) != _TAG_SIZE:
        raise ValueError(f"{name} must be {_TAG_SIZE} bytes, got {len(value)}")


def encode_client_meta(meta: hcomm.ClientMeta) -> bytes:
    """Encode handshake client metadata.

    Args:
        meta: Client metadata returned by ``HcclAgent.get_client_meta``.

    Returns:
        The versioned msgpack payload.

    Raises:
        ValueError: If a field is outside its C++ type's range.
    """
    _check_int(meta.dev_id, "client.dev_id", _INT32_MIN, _INT32_MAX)
    _check_int(meta.ipv4_addr, "client.ipv4_addr", 0, _UINT32_MAX)
    return _encode(_ClientMetaPayload(dev_id=meta.dev_id, ipv4_addr=meta.ipv4_addr))


def decode_client_meta(data: bytes) -> hcomm.ClientMeta:
    """Decode handshake client metadata.

    Args:
        data: A payload produced by :func:`encode_client_meta`.

    Returns:
        A reconstructed ``ClientMeta``.

    Raises:
        ValueError: If the payload is malformed or has an unknown version.
    """
    payload = _decode(data, _ClientMetaPayload)
    _check_int(payload.dev_id, "client.dev_id", _INT32_MIN, _INT32_MAX)
    _check_int(payload.ipv4_addr, "client.ipv4_addr", 0, _UINT32_MAX)
    meta = hcomm.ClientMeta()
    meta.dev_id = payload.dev_id
    meta.ipv4_addr = payload.ipv4_addr
    return meta


def encode_server_meta(meta: hcomm.ServerMeta) -> bytes:
    """Encode handshake server metadata.

    Args:
        meta: Server metadata returned by ``HcclAgent.get_server_meta``.

    Returns:
        The versioned msgpack payload.

    Raises:
        ValueError: If a field is invalid.
    """
    _check_int(meta.dev_id, "server.dev_id", _INT32_MIN, _INT32_MAX)
    _check_int(meta.ipv4_addr, "server.ipv4_addr", 0, _UINT32_MAX)
    _check_int(meta.listen_port, "server.listen_port", 0, _UINT32_MAX)
    _check_tag(meta.tag_ctrl, "server.tag_ctrl")
    _check_tag(meta.tag_data, "server.tag_data")
    return _encode(
        _ServerMetaPayload(
            dev_id=meta.dev_id,
            ipv4_addr=meta.ipv4_addr,
            listen_port=meta.listen_port,
            tag_ctrl=meta.tag_ctrl,
            tag_data=meta.tag_data,
        )
    )


def decode_server_meta(data: bytes) -> hcomm.ServerMeta:
    """Decode handshake server metadata.

    Args:
        data: A payload produced by :func:`encode_server_meta`.

    Returns:
        A reconstructed ``ServerMeta``.

    Raises:
        ValueError: If the payload is malformed or has an unknown version.
    """
    payload = _decode(data, _ServerMetaPayload)
    _check_int(payload.dev_id, "server.dev_id", _INT32_MIN, _INT32_MAX)
    _check_int(payload.ipv4_addr, "server.ipv4_addr", 0, _UINT32_MAX)
    _check_int(payload.listen_port, "server.listen_port", 0, _UINT32_MAX)
    _check_tag(payload.tag_ctrl, "server.tag_ctrl")
    _check_tag(payload.tag_data, "server.tag_data")
    meta = hcomm.ServerMeta()
    meta.dev_id = payload.dev_id
    meta.ipv4_addr = payload.ipv4_addr
    meta.listen_port = payload.listen_port
    meta.tag_ctrl = payload.tag_ctrl
    meta.tag_data = payload.tag_data
    return meta


def _encode_mem_handle(handle: HcclMemHandleMeta) -> _MemHandlePayload:
    _check_int(handle.buffer_ptr, "mem_handle.buffer_ptr", 0, _UINT64_MAX)
    _check_int(handle.buffer_size, "mem_handle.buffer_size", 0, _UINT64_MAX)
    _check_int(handle.page_size, "mem_handle.page_size", 1, _UINT64_MAX)
    if handle.local_buffer_addrs is not None:
        for addr in handle.local_buffer_addrs:
            _check_int(addr, "mem_handle.local_buffer_addrs", 0, _UINT64_MAX)
    if not handle.uuid:
        raise ValueError("mem_handle.uuid must not be empty")
    mem_handle = handle.mem_handle
    _check_int(mem_handle.local_rank_id, "mem_handle.local_rank_id", 0, _UINT32_MAX)
    _check_int(mem_handle.remote_rank_id, "mem_handle.remote_rank_id", 0, _UINT32_MAX)
    return _MemHandlePayload(
        mem_handle=_RmaMemDescPayload(
            local_rank_id=mem_handle.local_rank_id,
            remote_rank_id=mem_handle.remote_rank_id,
            mem_desc=mem_handle.mem_desc,
        ),
        buffer_ptr=handle.buffer_ptr,
        buffer_size=handle.buffer_size,
        page_size=handle.page_size,
        local_buffer_addrs=handle.local_buffer_addrs,
        buffer_type=handle.buffer_type.value,
        uuid=handle.uuid,
    )


def _decode_mem_handle(payload: _MemHandlePayload) -> HcclMemHandleMeta:
    _check_int(payload.buffer_ptr, "mem_handle.buffer_ptr", 0, _UINT64_MAX)
    _check_int(payload.buffer_size, "mem_handle.buffer_size", 0, _UINT64_MAX)
    _check_int(payload.page_size, "mem_handle.page_size", 1, _UINT64_MAX)
    if payload.local_buffer_addrs is not None:
        for addr in payload.local_buffer_addrs:
            _check_int(addr, "mem_handle.local_buffer_addrs", 0, _UINT64_MAX)
    if not payload.uuid:
        raise ValueError("mem_handle.uuid must not be empty")
    _check_int(
        payload.mem_handle.local_rank_id, "mem_handle.local_rank_id", 0, _UINT32_MAX
    )
    _check_int(
        payload.mem_handle.remote_rank_id,
        "mem_handle.remote_rank_id",
        0,
        _UINT32_MAX,
    )
    mem_handle = hcomm.RmaMemDesc()
    mem_handle.local_rank_id = payload.mem_handle.local_rank_id
    mem_handle.remote_rank_id = payload.mem_handle.remote_rank_id
    try:
        mem_handle.mem_desc = payload.mem_handle.mem_desc
    except RuntimeError as exc:
        logger.exception("Rejected HCCL mem_desc payload")
        raise ValueError("Invalid HCCL mem_desc payload") from exc
    try:
        buffer_type = BufferType(payload.buffer_type)
    except ValueError as exc:
        raise ValueError(f"Unknown HCCL buffer type: {payload.buffer_type}") from exc
    return HcclMemHandleMeta(
        mem_handle=mem_handle,
        buffer_ptr=payload.buffer_ptr,
        buffer_size=payload.buffer_size,
        page_size=payload.page_size,
        local_buffer_addrs=payload.local_buffer_addrs,
        buffer_type=buffer_type,
        uuid=payload.uuid,
    )


def encode_mem_handles(handles: list[HcclMemHandleMeta]) -> bytes:
    """Encode registered memory handle metadata.

    Args:
        handles: Registered buffers from ``HcclAgentWrapper.mem_handles``.

    Returns:
        The versioned msgpack payload.

    Raises:
        ValueError: If a handle contains an invalid field.
    """
    return _encode([_encode_mem_handle(handle) for handle in handles])


def decode_mem_handles(data: bytes) -> list[HcclMemHandleMeta]:
    """Decode registered memory handle metadata.

    Args:
        data: A payload produced by :func:`encode_mem_handles`.

    Returns:
        The reconstructed handle metadata list.

    Raises:
        ValueError: If the payload is malformed or has an unknown version.
    """
    payloads = _decode(data, list[_MemHandlePayload])
    return [_decode_mem_handle(payload) for payload in payloads]
