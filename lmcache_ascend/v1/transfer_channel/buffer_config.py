# SPDX-License-Identifier: Apache-2.0
"""Shared buffer configuration types used by all transfer channels."""

# Standard
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable, List, Optional
import uuid as _uuid

# Third Party
import msgspec


class BufferType(Enum):
    CPU = auto()
    NPU = auto()


@dataclass
class BufferConfig:
    ptr: int
    size: int
    device_id: int
    device_type: BufferType
    align_bytes: int


@dataclass
class MemHandleMeta:
    """Transport-agnostic metadata for a registered memory buffer.

    Each channel implementation stores an opaque library-specific handle
    in ``mem_handle`` (e.g. ``hcomm.RmaMemDesc``, ``int`` for hixl /
    hcomm_onesided).
    """

    mem_handle: Any
    buffer_ptr: int
    buffer_size: int
    page_size: int
    local_buffer_addrs: Optional[List[int]] = None
    buffer_type: BufferType = BufferType.CPU
    uuid: str = field(default_factory=lambda: str(_uuid.uuid4()))


class PeerBufferInfo(msgspec.Struct):
    uuid: str
    buffer_ptr: int
    buffer_size: int
    page_size: int
    is_device: bool = False


class RemotePeerBufferHandle:
    __slots__ = ("uuid", "buffer_ptr", "buffer_size", "page_size", "num_pages")

    def __init__(self, info: Any):
        self.uuid = info.uuid
        self.buffer_ptr = info.buffer_ptr
        self.buffer_size = info.buffer_size
        self.page_size = info.page_size
        self.num_pages = self.buffer_size // self.page_size


class RemotePeerBufferList:
    def __init__(self, buffer_infos: Iterable[Any]):
        self.handles = [RemotePeerBufferHandle(info) for info in buffer_infos]
        self.peer_mem_handles = self.handles
        self._uuid_to_handle = {h.uuid: h for h in self.handles}

    def extend_handles(self, buffer_infos: Iterable[Any]):
        for info in buffer_infos:
            handle = RemotePeerBufferHandle(info)
            self.handles.append(handle)
            self._uuid_to_handle[handle.uuid] = handle

    def get_handle_by_uuid(self, buffer_uuid: str) -> RemotePeerBufferHandle:
        handle = self._uuid_to_handle.get(buffer_uuid)
        if handle is None:
            raise ValueError(f"Buffer UUID {buffer_uuid} not found in remote peer buffers")
        return handle

    def resolve_addr(self, buffer_uuid: str, page_index: int) -> int:
        handle = self.get_handle_by_uuid(buffer_uuid)
        if not (0 <= page_index < handle.num_pages):
            raise IndexError(
                f"page_index {page_index} out of range [0, {handle.num_pages}) "
                f"for remote buffer {buffer_uuid}"
            )
        return handle.buffer_ptr + page_index * handle.page_size


def resolve_buffer_ref(
    mem_handles: Iterable[Any], data_ptr: int, page_index: int
) -> tuple[str, int]:
    for meta in mem_handles:
        if meta.buffer_ptr <= data_ptr < meta.buffer_ptr + meta.buffer_size:
            if meta.local_buffer_addrs is not None:
                num_pages = len(meta.local_buffer_addrs)
                if not (0 <= page_index < num_pages):
                    raise IndexError(
                        f"page_index {page_index} out of range [0, {num_pages}) "
                        f"for buffer {meta.uuid}"
                    )
            return (meta.uuid, page_index)
    raise ValueError(f"Pointer {data_ptr} not found in any registered memory handle.")


def resolve_local_addr(mem_handles: Iterable[Any], ptr: int, idx: int) -> int:
    for meta in mem_handles:
        if meta.buffer_ptr <= ptr < meta.buffer_ptr + meta.buffer_size:
            if meta.local_buffer_addrs is None:
                raise ValueError(f"Buffer {meta.uuid} has no local_buffer_addrs")
            return meta.local_buffer_addrs[idx]
    raise ValueError(f"Pointer {ptr} not found in any registered memory handle.")


def get_device_buffer_type(device: str) -> BufferType:
    if device == "cpu":
        return BufferType.CPU
    elif device.startswith("npu"):
        return BufferType.NPU
    else:
        raise ValueError(f"Invalid device: {device}")
