# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.transfer_channel.buffer_config import BufferType
from lmcache_ascend.v1.transfer_channel.hccl_agent import HcclMemHandleMeta
from lmcache_ascend.v1.transfer_channel.hccl_metadata import (
    decode_client_meta,
    decode_mem_handles,
    decode_server_meta,
    encode_client_meta,
    encode_mem_handles,
    encode_server_meta,
)
import lmcache_ascend.hccl_npu_comms as hcomm


def _make_mem_handle(uuid: str, buffer_type: BufferType) -> HcclMemHandleMeta:
    mem_handle = hcomm.RmaMemDesc()
    mem_handle.local_rank_id = 3
    mem_handle.remote_rank_id = 4
    mem_handle.mem_desc = b"\x01" * len(mem_handle.mem_desc)
    return HcclMemHandleMeta(
        mem_handle=mem_handle,
        buffer_ptr=0x1000,
        buffer_size=0x2000,
        page_size=0x100,
        local_buffer_addrs=[0x1000, 0x1100],
        buffer_type=buffer_type,
        uuid=uuid,
    )


def test_client_meta_round_trip() -> None:
    meta = hcomm.ClientMeta()
    meta.dev_id = 1
    meta.ipv4_addr = 0x0A000001

    decoded = decode_client_meta(encode_client_meta(meta))

    assert decoded.dev_id == meta.dev_id
    assert decoded.ipv4_addr == meta.ipv4_addr


def test_server_meta_round_trip() -> None:
    meta = hcomm.ServerMeta()
    meta.dev_id = 2
    meta.ipv4_addr = 0x0A000002
    meta.listen_port = 12345
    meta.tag_ctrl = b"\x02" * 32
    meta.tag_data = b"\x03" * 32

    decoded = decode_server_meta(encode_server_meta(meta))

    assert decoded.dev_id == meta.dev_id
    assert decoded.ipv4_addr == meta.ipv4_addr
    assert decoded.listen_port == meta.listen_port
    assert decoded.tag_ctrl == meta.tag_ctrl
    assert decoded.tag_data == meta.tag_data


def test_mem_handles_round_trip() -> None:
    handles = [
        _make_mem_handle("buffer-0", BufferType.CPU),
        _make_mem_handle("buffer-1", BufferType.NPU),
    ]

    decoded = decode_mem_handles(encode_mem_handles(handles))

    assert len(decoded) == len(handles)
    for original, restored in zip(handles, decoded, strict=True):
        assert restored.mem_handle.local_rank_id == original.mem_handle.local_rank_id
        assert restored.mem_handle.remote_rank_id == original.mem_handle.remote_rank_id
        assert restored.mem_handle.mem_desc == original.mem_handle.mem_desc
        assert restored.buffer_ptr == original.buffer_ptr
        assert restored.buffer_size == original.buffer_size
        assert restored.page_size == original.page_size
        assert restored.local_buffer_addrs == original.local_buffer_addrs
        assert restored.buffer_type == original.buffer_type
        assert restored.uuid == original.uuid


def test_empty_mem_handles_round_trip() -> None:
    assert decode_mem_handles(encode_mem_handles([])) == []


def test_decode_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported HCCL metadata encoding"):
        decode_client_meta(b"\x80\x04legacy-pickle-payload")


def test_decode_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="Malformed HCCL metadata payload"):
        decode_client_meta(b"LMC-A-HCCL-META\x01\xc1")
