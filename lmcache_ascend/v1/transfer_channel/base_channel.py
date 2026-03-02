# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import abstractmethod
from typing import Optional, Union
import asyncio
import threading
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
)
import msgspec
import torch
import zmq

# Local
from .buffer_config import BufferConfig, BufferType, RemotePeerBufferList, resolve_buffer_ref

logger = init_logger(__name__)


class BaseMultiBufferChannel(BaseTransferChannel):
    _init_msg_type: type
    _channel_name = "multi-buffer"

    def __init__(
        self,
        async_mode: bool = False,
        buffers: Optional[list[BufferConfig]] = None,
        **kwargs,
    ):
        assert "role" in kwargs
        if buffers is None:
            logger.warning(
                "Buffers not provided, "
                "using legacy initialization with buffer_ptr, "
                "buffer_size, and align_bytes"
            )
            assert "buffer_ptr" in kwargs
            assert "buffer_size" in kwargs
            assert "align_bytes" in kwargs
            buffers = [
                BufferConfig(
                    ptr=kwargs["buffer_ptr"],
                    size=kwargs["buffer_size"],
                    device_id=-1,
                    device_type=BufferType.CPU,
                    align_bytes=kwargs["align_bytes"],
                )
            ]

        assert "peer_init_url" in kwargs
        self.role = kwargs["role"]
        self.page_size = buffers[0].align_bytes
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self._state_lock = threading.Lock()
        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        self.zmq_context = get_zmq_context(use_asyncio=self.async_mode)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)
        self.handle_device = torch.npu.current_device()

        self._register_buffers(buffers)
        self._init_side_channels()

    @abstractmethod
    def _register_buffers(self, buffers: list[BufferConfig]) -> None:
        raise NotImplementedError("Subclasses must implement _register_buffers")

    @abstractmethod
    def _handle_init_msg(self, req):
        raise NotImplementedError("Subclasses must implement _handle_init_msg")

    @abstractmethod
    def _make_error_response(self):
        raise NotImplementedError("Subclasses must implement _make_error_response")

    def get_local_buffer_refs(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> tuple[list[str], list[int]]:
        buffer_uuids: list[str] = []
        mem_indexes: list[int] = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert mem_obj is not None and isinstance(mem_obj, MemoryObj)
                buf_uuid, mem_idx = resolve_buffer_ref(
                    self.mem_handles, mem_obj.data_ptr, mem_obj.meta.address
                )
                buffer_uuids.append(buf_uuid)
                mem_indexes.append(mem_idx)
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                f"Sending raw bytes is not supported in {self._channel_name} channel"
            )
        return buffer_uuids, mem_indexes

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        local_indices = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                local_indices.append(mem_obj.meta.address)
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                f"Sending raw bytes is not supported in {self._channel_name} channel"
            )
        return local_indices

    def _init_side_channels(self):
        if self.peer_init_url is None:
            raise ValueError("Peer init URL is not set")

        if self.async_mode:
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            t = threading.Thread(target=self._init_loop, daemon=True)
            t.start()
            self.running_threads.append(t)

    def _init_loop(self):
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)

        torch.npu.set_device(self.handle_device)
        self.init_side_channel.setsockopt(zmq.RCVTIMEO, 1000)

        while self.running:
            try:
                req_bytes = self.init_side_channel.recv()
                req = msgspec.msgpack.decode(req_bytes, type=self._init_msg_type)
                resp = self._handle_init_msg(req)
                self.init_side_channel.send(msgspec.msgpack.encode(resp))
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(
                    "Failed to process %s initialization loop: %s",
                    self._channel_name,
                    str(e),
                )
                try:
                    self.init_side_channel.send(
                        msgspec.msgpack.encode(self._make_error_response())
                    )
                except Exception:
                    logger.error("Failed to send %s error response: %s", self._channel_name, e)
                if self.running:
                    time.sleep(0.01)

        self.init_side_channel.close()

    async def _async_init_loop(self):
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            self.peer_init_url,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)

        torch.npu.set_device(self.handle_device)
        loop = asyncio.get_running_loop()

        while self.running:
            try:
                req_bytes = await asyncio.wait_for(
                    self.init_side_channel.recv(), timeout=1.0
                )
                req = msgspec.msgpack.decode(req_bytes, type=self._init_msg_type)
                resp = await loop.run_in_executor(None, self._handle_init_msg, req)
                await self.init_side_channel.send(msgspec.msgpack.encode(resp))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(
                    "Failed to process async %s initialization loop: %s",
                    self._channel_name,
                    str(e),
                )
                try:
                    await self.init_side_channel.send(
                        msgspec.msgpack.encode(self._make_error_response())
                    )
                except Exception:
                    logger.error("Failed to send %s error response: %s", self._channel_name, e)
                if self.running:
                    time.sleep(0.01)

        self.init_side_channel.close()

    def _resolve_transfer_addrs(
        self, remote_buffers: RemotePeerBufferList, transfer_spec: dict
    ) -> list[int]:
        if (
            "remote_buffer_uuids" in transfer_spec
            and "remote_mem_indexes" in transfer_spec
        ):
            return [
                remote_buffers.resolve_addr(buf_uuid, page_idx)
                for buf_uuid, page_idx in zip(
                    transfer_spec["remote_buffer_uuids"],
                    transfer_spec["remote_mem_indexes"],
                    strict=True,
                )
            ]

        if "remote_indexes" in transfer_spec:
            first_handle = remote_buffers.handles[0]
            return [
                first_handle.buffer_ptr + remote_index * first_handle.page_size
                for remote_index in transfer_spec["remote_indexes"]
            ]

        raise ValueError(
            "transfer_spec must contain either "
            "(remote_buffer_uuids, remote_mem_indexes) or remote_indexes"
        )

    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        raise NotImplementedError("Subclasses must implement async_lazy_init_peer_connection")

    @abstractmethod
    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        raise NotImplementedError("Subclasses must implement async_lazy_init_peer_connection")
