# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Union
import asyncio
import threading
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_zmq_socket
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)
import msgspec
import torch
import zmq

# First Party
import lmcache_ascend.hixl_npu_comms as hixl_comms

# Local
from .base_channel import BaseMultiBufferChannel
from .buffer_config import BufferConfig, PeerBufferInfo, RemotePeerBufferList
from .hixl_engine import HixlEngineWrapper
from .hixl_protocol import (
    HixlInitRequest,
    HixlInitResponse,
    HixlMemInfoRequest,
    HixlMemInfoResponse,
    HixlMsg,
    HixlReadyRequest,
    HixlReadyResponse,
)
from .transfer_spec import resolve_peer_id

logger = init_logger(__name__)


class HixlChannel(BaseMultiBufferChannel):
    _init_msg_type = Union[HixlMsg, SideMsg]
    _channel_name = "hixl"

    def __init__(
        self,
        async_mode: bool = False,
        buffers: Optional[List[BufferConfig]] = None,
        **kwargs,
    ):
        self.hixl_wrapper: Optional[HixlEngineWrapper] = None
        self._buffer_pool = kwargs.get("buffer_pool", "0:0")

        # Maps peer_id -> remote engine string (ip:port).
        self.remote_engine_dict: dict[str, str] = {}
        # Maps peer_id -> RemotePeerBufferList.
        self.remote_peer_buffers: dict[str, RemotePeerBufferList] = {}
        super().__init__(async_mode=async_mode, buffers=buffers, **kwargs)

    def _register_buffers(self, buffers: list[BufferConfig]) -> None:
        self.hixl_wrapper = HixlEngineWrapper(
            buffers=buffers,
            buffer_pool=self._buffer_pool,
        )
        self.mem_handles = self.hixl_wrapper.mem_handles

    def _connect_to_peer(self, peer_id: str, remote_engine_id: str) -> None:
        logger.info("Connecting to remote HIXL engine: %s", remote_engine_id)
        self.hixl_wrapper.engine.connect(remote_engine_id)
        with self._state_lock:
            self.remote_engine_dict[peer_id] = remote_engine_id
        logger.info("Connected to remote HIXL engine: %s", remote_engine_id)

    def _store_remote_mem_info(
        self, peer_id: str, resp_buffers: List[PeerBufferInfo]
    ) -> None:
        remote_buffers = RemotePeerBufferList(resp_buffers)
        with self._state_lock:
            self.remote_peer_buffers[peer_id] = remote_buffers

    def _make_buffer_infos(self) -> List[PeerBufferInfo]:
        """Build handshake buffer-info list from our registered handles."""
        return [
            PeerBufferInfo(
                uuid=meta.uuid,
                buffer_ptr=meta.buffer_ptr,
                buffer_size=meta.buffer_size,
                page_size=meta.page_size,
            )
            for meta in self.hixl_wrapper.mem_handles
        ]

    def _make_mem_info_request(self, local_id: str) -> HixlMemInfoRequest:
        return HixlMemInfoRequest(
            local_id=local_id,
            buffers=self._make_buffer_infos(),
        )

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        # Step 1: exchange engine IDs
        init_req = HixlInitRequest(
            local_id=local_id,
            engine_id=self.hixl_wrapper.engine_id,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(init_req))
        resp = msgspec.msgpack.decode(init_tmp_socket.recv(), type=HixlMsg)
        if not isinstance(resp, HixlInitResponse):
            raise ValueError(f"Expected HixlInitResponse, got {type(resp).__name__}")
        self._connect_to_peer(peer_id, resp.engine_id)

        # Step 2: signal ready so server knows connect() finished
        init_tmp_socket.send(
            msgspec.msgpack.encode(HixlReadyRequest(local_id=local_id))
        )
        ready_bytes = init_tmp_socket.recv()
        ready_resp = msgspec.msgpack.decode(ready_bytes, type=HixlMsg)
        if isinstance(ready_resp, HixlReadyResponse) and not ready_resp.ok:
            raise ConnectionError(
                f"Server failed to complete handshake for peer {peer_id}"
            )

        # Step 3: exchange buffer layout info
        init_tmp_socket.send(
            msgspec.msgpack.encode(self._make_mem_info_request(local_id))
        )
        mem_resp = msgspec.msgpack.decode(init_tmp_socket.recv(), type=HixlMsg)
        if not isinstance(mem_resp, HixlMemInfoResponse):
            raise ValueError(
                f"Expected HixlMemInfoResponse, got {type(mem_resp).__name__}"
            )
        self._store_remote_mem_info(peer_id, mem_resp.buffers)

        # Step 4: optional side message
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = self.send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        # Step 1: exchange engine IDs
        init_req = HixlInitRequest(
            local_id=local_id,
            engine_id=self.hixl_wrapper.engine_id,
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(init_req))
        resp = msgspec.msgpack.decode(await init_tmp_socket.recv(), type=HixlMsg)
        if not isinstance(resp, HixlInitResponse):
            raise ValueError(f"Expected HixlInitResponse, got {type(resp).__name__}")
        self._connect_to_peer(peer_id, resp.engine_id)

        # Step 2: signal ready so server knows connect() finished
        await init_tmp_socket.send(
            msgspec.msgpack.encode(HixlReadyRequest(local_id=local_id))
        )
        ready_bytes = await init_tmp_socket.recv()
        ready_resp = msgspec.msgpack.decode(ready_bytes, type=HixlMsg)
        if isinstance(ready_resp, HixlReadyResponse) and not ready_resp.ok:
            raise ConnectionError(
                f"Server failed to complete handshake for peer {peer_id}"
            )

        # Step 3: exchange buffer layout info
        await init_tmp_socket.send(
            msgspec.msgpack.encode(self._make_mem_info_request(local_id))
        )
        mem_resp = msgspec.msgpack.decode(await init_tmp_socket.recv(), type=HixlMsg)
        if not isinstance(mem_resp, HixlMemInfoResponse):
            raise ValueError(
                f"Expected HixlMemInfoResponse, got {type(mem_resp).__name__}"
            )
        self._store_remote_mem_info(peer_id, mem_resp.buffers)

        # Step 4: optional side message
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self.remote_engine_dict

    def _handle_init_msg(
        self, req: Union[HixlMsg, InitSideMsgBase]
    ) -> Union[HixlMsg, InitSideRetMsgBase]:
        resp: Union[HixlMsg, InitSideRetMsgBase]
        if isinstance(req, HixlInitRequest):
            logger.info("Processing HixlInitRequest from %s", req.local_id)

            resp = HixlInitResponse(
                engine_id=self.hixl_wrapper.engine_id,
            )

            remote_engine_id = req.engine_id
            connect_started_event = threading.Event()

            def complete_connection():
                torch.npu.set_device(self.handle_device)
                logger.info(
                    "Background: Connecting to remote engine %s",
                    remote_engine_id,
                )
                try:
                    connect_started_event.set()
                    self.hixl_wrapper.engine.connect(remote_engine_id)
                    with self._state_lock:
                        self.remote_engine_dict[req.local_id] = remote_engine_id
                    logger.info(
                        "Background: Connection established with %s",
                        req.local_id,
                    )
                except Exception as e:
                    logger.error("Connection failed: %s", e)

            t = threading.Thread(target=complete_connection, daemon=True)
            t.start()

            is_ready = connect_started_event.wait(timeout=20.0)
            if not is_ready:
                raise TimeoutError(
                    "Timed out waiting for connection thread to start connect()"
                )

            logger.info("Replying initialization response")

        elif isinstance(req, HixlReadyRequest):
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                with self._state_lock:
                    if req.local_id in self.remote_engine_dict:
                        break
                time.sleep(0.05)
            resp = HixlReadyResponse(
                ok=req.local_id in self.remote_engine_dict,
            )

        elif isinstance(req, HixlMemInfoRequest):
            logger.info("Processing HixlMemInfoRequest from %s", req.local_id)

            self._store_remote_mem_info(req.local_id, req.buffers)

            resp = HixlMemInfoResponse(
                buffers=self._make_buffer_infos(),
            )

            logger.info("Replying mem info response")

        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _make_error_response(self) -> HixlReadyResponse:
        return HixlReadyResponse(ok=False)

    def _build_op_descs(
        self,
        items: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict,
    ) -> tuple[str, list]:
        peer_id = resolve_peer_id(transfer_spec)

        with self._state_lock:
            remote_engine = self.remote_engine_dict[peer_id]
            remote_buffers = self.remote_peer_buffers[peer_id]
        remote_addrs = self._resolve_transfer_addrs(remote_buffers, transfer_spec)

        op_descs = []
        for mem_obj, remote_addr in zip(items, remote_addrs, strict=True):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HIXL channel"
                )
            op_descs.append(
                hixl_comms.TransferOpDesc(
                    local_addr=self.hixl_wrapper.get_local_addr(
                        mem_obj.data_ptr, mem_obj.meta.address
                    ),
                    remote_addr=remote_addr,
                    len=self.page_size,
                )
            )

        return remote_engine, op_descs

    async def _poll_transfer(self, req, op_name: str) -> None:
        while True:
            status = self.hixl_wrapper.engine.get_transfer_status(req)
            if status == hixl_comms.TransferStatus.COMPLETED:
                return
            if status == hixl_comms.TransferStatus.FAILED:
                raise RuntimeError(f"HIXL async {op_name} transfer failed")
            if status == hixl_comms.TransferStatus.TIMEOUT:
                raise TimeoutError(f"HIXL async {op_name} transfer timed out")
            await asyncio.sleep(0.001)

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        remote_engine, op_descs = self._build_op_descs(objects, transfer_spec)
        self.hixl_wrapper.engine.transfer_sync(
            remote_engine, hixl_comms.WRITE, op_descs
        )
        return len(objects)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        remote_engine, op_descs = self._build_op_descs(buffers, transfer_spec)
        self.hixl_wrapper.engine.transfer_sync(remote_engine, hixl_comms.READ, op_descs)
        return len(buffers)

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        remote_engine, op_descs = self._build_op_descs(objects, transfer_spec)
        req = self.hixl_wrapper.engine.transfer_async(
            remote_engine, hixl_comms.WRITE, op_descs
        )
        await self._poll_transfer(req, "write")
        return len(objects)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        remote_engine, op_descs = self._build_op_descs(buffers, transfer_spec)
        req = self.hixl_wrapper.engine.transfer_async(
            remote_engine, hixl_comms.READ, op_descs
        )
        await self._poll_transfer(req, "read")
        return len(buffers)

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
        self.hixl_wrapper.close()
