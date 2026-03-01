# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional, Union
import asyncio
import socket
import threading
import time


# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_ip, get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)
import msgspec
import torch
import zmq

# First Party
import lmcache_ascend.c_ops as lmc_ops
import lmcache_ascend.hixl_npu_comms as hixl_comms

logger = init_logger(__name__)


class HixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all HIXL-related handshake messages"""

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
    buffer_ptr: int
    buffer_size: int
    page_size: int


class HixlMemInfoResponse(HixlMsgBase):
    buffer_ptr: int
    buffer_size: int
    page_size: int


HixlMsg = Union[
    HixlInitRequest,
    HixlInitResponse,
    HixlReadyRequest,
    HixlReadyResponse,
    HixlMemInfoRequest,
    HixlMemInfoResponse,
]


class HixlChannel(BaseTransferChannel):
    def __init__(
        self,
        async_mode: bool = False,
        **kwargs,
    ):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "peer_init_url" in kwargs
        assert "align_bytes" in kwargs

        self.role = kwargs["role"]

        self.hixl_wrapper = HixlEngineWrapper(
            buffer_ptr=kwargs["buffer_ptr"],
            buffer_size=kwargs["buffer_size"],
            page_size=kwargs["align_bytes"],
            buffer_pool=kwargs.get("buffer_pool", "0:0"),
        )
        self.page_size = kwargs["align_bytes"]

        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self._state_lock = threading.Lock()

        # Maps peer_id -> remote engine string (ip:port)
        self.remote_engine_dict: dict[str, str] = {}
        # Maps peer_id -> list of remote buffer addresses per page
        self.remote_index_addr_dict: dict[str, list[int]] = {}

        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        self.handle_device = torch.npu.current_device()

        self._init_side_channels()

    def _connect_to_peer(self, peer_id: str, remote_engine_id: str) -> None:
        logger.info("Connecting to remote HIXL engine: %s", remote_engine_id)
        self.hixl_wrapper.engine.connect(remote_engine_id)
        with self._state_lock:
            self.remote_engine_dict[peer_id] = remote_engine_id
        logger.info("Connected to remote HIXL engine: %s", remote_engine_id)

    def _store_remote_mem_info(self, peer_id: str, mem_resp) -> None:
        addr_list = _build_addr_list(
            mem_resp.buffer_ptr, mem_resp.buffer_size, mem_resp.page_size
        )
        with self._state_lock:
            self.remote_index_addr_dict[peer_id] = addr_list

    def _make_mem_info_request(self, local_id: str) -> HixlMemInfoRequest:
        return HixlMemInfoRequest(
            local_id=local_id,
            buffer_ptr=self.hixl_wrapper.buffer_ptr,
            buffer_size=self.hixl_wrapper.buffer_size,
            page_size=self.hixl_wrapper.page_size,
        )

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        init_tmp_socket = get_zmq_socket(
            self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect",
        )

        # Step 1: exchange engine IDs
        init_req = HixlInitRequest(
            local_id=local_id, engine_id=self.hixl_wrapper.engine_id,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(init_req))
        resp = msgspec.msgpack.decode(init_tmp_socket.recv(), type=HixlMsg)
        self._connect_to_peer(peer_id, resp.engine_id)

        # Step 2: signal ready so server knows connect() finished
        init_tmp_socket.send(
            msgspec.msgpack.encode(HixlReadyRequest(local_id=local_id))
        )
        init_tmp_socket.recv()  # ack

        # Step 3: exchange buffer layout info
        init_tmp_socket.send(
            msgspec.msgpack.encode(self._make_mem_info_request(local_id))
        )
        mem_resp = msgspec.msgpack.decode(init_tmp_socket.recv(), type=HixlMsg)
        self._store_remote_mem_info(peer_id, mem_resp)

        # Step 4: optional side message
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = self.send_init_side_msg(
                init_tmp_socket, init_side_msg,
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
            self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect",
        )

        # Step 1: exchange engine IDs
        init_req = HixlInitRequest(
            local_id=local_id, engine_id=self.hixl_wrapper.engine_id,
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(init_req))
        resp = msgspec.msgpack.decode(
            await init_tmp_socket.recv(), type=HixlMsg
        )
        self._connect_to_peer(peer_id, resp.engine_id)

        # Step 2: signal ready so server knows connect() finished
        await init_tmp_socket.send(
            msgspec.msgpack.encode(HixlReadyRequest(local_id=local_id))
        )
        await init_tmp_socket.recv()  # ack

        # Step 3: exchange buffer layout info
        await init_tmp_socket.send(
            msgspec.msgpack.encode(self._make_mem_info_request(local_id))
        )
        mem_resp = msgspec.msgpack.decode(
            await init_tmp_socket.recv(), type=HixlMsg
        )
        self._store_remote_mem_info(peer_id, mem_resp)

        # Step 4: optional side message
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket, init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    def _init_side_channels(self):
        if self.peer_init_url is None:
            logger.warning("Peer init URL is not set, skipping initialization")
            return

        if self.async_mode:
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

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

            addr_list = _build_addr_list(
                req.buffer_ptr, req.buffer_size, req.page_size
            )
            with self._state_lock:
                self.remote_index_addr_dict[req.local_id] = addr_list

            resp = HixlMemInfoResponse(
                buffer_ptr=self.hixl_wrapper.buffer_ptr,
                buffer_size=self.hixl_wrapper.buffer_size,
                page_size=self.hixl_wrapper.page_size,
            )

            logger.info("Replying mem info response")

        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

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

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=Union[HixlMsg, SideMsg])
                resp = self._handle_init_msg(req)
                self.init_side_channel.send(msgspec.msgpack.encode(resp))

                logger.info("Sent initialization request response")

            except zmq.Again:
                continue

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
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
        logger.info("Starting async initialization loop")

        torch.npu.set_device(self.handle_device)

        loop = asyncio.get_running_loop()

        while self.running:
            try:
                req_bytes = await asyncio.wait_for(
                    self.init_side_channel.recv(), timeout=1.0
                )

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=Union[HixlMsg, SideMsg])
                resp = await loop.run_in_executor(None, self._handle_init_msg, req)

                logger.info("Handled init msg")

                await self.init_side_channel.send(msgspec.msgpack.encode(resp))

                logger.info("Sent initialization request response")

            except asyncio.TimeoutError:
                continue

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

        self.init_side_channel.close()

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
                "Sending raw bytes is not supported in HIXL channel"
            )
        return local_indices

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

    def _build_op_descs(
        self,
        items: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict,
    ) -> tuple[str, list]:
        peer_id = transfer_spec.get("receiver_id") or transfer_spec["sender_id"]
        with self._state_lock:
            remote_engine = self.remote_engine_dict[peer_id]
            remote_index_addr = self.remote_index_addr_dict[peer_id]

        op_descs = []
        for mem_obj, remote_index in zip(
            items, transfer_spec["remote_indexes"], strict=True
        ):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HIXL channel"
                )
            op_descs.append(
                hixl_comms.TransferOpDesc(
                    local_addr=self.hixl_wrapper.local_index_addr[mem_obj.meta.address],
                    remote_addr=remote_index_addr[remote_index],
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
        self.hixl_wrapper.engine.transfer_sync(
            remote_engine, hixl_comms.READ, op_descs
        )
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


@dataclass
class HixlEngineWrapper:
    engine: hixl_comms.Hixl
    engine_id: str
    buffer_ptr: int
    buffer_size: int
    page_size: int
    local_index_addr: list[int]
    mem_handle: int  # opaque MemHandle as uintptr_t

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        buffer_pool: str = "0:0",
    ):
        device_id = torch.npu.current_device()

        is_device = _is_device_memory(buffer_ptr)

        already_registered = lmc_ops.get_device_ptr(buffer_ptr) is not None
        if already_registered:
            lmc_ops.unregister_ptr(buffer_ptr)

        self.engine = hixl_comms.Hixl()

        ip = get_ip()
        port = _find_free_port()
        self.engine_id = f"{ip}:{port}"

        options = {"BufferPool": buffer_pool}
        self.engine.initialize(self.engine_id, options)

        mem_type = hixl_comms.MEM_DEVICE if is_device else hixl_comms.MEM_HOST

        logger.debug(
            "Registering HIXL memory with type: %s, "
            "buffer_pool: %s, ptr in hex: %s, and the ptr: %s",
            mem_type,
            buffer_pool,
            hex(buffer_ptr),
            buffer_ptr,
        )

        self.mem_handle = self.engine.register_mem(buffer_ptr, buffer_size, mem_type)

        logger.info(
            "HIXL memory registered mem_handle=0x%x",
            self.mem_handle,
        )

        dev_ptr = hixl_comms.get_dev_va(device_id, buffer_ptr, buffer_size)
        if dev_ptr is not None:
            lmc_ops.register_mapping(buffer_ptr, dev_ptr, buffer_size)
            logger.info(
                "Re-registered lmc_ops mapping via "
                "MemMappingManager (devVA=0x%x)",
                dev_ptr,
            )

        self.buffer_ptr = buffer_ptr
        self.buffer_size = buffer_size
        self.page_size = page_size

        self.local_index_addr = _build_addr_list(buffer_ptr, buffer_size, page_size)

    def close(self):
        if self.mem_handle is not None:
            try:
                self.engine.deregister_mem(self.mem_handle)
            except Exception as e:
                logger.warning("Failed to deregister HIXL memory: %s", e)
        self.engine.finalize()


def _build_addr_list(
    buffer_ptr: int, buffer_size: int, page_size: int
) -> list[int]:
    return list(range(buffer_ptr, buffer_ptr + buffer_size, page_size))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _is_device_memory(ptr: int) -> bool:
    """Check if the pointer refers to device or host memory via ACL runtime."""
    return hixl_comms.is_device_memory(ptr)
