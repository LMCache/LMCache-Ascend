# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
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
    HixlInitRequest, HixlInitResponse, HixlMemInfoRequest, HixlMemInfoResponse
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
        hixl_init_req = HixlInitRequest(
            local_id=local_id,
            engine_id=self.hixl_wrapper.engine_id,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(hixl_init_req))

        hixl_init_resp_bytes = init_tmp_socket.recv()
        hixl_init_resp = msgspec.msgpack.decode(hixl_init_resp_bytes, type=HixlMsg)
        remote_engine_id = hixl_init_resp.engine_id

        logger.info("Connecting to remote HIXL engine: %s", remote_engine_id)
        self.hixl_wrapper.engine.connect(remote_engine_id)

        with self._state_lock:
            self.remote_engine_dict[peer_id] = remote_engine_id

        logger.info("Connected to remote HIXL engine: %s", remote_engine_id)

        # Step 2: exchange buffer layout info
        hixl_mem_req = HixlMemInfoRequest(
            local_id=local_id,
            buffer_ptr=self.hixl_wrapper.buffer_ptr,
            buffer_size=self.hixl_wrapper.buffer_size,
            page_size=self.hixl_wrapper.page_size,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(hixl_mem_req))
        hixl_mem_resp_bytes = init_tmp_socket.recv()
        hixl_mem_resp = msgspec.msgpack.decode(hixl_mem_resp_bytes, type=HixlMsg)

        addr_list = []
        with self._state_lock:
            self.remote_index_addr_dict[peer_id] = addr_list
        for base_addr in range(
            hixl_mem_resp.buffer_ptr,
            hixl_mem_resp.buffer_ptr + hixl_mem_resp.buffer_size,
            hixl_mem_resp.page_size,
        ):
            addr_list.append(base_addr)

        # Step 3: optional side message
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
        hixl_init_req = HixlInitRequest(
            local_id=local_id,
            engine_id=self.hixl_wrapper.engine_id,
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(hixl_init_req))

        hixl_init_resp_bytes = await init_tmp_socket.recv()
        hixl_init_resp = msgspec.msgpack.decode(hixl_init_resp_bytes, type=HixlMsg)
        remote_engine_id = hixl_init_resp.engine_id

        self.hixl_wrapper.engine.connect(remote_engine_id)
        with self._state_lock:
            self.remote_engine_dict[peer_id] = remote_engine_id

        # Step 2: exchange buffer layout info
        hixl_mem_req = HixlMemInfoRequest(
            local_id=local_id,
            buffer_ptr=self.hixl_wrapper.buffer_ptr,
            buffer_size=self.hixl_wrapper.buffer_size,
            page_size=self.hixl_wrapper.page_size,
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(hixl_mem_req))
        hixl_mem_resp_bytes = await init_tmp_socket.recv()
        hixl_mem_resp = msgspec.msgpack.decode(hixl_mem_resp_bytes, type=HixlMsg)

        addr_list = []
        with self._state_lock:
            self.remote_index_addr_dict[peer_id] = addr_list
        for base_addr in range(
            hixl_mem_resp.buffer_ptr,
            hixl_mem_resp.buffer_ptr + hixl_mem_resp.buffer_size,
            hixl_mem_resp.page_size,
        ):
            addr_list.append(base_addr)

        # Step 3: optional side message
        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
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

        elif isinstance(req, HixlMemInfoRequest):
            logger.info("Processing HixlMemInfoRequest from %s", req.local_id)

            addr_list = []
            with self._state_lock:
                self.remote_index_addr_dict[req.local_id] = addr_list

            for base_addr in range(
                req.buffer_ptr,
                req.buffer_ptr + req.buffer_size,
                req.page_size,
            ):
                addr_list.append(base_addr)

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

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the HIXL channel (synchronous).
        """
        assert transfer_spec is not None

        remote_engine = ""
        remote_index_addr: list[int] = []
        with self._state_lock:
            remote_engine = self.remote_engine_dict[transfer_spec["receiver_id"]]
            remote_index_addr = self.remote_index_addr_dict[
                transfer_spec["receiver_id"]
            ]

        op_descs = []
        for mem_obj, remote_index in zip(
            objects, transfer_spec["remote_indexes"], strict=False
        ):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HIXL channel"
                )

            # TODO: Potentially use the actual size of the object
            op_descs.append(
                hixl_comms.TransferOpDesc(
                    local_addr=self.hixl_wrapper.local_index_addr[mem_obj.meta.address],
                    remote_addr=remote_index_addr[remote_index],
                    len=self.page_size,
                )
            )

        self.hixl_wrapper.engine.transfer_sync(
            remote_engine, hixl_comms.WRITE, op_descs
        )
        return len(objects)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the HIXL channel (asynchronous).
        Uses TransferAsync + polling GetTransferStatus.
        """
        assert transfer_spec is not None

        remote_engine = ""
        remote_index_addr: list[int] = []
        with self._state_lock:
            remote_engine = self.remote_engine_dict[transfer_spec["receiver_id"]]
            remote_index_addr = self.remote_index_addr_dict[
                transfer_spec["receiver_id"]
            ]

        op_descs = []
        for mem_obj, remote_index in zip(
            objects, transfer_spec["remote_indexes"], strict=False
        ):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HIXL channel"
                )

            # TODO: Potentially use the actual size of the object
            op_descs.append(
                hixl_comms.TransferOpDesc(
                    local_addr=self.hixl_wrapper.local_index_addr[mem_obj.meta.address],
                    remote_addr=remote_index_addr[remote_index],
                    len=self.page_size,
                )
            )

        req = self.hixl_wrapper.engine.transfer_async(
            remote_engine, hixl_comms.WRITE, op_descs
        )

        while True:
            status = self.hixl_wrapper.engine.get_transfer_status(req)
            if status == hixl_comms.TransferStatus.COMPLETED:
                break
            if status == hixl_comms.TransferStatus.FAILED:
                raise RuntimeError("HIXL async write transfer failed")
            if status == hixl_comms.TransferStatus.TIMEOUT:
                raise TimeoutError("HIXL async write transfer timed out")
            await asyncio.sleep(0.001)

        return len(objects)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the HIXL channel (asynchronous).
        Uses TransferAsync + polling GetTransferStatus.
        """
        assert transfer_spec is not None

        remote_engine = ""
        remote_index_addr: list[int] = []
        with self._state_lock:
            remote_engine = self.remote_engine_dict[transfer_spec["receiver_id"]]
            remote_index_addr = self.remote_index_addr_dict[
                transfer_spec["receiver_id"]
            ]

        op_descs = []
        for mem_obj, remote_index in zip(
            buffers, transfer_spec["remote_indexes"], strict=False
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

        req = self.hixl_wrapper.engine.transfer_async(
            remote_engine, hixl_comms.READ, op_descs
        )

        while True:
            status = self.hixl_wrapper.engine.get_transfer_status(req)
            if status == hixl_comms.TransferStatus.COMPLETED:
                break
            if status == hixl_comms.TransferStatus.FAILED:
                raise RuntimeError("HIXL async read transfer failed")
            if status == hixl_comms.TransferStatus.TIMEOUT:
                raise TimeoutError("HIXL async read transfer timed out")
            await asyncio.sleep(0.001)

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
    ):
        device_id = torch.npu.current_device()

        is_device = _is_device_memory(buffer_ptr)

        self.engine = hixl_comms.Hixl()

        ip = _get_device_ip(device_id)
        port = _find_free_port()
        self.engine_id = f"{ip}:{port}"

        # NOTE (gingfung): this option is for the buffer pool size
        # and the number of buffers the default is 4:8,
        # which means 4 buffers of 8MB each
        # we currently hardcoded to 4:8 because LMCache supports H2D, H2H
        # and these patterns require a buffer pool for now
        # see Hixl Tests for more details.
        options = {"BufferPool": "4:8"}
        self.engine.initialize(self.engine_id, options)

        if is_device:
            already_registered = lmc_ops.get_device_ptr(buffer_ptr) is not None
            if already_registered:
                lmc_ops.unregister_ptr(buffer_ptr)

            self.mem_handle = self.engine.register_mem(
                buffer_ptr, buffer_size, hixl_comms.MEM_DEVICE
            )

            if already_registered:
                dev_ptr = hixl_comms.get_dev_va(device_id, buffer_ptr, buffer_size)
                if dev_ptr is not None:
                    lmc_ops.register_mapping(buffer_ptr, dev_ptr, buffer_size)
                    logger.info(
                        "Re-registered lmc_ops mapping via "
                        "MemMappingManager (devVA=0x%x) dev ptr: %s",
                        dev_ptr,
                        dev_ptr,
                    )
        else:
            self.mem_handle = None
            logger.info(
                "Host memory: skipping RegisterMem, relying on "
                "BufferPool for staging (H2H pattern)"
            )

        self.buffer_ptr = buffer_ptr
        self.buffer_size = buffer_size
        self.page_size = page_size

        self.local_index_addr = []
        for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
            self.local_index_addr.append(base_addr)

    def close(self):
        if self.mem_handle is not None:
            try:
                self.engine.deregister_mem(self.mem_handle)
            except Exception as e:
                logger.warning("Failed to deregister HIXL memory: %s", e)
        self.engine.finalize()


def _get_device_ip(device_id: int) -> str:
    """Get the RDMA-reachable IP for the given NPU device."""
    try:
        # Standard
        import subprocess

        result = subprocess.run(
            ["hccn_tool", "-i", str(device_id), "-ip", "-g"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().splitlines():
            if "ipaddr" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return "127.0.0.1"


def _find_free_port() -> int:
    """Find an available TCP port."""
    # Standard
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _is_device_memory(ptr: int) -> bool:
    """Check if the pointer refers to device or host memory via ACL runtime."""
    return hixl_comms.is_device_memory(ptr)
