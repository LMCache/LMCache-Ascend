# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, Optional, Union
import asyncio
import pickle
import threading

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
import lmcache_ascend.hccl_npu_comms as hcomm

# Local
from .base_channel import BaseMultiBufferChannel
from .buffer_config import (
    BufferConfig,
    RemotePeerBufferList,
)
from .hccl_agent import HcclAgentWrapper
from .transfer_spec import TS_RECEIVER_ID

logger = init_logger(__name__)


class HcclMsgBase(msgspec.Struct, tag=True):
    """Base class for all hccl-related messages"""

    pass


class HcclInitRequest(HcclMsgBase):
    local_id: str  # Id of client connecting to server
    client_meta_bytes: bytes  # Client meta from client (sender) for server to accept


class HcclInitResponse(HcclMsgBase):
    server_meta_bytes: bytes  # Server meta for client (sender) to connect to


class HcclMemRegRequest(HcclMsgBase):
    local_id: str
    client_mem_handle_bytes: (
        bytes  # Handle of client (sender) memory to register to server
    )


class HcclMemRegResponse(HcclMsgBase):
    server_mem_handle_bytes: (
        bytes  # Handle of server (receiver) memory to register to client
    )


class HcclReadyRequest(HcclMsgBase):
    local_id: str


class HcclReadyResponse(HcclMsgBase):
    ok: bool


class HcclErrorResponse(HcclMsgBase):
    ok: bool = False


HcclMsg = Union[
    HcclInitRequest,
    HcclInitResponse,
    HcclMemRegRequest,
    HcclMemRegResponse,
    HcclReadyRequest,
    HcclReadyResponse,
    HcclErrorResponse,
]


class HcclChannel(BaseMultiBufferChannel):
    _init_msg_type = Union[HcclMsg, SideMsg]
    _channel_name = "hccl"

    def __init__(
        self,
        async_mode: bool = False,
        buffers: Optional[list[BufferConfig]] = None,
        **kwargs,
    ):
        self.conn_handles_dict: Dict[str, object] = {}
        self.remote_index_addr_dict: Dict[str, RemotePeerBufferList] = {}
        self._peer_ready_events: Dict[str, threading.Event] = {}

        super().__init__(async_mode=async_mode, buffers=buffers, **kwargs)

        self.transport_stream = torch.npu.Stream(self.handle_device)

    def _register_buffers(self, buffers: list[BufferConfig]) -> None:
        self.hccl_wrapper = HcclAgentWrapper(buffers=buffers)
        self.hccl_agent = self.hccl_wrapper.agent
        self.mem_handles = self.hccl_wrapper.mem_handles

    def _make_error_response(self) -> HcclErrorResponse:
        return HcclErrorResponse(ok=False)

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        with self._state_lock:
            already_connected = peer_id in self.conn_handles_dict
        if already_connected:
            if init_side_msg is None:
                return None
            init_tmp_socket = get_zmq_socket(
                self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
            )
            try:
                return self.send_init_side_msg(init_tmp_socket, init_side_msg)
            except Exception as e:
                logger.error("Failed to send init side message: %s", e)
                return None
            finally:
                init_tmp_socket.close()

        with self._state_lock:
            self.conn_handles_dict.pop(peer_id, None)
            self.remote_index_addr_dict.pop(peer_id, None)

        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        hccl_init_req = HcclInitRequest(
            local_id=local_id,
            client_meta_bytes=pickle.dumps(self.hccl_agent.get_client_meta()),
        )
        init_tmp_socket.send(msgspec.msgpack.encode(hccl_init_req))

        # Wait remote agent metadata and connect to remote agent
        hccl_init_resp_bytes = init_tmp_socket.recv()
        hccl_init_resp = msgspec.msgpack.decode(hccl_init_resp_bytes, type=HcclMsg)
        server_meta = pickle.loads(hccl_init_resp.server_meta_bytes)

        logger.info("Connecting to remote")

        conn_handle = self.hccl_agent.connect(server_meta)
        with self._state_lock:
            self.conn_handles_dict[peer_id] = conn_handle

        logger.info("Connected to remote")

        # Exchange and register memory with peer
        mem_handles = self.hccl_wrapper.mem_handles
        # Backward compatibility: wrap single handle if needed or send list
        # For now, we assume protocol upgrade to send list of handles
        hccl_mem_reg_req = HcclMemRegRequest(
            local_id=local_id,
            client_mem_handle_bytes=pickle.dumps(mem_handles),
        )
        init_tmp_socket.send(msgspec.msgpack.encode(hccl_mem_reg_req))
        hccl_mem_reg_resp_bytes = init_tmp_socket.recv()
        hccl_mem_reg_resp = msgspec.msgpack.decode(
            hccl_mem_reg_resp_bytes, type=HcclMsg
        )
        server_mem_handles = pickle.loads(hccl_mem_reg_resp.server_mem_handle_bytes)

        # Handle both single item (legacy) and list (new)
        if not isinstance(server_mem_handles, list):
            server_mem_handles = [server_mem_handles]

        for handle in server_mem_handles:
            self.hccl_agent.import_mem(conn_handle, handle.mem_handle)

        with self._state_lock:
            self.remote_index_addr_dict[peer_id] = RemotePeerBufferList(
                server_mem_handles
            )

        ready_req = HcclReadyRequest(local_id=local_id)
        init_tmp_socket.send(msgspec.msgpack.encode(ready_req))
        ready_bytes = init_tmp_socket.recv()
        ready_resp = msgspec.msgpack.decode(ready_bytes, type=HcclMsg)
        if isinstance(ready_resp, HcclReadyResponse) and not ready_resp.ok:
            raise ConnectionError(
                f"Server failed to complete handshake for peer {peer_id}"
            )

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
        with self._state_lock:
            already_connected = peer_id in self.conn_handles_dict
        if already_connected:
            if init_side_msg is None:
                return None
            init_tmp_socket = get_zmq_socket(
                self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
            )
            try:
                return await self.async_send_init_side_msg(
                    init_tmp_socket, init_side_msg
                )
            except Exception as e:
                logger.error("Failed to send init side message: %s", e)
                return None
            finally:
                init_tmp_socket.close()

        with self._state_lock:
            self.conn_handles_dict.pop(peer_id, None)
            self.remote_index_addr_dict.pop(peer_id, None)

        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        hccl_init_req = HcclInitRequest(
            local_id=local_id,
            client_meta_bytes=pickle.dumps(self.hccl_agent.get_client_meta()),
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(hccl_init_req))

        # Wait remote agent metadata and connect to remote agent
        hccl_init_resp_bytes = await init_tmp_socket.recv()
        hccl_init_resp = msgspec.msgpack.decode(hccl_init_resp_bytes, type=HcclMsg)
        server_meta = pickle.loads(hccl_init_resp.server_meta_bytes)

        conn_handle = self.hccl_agent.connect(server_meta)
        with self._state_lock:
            self.conn_handles_dict[peer_id] = conn_handle

        # Exchange and register memory with peer
        mem_handles = self.hccl_wrapper.mem_handles

        hccl_mem_reg_req = HcclMemRegRequest(
            local_id=local_id,
            client_mem_handle_bytes=pickle.dumps(mem_handles),
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(hccl_mem_reg_req))
        hccl_mem_reg_resp_bytes = await init_tmp_socket.recv()
        hccl_mem_reg_resp = msgspec.msgpack.decode(
            hccl_mem_reg_resp_bytes, type=HcclMsg
        )
        server_mem_handles = pickle.loads(hccl_mem_reg_resp.server_mem_handle_bytes)

        if not isinstance(server_mem_handles, list):
            server_mem_handles = [server_mem_handles]

        for handle in server_mem_handles:
            self.hccl_agent.import_mem(conn_handle, handle.mem_handle)

        with self._state_lock:
            self.remote_index_addr_dict[peer_id] = RemotePeerBufferList(
                server_mem_handles
            )

        ready_req = HcclReadyRequest(local_id=local_id)
        await init_tmp_socket.send(msgspec.msgpack.encode(ready_req))
        ready_bytes = await init_tmp_socket.recv()
        ready_resp = msgspec.msgpack.decode(ready_bytes, type=HcclMsg)
        if isinstance(ready_resp, HcclReadyResponse) and not ready_resp.ok:
            raise ConnectionError(
                f"Server failed to complete handshake for peer {peer_id}"
            )

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket,
                init_side_msg,
            )

        init_tmp_socket.close()
        return init_ret_msg

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self.conn_handles_dict

    def _handle_init_msg(
        self, req: Union[HcclMsg, InitSideMsgBase]
    ) -> Union[HcclMsg, InitSideRetMsgBase]:
        resp: Union[HcclMsg, InitSideRetMsgBase]
        if isinstance(req, HcclInitRequest):
            logger.info("Processing HcclInitRequest")
            server_meta = self.hccl_agent.get_server_meta()
            resp = HcclInitResponse(
                server_meta_bytes=pickle.dumps(server_meta),
            )

            client_meta = pickle.loads(req.client_meta_bytes)
            accept_started_event = threading.Event()
            ready_event = threading.Event()
            self._peer_ready_events[req.local_id] = ready_event

            def complete_handshake():
                torch.npu.set_device(self.handle_device)

                logger.info(
                    f"Background: Waiting for connection from {req.local_id}..."
                )
                try:
                    accept_started_event.set()

                    conn_handle = self.hccl_agent.accept(client_meta, server_meta)

                    with self._state_lock:
                        self.conn_handles_dict[req.local_id] = conn_handle
                    logger.info(
                        f"Background: Connection established with {req.local_id}"
                    )
                except Exception as e:
                    logger.error(f"Handshake failed: {e}")
                finally:
                    ready_event.set()

            t = threading.Thread(target=complete_handshake, daemon=True)
            t.start()

            is_ready = accept_started_event.wait(timeout=10.0)

            if not is_ready:
                raise TimeoutError(
                    "Timed out waiting for handshake thread to start accept()"
                )

            logger.info("Replying initialization response")

        elif isinstance(req, HcclMemRegRequest):
            logger.info("Processing HcclMemRegRequest")
            conn_handle = None

            with self._state_lock:
                conn_handle = self.conn_handles_dict[req.local_id]

            client_mem_handles = pickle.loads(req.client_mem_handle_bytes)
            if not isinstance(client_mem_handles, list):
                client_mem_handles = [client_mem_handles]

            for handle in client_mem_handles:
                self.hccl_agent.import_mem(conn_handle, handle.mem_handle)

            with self._state_lock:
                if req.local_id not in self.remote_index_addr_dict:
                    self.remote_index_addr_dict[req.local_id] = RemotePeerBufferList(
                        client_mem_handles
                    )
                else:
                    self.remote_index_addr_dict[req.local_id].extend_handles(
                        client_mem_handles
                    )

            # Send back our handles (all of them)
            resp = HcclMemRegResponse(
                server_mem_handle_bytes=pickle.dumps(self.hccl_wrapper.mem_handles),
            )

            logger.info("Replying mem register response")
        elif isinstance(req, HcclReadyRequest):
            event = self._peer_ready_events.get(req.local_id)
            if event is not None:
                event.wait(timeout=120)
            ok = req.local_id in self.conn_handles_dict
            if not ok:
                logger.error("Ready check timed out for peer %s", req.local_id)
            resp = HcclReadyResponse(ok=ok)

        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _resolve_transfer(self, transfer_spec: dict):
        """Return (conn_handle, remote_buffers) for the peer in transfer_spec."""
        peer_id = transfer_spec[TS_RECEIVER_ID]
        with self._state_lock:
            conn_handle = self.conn_handles_dict[peer_id]
            remote_buffers = self.remote_index_addr_dict[peer_id]
        return conn_handle, remote_buffers

    def _build_write_ops(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict,
    ) -> tuple:
        """Build write operations and resolve connection handle.

        Returns (conn_handle, write_ops) for use by write methods.
        """
        conn_handle, remote_buffers = self._resolve_transfer(transfer_spec)
        remote_addrs = self._resolve_transfer_addrs(remote_buffers, transfer_spec)

        write_ops = []
        for mem_obj, remote_addr in zip(objects, remote_addrs, strict=False):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HCCL channel"
                )

            write_ops.append(
                hcomm.HcclWriteOp(
                    src=self.hccl_wrapper.get_local_addr(
                        mem_obj.data_ptr, mem_obj.meta.address
                    ),
                    dst=remote_addr,
                    s=self.page_size,
                )
            )
        return conn_handle, write_ops

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the hccl channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        assert transfer_spec is not None
        conn_handle, write_ops = self._build_write_ops(objects, transfer_spec)

        self.hccl_agent.write_batch(
            conn_handle, write_ops, self.transport_stream.npu_stream
        )
        self.transport_stream.synchronize()
        return len(objects)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        self.submit_batched_read(buffers, transfer_spec)
        self.transport_stream.synchronize()
        return len(buffers)

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.
            Should contain 'receiver_id' and 'dev_ptrs'.

        :return: Number of successfully transferred objects.
        """
        assert transfer_spec is not None
        conn_handle, write_ops = self._build_write_ops(objects, transfer_spec)

        self.hccl_agent.write_batch(
            conn_handle, write_ops, self.transport_stream.npu_stream
        )

        event = torch.npu.Event()
        event.record(self.transport_stream)
        while not event.query():
            await asyncio.sleep(0.001)

        return len(objects)

    def _build_read_ops(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict,
    ) -> tuple:
        """Build read operations and resolve connection handle.

        Returns (conn_handle, read_ops) for use by read methods.
        """
        conn_handle, remote_buffers = self._resolve_transfer(transfer_spec)
        remote_addrs = self._resolve_transfer_addrs(remote_buffers, transfer_spec)

        read_ops = []
        for mem_obj, remote_addr in zip(buffers, remote_addrs, strict=False):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HCCL channel"
                )

            read_ops.append(
                hcomm.HcclReadOp(
                    src=remote_addr,
                    dst=self.hccl_wrapper.get_local_addr(
                        mem_obj.data_ptr, mem_obj.meta.address
                    ),
                    s=self.page_size,
                )
            )
        return conn_handle, read_ops

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        assert transfer_spec is not None
        conn_handle, read_ops = self._build_read_ops(buffers, transfer_spec)

        self.hccl_agent.read_batch(
            conn_handle, read_ops, self.transport_stream.npu_stream
        )

        event = torch.npu.Event()
        event.record(self.transport_stream)
        while not event.query():
            await asyncio.sleep(0.001)

        return len(buffers)

    def submit_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> torch.npu.Event:
        """Submit a batched read to the transport stream without waiting.

        Unlike async_batched_read, this returns immediately after submitting
        the read operations and recording an event. The caller can use
        the returned event for cross-stream synchronization, e.g.:

            event = channel.submit_batched_read(buffers, spec)
            load_stream.wait_event(event)  # load_stream waits for read

        This enables pipelining: submit reads on transport_stream while
        processing previously fetched data on load_stream.

        :param buffers: A list of MemoryObj to store the read data.
        :param transfer_spec: Must contain 'receiver_id' and 'remote_addrs'.

        :return: An NPU event recorded on transport_stream after submission.
        """
        assert transfer_spec is not None
        conn_handle, read_ops = self._build_read_ops(buffers, transfer_spec)

        self.hccl_agent.read_batch(
            conn_handle, read_ops, self.transport_stream.npu_stream
        )

        event = torch.npu.Event()
        event.record(self.transport_stream)
        return event

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
        with self._state_lock:
            self.conn_handles_dict.clear()
            self.remote_index_addr_dict.clear()
        self._peer_ready_events.clear()
        self.hccl_wrapper.close()
