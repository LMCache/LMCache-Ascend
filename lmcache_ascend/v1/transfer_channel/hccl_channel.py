# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
import asyncio
import threading
import time
import uuid
import pickle

# Third Party
import msgspec
import zmq

import torch
import torch_npu

import lmcache_ascend.hccl_npu_comms as hcomm
import lmcache_ascend.c_ops as lmc_ops

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryObj,
)

# First Party
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)

logger = init_logger(__name__)


class HcclMsgBase(msgspec.Struct, tag=True):
    """Base class for all hccl-related messages"""

    pass


class HcclInitRequest(HcclMsgBase):
    local_id: str # Id of client connecting to server
    client_meta_bytes: bytes # Client meta from client (sender) for server to accept


class HcclInitResponse(HcclMsgBase):
    server_meta_bytes: bytes # Server meta for client (sender) to connect to

class HcclMemRegRequest(HcclMsgBase):
    local_id: str
    client_mem_handle_bytes: bytes # Handle of client (sender) memory to register to server

class HcclMemRegResponse(HcclMsgBase):
    server_mem_handle_bytes: bytes # Handle of server (receiver) memory to register to client

HcclMsg = Union[
    HcclInitRequest, HcclInitResponse, HcclMemRegRequest, HcclMemRegResponse
]


class HcclChannel(BaseTransferChannel):
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

        self.hccl_wrapper = HcclAgentWrapper(
            buffer_ptr=kwargs["buffer_ptr"],
            buffer_size=kwargs["buffer_size"],
            page_size=kwargs["align_bytes"],
        )
        self.page_size = kwargs["align_bytes"]
        self.hccl_agent = self.hccl_wrapper.agent

        # Used for P2P
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self.conn_handles_dict = {}

        self._state_lock = threading.Lock()
        self.remote_index_addr_dict = {}

        self.side_channels: list[zmq.Socket] = []
        self.running_threads: list[threading.Thread] = []

        self.async_mode = async_mode
        if self.async_mode:
            self.zmq_context = get_zmq_context(use_asyncio=True)
        else:
            self.zmq_context = get_zmq_context(use_asyncio=False)
        self.peer_init_url = kwargs["peer_init_url"]
        self.event_loop = kwargs.get("event_loop", None)

        self.transport_stream = torch.npu.Stream(torch.npu.current_device())
        self.handle_device = torch.npu.current_device()

        self._init_side_channels()

    ############################################################
    # Initialization functions
    ############################################################
    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        # Initialize temporary socket for hccl initialization
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        # Build and send init request
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
        mem_handle = self.hccl_wrapper.mem_handles[0] # TODO: make this work for multiple mem_handles, for now there's only one
        hccl_mem_reg_req = HcclMemRegRequest(
            local_id=local_id,
            client_mem_handle_bytes=pickle.dumps(mem_handle),
        )
        init_tmp_socket.send(msgspec.msgpack.encode(hccl_mem_reg_req))
        hccl_mem_reg_resp_bytes = init_tmp_socket.recv()
        hccl_mem_reg_resp = msgspec.msgpack.decode(
            hccl_mem_reg_resp_bytes, type=HcclMsg
        )
        server_mem_handle = pickle.loads(hccl_mem_reg_resp.server_mem_handle_bytes)
        self.hccl_agent.import_mem(conn_handle, server_mem_handle.mem_handle)

        addr_list = []
        with self._state_lock:
            self.remote_index_addr_dict[peer_id] = addr_list

        for base_addr in range(server_mem_handle.buffer_ptr, server_mem_handle.buffer_ptr + server_mem_handle.buffer_size, server_mem_handle.page_size):
            addr_list.append(base_addr)

        # Send side message if any
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
        # Initialize temporary socket for hccl initialization
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        # Build and send init request
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
        mem_handle = self.hccl_wrapper.mem_handles[0] # TODO: make this work for multiple mem_handles, for now there's only one
        hccl_mem_reg_req = HcclMemRegRequest(
            local_id=local_id,
            client_mem_handle_bytes=pickle.dumps(mem_handle),
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(hccl_mem_reg_req))
        hccl_mem_reg_resp_bytes = await init_tmp_socket.recv()
        hccl_mem_reg_resp = msgspec.msgpack.decode(
            hccl_mem_reg_resp_bytes, type=HcclMsg
        )
        server_mem_handle = pickle.loads(hccl_mem_reg_resp.server_mem_handle_bytes)
        self.hccl_agent.import_mem(conn_handle, server_mem_handle.mem_handle)

        addr_list = []
        with self._state_lock:
            self.remote_index_addr_dict[peer_id] = addr_list

        for base_addr in range(server_mem_handle.buffer_ptr, server_mem_handle.buffer_ptr + server_mem_handle.buffer_size, server_mem_handle.page_size):
            addr_list.append(base_addr)

        # Send side message if any
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
            return

        if self.async_mode:
            # Start listening coroutine for initialization side channel
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            # Start listening thread for initialization side channel
            self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
            self.init_thread.start()
            self.running_threads.append(self.init_thread)

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

            # In nixl, add_remote_agent is executed before returning the response.
            # Note: just put in separate thread for now, make sure also works for asyncio case later
            def complete_handshake():

                torch.npu.set_device(self.handle_device)
                
                logger.info(f"Background: Waiting for connection from {req.local_id}...")
                try:
                    accept_started_event.set()

                    # This blocks, but now it's in a thread so it doesn't stop the return
                    conn_handle = self.hccl_agent.accept(client_meta, server_meta)
                    
                    # Update the state once the connection is established
                    with self._state_lock:
                        self.conn_handles_dict[req.local_id] = conn_handle
                    logger.info(f"Background: Connection established with {req.local_id}")
                except Exception as e:
                    logger.error(f"Handshake failed: {e}")

            t = threading.Thread(target=complete_handshake, daemon=True)
            t.start()

            is_ready = accept_started_event.wait(timeout=10.0)

            if not is_ready:
                logger.error("Timed out waiting for handshake thread to start accept()")

            logger.info("Replying initialization response")

        elif isinstance(req, HcclMemRegRequest):
            logger.info("Processing HcclMemRegRequest")
            conn_handle = None
            addr_list = []
            with self._state_lock:
                conn_handle = self.conn_handles_dict[req.local_id] # TODO: check if exists
                self.remote_index_addr_dict[req.local_id] = addr_list

            client_mem_handle = pickle.loads(req.client_mem_handle_bytes)
            self.hccl_agent.import_mem(conn_handle, client_mem_handle.mem_handle)

            for base_addr in range(client_mem_handle.buffer_ptr, client_mem_handle.buffer_ptr + client_mem_handle.buffer_size, client_mem_handle.page_size):
                addr_list.append(base_addr)

            resp = HcclMemRegResponse(
                server_mem_handle_bytes=pickle.dumps(self.hccl_wrapper.mem_handles[0]), # TODO: make this work for multiple mem_handles, for now there's only one
            )

            logger.info("Replying mem register response")
        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)
            logger.info("Replying P2P init side response")
        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _init_loop(self):
        # Initialize initialization side channels
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

                req = msgspec.msgpack.decode(req_bytes, type=Union[HcclMsg, SideMsg])

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
        # Initialize initialization side channels
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
                req_bytes = await asyncio.wait_for(self.init_side_channel.recv(), timeout=1.0)

                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=Union[HcclMsg, SideMsg])

                resp = await loop.run_in_executor(None, self._handle_init_msg, req)

                logger.info("handled init msg")

                await self.init_side_channel.send(msgspec.msgpack.encode(resp))

                logger.info("Sent initialization request response")

            except asyncio.TimeoutError:
                continue

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

        self.init_side_channel.close()

    ############################################################
    # Utility functions
    ############################################################

    def get_local_mem_indices( # TODO: unused be necessary for abstract class
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        local_indices = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                local_indices.append(mem_obj.meta.address)
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in hccl channel"
            )
        return local_indices

    ############################################################
    # Send/Recv functions
    ############################################################

    ### Send and Recv must be called in pair ###
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

    ############################################################
    # Read/Write functions
    ############################################################

    ### Read and Write only need to be called on one side ###
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

        conn_handle = None
        remote_index_addr = []
        with self._state_lock:
            conn_handle = self.conn_handles_dict[transfer_spec["receiver_id"]]
            remote_index_addr = self.remote_index_addr_dict[transfer_spec["receiver_id"]]

        write_ops = []
        for mem_obj, remote_index in zip(objects, transfer_spec["remote_indexes"]):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HCCL channel"
                )

            write_ops.append(hcomm.HcclWriteOp(
                src=self.hccl_wrapper.local_index_addr[mem_obj.meta.address], dst=remote_index_addr[remote_index], s=self.page_size
            ))

        self.hccl_agent.write_batch(conn_handle, write_ops, self.transport_stream.npu_stream)
        self.transport_stream.synchronize()
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
        Write a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.
            Should contain 'receiver_id' and 'dev_ptrs'.

        :return: Number of successfully transferred objects.
        """

        assert transfer_spec is not None

        conn_handle = None
        remote_index_addr = []
        with self._state_lock:
            conn_handle = self.conn_handles_dict[transfer_spec["receiver_id"]]
            remote_index_addr = self.remote_index_addr_dict[transfer_spec["receiver_id"]]

        write_ops = []
        for mem_obj, remote_index in zip(objects, transfer_spec["remote_indexes"]):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HCCL channel"
                )

            write_ops.append(hcomm.HcclWriteOp(
                src=self.hccl_wrapper.local_index_addr[mem_obj.meta.address], dst=remote_index_addr[remote_index], s=self.page_size
            ))
 
        self.hccl_agent.write_batch(conn_handle, write_ops, self.transport_stream.npu_stream)
                
        event = torch.npu.Event()
        event.record(self.transport_stream)
        while not event.query():
            await asyncio.sleep(0.001)

        return len(objects)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """

        assert transfer_spec is not None

        conn_handle = None
        remote_index_addr = []
        with self._state_lock:
            conn_handle = self.conn_handles_dict[transfer_spec["receiver_id"]]
            remote_index_addr = self.remote_index_addr_dict[transfer_spec["receiver_id"]]

        write_ops = []
        for mem_obj, remote_index in zip(buffers, transfer_spec["remote_indexes"]):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HCCL channel"
                )

            write_ops.append(hcomm.HcclReadOp(
                src=remote_index_addr[remote_index], dst=self.hccl_wrapper.local_index_addr[mem_obj.meta.address], s=self.page_size
            ))

        self.hccl_agent.read_batch(conn_handle, write_ops, self.transport_stream.npu_stream)
        
        event = torch.npu.Event()
        event.record(self.transport_stream)
        while not event.query():
            await asyncio.sleep(0.001)

        return len(buffers)

    ############################################################
    # Cleanup-related functions
    ############################################################

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
        self.hccl_wrapper.close()

@dataclass
class HcclMemHandleMeta():
    mem_handle: hcomm.RmaMemDesc
    buffer_ptr: int
    buffer_size: int
    page_size: int

@dataclass
class HcclAgentWrapper:
    agent: "HcclAgent"
    mem_handles: Any
    conn_handle: Any
    buffer_ptrs: Any
    # xfer_handler is removed

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
    ):
        """
        Initialize the hccl agent.

        Args:
            buffer_size (int): The size of the buffer.
            buffer_ptr (int): The pointer to the buffer.
        """

        device_id = torch.npu.current_device()
        hccl_agent = hcomm.HcclAgent.get_instance(device_id)
        hccl_agent.init()

        device_ptr = lmc_ops.get_device_ptr(buffer_ptr) or buffer_ptr

        # Register the memory
        mem_handle = hccl_agent.register_mem(device_ptr, buffer_size) 
        mem_handle_meta = HcclMemHandleMeta(
            mem_handle=mem_handle,
            buffer_ptr=device_ptr,
            buffer_size=buffer_size,
            page_size=page_size,
        )

        self.local_index_addr = []
        for base_addr in range(device_ptr, device_ptr + buffer_size, page_size):
            self.local_index_addr.append(base_addr)

        self.agent = hccl_agent
        self.mem_handles = [mem_handle_meta]
        self.buffer_ptrs = [device_ptr]

    def close(self):
        for buffer_ptr in self.buffer_ptrs:
            self.agent.deregister_mem(buffer_ptr)
