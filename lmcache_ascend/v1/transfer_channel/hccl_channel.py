# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, List, Optional, Union
import asyncio
import pickle
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
import lmcache_ascend.hccl_npu_comms as hcomm

# Local
from .hccl_agent import BufferConfig, BufferType, HcclAgentWrapper, HcclMemHandleMeta

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


class PeerMemHandle:
    def __init__(self, mem_handle: HcclMemHandleMeta):
        self.mem_handle = mem_handle
        self.uuid = mem_handle.uuid
        self.buffer_ptr = mem_handle.buffer_ptr
        self.buffer_size = mem_handle.buffer_size
        self.page_size = mem_handle.page_size
        self.num_pages = self.buffer_size // self.page_size


class PeerMemHandleList:
    def __init__(self, mem_handles: List[HcclMemHandleMeta]):
        self.peer_mem_handles = [PeerMemHandle(handle) for handle in mem_handles]
        self._uuid_to_handle = {h.uuid: h for h in self.peer_mem_handles}

    def extend_handles(self, mem_handles: List[HcclMemHandleMeta]):
        for handle in mem_handles:
            peer_handle = PeerMemHandle(handle)
            self.peer_mem_handles.append(peer_handle)
            self._uuid_to_handle[peer_handle.uuid] = peer_handle

    def get_handle_by_uuid(self, buffer_uuid: str) -> PeerMemHandle:
        """Look up a peer mem handle by its UUID.

        Raises ValueError if UUID is not found.
        """
        handle = self._uuid_to_handle.get(buffer_uuid)
        if handle is None:
            raise ValueError(f"Buffer UUID {buffer_uuid} not found in peer mem handles")
        return handle

    def resolve_addr(self, buffer_uuid: str, page_index: int) -> int:
        """Resolve a (buffer_uuid, page_index) to an actual remote memory address.

        Performs bounds checking: page_index must be in [0, num_pages).
        Raises ValueError for unknown UUID, IndexError for out-of-range page.
        """
        handle = self.get_handle_by_uuid(buffer_uuid)
        if not (0 <= page_index < handle.num_pages):
            raise IndexError(
                f"page_index {page_index} out of range [0, {handle.num_pages}) "
                f"for remote buffer {buffer_uuid}"
            )
        return handle.buffer_ptr + page_index * handle.page_size


HcclMsg = Union[
    HcclInitRequest, HcclInitResponse, HcclMemRegRequest, HcclMemRegResponse
]


class HcclChannel(BaseTransferChannel):
    def __init__(
        self,
        async_mode: bool = False,
        buffers: Optional[list[BufferConfig]] = None,
        **kwargs,
    ):
        assert "role" in kwargs

        self.role = kwargs["role"]

        if buffers is None:
            logger.warning(
                "Buffers not provided, "
                "using legacy initialization with buffer_ptr, "
                "buffer_size, and align_bytes"
            )
            # Legacy initialization support
            assert "buffer_ptr" in kwargs
            assert "buffer_size" in kwargs
            assert "align_bytes" in kwargs

            buffers = [
                BufferConfig(
                    ptr=kwargs["buffer_ptr"],
                    size=kwargs["buffer_size"],
                    device_id=-1,  # Deprecated, not used in CPU implementation
                    device_type=BufferType.CPU,
                    align_bytes=kwargs["align_bytes"],
                )
            ]

        # Take the page size from the first buffer (assuming uniform for now)
        self.page_size = buffers[0].align_bytes

        self.hccl_wrapper = HcclAgentWrapper(
            buffers=buffers,
        )
        self.hccl_agent = self.hccl_wrapper.agent

        # Used for P2P
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.running = True
        self.conn_handles_dict = {}

        self._state_lock = threading.Lock()
        self.remote_index_addr_dict: Dict[str, PeerMemHandleList] = {}

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
            self.remote_index_addr_dict[peer_id] = PeerMemHandleList(server_mem_handles)

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
            self.remote_index_addr_dict[peer_id] = PeerMemHandleList(server_mem_handles)

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

            # In nixl, add_remote_agent is executed before returning the response.
            # Note: just put in separate thread for now, make sure also works for
            #       asyncio case later
            def complete_handshake():
                torch.npu.set_device(self.handle_device)

                logger.info(
                    f"Background: Waiting for connection from {req.local_id}..."
                )
                try:
                    accept_started_event.set()

                    conn_handle = self.hccl_agent.accept(client_meta, server_meta)

                    # Update the state once the connection is established
                    with self._state_lock:
                        self.conn_handles_dict[req.local_id] = conn_handle
                    logger.info(
                        f"Background: Connection established with {req.local_id}"
                    )
                except Exception as e:
                    logger.error(f"Handshake failed: {e}")

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

            # FIXME (gingfung): Seems to have race condition
            # if I move to the next lock position...
            with self._state_lock:
                conn_handle = self.conn_handles_dict[req.local_id]

            client_mem_handles = pickle.loads(req.client_mem_handle_bytes)
            if not isinstance(client_mem_handles, list):
                client_mem_handles = [client_mem_handles]

            for handle in client_mem_handles:
                self.hccl_agent.import_mem(conn_handle, handle.mem_handle)

            with self._state_lock:
                if req.local_id not in self.remote_index_addr_dict:
                    self.remote_index_addr_dict[req.local_id] = PeerMemHandleList(
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
            except zmq.Again:
                continue
            except Exception as e:
                logger.error("Failed to receive in initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)
                continue

            try:
                logger.info("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=Union[HcclMsg, SideMsg])

                resp = self._handle_init_msg(req)

                self.init_side_channel.send(msgspec.msgpack.encode(resp))

                logger.info("Sent initialization request response")

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                # Must send *something* to keep the REP socket in sync,
                # otherwise it enters the "must send" state permanently
                # and every subsequent recv() fails with
                # "Operation cannot be accomplished in current state".
                try:
                    self.init_side_channel.send(b"")
                except Exception:
                    pass
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
                req_bytes = await asyncio.wait_for(
                    self.init_side_channel.recv(), timeout=1.0
                )

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

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        raise NotImplementedError("When using Ascend, this should not be used.")

    def get_local_buffer_refs(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> tuple[list[str], list[int]]:
        """Return (buffer_uuids, mem_indexes) for each object.

        These references can be sent over the wire instead of mem indexes.
        The remote peer resolves them.

        Returns:
            A tuple of (buffer_uuids, mem_indexes) parallel lists.
        """
        buffer_uuids: list[str] = []
        mem_indexes: list[int] = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert mem_obj is not None and isinstance(mem_obj, MemoryObj), (
                    "Expected MemoryObj, got {}".format(type(mem_obj))
                )
                buf_uuid, mem_idx = self.hccl_wrapper.get_buffer_ref(
                    mem_obj.data_ptr, mem_obj.meta.address
                )
                buffer_uuids.append(buf_uuid)
                mem_indexes.append(mem_idx)
        elif isinstance(objects[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in hccl channel"
            )
        return buffer_uuids, mem_indexes

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

    def _resolve_remote_addrs(self, transfer_spec: dict) -> list[int]:
        """Resolve remote memory addresses from transfer_spec.

        Supports the format (remote_buffer_uuids + remote_mem_indexes)

        The format avoids passing raw pointers or single indexes over the wire by using
        buffer UUIDs and bounded mem indexes, which are resolved
        locally against the peer's registered memory handles.

        Returns:
            List of resolved remote memory addresses for RDMA operations.
        """
        if (
            "remote_buffer_uuids" in transfer_spec
            and "remote_mem_indexes" in transfer_spec
        ):
            peer_id = transfer_spec["receiver_id"]
            with self._state_lock:
                remote_mem_handles = self.remote_index_addr_dict[peer_id]
            return [
                remote_mem_handles.resolve_addr(buf_uuid, page_idx)
                for buf_uuid, page_idx in zip(
                    transfer_spec["remote_buffer_uuids"],
                    transfer_spec["remote_mem_indexes"],
                    strict=True,
                )
            ]
        else:
            raise ValueError(
                "transfer_spec must contain either "
                "(remote_buffer_uuids, remote_mem_indexes) or remote_addrs"
            )

    def _build_write_ops(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict,
    ) -> tuple:
        """Build write operations and resolve connection handle.

        Returns (conn_handle, write_ops) for use by write methods.
        """
        with self._state_lock:
            conn_handle = self.conn_handles_dict[transfer_spec["receiver_id"]]

        remote_addrs = self._resolve_remote_addrs(transfer_spec)

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
        conn_handle = None
        with self._state_lock:
            conn_handle = self.conn_handles_dict[transfer_spec["receiver_id"]]

        remote_addrs = self._resolve_remote_addrs(transfer_spec)

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

    ############################################################
    # functions added for better pipelining control
    # These allow submitting reads/writes without waiting for completion,
    # and using events for synchronization
    ############################################################
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

    ############################################################
    # Cleanup-related functions
    ############################################################

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
        self.hccl_wrapper.close()
