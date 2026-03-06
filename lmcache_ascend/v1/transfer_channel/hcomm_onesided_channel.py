# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import threading
import time
from typing import Dict, List, Optional, Union

# First Party
import lmcache_ascend.c_ops as lmc_ops
import lmcache_ascend.hcomm_onesided as hcomm_os
import msgspec
import torch
import zmq
# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_zmq_socket
from lmcache.v1.transfer_channel.transfer_utils import (InitSideMsgBase,
                                                        InitSideRetMsgBase,
                                                        SideMsg)

# Local
from .base_channel import BaseMultiBufferChannel
from .buffer_config import (BufferConfig, BufferType, MemHandleMeta,
                            PeerBufferInfo, RemotePeerBufferList,
                            resolve_buffer_ref, resolve_local_addr)
from .hcomm_onesided_protocol import (HcommOsInitRequest, HcommOsInitResponse,
                                      HcommOsMsg, HcommOsReadyRequest,
                                      HcommOsReadyResponse, _PeerState)
from .hcomm_onesided_runtime import (_build_rank_table_json,
                                     _get_local_device_info,
                                     _init_comm_and_prepare)
from .transfer_spec import TS_STREAM, resolve_peer_id

logger = init_logger(__name__)


class HcommOneSidedChannel(BaseMultiBufferChannel):
    """Transfer channel using hcomm one-sided service API with stream support.

    Each peer pair gets its own HcclComm (nRanks=2).  The server side is
    always rank 0 and the client is rank 1.
    """

    SERVER_RANK = 0
    CLIENT_RANK = 1
    _init_msg_type = Union[HcommOsMsg, SideMsg]
    _channel_name = "hcomm_onesided"

    def __init__(
        self,
        async_mode: bool = False,
        buffers: Optional[List[BufferConfig]] = None,
        **kwargs,
    ):
        self.device_info = _get_local_device_info()
        self.mem_handles: List[MemHandleMeta] = []
        self._uuid_to_handle: Dict[str, MemHandleMeta] = {}
        # peer_id -> PeerState.
        self._peers: dict[str, _PeerState] = {}
        super().__init__(async_mode=async_mode, buffers=buffers, **kwargs)
        self.transport_stream = torch.npu.Stream()

    def _register_buffers(self, buffers: list[BufferConfig]) -> None:
        self.mem_handles = []
        self._uuid_to_handle = {}
        for buf in buffers:
            meta = self._register_buffer(buf)
            self.mem_handles.append(meta)
            self._uuid_to_handle[meta.uuid] = meta

    def _register_buffer(self, buf: BufferConfig) -> MemHandleMeta:
        """Register a single buffer with the hcomm one-sided library."""
        buffer_ptr = buf.ptr
        buffer_size = buf.size
        page_size = buf.align_bytes
        is_device = buf.device_type == BufferType.NPU

        already_registered = lmc_ops.get_device_ptr(buffer_ptr) is not None
        if already_registered:
            lmc_ops.unregister_ptr(buffer_ptr)

        mem_handle = hcomm_os.register_global_mem(buffer_ptr, buffer_size, is_device)
        device_id = torch.npu.current_device()
        dev_ptr = hcomm_os.get_dev_va(device_id, buffer_ptr, buffer_size)
        if dev_ptr is not None:
            lmc_ops.register_mapping(buffer_ptr, dev_ptr, buffer_size)
            logger.info(
                "Re-registered lmc_ops mapping via MemMappingManager (devVA=0x%x)",
                dev_ptr,
            )

        buffer_addrs = list(range(buffer_ptr, buffer_ptr + buffer_size, page_size))

        logger.info(
            "Registered global mem: ptr=0x%x size=%d is_device=%s",
            buffer_ptr,
            buffer_size,
            is_device,
        )

        return MemHandleMeta(
            mem_handle=mem_handle,
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            page_size=page_size,
            local_buffer_addrs=buffer_addrs,
            buffer_type=buf.device_type,
        )

    def _make_buffer_infos(self) -> List[PeerBufferInfo]:
        """Build handshake buffer-info list from our registered handles."""
        return [
            PeerBufferInfo(
                uuid=meta.uuid,
                buffer_ptr=meta.buffer_ptr,
                buffer_size=meta.buffer_size,
                page_size=meta.page_size,
                is_device=(meta.buffer_type == BufferType.NPU),
            )
            for meta in self.mem_handles
        ]

    def _get_buffer_ref(self, data_ptr: int, page_index: int) -> tuple:
        """Find the buffer UUID for a given data pointer."""
        return resolve_buffer_ref(self.mem_handles, data_ptr, page_index)

    def _get_local_addr(self, ptr: int, idx: int) -> int:
        """Resolve a pointer + page index to a local buffer address."""
        return resolve_local_addr(self.mem_handles, ptr, idx)

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        init_tmp_socket = get_zmq_socket(
            self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
        )

        req = HcommOsInitRequest(
            local_id=local_id,
            buffers=self._make_buffer_infos(),
            device_info=self.device_info,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(req))
        resp_bytes = init_tmp_socket.recv()
        resp = msgspec.msgpack.decode(resp_bytes, type=HcommOsMsg)
        if not isinstance(resp, HcommOsInitResponse):
            raise ValueError(f"Expected HcommOsInitResponse, got {type(resp).__name__}")

        my_rank = resp.client_rank
        remote_rank = resp.server_rank

        logger.info(
            "Client: init comm cluster_info rank=%d comm_name=%s",
            my_rank,
            resp.comm_name,
        )
        torch.npu.set_device(self.handle_device)
        old_peer = self._pop_stale_peer(peer_id)
        if old_peer is not None:
            logger.info(
                "Client: destroying stale comm for peer %s before reconnect",
                peer_id,
            )
            self._destroy_peer_comm(old_peer, peer_id)

        all_mem_handles = [m.mem_handle for m in self.mem_handles]
        comm = _init_comm_and_prepare(
            resp.cluster_json, resp.comm_name, my_rank, all_mem_handles
        )

        remote_buffers = RemotePeerBufferList(resp.buffers)

        peer_state = _PeerState(
            comm=comm,
            my_rank=my_rank,
            remote_rank=remote_rank,
            remote_buffers=remote_buffers,
        )
        with self._state_lock:
            self._peers[peer_id] = peer_state

        logger.info("Client: peer %s connected", peer_id)

        ready_req = HcommOsReadyRequest(local_id=local_id)
        init_tmp_socket.send(msgspec.msgpack.encode(ready_req))
        ready_bytes = init_tmp_socket.recv()
        ready_resp = msgspec.msgpack.decode(ready_bytes, type=HcommOsMsg)
        if isinstance(ready_resp, HcommOsReadyResponse) and not ready_resp.ok:
            raise ConnectionError(
                f"Server failed to complete handshake for peer {peer_id}"
            )

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = self.send_init_side_msg(init_tmp_socket, init_side_msg)

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
            self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
        )

        req = HcommOsInitRequest(
            local_id=local_id,
            buffers=self._make_buffer_infos(),
            device_info=self.device_info,
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(req))
        resp_bytes = await init_tmp_socket.recv()
        resp = msgspec.msgpack.decode(resp_bytes, type=HcommOsMsg)
        if not isinstance(resp, HcommOsInitResponse):
            raise ValueError(f"Expected HcommOsInitResponse, got {type(resp).__name__}")

        my_rank = resp.client_rank
        remote_rank = resp.server_rank

        old_peer = self._pop_stale_peer(peer_id)

        loop = asyncio.get_running_loop()
        device = self.handle_device
        all_mem_handles = [m.mem_handle for m in self.mem_handles]

        def _client_init_comm():
            torch.npu.set_device(device)
            if old_peer is not None:
                logger.info(
                    "Client: destroying stale comm for peer %s before reconnect",
                    peer_id,
                )
                self._destroy_peer_comm(old_peer, peer_id)
            return _init_comm_and_prepare(
                resp.cluster_json, resp.comm_name, my_rank, all_mem_handles
            )

        comm = await loop.run_in_executor(None, _client_init_comm)

        remote_buffers = RemotePeerBufferList(resp.buffers)
        peer_state = _PeerState(
            comm=comm,
            my_rank=my_rank,
            remote_rank=remote_rank,
            remote_buffers=remote_buffers,
        )
        with self._state_lock:
            self._peers[peer_id] = peer_state

        ready_req = HcommOsReadyRequest(local_id=local_id)
        await init_tmp_socket.send(msgspec.msgpack.encode(ready_req))
        ready_bytes = await init_tmp_socket.recv()
        ready_resp = msgspec.msgpack.decode(ready_bytes, type=HcommOsMsg)
        if isinstance(ready_resp, HcommOsReadyResponse) and not ready_resp.ok:
            raise ConnectionError(
                f"Server failed to complete handshake for peer {peer_id}"
            )

        init_ret_msg: Optional[InitSideRetMsgBase] = None
        if init_side_msg is not None:
            init_ret_msg = await self.async_send_init_side_msg(
                init_tmp_socket, init_side_msg
            )

        init_tmp_socket.close()
        return init_ret_msg

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self._peers

    def _handle_init_msg(
        self, req: Union[HcommOsMsg, InitSideMsgBase]
    ) -> Union[HcommOsMsg, InitSideRetMsgBase]:
        if isinstance(req, HcommOsInitRequest):
            logger.info("Server: HcommOsInitRequest from %s", req.local_id)

            my_rank = self.SERVER_RANK
            client_rank = self.CLIENT_RANK
            remote_rank = client_rank

            cluster_json = _build_rank_table_json(
                self.device_info,
                my_rank,
                req.device_info,
                client_rank,
            )
            local_id = self.peer_init_url.replace(":", "_")
            remote_id = req.local_id.replace(":", "_")
            comm_name = f"lmcache_{local_id}_{remote_id}"
            logger.info(
                "Server: built rank table: %s  comm_name=%s", cluster_json, comm_name
            )

            resp = HcommOsInitResponse(
                cluster_json=cluster_json,
                comm_name=comm_name,
                server_rank=my_rank,
                client_rank=client_rank,
                buffers=self._make_buffer_infos(),
            )

            old_peer = self._pop_stale_peer(req.local_id)
            all_mem_handles = [m.mem_handle for m in self.mem_handles]

            def _setup_server_comm():
                torch.npu.set_device(self.handle_device)
                try:
                    if old_peer is not None:
                        logger.info(
                            "Server: destroying stale comm for peer %s "
                            "before reconnect",
                            req.local_id,
                        )
                        self._destroy_peer_comm(old_peer, req.local_id)
                    comm = _init_comm_and_prepare(
                        cluster_json, comm_name, my_rank, all_mem_handles
                    )
                    remote_buffers = RemotePeerBufferList(req.buffers)
                    peer_state = _PeerState(
                        comm=comm,
                        my_rank=my_rank,
                        remote_rank=remote_rank,
                        remote_buffers=remote_buffers,
                    )
                    with self._state_lock:
                        self._peers[req.local_id] = peer_state
                    logger.info("Server: peer %s connected", req.local_id)
                except Exception as e:
                    logger.error("Server comm setup failed: %s", e)

            t = threading.Thread(target=_setup_server_comm, daemon=True)
            t.start()
            return resp

        elif isinstance(req, HcommOsReadyRequest):
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                with self._state_lock:
                    if req.local_id in self._peers:
                        break
                time.sleep(0.05)
            return HcommOsReadyResponse(ok=req.local_id in self._peers)

        elif isinstance(req, InitSideMsgBase):
            return self.handle_init_side_msg(req)

        raise ValueError(f"Unsupported message type: {type(req)}")

    def _make_error_response(self) -> HcommOsReadyResponse:
        return HcommOsReadyResponse(ok=False)

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        peer_state, stream_ptr = self._resolve_transfer(transfer_spec)

        op_descs = self._build_op_descs(objects, peer_state, transfer_spec)
        hcomm_os.batch_put(
            peer_state.comm, peer_state.remote_rank, op_descs, stream_ptr
        )

        stream = self._get_torch_stream(transfer_spec)
        stream.synchronize()
        return len(objects)

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        peer_state, stream_ptr = self._resolve_transfer(transfer_spec)

        op_descs = self._build_op_descs(objects, peer_state, transfer_spec)
        hcomm_os.batch_put(
            peer_state.comm, peer_state.remote_rank, op_descs, stream_ptr
        )

        stream = self._get_torch_stream(transfer_spec)
        event = torch.npu.Event()
        event.record(stream)
        while not event.query():
            await asyncio.sleep(0.001)
        return len(objects)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        peer_state, stream_ptr = self._resolve_transfer(transfer_spec)

        op_descs = self._build_op_descs(buffers, peer_state, transfer_spec)
        hcomm_os.batch_get(
            peer_state.comm, peer_state.remote_rank, op_descs, stream_ptr
        )

        stream = self._get_torch_stream(transfer_spec)
        stream.synchronize()
        return len(buffers)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        peer_state, stream_ptr = self._resolve_transfer(transfer_spec)

        op_descs = self._build_op_descs(buffers, peer_state, transfer_spec)
        hcomm_os.batch_get(
            peer_state.comm, peer_state.remote_rank, op_descs, stream_ptr
        )

        stream = self._get_torch_stream(transfer_spec)
        event = torch.npu.Event()
        event.record(stream)
        while not event.query():
            await asyncio.sleep(0.001)
        return len(buffers)

    def submit_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> torch.npu.Event:
        """Submit a batched read without waiting for completion.

        The returned event is recorded on the read stream and can be used
        by callers for cross-stream synchronization.
        """
        assert transfer_spec is not None
        peer_state, stream_ptr = self._resolve_transfer(transfer_spec)

        op_descs = self._build_op_descs(buffers, peer_state, transfer_spec)
        hcomm_os.batch_get(
            peer_state.comm, peer_state.remote_rank, op_descs, stream_ptr
        )

        stream = self._get_torch_stream(transfer_spec)
        event = torch.npu.Event()
        event.record(stream)
        return event

    def _destroy_peer_comm(self, peer: "_PeerState", peer_id: str) -> None:
        try:
            for meta in self.mem_handles:
                hcomm_os.unbind_mem(peer.comm, meta.mem_handle)
            hcomm_os.destroy_comm(peer.comm)
        except Exception as e:
            logger.warning("Failed to destroy comm for peer %s: %s", peer_id, e)

    def _pop_stale_peer(self, peer_id: str) -> Optional["_PeerState"]:
        with self._state_lock:
            return self._peers.pop(peer_id, None)

    def _resolve_transfer(self, transfer_spec: dict):
        """Return (peer_state, stream_ptr) from transfer_spec."""
        peer_id = resolve_peer_id(transfer_spec)
        with self._state_lock:
            peer_state = self._peers[peer_id]
        stream_ptr = self._get_stream_ptr(transfer_spec)
        return peer_state, stream_ptr

    def _get_stream_ptr(self, transfer_spec: dict) -> int:
        stream = transfer_spec.get(TS_STREAM, None)
        if stream is not None:
            if isinstance(stream, int):
                return stream
            return stream.npu_stream
        return self.transport_stream.npu_stream

    def _get_torch_stream(self, transfer_spec: dict) -> torch.npu.Stream:
        stream = transfer_spec.get(TS_STREAM, None)
        if stream is not None and isinstance(stream, torch.npu.Stream):
            return stream
        return self.transport_stream

    def _build_op_descs(self, objects, peer_state, transfer_spec):
        descs = []
        remote_addrs = self._resolve_transfer_addrs(
            peer_state.remote_buffers, transfer_spec
        )
        for mem_obj, remote_addr in zip(objects, remote_addrs, strict=True):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError("Sending raw bytes is not supported")
            descs.append(
                hcomm_os.OpDesc(
                    local_addr=self._get_local_addr(
                        mem_obj.data_ptr, mem_obj.meta.address
                    ),
                    remote_addr=remote_addr,
                    num_bytes=self.page_size,
                )
            )

        return descs

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()

        with self._state_lock:
            peers_snapshot = list(self._peers.items())
            self._peers.clear()
        for peer_id, ps in peers_snapshot:
            self._destroy_peer_comm(ps, peer_id)

        for meta in self.mem_handles:
            try:
                hcomm_os.deregister_global_mem(meta.mem_handle)
            except Exception as e:
                logger.warning("Error deregistering global mem: %s", e)
