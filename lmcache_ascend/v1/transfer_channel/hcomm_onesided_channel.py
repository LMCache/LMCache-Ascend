# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import asyncio
import json
import os
import random
import socket
import subprocess
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
import lmcache_ascend.hcomm_onesided as hcomm_os

logger = init_logger(__name__)


class HcommOsMsgBase(msgspec.Struct, tag=True):
    pass


# SoC models that require v1.2 rank table with super_device_id / super_pod_list.
# All other SoCs use the simpler v1.0 format.
# Reference: hixl/src/llm_datadist/common/rank_table_generator.cc (kV2Version)
_V2_SOC_NAMES = frozenset(
    {
        "Ascend910_9391",
        "Ascend910_9381",
        "Ascend910_9392",
        "Ascend910_9382",
        "Ascend910_9372",
        "Ascend910_9362",
    }
)


class HcommDeviceInfo(msgspec.Struct):
    """Device info exchanged during handshake to build the rank table."""

    server_id: str
    phy_device_id: str
    device_ip: str
    super_device_id: str = "0"
    super_pod_id: str = "0"
    use_v2: bool = False


class HcommOsInitRequest(HcommOsMsgBase):
    local_id: str
    buffer_ptr: int
    buffer_size: int
    page_size: int
    is_device: bool
    device_info: HcommDeviceInfo


class HcommOsInitResponse(HcommOsMsgBase):
    cluster_json: str
    comm_name: str
    server_rank: int
    client_rank: int
    buffer_ptr: int
    buffer_size: int
    page_size: int
    is_device: bool


class HcommOsReadyRequest(HcommOsMsgBase):
    local_id: str


class HcommOsReadyResponse(HcommOsMsgBase):
    ok: bool


HcommOsMsg = Union[
    HcommOsInitRequest,
    HcommOsInitResponse,
    HcommOsReadyRequest,
    HcommOsReadyResponse,
]


class HcommOneSidedChannel(BaseTransferChannel):
    """Transfer channel using hcomm one-sided service API with stream support.

    Each peer pair gets its own HcclComm (nRanks=2).  The server side is
    always rank 0 and the client is rank 1.
    """

    SERVER_RANK = 0
    CLIENT_RANK = 1

    def __init__(self, async_mode: bool = False, **kwargs):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "peer_init_url" in kwargs
        assert "align_bytes" in kwargs

        self.role = kwargs["role"]
        self.buffer_ptr: int = kwargs["buffer_ptr"]
        self.buffer_size: int = kwargs["buffer_size"]
        self.page_size: int = kwargs["align_bytes"]

        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        self.is_device = _is_device_memory(self.buffer_ptr)
        self.device_info = _get_local_device_info()

        self.mem_handle: Optional[int] = None

        self._register_global_mem()

        self.local_index_addr: list[int] = []
        for addr in range(
            self.buffer_ptr,
            self.buffer_ptr + self.buffer_size,
            self.page_size,
        ):
            self.local_index_addr.append(addr)

        self.running = True
        self._state_lock = threading.Lock()

        # peer_id -> PeerState
        self._peers: dict[str, _PeerState] = {}

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

    def _register_global_mem(self):
        already_registered = lmc_ops.get_device_ptr(self.buffer_ptr) is not None
        if already_registered:
            lmc_ops.unregister_ptr(self.buffer_ptr)

        self.mem_handle = hcomm_os.register_global_mem(
            self.buffer_ptr, self.buffer_size, self.is_device
        )
        device_id = torch.npu.current_device()
        dev_ptr = hcomm_os.get_dev_va(device_id, self.buffer_ptr, self.buffer_size)
        if dev_ptr is not None:
            lmc_ops.register_mapping(self.buffer_ptr, dev_ptr, self.buffer_size)
            logger.info(
                "Re-registered lmc_ops mapping via MemMappingManager (devVA=0x%x)",
                dev_ptr,
            )

        logger.info(
            "Registered global mem: ptr=0x%x size=%d is_device=%s",
            self.buffer_ptr,
            self.buffer_size,
            self.is_device,
        )

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

        # Step 1: send init request with device info, get cluster JSON
        req = HcommOsInitRequest(
            local_id=local_id,
            buffer_ptr=self.buffer_ptr,
            buffer_size=self.buffer_size,
            page_size=self.page_size,
            is_device=self.is_device,
            device_info=self.device_info,
        )
        init_tmp_socket.send(msgspec.msgpack.encode(req))
        resp_bytes = init_tmp_socket.recv()
        resp = msgspec.msgpack.decode(resp_bytes, type=HcommOsMsg)
        if not isinstance(resp, HcommOsInitResponse):
            raise ValueError(
                f"Expected HcommOsInitResponse, got {type(resp).__name__}"
            )
        if resp.page_size <= 0:
            raise ValueError(
                f"Peer returned invalid page_size={resp.page_size}; "
                "expected a positive value"
            )

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
        comm = _init_comm_and_prepare(
            resp.cluster_json, resp.comm_name, my_rank, self.mem_handle
        )

        remote_addrs = _build_remote_index_addr(
            resp.buffer_ptr, resp.buffer_size, resp.page_size
        )

        peer_state = _PeerState(
            comm=comm,
            my_rank=my_rank,
            remote_rank=remote_rank,
            remote_index_addr=remote_addrs,
        )
        with self._state_lock:
            self._peers[peer_id] = peer_state

        logger.info("Client: peer %s connected", peer_id)

        # Step 2: signal ready so server knows prepare finished
        ready_req = HcommOsReadyRequest(local_id=local_id)
        init_tmp_socket.send(msgspec.msgpack.encode(ready_req))
        ready_bytes = init_tmp_socket.recv()
        ready_resp = msgspec.msgpack.decode(ready_bytes, type=HcommOsMsg)
        if isinstance(ready_resp, HcommOsReadyResponse) and not ready_resp.ok:
            raise ConnectionError(
                f"Server failed to complete handshake for peer {peer_id}"
            )

        # Step 3: optional side message
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
            buffer_ptr=self.buffer_ptr,
            buffer_size=self.buffer_size,
            page_size=self.page_size,
            is_device=self.is_device,
            device_info=self.device_info,
        )
        await init_tmp_socket.send(msgspec.msgpack.encode(req))
        resp_bytes = await init_tmp_socket.recv()
        resp = msgspec.msgpack.decode(resp_bytes, type=HcommOsMsg)
        if not isinstance(resp, HcommOsInitResponse):
            raise ValueError(
                f"Expected HcommOsInitResponse, got {type(resp).__name__}"
            )
        if resp.page_size <= 0:
            raise ValueError(
                f"Peer returned invalid page_size={resp.page_size}; "
                "expected a positive value"
            )

        my_rank = resp.client_rank
        remote_rank = resp.server_rank

        old_peer = self._pop_stale_peer(peer_id)

        loop = asyncio.get_running_loop()
        device = self.handle_device
        mem_handle = self.mem_handle

        def _client_init_comm():
            torch.npu.set_device(device)
            if old_peer is not None:
                logger.info(
                    "Client: destroying stale comm for peer %s before reconnect",
                    peer_id,
                )
                self._destroy_peer_comm(old_peer, peer_id)
            return _init_comm_and_prepare(
                resp.cluster_json, resp.comm_name, my_rank, mem_handle
            )

        comm = await loop.run_in_executor(None, _client_init_comm)

        remote_addrs = _build_remote_index_addr(
            resp.buffer_ptr, resp.buffer_size, resp.page_size
        )
        peer_state = _PeerState(
            comm=comm,
            my_rank=my_rank,
            remote_rank=remote_rank,
            remote_index_addr=remote_addrs,
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

    def _init_side_channels(self):
        if self.peer_init_url is None:
            logger.warning("Peer init URL not set, skipping init loop")
            return
        if self.async_mode:
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            t = threading.Thread(target=self._init_loop, daemon=True)
            t.start()
            self.running_threads.append(t)

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self._peers

    def _handle_init_msg(
        self, req: Union[HcommOsMsg, InitSideMsgBase]
    ) -> Union[HcommOsMsg, InitSideRetMsgBase]:
        if isinstance(req, HcommOsInitRequest):
            logger.info("Server: HcommOsInitRequest from %s", req.local_id)
            if req.page_size <= 0:
                raise ValueError(
                    f"Peer sent invalid page_size={req.page_size}; "
                    "expected a positive value"
                )

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
                buffer_ptr=self.buffer_ptr,
                buffer_size=self.buffer_size,
                page_size=self.page_size,
                is_device=self.is_device,
            )

            old_peer = self._pop_stale_peer(req.local_id)

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
                        cluster_json, comm_name, my_rank, self.mem_handle
                    )
                    remote_addrs = _build_remote_index_addr(
                        req.buffer_ptr, req.buffer_size, req.page_size
                    )
                    peer_state = _PeerState(
                        comm=comm,
                        my_rank=my_rank,
                        remote_rank=remote_rank,
                        remote_index_addr=remote_addrs,
                    )
                    with self._state_lock:
                        self._peers[req.local_id] = peer_state
                    logger.info("Server: peer %s connected", req.local_id)
                except Exception as e:
                    logger.error("Server comm setup failed: %s", e)

            t = threading.Thread(target=_setup_server_comm, daemon=True)
            t.start()
            # Reply immediately so the client can start its own
            # init_comm_cluster_info + prepare in parallel.
            # The ready handshake later ensures both sides finished.
            return resp

        elif isinstance(req, HcommOsReadyRequest):
            # Client signals that its prepare() has returned.  Wait until
            # our own peer state is ready too.
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

    def _init_loop(self):
        self.init_side_channel = get_zmq_socket(
            self.zmq_context, self.peer_init_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self.init_side_channel)
        torch.npu.set_device(self.handle_device)
        self.init_side_channel.setsockopt(zmq.RCVTIMEO, 1000)

        while self.running:
            try:
                req_bytes = self.init_side_channel.recv()
                logger.info("Received init request")
                req = msgspec.msgpack.decode(req_bytes, type=Union[HcommOsMsg, SideMsg])
                resp = self._handle_init_msg(req)
                self.init_side_channel.send(msgspec.msgpack.encode(resp))
                logger.info("Sent init response")
            except zmq.Again:
                continue
            except Exception as e:
                logger.error("Init loop error: %s", e)
                try:
                    self.init_side_channel.send(
                        msgspec.msgpack.encode(HcommOsReadyResponse(ok=False))
                    )
                except Exception:
                    logger.error("Failed to send HcommOsReadyResponse: %s", e)
                if self.running:
                    time.sleep(0.01)
        self.init_side_channel.close()

    async def _async_init_loop(self):
        self.init_side_channel = get_zmq_socket(
            self.zmq_context, self.peer_init_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self.init_side_channel)
        torch.npu.set_device(self.handle_device)
        loop = asyncio.get_running_loop()

        while self.running:
            try:
                req_bytes = await asyncio.wait_for(
                    self.init_side_channel.recv(), timeout=1.0
                )
                logger.info("Received init request (async)")
                req = msgspec.msgpack.decode(req_bytes, type=Union[HcommOsMsg, SideMsg])
                resp = await loop.run_in_executor(None, self._handle_init_msg, req)
                await self.init_side_channel.send(msgspec.msgpack.encode(resp))
                logger.info("Sent init response (async)")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Async init loop error: %s", e)
                try:
                    await self.init_side_channel.send(
                        msgspec.msgpack.encode(HcommOsReadyResponse(ok=False))
                    )
                except Exception:
                    logger.error("Failed to send HcommOsReadyResponse: %s", e)
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
                "Sending raw bytes is not supported in hcomm one-sided channel"
            )
        return local_indices

    def batched_send(self, objects, transfer_spec=None) -> int:
        raise NotImplementedError

    def batched_recv(self, buffers, transfer_spec=None) -> int:
        raise NotImplementedError

    async def async_batched_send(self, objects, transfer_spec=None) -> int:
        raise NotImplementedError

    async def async_batched_recv(self, buffers, transfer_spec=None) -> int:
        raise NotImplementedError

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

    def _destroy_peer_comm(self, peer: "_PeerState", peer_id: str) -> None:
        try:
            hcomm_os.unbind_mem(peer.comm, self.mem_handle)
            hcomm_os.destroy_comm(peer.comm)
        except Exception as e:
            logger.warning("Failed to destroy comm for peer %s: %s", peer_id, e)

    def _pop_stale_peer(self, peer_id: str) -> Optional["_PeerState"]:
        with self._state_lock:
            return self._peers.pop(peer_id, None)

    def _resolve_transfer(self, transfer_spec: dict):
        """Return (peer_state, stream_ptr) from transfer_spec."""
        peer_id = transfer_spec.get("receiver_id") or transfer_spec["sender_id"]
        with self._state_lock:
            peer_state = self._peers[peer_id]
        stream_ptr = self._get_stream_ptr(transfer_spec)
        return peer_state, stream_ptr

    def _get_stream_ptr(self, transfer_spec: dict) -> int:
        """Extract the raw stream pointer for the C API.

        If ``transfer_spec["stream"]`` is provided it is used directly,
        otherwise falls back to the internal transport stream.
        """
        stream = transfer_spec.get("stream", None)
        if stream is not None:
            if isinstance(stream, int):
                return stream
            return stream.npu_stream
        return self.transport_stream.npu_stream

    def _get_torch_stream(self, transfer_spec: dict) -> torch.npu.Stream:
        stream = transfer_spec.get("stream", None)
        if stream is not None and isinstance(stream, torch.npu.Stream):
            return stream
        return self.transport_stream

    def _build_op_descs(self, objects, peer_state, transfer_spec):
        descs = []
        for mem_obj, remote_index in zip(
            objects, transfer_spec["remote_indexes"], strict=True
        ):
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError("Sending raw bytes is not supported")
            descs.append(
                hcomm_os.OpDesc(
                    local_addr=self.local_index_addr[mem_obj.meta.address],
                    remote_addr=peer_state.remote_index_addr[remote_index],
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

        if self.mem_handle is not None:
            try:
                hcomm_os.deregister_global_mem(self.mem_handle)
            except Exception as e:
                logger.warning("Error deregistering global mem: %s", e)


class _PeerState:
    __slots__ = ("comm", "my_rank", "remote_rank", "remote_index_addr")

    def __init__(
        self,
        comm: int,
        my_rank: int,
        remote_rank: int,
        remote_index_addr: list[int],
    ):
        self.comm = comm
        self.my_rank = my_rank
        self.remote_rank = remote_rank
        self.remote_index_addr = remote_index_addr


def _build_remote_index_addr(
    buffer_ptr: int, buffer_size: int, page_size: int
) -> list[int]:
    return list(range(buffer_ptr, buffer_ptr + buffer_size, page_size))


_HCOMM_INIT_MAX_RETRIES = int(os.environ.get("LMCACHE_HCOMM_INIT_MAX_RETRIES", "5"))
_HCOMM_INIT_BASE_DELAY = 0.1
_HCOMM_INIT_MAX_DELAY = 5.0


def _init_comm_and_prepare(
    cluster_json: str, comm_name: str, rank: int, mem_handle: int
) -> int:
    """Blocking helper: init comm via cluster-info JSON, bind mem, prepare.

    Retries ``init_comm_cluster_info`` with exponential back-off because
    concurrent calls to HcclCommInitClusterInfoMemConfig from different
    processes on the same node can transiently fail (HCCL error 7).
    """
    last_err: Optional[RuntimeError] = None
    for attempt in range(_HCOMM_INIT_MAX_RETRIES):
        try:
            comm = hcomm_os.init_comm_cluster_info(cluster_json, rank, comm_name)
            break
        except RuntimeError as e:
            last_err = e
            delay = min(
                _HCOMM_INIT_BASE_DELAY * (2**attempt),
                _HCOMM_INIT_MAX_DELAY,
            )
            delay *= random.uniform(0.5, 1.5)
            logger.warning(
                "init_comm_cluster_info failed (attempt %d/%d): %s  retrying in %.2fs",
                attempt + 1,
                _HCOMM_INIT_MAX_RETRIES,
                e,
                delay,
            )
            time.sleep(delay)
    else:
        raise RuntimeError(
            f"init_comm_cluster_info failed after {_HCOMM_INIT_MAX_RETRIES} attempts"
        ) from last_err

    hcomm_os.bind_mem(comm, mem_handle)
    hcomm_os.prepare(comm, timeout=120)
    return comm


def _is_device_memory(ptr: int) -> bool:
    return hcomm_os.is_device_memory(ptr)


def _get_device_ip(phy_device_id: int) -> str:
    """Read device IP from /etc/hccn.conf or fall back to hccn_tool."""
    hccn_conf = "/etc/hccn.conf"
    if os.path.isfile(hccn_conf):
        key = f"address_{phy_device_id}="
        with open(hccn_conf) as f:
            for line in f:
                if line.startswith(key):
                    return line.strip().split("=", 1)[1]
    try:
        result = subprocess.run(
            ["hccn_tool", "-i", str(phy_device_id), "-ip", "-g"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "ipaddr:" in line:
                return line.split("ipaddr:")[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("Failed to get device IP from hccn_tool")
    return ""


def _get_local_device_info() -> HcommDeviceInfo:
    """Gather local device metadata needed for the rank table."""
    device_id = torch.npu.current_device()
    info = hcomm_os.get_device_info(device_id)
    device_ip = _get_device_ip(info["phy_device_id"])
    soc_name = info.get("soc_name", "")
    use_v2 = soc_name in _V2_SOC_NAMES

    result = HcommDeviceInfo(
        server_id=socket.gethostname(),
        phy_device_id=str(info["phy_device_id"]),
        device_ip=device_ip,
        use_v2=use_v2,
    )
    if use_v2:
        result.super_device_id = str(info["super_device_id"])
        result.super_pod_id = str(info["super_pod_id"])
    logger.info(
        "Local device info: soc=%s v2=%s phy_dev=%s ip=%s",
        soc_name,
        use_v2,
        result.phy_device_id,
        result.device_ip,
    )
    return result


def _build_rank_table_json(
    server_info: HcommDeviceInfo,
    server_rank: int,
    client_info: HcommDeviceInfo,
    client_rank: int,
) -> str:
    """Build a rank-table JSON string for HcclCommInitClusterInfoMemConfig.

    Uses v1.0 (no super_device_id / super_pod_list) unless both peers
    report use_v2, in which case v1.2 is used.  Groups devices by
    server_id (hostname).
    """
    use_v2 = server_info.use_v2 and client_info.use_v2

    servers: dict[str, list[dict]] = {}
    for dev_info, rank_id in [
        (server_info, server_rank),
        (client_info, client_rank),
    ]:
        dev: dict = {
            "device_id": dev_info.phy_device_id,
            "rank_id": str(rank_id),
        }
        if dev_info.device_ip:
            dev["device_ip"] = dev_info.device_ip
        if use_v2:
            dev["super_device_id"] = dev_info.super_device_id
        servers.setdefault(dev_info.server_id, []).append(dev)

    server_list = []
    for sid, devices in servers.items():
        devices.sort(key=lambda d: int(d["rank_id"]))
        server_list.append({"server_id": sid, "device": devices})

    rank_table: dict = {
        "version": "1.2" if use_v2 else "1.0",
        "server_count": str(len(server_list)),
        "server_list": server_list,
        "status": "completed",
    }

    if use_v2:
        pod_map: dict[str, set[str]] = {}
        for dev_info in (server_info, client_info):
            pod_map.setdefault(dev_info.super_pod_id, set()).add(dev_info.server_id)
        super_pod_list = []
        for pod_id, sids in sorted(pod_map.items()):
            super_pod_list.append(
                {
                    "super_pod_id": pod_id,
                    "server_list": [{"server_id": s} for s in sorted(sids)],
                }
            )
        rank_table["super_pod_list"] = super_pod_list

    return json.dumps(rank_table)
