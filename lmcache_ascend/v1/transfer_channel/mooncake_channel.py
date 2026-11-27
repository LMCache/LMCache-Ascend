# SPDX-License-Identifier: Apache-2.0
"""Mooncake TE transfer channel — an optional NPU transfer transport.

Uses Mooncake Transfer Engine (AscendDirect transport) for KV cache
transfer between NPUs, as an alternative to the built-in hccl/hixl
channels. Peer discovery works in ``P2PHANDSHAKE`` mode: each side
exposes a Mooncake RPC endpoint (``ip:rpc_port``) that is exchanged
over the existing ZMQ init handshake, and KV pages are pushed/pulled
as RDMA ops against peer page addresses resolved from the exchanged
buffer layout tables.

Note: Mooncake TE provides the sync transfer surface plus a
sync-backed ``async_batched_read``; it does not (yet) provide the full
async + host-staging surface, so it is not usable for P2P KV sharing.
"""

# Standard
from typing import List, Optional, Union
import importlib
import os

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_ip, get_zmq_socket
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)
import msgspec
import zmq

# Local
from .base_channel import BaseMultiBufferChannel
from .buffer_config import (
    BufferConfig,
    BufferType,
    MemHandleMeta,
    PeerBufferInfo,
    RemotePeerBufferList,
    resolve_local_addr,
)
from .mooncake_protocol import (
    MooncakeErrorResponse,
    MooncakeInitRequest,
    MooncakeInitResponse,
    MooncakeMemInfoRequest,
    MooncakeMemInfoResponse,
    MooncakeMsg,
)
from .transfer_spec import resolve_peer_id

logger = init_logger(__name__)

# Mooncake native timeout knobs (read by the engine itself):
#   MC_TRANSFER_TIMEOUT      — seconds, default 30
#   ASCEND_TRANSFER_TIMEOUT  — milliseconds, default 10000
_MC_TRANSFER_TIMEOUT_DEFAULT_S = 30.0
_ASCEND_TRANSFER_TIMEOUT_DEFAULT_MS = 10000.0


# API locations seen in the wild: the canonical wheel exposes
# ``TransferEngine``/``TransferOpcode`` in ``mooncake.transfer_engine``;
# some vllm-ascend images ship a repackaged tree where the pybind module
# is ``mooncake.engine`` (``engine.so``) and the standard wrapper is
# absent.
_MOONCAKE_API_MODULES = ("mooncake.transfer_engine", "mooncake.engine")


def _load_mooncake():
    """Import the mooncake package and locate the transfer-engine API.

    Done lazily so that importing this module (and running unit tests
    with a fake mooncake module) does not require the real package.

    Depending on the installed layout, ``TransferEngine`` /
    ``TransferOpcode`` live in the ``mooncake.transfer_engine``
    submodule (canonical), in a repackaged ``mooncake.engine`` pybind
    module, or at the package top level. Returns the module object
    that provides both symbols.
    """
    try:
        # Third Party
        import mooncake
    except ImportError as e:
        raise ImportError(
            "The 'mooncake' package is required for the Mooncake transfer "
            "channel (build mooncake-transfer-engine with "
            "-DUSE_ASCEND_DIRECT=ON for Ascend support)"
        ) from e

    candidates = []
    for module_name in _MOONCAKE_API_MODULES:
        try:
            candidates.append(importlib.import_module(module_name))
        except (ImportError, OSError):
            pass
    candidates.append(mooncake)

    for mod in candidates:
        if all(hasattr(mod, n) for n in ("TransferEngine", "TransferOpcode")):
            return mod

    origin = getattr(mooncake, "__file__", None) or "<namespace package>"
    visible = {}
    for mod in candidates:
        names = [n for n in dir(mod) if not n.startswith("_")]
        visible[mod.__name__] = names[:20] + (["..."] if len(names) > 20 else [])
    raise ImportError(
        "mooncake package found but the transfer-engine API "
        "(TransferEngine/TransferOpcode) is missing. Inspected: "
        f"{origin}; visible symbols: {visible}. Install/build "
        "mooncake-transfer-engine (with -DUSE_ASCEND_DIRECT=ON for "
        "Ascend) and make sure its mooncake package is the one imported."
    )


def _check_status(ret, what: str) -> None:
    """Mooncake pybind calls report success as 0/None/True depending on
    the binding version — treat anything else as a failure."""
    if ret is None or ret == 0 or ret is True:
        return
    raise RuntimeError(f"Mooncake {what} failed with status {ret}")


def _validate_transfer_timeouts() -> None:
    """Warn when MC_TRANSFER_TIMEOUT is not strictly greater than 2x
    ASCEND_TRANSFER_TIMEOUT (Mooncake-documented pitfall: transfers may
    otherwise time out before the AscendDirect transport aborts)."""
    try:
        transfer_timeout_s = float(
            os.environ.get("MC_TRANSFER_TIMEOUT", _MC_TRANSFER_TIMEOUT_DEFAULT_S)
        )
        ascend_timeout_ms = float(
            os.environ.get(
                "ASCEND_TRANSFER_TIMEOUT", _ASCEND_TRANSFER_TIMEOUT_DEFAULT_MS
            )
        )
    except ValueError:
        logger.warning(
            "Non-numeric Mooncake timeout env vars; skipping timeout check"
        )
        return
    if transfer_timeout_s <= 2.0 * ascend_timeout_ms / 1000.0:
        logger.warning(
            "MC_TRANSFER_TIMEOUT (%.1fs) should be strictly greater than "
            "2x ASCEND_TRANSFER_TIMEOUT (%.0fms = %.1fs); transfers may "
            "time out before the AscendDirect transport aborts",
            transfer_timeout_s,
            ascend_timeout_ms,
            ascend_timeout_ms / 1000.0,
        )


# Enum-member spellings seen across mooncake builds: the canonical
# wheel uses WRITE/READ; repackaged NPU trees have been observed with
# other spellings/prefixes.
_WRITE_NAMES = ("WRITE", "OP_WRITE", "Write", "write")
_READ_NAMES = ("READ", "OP_READ", "Read", "read")

# No-op TransferNotify member spellings for the repackaged builds that
# require a sixth batch_transfer_sync argument (the canonical wheel
# takes only five and has no TransferNotify at all).
_NOTIFY_NONE_NAMES = ("NONE", "None", "NOTIFY_NONE", "none")
_NO_NOTIFY_ARG = object()


def _resolve_opcodes(mod) -> tuple:
    """Resolve (write, read) opcode values from a mooncake API module.

    Tolerates enum-member naming differences between builds. Raises
    with the visible member names when nothing matches, so yet another
    spelling is identifiable at a glance. No silent integer fallback:
    mixing up the transfer direction would corrupt data, so an unknown
    layout must fail loudly.
    """
    enum = getattr(mod, "TransferOpcode", None)
    if enum is None:
        raise AttributeError(
            f"{mod.__name__} has no TransferOpcode; cannot resolve opcodes"
        )
    write_op = next(
        (getattr(enum, n) for n in _WRITE_NAMES if hasattr(enum, n)), None
    )
    read_op = next(
        (getattr(enum, n) for n in _READ_NAMES if hasattr(enum, n)), None
    )
    if write_op is None or read_op is None:
        members = [n for n in dir(enum) if not n.startswith("_")]
        raise AttributeError(
            f"TransferOpcode members {members} match no known WRITE/READ "
            f"spelling (tried write={_WRITE_NAMES}, read={_READ_NAMES})"
        )
    return write_op, read_op


def _resolve_notify_arg(mod):
    """Resolve the no-op TransferNotify argument for builds that require
    it on batch_transfer_sync (repackaged NPU trees expose a trailing
    notify parameter the canonical wheel does not have).

    Two layouts exist: enum-style builds (no-op member NONE/None/...) and
    class-style builds where TransferNotify is a struct with msg/name
    fields — a default-constructed instance is the no-op either way:
    an empty target sends nothing under input semantics, and the
    container simply stays unfilled under output semantics. PD
    completion signaling goes through the proxy protocol, so no-op is
    correct either way.
    """
    notify = getattr(mod, "TransferNotify", None)
    if notify is None:
        return _NO_NOTIFY_ARG
    for name in _NOTIFY_NONE_NAMES:
        member = getattr(notify, name, None)
        if member is not None:
            return member
    try:
        return notify()
    except TypeError as e:
        members = [n for n in dir(notify) if not n.startswith("_")]
        raise AttributeError(
            f"TransferNotify is neither a no-op enum (members {members}) "
            f"nor default-constructible ({e}); inspect how the installed "
            "stack builds TransferNotify and adapt"
        ) from e


def _build_addr_list(buffer_ptr: int, buffer_size: int, page_size: int) -> list[int]:
    """Page-start addresses inside a registered buffer (used to map a
    MemoryObj's (data_ptr, page index) pair to a local transfer address)."""
    return list(range(buffer_ptr, buffer_ptr + buffer_size, page_size))


class MooncakeChannel(BaseMultiBufferChannel):
    _init_msg_type = Union[MooncakeMsg, SideMsg]
    _channel_name = "mooncake"

    def __init__(
        self,
        async_mode: bool = False,
        buffers: Optional[List[BufferConfig]] = None,
        **kwargs,
    ):
        self.mooncake_metadata_server = kwargs.pop(
            "mooncake_metadata_server", "P2PHANDSHAKE"
        )
        self.mooncake_protocol = kwargs.pop("mooncake_protocol", "ascend")
        self.mooncake_device_name = kwargs.pop("mooncake_device_name", "")
        self.mooncake_rpc_port = int(kwargs.pop("mooncake_rpc_port", 0))

        # Maps peer_id -> remote engine name (ip:rpc_port).
        self.remote_engine_dict: dict[str, str] = {}
        # Maps peer_id -> RemotePeerBufferList.
        self.remote_peer_buffers: dict[str, RemotePeerBufferList] = {}

        self.engine = None  # set by _register_buffers
        self.engine_name: Optional[str] = None

        super().__init__(async_mode=async_mode, buffers=buffers, **kwargs)
        _validate_transfer_timeouts()

    # ------------------------------------------------------------------
    # Buffer registration
    # ------------------------------------------------------------------

    def _register_buffers(self, buffers: list[BufferConfig]) -> None:
        mooncake = _load_mooncake()
        self._mooncake = mooncake
        self._opcode_write, self._opcode_read = _resolve_opcodes(mooncake)
        self._notify_arg = _resolve_notify_arg(mooncake)

        self.engine = mooncake.TransferEngine()
        ip = get_ip()
        local_hostname = f"{ip}:{self.mooncake_rpc_port}"
        ret = self.engine.initialize(
            local_hostname,
            self.mooncake_metadata_server,
            self.mooncake_protocol,
            self.mooncake_device_name,
        )
        _check_status(ret, f"engine initialize ({local_hostname})")

        # The transfer target is always the *actually bound* RPC port.
        self.engine_name = f"{ip}:{self.engine.get_rpc_port()}"
        logger.info(
            "Mooncake TE initialized: engine_name=%s protocol=%s "
            "metadata_server=%s device_name=%r",
            self.engine_name,
            self.mooncake_protocol,
            self.mooncake_metadata_server,
            self.mooncake_device_name,
        )

        self.mem_handles: List[MemHandleMeta] = []
        for buf in buffers:
            _check_status(
                self.engine.register_memory(buf.ptr, buf.size),
                f"register_memory(ptr={hex(buf.ptr)}, size={buf.size})",
            )
            self.mem_handles.append(
                MemHandleMeta(
                    mem_handle=(buf.ptr, buf.size),
                    buffer_ptr=buf.ptr,
                    buffer_size=buf.size,
                    page_size=buf.align_bytes,
                    local_buffer_addrs=_build_addr_list(
                        buf.ptr, buf.size, buf.align_bytes
                    ),
                    buffer_type=buf.device_type,
                )
            )
            logger.info(
                "Mooncake memory registered: ptr=%s size=%d page_size=%d type=%s",
                hex(buf.ptr),
                buf.size,
                buf.align_bytes,
                buf.device_type,
            )

    # ------------------------------------------------------------------
    # Peer bookkeeping (mirrors the HIXL channel's registry pattern)
    # ------------------------------------------------------------------

    def _connect_to_peer(self, peer_id: str, engine_name: str) -> None:
        # Mooncake needs no explicit connect(): the target engine name is
        # passed to batch_transfer_sync directly (P2PHANDSHAKE mode).
        with self._state_lock:
            self.remote_engine_dict[peer_id] = engine_name
        logger.info(
            "Registered remote Mooncake engine for %s: %s", peer_id, engine_name
        )

    def _store_remote_mem_info(
        self, peer_id: str, resp_buffers: List[PeerBufferInfo]
    ) -> None:
        remote_buffers = RemotePeerBufferList(resp_buffers)
        with self._state_lock:
            self.remote_peer_buffers[peer_id] = remote_buffers

    def _make_buffer_infos(self) -> List[PeerBufferInfo]:
        """Build handshake buffer-info list from our registered buffers."""
        return [
            PeerBufferInfo(
                uuid=meta.uuid,
                buffer_ptr=meta.buffer_ptr,
                buffer_size=meta.buffer_size,
                page_size=meta.page_size,
                is_device=meta.buffer_type == BufferType.NPU,
            )
            for meta in self.mem_handles
        ]

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        return receiver_or_sender_id in self.remote_engine_dict

    # ------------------------------------------------------------------
    # Client-side handshake
    # ------------------------------------------------------------------

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
        # Bound handshake: a dead peer must fail the connection attempt
        # (the caller retries on the next transfer) instead of wedging
        # the engine's save path forever.
        init_tmp_socket.setsockopt(zmq.RCVTIMEO, 10000)
        init_tmp_socket.setsockopt(zmq.SNDTIMEO, 10000)
        try:
            # Step 1: exchange engine names
            init_req = MooncakeInitRequest(
                local_id=local_id,
                engine_name=self.engine_name,
            )
            init_tmp_socket.send(msgspec.msgpack.encode(init_req))
            resp = msgspec.msgpack.decode(init_tmp_socket.recv(), type=MooncakeMsg)
            if isinstance(resp, MooncakeErrorResponse):
                raise ConnectionError(
                    f"Peer {peer_id} failed the Mooncake init handshake"
                )
            if not isinstance(resp, MooncakeInitResponse):
                raise ValueError(
                    f"Expected MooncakeInitResponse, got {type(resp).__name__}"
                )
            self._connect_to_peer(peer_id, resp.engine_name)

            # Step 2: exchange buffer layout tables
            mem_req = MooncakeMemInfoRequest(
                local_id=local_id,
                buffers=self._make_buffer_infos(),
            )
            init_tmp_socket.send(msgspec.msgpack.encode(mem_req))
            mem_resp = msgspec.msgpack.decode(init_tmp_socket.recv(), type=MooncakeMsg)
            if isinstance(mem_resp, MooncakeErrorResponse):
                raise ConnectionError(
                    f"Peer {peer_id} failed the Mooncake mem-info handshake"
                )
            if not isinstance(mem_resp, MooncakeMemInfoResponse):
                raise ValueError(
                    f"Expected MooncakeMemInfoResponse, got "
                    f"{type(mem_resp).__name__}"
                )
            self._store_remote_mem_info(peer_id, mem_resp.buffers)

            # Step 3: optional side message
            init_ret_msg: Optional[InitSideRetMsgBase] = None
            if init_side_msg is not None:
                init_ret_msg = self.send_init_side_msg(
                    init_tmp_socket,
                    init_side_msg,
                )
            return init_ret_msg
        finally:
            init_tmp_socket.close()

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
        try:
            # Step 1: exchange engine names
            init_req = MooncakeInitRequest(
                local_id=local_id,
                engine_name=self.engine_name,
            )
            await init_tmp_socket.send(msgspec.msgpack.encode(init_req))
            resp = msgspec.msgpack.decode(
                await init_tmp_socket.recv(), type=MooncakeMsg
            )
            if isinstance(resp, MooncakeErrorResponse):
                raise ConnectionError(
                    f"Peer {peer_id} failed the Mooncake init handshake"
                )
            if not isinstance(resp, MooncakeInitResponse):
                raise ValueError(
                    f"Expected MooncakeInitResponse, got {type(resp).__name__}"
                )
            self._connect_to_peer(peer_id, resp.engine_name)

            # Step 2: exchange buffer layout tables
            mem_req = MooncakeMemInfoRequest(
                local_id=local_id,
                buffers=self._make_buffer_infos(),
            )
            await init_tmp_socket.send(msgspec.msgpack.encode(mem_req))
            mem_resp = msgspec.msgpack.decode(
                await init_tmp_socket.recv(), type=MooncakeMsg
            )
            if isinstance(mem_resp, MooncakeErrorResponse):
                raise ConnectionError(
                    f"Peer {peer_id} failed the Mooncake mem-info handshake"
                )
            if not isinstance(mem_resp, MooncakeMemInfoResponse):
                raise ValueError(
                    f"Expected MooncakeMemInfoResponse, got "
                    f"{type(mem_resp).__name__}"
                )
            self._store_remote_mem_info(peer_id, mem_resp.buffers)

            # Step 3: optional side message
            init_ret_msg: Optional[InitSideRetMsgBase] = None
            if init_side_msg is not None:
                init_ret_msg = await self.async_send_init_side_msg(
                    init_tmp_socket,
                    init_side_msg,
                )
            return init_ret_msg
        finally:
            init_tmp_socket.close()

    # ------------------------------------------------------------------
    # Server-side handshake
    # ------------------------------------------------------------------

    def _handle_init_msg(
        self, req: Union[MooncakeMsg, InitSideMsgBase]
    ) -> Union[MooncakeMsg, InitSideRetMsgBase]:
        resp: Union[MooncakeMsg, InitSideRetMsgBase]
        if isinstance(req, MooncakeInitRequest):
            logger.info(
                "Processing MooncakeInitRequest from %s (engine=%s)",
                req.local_id,
                req.engine_name,
            )
            self._connect_to_peer(req.local_id, req.engine_name)
            resp = MooncakeInitResponse(engine_name=self.engine_name)

        elif isinstance(req, MooncakeMemInfoRequest):
            logger.info(
                "Processing MooncakeMemInfoRequest from %s (%d buffers)",
                req.local_id,
                len(req.buffers),
            )
            self._store_remote_mem_info(req.local_id, req.buffers)
            resp = MooncakeMemInfoResponse(buffers=self._make_buffer_infos())

        elif isinstance(req, InitSideMsgBase):
            resp = self.handle_init_side_msg(req)

        else:
            raise ValueError(f"Unsupported InitMsg type: {type(req)}")

        return resp

    def _make_error_response(self) -> MooncakeErrorResponse:
        return MooncakeErrorResponse(ok=False)

    # ------------------------------------------------------------------
    # Data transfer
    # ------------------------------------------------------------------

    def _build_transfer_args(
        self,
        items: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict,
    ) -> tuple[str, List[int], List[int], List[int]]:
        """Resolve (target engine, local addrs, remote addrs, lengths) for
        a batch transfer. Items are MemoryObj-like objects exposing
        ``data_ptr`` and ``meta.address`` (page index)."""
        peer_id = resolve_peer_id(transfer_spec)

        with self._state_lock:
            target = self.remote_engine_dict[peer_id]
            remote_buffers = self.remote_peer_buffers[peer_id]
        remote_addrs = self._resolve_transfer_addrs(remote_buffers, transfer_spec)

        local_addrs: List[int] = []
        for item in items:
            if isinstance(item, (bytes, bytearray)):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in Mooncake channel"
                )
            local_addrs.append(
                resolve_local_addr(self.mem_handles, item.data_ptr, item.meta.address)
            )
        lengths = [self.page_size] * len(items)
        return target, local_addrs, remote_addrs, lengths

    def _batch_transfer_sync(
        self,
        target: str,
        local_addrs: List[int],
        remote_addrs: List[int],
        lengths: List[int],
        opcode,
        what: str,
    ) -> None:
        """Layout-adaptive batch_transfer_sync: the repackaged NPU build
        requires a trailing TransferNotify argument, the canonical wheel
        does not."""
        args = [target, local_addrs, remote_addrs, lengths, opcode]
        if self._notify_arg is not _NO_NOTIFY_ARG:
            args.append(self._notify_arg)
        _check_status(self.engine.batch_transfer_sync(*args), what)

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        target, local_addrs, remote_addrs, lengths = self._build_transfer_args(
            objects, transfer_spec
        )
        self._batch_transfer_sync(
            target,
            local_addrs,
            remote_addrs,
            lengths,
            self._opcode_write,
            f"batched_write to {target} ({len(objects)} pages)",
        )
        return len(objects)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        target, local_addrs, remote_addrs, lengths = self._build_transfer_args(
            buffers, transfer_spec
        )
        self._batch_transfer_sync(
            target,
            local_addrs,
            remote_addrs,
            lengths,
            self._opcode_read,
            f"batched_read from {target} ({len(buffers)} pages)",
        )
        return len(buffers)

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """ABC-required override; no async write is wired for Mooncake yet.

        The upstream ``BaseTransferChannel`` declares
        ``async_batched_write`` abstract, so the channel must override it
        to be instantiable at all. Raising here keeps the documented
        contract explicit: the PD push path drives the synchronous
        ``batched_write``, and P2P KV sharing (which drives writes from
        an asyncio loop and needs the host-staging surface) stays on the
        hccl/hixl channels.
        """
        raise NotImplementedError(
            "MooncakeChannel does not implement async_batched_write; "
            "use the synchronous batched_write (PD push path) or the "
            "hccl/hixl channels for P2P KV sharing"
        )

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async read used by the PD pull-delay path (ProxyMemoryObj).

        Backed by the synchronous transfer: the callers run the coroutine
        via ``run_coroutine_threadsafe(...).result()`` and block until the
        read completes, so there is no concurrency to exploit yet. Future
        work swaps this for Mooncake's async API (batch_transfer + status
        polling). The write counterpart raises NotImplementedError so
        P2P KV sharing stays on the hccl/hixl channels.
        """
        return self.batched_read(buffers, transfer_spec)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _unregister_memory(self, ptr: int, size: int) -> None:
        """Layout-adaptive unregister: the repackaged NPU build takes the
        address only; the canonical wheel takes (addr, length)."""
        try:
            ret = self.engine.unregister_memory(ptr)
        except TypeError:
            ret = self.engine.unregister_memory(ptr, size)
        _check_status(ret, f"unregister_memory(ptr={hex(ptr)})")

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
        if self.engine is not None:
            for meta in self.mem_handles:
                try:
                    ptr, size = meta.mem_handle
                    self._unregister_memory(ptr, size)
                except Exception as e:
                    logger.warning("Failed to unregister Mooncake memory: %s", e)
