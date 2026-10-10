# SPDX-License-Identifier: Apache-2.0
"""Ascend PD backend — shared core logic.

Defines :class:`AscendPDBackend` which composes the sender and receiver
mixins with the upstream :class:`PDBackend` base class.  Only shared /
role-neutral code lives here: initialisation, allocator setup, memory
allocation, and key lookup/partitioning.
"""

# Standard
from dataclasses import dataclass, field
from typing import Optional, Union
import threading
import time

# Third Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import get_zmq_context
from lmcache.v1.storage_backend.pd_backend import PDBackend, PDConfig
import torch
import torch_npu  # noqa: F401
import zmq

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.rpc_utils import _find_free_port
from lmcache_ascend.v1.storage_backend.pd.handoff import (
    make_pd_handoff_lease_id,
)
from lmcache_ascend.v1.storage_backend.pd.receiver_mixin import AscendPDReceiverMixin
from lmcache_ascend.v1.storage_backend.pd.sender_mixin import AscendPDSenderMixin
from lmcache_ascend.v1.storage_backend.utils import resolve_memory_format
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel, get_correct_device

logger = init_logger(__name__)


@dataclass
class PDEntry:
    """Receiver-side PD object plus request ownership metadata."""

    base_obj: MemoryObj
    owners: set[str] = field(default_factory=set)
    # Request-local proxy clones used for delay-pull mode.
    proxy_leases: dict[str, ProxyMemoryObj] = field(default_factory=dict)
    pending_delete: bool = False


class AscendPDBackend(AscendPDSenderMixin, AscendPDReceiverMixin, PDBackend):
    """PD backend for Ascend (NPU) using HCCL transfer channel.

    Overrides the base :class:`PDBackend` to:

    * initialize **both** CPU and NPU allocators so that the sender can
      offload KV caches to CPU first (pd_use_cpu_offload) and
      transfer via RDMA from host memory,
      while the receiver allocates directly on NPU (pd_buffer_device),
    * create a transfer channel via
      :func:`lmcache_ascend.v1.transfer_channel.CreateTransferChannel`
      with both CPU and NPU buffers registered (multi-buffer pattern),
    * use UUID-based buffer references in alloc responses and transfer specs
      (required by the channel's ``_resolve_remote_addrs``).
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        self.running = True
        self.tp_rank = metadata.worker_id

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )

        # CPU offload: sender offloads KV to CPU first, then RDMA from CPU.
        # Read from LMCacheEngineConfig (not PDConfig, which is upstream).
        self.use_cpu_offload: bool = getattr(config, "pd_use_cpu_offload", False)

        # Receiver-side KV store
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self._pd_entries: dict[CacheEngineKey, PDEntry] = {}
        self._pd_request_keys: dict[str, set[CacheEngineKey]] = {}
        self._pd_handoff_deadlines: dict[str, float] = {}
        self._pd_handoff_lease_ttl = float(
            getattr(config, "pd_handoff_lease_ttl", 300.0)
        )
        if self._pd_handoff_lease_ttl <= 0:
            raise ValueError("pd_handoff_lease_ttl must be greater than zero")
        self.data_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

        # Pull mode: the receiver reads from the sender instead of the
        # sender writing to the receiver.
        self.pull_mode: bool = getattr(config, "pd_pull_mode", False)
        if self.pull_mode:
            logger.info("PD pull mode enabled.")

        self.delay_pull: bool = getattr(config, "pd_delay_pull", False)
        if self.delay_pull:
            assert self.pull_mode, "Delay pull only works when pull mode is enabled"
            assert self.pd_config.buffer_device.startswith("npu"), (
                "Delay pull only works when buffer device is NPU"
            )

        # Keep config ref for extra_config access (e.g., pull_done_port)
        self._config = config

        # Per-peer circuit breaker: when a receiver fails to allocate,
        # skip all transfers to that peer until the TTL expires.
        # Protected by _peer_alloc_backoff_lock to prevent a race
        # where a concurrent call slips past the check before the
        # failing call has set the backoff timestamp.
        self._peer_alloc_backoff: dict[str, float] = {}
        self._peer_alloc_backoff_lock = threading.Lock()
        self._peer_alloc_backoff_ttl: float = getattr(
            config, "pd_alloc_fail_backoff_ttl", 2.0
        )

        # Peer init URL / local id
        peer_init_url = None
        self.local_id = ""
        if self.pd_config.peer_init_port is not None:
            peer_init_url = (
                f"{self.pd_config.peer_host}:{self.pd_config.peer_init_port}"
            )
            self.local_id = self.pd_config.peer_host + str(
                self.pd_config.peer_init_port
            )
        else:
            port = _find_free_port()
            # generate a random port and obtain ip
            peer_init_url = f"{self.pd_config.peer_host}:{port}"
            self.local_id = self.pd_config.peer_host + str(port)

        # Register both CPU and NPU buffers with the transfer channel
        # so that RDMA can operate on either memory region.
        # (Mirrors the multi-buffer pattern used by AscendP2PBackend.)
        buffer_ptr = []
        buffer_size = []
        buffer_type = []
        align_bytes = []
        if self.pd_config.buffer_device.startswith("npu"):
            buffer_ptr.append(self.memory_allocator.gpu_allocator.buffer_ptr)
            buffer_size.append(self.memory_allocator.gpu_allocator.buffer_size)
            buffer_type.append("npu")
            align_bytes.append(self.memory_allocator.gpu_allocator.align_bytes)

        if self.pd_config.buffer_device == "cpu" or self.use_cpu_offload:
            buffer_ptr.append(self.memory_allocator.cpu_allocator.buffer_ptr)
            buffer_size.append(self.memory_allocator.cpu_allocator.buffer_size)
            buffer_type.append("cpu")
            align_bytes.append(self.memory_allocator.cpu_allocator.align_bytes)

        assert buffer_ptr, (
            "No buffers registered — at least one of NPU or CPU must be configured"
        )

        self.transfer_channel = CreateTransferChannel(
            channel_type=config.transfer_channel,
            async_mode=False,
            role=self.pd_config.role,
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            buffer_type=buffer_type,
            align_bytes=align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=peer_init_url,
        )

        # Role-specific initialization
        if self.pd_config.role == "sender":
            self._init_sender()
            self.initialized_peers: set[str] = set()
            self.mem_alloc_sockets: dict[str, zmq.Socket] = {}
        elif self.pd_config.role == "receiver":
            self._init_receiver()
        else:
            raise ValueError("Invalid PD role.")

        self.full_chunk_size = config.chunk_size

        # Cache metadata for proxy creation on receiver side
        self._metadata = metadata
        self._fmt = resolve_memory_format(metadata.use_mla)
        self._kv_shapes = [torch.Size(metadata.kv_shape)]
        self._kv_dtypes = [metadata.kv_dtype]

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        npu_corrected_device = get_correct_device("npu", metadata.worker_id)
        logger.debug("Setting NPU device to %s", npu_corrected_device)
        torch.npu.set_device(npu_corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()
        fmt = resolve_memory_format(metadata.use_mla)
        # Multi-group models (e.g. DSv4): page size must cover all KV layer
        # groups so allocate() keeps matching shapes and group_prefix_sum.
        # Single-group: get_shapes()/get_dtypes() match kv_shape/kv_dtype.
        sizes = metadata.get_shapes()
        dtypes = metadata.get_dtypes()
        total_size = get_size_bytes(sizes, dtypes)

        if self.pd_config.buffer_device.startswith("npu"):
            # NPU allocator — needed for RDMA buffer registration and
            # receiver-side allocation (incoming KV lands directly on NPU).
            npu_aligned_byte = (
                (config.pd_buffer_size + total_size - 1) // total_size * total_size
            )
            paged_mem_allocator.init_gpu_memory_allocator(
                npu_aligned_byte, sizes, dtypes, fmt, npu_corrected_device
            )
            logger.info(
                "Initialized NPU allocator: %.2f MB",
                npu_aligned_byte / (1024 * 1024),
            )

        if self.pd_config.buffer_device == "cpu" or self.use_cpu_offload:
            # CPU allocator — for sender-side KV offload (NPU -> CPU -> RDMA).
            # or configured to use CPU as the buffer device.
            # Falls back to pd_buffer_size when pd_cpu_buffer_size is not set.
            cpu_buffer_size = config.pd_cpu_buffer_size or config.pd_buffer_size
            cpu_aligned_byte = (
                (cpu_buffer_size + total_size - 1) // total_size * total_size
            )
            paged_mem_allocator.init_cpu_memory_allocator(
                cpu_aligned_byte, sizes, dtypes, fmt
            )

            logger.info(
                "Initialized CPU allocator: %.2f MB",
                cpu_aligned_byte / (1024 * 1024),
            )

        return paged_mem_allocator

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate memory with role-aware placement.

        * **Sender** (prefiller): allocates on **CPU** so that
          ``gpu_connector.batched_from_gpu()`` performs an NPU -> CPU
          offload.  The CPU buffer is registered for RDMA, enabling the
          receiver to pull (or the sender to push) directly from host
          memory.
        * **Receiver** (decoder): allocates on **NPU** so that incoming
          KV data lands directly on the accelerator.
        """
        if fmt is None:
            # NOTE (gingfung): this currently can happen
            # because we don't have a default fmt in the config
            fmt = MemoryFormat.KV_2LTD
        # Sender + cpu_offload: offload to CPU first  ->  RDMA from CPU
        # Otherwise (receiver, or sender without offload): allocate on NPU
        use_cpu = self.pd_config.buffer_device == "cpu" or (
            self.pd_config.role == "sender" and self.use_cpu_offload
        )
        alloc_type = "cpu" if use_cpu else "gpu"
        return self.memory_allocator.allocate(
            shapes, dtypes, fmt=fmt, allocator_type=alloc_type
        )

    def _lookup(self, key: CacheEngineKey, pin: bool = False) -> Optional[MemoryObj]:
        """Look up *key*, optionally pin it, and return the :class:`MemoryObj`.

        Consumed :class:`ProxyMemoryObj` instances are evicted from the
        store and treated as absent.

        Pinning is safe for both regular ``MemoryObj`` instances and proxies.
        ``ProxyMemoryObj`` tracks temporary lookup pins as logical references
        above its base transfer owner, so releasing a lookup pin does not
        release the shared transfer context.

        The caller **must** call ``ref_count_down()`` on every returned
        object when *pin* is ``True`` once the pin is no longer needed.
        """
        with self.data_lock:
            entry = self._pd_entries.get(key)
            if entry is None:
                return None

            mem_obj = entry.base_obj
            if isinstance(mem_obj, ProxyMemoryObj) and mem_obj.consumed:
                self._delete_pd_entry_locked(key, entry, release_obj=False)
                return None

            if pin:
                mem_obj.ref_count_up()
            return mem_obj

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if *key* exists in the receiver's data store.

        Overrides the base :meth:`PDBackend.contains` to evict consumed
        :class:`ProxyMemoryObj` instances whose remote buffer
        references are stale.
        """
        assert isinstance(key, CacheEngineKey)
        return self._lookup(key, pin=pin) is not None

    def _contains_and_pin(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Check if *key* exists, pin it, and return the object.

        Combines the existence check with an atomic ``ref_count_up()``
        under ``data_lock``, and returns the **object reference** so the
        caller can later call ``ref_count_down()`` to release the pin.

        Returns ``None`` when the key is absent or is a consumed proxy.
        """
        return self._lookup(key, pin=True)

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ):
        with self.data_lock:
            old_entry = self._pd_entries.get(key)
            if old_entry is not None and old_entry.owners:
                logger.warning(
                    "PD receiver overwriting key %s with active owners: %s",
                    key,
                    sorted(old_entry.owners),
                )
            self.data[key] = mem_obj
            self._pd_entries[key] = PDEntry(base_obj=mem_obj)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.data_lock:
            entry = self._pd_entries.get(key)
            assert entry is not None, f"Key {key} not found in local data."
            return entry.base_obj

    def batched_get_blocking_for_request(
        self,
        keys: list[CacheEngineKey],
        request_id: str,
    ) -> list[Optional[MemoryObj]]:
        memory_objs: list[Optional[MemoryObj]] = []
        with self.data_lock:
            for key in keys:
                entry = self._pd_entries.get(key)
                if entry is None:
                    logger.warning(
                        "PD request %s declared hit for missing key %s.",
                        request_id,
                        key,
                    )
                    memory_objs.append(None)
                    continue

                if request_id not in entry.owners:
                    logger.debug(
                        "PD request %s retrieves key %s without a prior lease; "
                        "registering a lazy lease.",
                        request_id,
                        key,
                    )
                    self._ensure_request_lease_locked(key, entry, request_id)

                base_obj = entry.base_obj
                if isinstance(base_obj, ProxyMemoryObj):
                    proxy = entry.proxy_leases.get(request_id)
                    if proxy is None or proxy.consumed:
                        logger.warning(
                            "PD request %s has no usable proxy lease for key %s.",
                            request_id,
                            key,
                        )
                        memory_objs.append(None)
                    else:
                        memory_objs.append(proxy)
                    continue

                base_obj.ref_count_up()
                memory_objs.append(base_obj)
        return memory_objs

    def batched_contains_and_lease(
        self,
        keys: list[CacheEngineKey],
        request_id: str,
    ) -> int:
        hit_chunks = 0
        with self.data_lock:
            for key in keys:
                entry = self._pd_entries.get(key)
                if entry is None:
                    break

                base_obj = entry.base_obj
                if isinstance(base_obj, ProxyMemoryObj) and base_obj.consumed:
                    self._delete_pd_entry_locked(key, entry, release_obj=False)
                    break

                self._ensure_request_lease_locked(key, entry, request_id)

                hit_chunks += 1
        return hit_chunks

    def _ensure_request_lease_locked(
        self,
        key: CacheEngineKey,
        entry: PDEntry,
        request_id: str,
    ) -> bool:
        """Acquire one key lease while ``data_lock`` is held."""
        if request_id in entry.owners:
            return False

        request_proxy = None
        base_obj = entry.base_obj
        if isinstance(base_obj, ProxyMemoryObj):
            request_proxy = base_obj.clone_for_request(request_id)
            acquire_request = getattr(
                base_obj.transfer_context,
                "acquire_request",
                None,
            )
            if callable(acquire_request):
                acquire_request(request_id)

        entry.owners.add(request_id)
        self._pd_request_keys.setdefault(request_id, set()).add(key)
        if request_proxy is not None:
            entry.proxy_leases[request_id] = request_proxy
        return True

    def _detach_request_lease_locked(
        self,
        key: CacheEngineKey,
        entry: PDEntry,
        request_id: str,
    ):
        """Detach owner metadata and return its proxy transfer context."""
        if request_id not in entry.owners:
            return None

        entry.owners.remove(request_id)
        owned_keys = self._pd_request_keys.get(request_id)
        if owned_keys is not None:
            owned_keys.discard(key)
            if not owned_keys:
                self._pd_request_keys.pop(request_id, None)

        proxy = entry.proxy_leases.pop(request_id, None)
        return proxy.transfer_context if proxy is not None else None

    @staticmethod
    def _release_context_leases(transfer_contexts, request_id: str) -> None:
        for transfer_context in transfer_contexts:
            release_request = getattr(
                transfer_context,
                "release_request",
                None,
            )
            if callable(release_request):
                try:
                    release_request(request_id)
                except Exception as e:
                    logger.error(
                        "Failed to release PD context lease for request %s: %s",
                        request_id,
                        e,
                    )

    def _refresh_handoff_deadline_locked(self, lease_id: str) -> None:
        self._pd_handoff_deadlines[lease_id] = (
            time.monotonic() + self._pd_handoff_lease_ttl
        )

    def put_with_handoff_lease(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
        handoff_id: str,
    ) -> tuple[bool, Optional[MemoryObj]]:
        """Atomically publish a received object with its temporary owner.

        Returns ``(True, None)`` when ``mem_obj`` was inserted. If another
        thread populated the key, its object is leased, pinned, and returned.
        """
        lease_id = make_pd_handoff_lease_id(handoff_id)
        with self.data_lock:
            existing = self._pd_entries.get(key)
            if existing is not None:
                base_obj = existing.base_obj
                if isinstance(base_obj, ProxyMemoryObj) and base_obj.consumed:
                    self._delete_pd_entry_locked(key, existing, release_obj=False)
                    existing = None

            if existing is not None:
                self._ensure_request_lease_locked(key, existing, lease_id)
                existing.base_obj.ref_count_up()
                self._refresh_handoff_deadline_locked(lease_id)
                return False, existing.base_obj

            entry = PDEntry(base_obj=mem_obj)
            self._ensure_request_lease_locked(key, entry, lease_id)
            self.data[key] = mem_obj
            self._pd_entries[key] = entry
            self._refresh_handoff_deadline_locked(lease_id)
            return True, None

    def promote_handoff_lease(self, handoff_id: str, request_id: str) -> int:
        """Replace a PullReady handoff owner with the real decoder request."""
        if not handoff_id or not request_id or request_id == "unspecified":
            return 0

        lease_id = make_pd_handoff_lease_id(handoff_id)
        acquired_keys: list[CacheEngineKey] = []
        request_contexts_to_rollback = []
        handoff_contexts_to_release = []
        claimed_keys = 0
        error: Optional[Exception] = None

        with self.data_lock:
            handoff_keys = list(self._pd_request_keys.get(lease_id, set()))
            if not handoff_keys:
                return 0

            try:
                # Acquire every real owner before releasing any synthetic
                # owner, so a shared transfer context never reaches zero.
                for key in handoff_keys:
                    entry = self._pd_entries.get(key)
                    if entry is None or lease_id not in entry.owners:
                        raise RuntimeError(
                            f"PD handoff {handoff_id} lost protected key {key}"
                        )
                    if self._ensure_request_lease_locked(key, entry, request_id):
                        acquired_keys.append(key)

                for key in handoff_keys:
                    entry = self._pd_entries.get(key)
                    assert entry is not None
                    context = self._detach_request_lease_locked(key, entry, lease_id)
                    if context is not None:
                        handoff_contexts_to_release.append(context)
                self._pd_handoff_deadlines.pop(lease_id, None)
                claimed_keys = len(handoff_keys)
            except Exception as exc:
                error = exc
                for key in acquired_keys:
                    entry = self._pd_entries.get(key)
                    if entry is None:
                        continue
                    context = self._detach_request_lease_locked(key, entry, request_id)
                    if context is not None:
                        request_contexts_to_rollback.append(context)

        self._release_context_leases(request_contexts_to_rollback, request_id)
        if error is not None:
            logger.error(
                "Failed to promote PD handoff %s to request %s: %s",
                handoff_id,
                request_id,
                error,
            )
            return 0

        self._release_context_leases(handoff_contexts_to_release, lease_id)
        logger.debug(
            "Promoted PD handoff %s to request %s for %d keys.",
            handoff_id,
            request_id,
            claimed_keys,
        )
        return claimed_keys

    def release_expired_handoff_leases(self) -> int:
        """Release PullReady handoffs whose decoder request never arrived."""
        now = time.monotonic()
        with self.data_lock:
            expired = [
                (lease_id, deadline)
                for lease_id, deadline in self._pd_handoff_deadlines.items()
                if deadline <= now
            ]

        released = 0
        for lease_id, deadline in expired:
            if self.release_request_lease(
                lease_id,
                expected_handoff_deadline=deadline,
            ):
                released += 1
                logger.warning(
                    "Expired unclaimed PD handoff lease: rank=%s lease=%s",
                    self.tp_rank,
                    lease_id,
                )
        return released

    def release_handoff_lease(self, handoff_id: str) -> bool:
        """Release one synthetic owner by its logical handoff ID."""
        return self.release_request_lease(make_pd_handoff_lease_id(handoff_id))

    def release_request_lease(
        self,
        request_id: str,
        expected_handoff_deadline: Optional[float] = None,
    ) -> bool:
        proxy_contexts = []
        with self.data_lock:
            if expected_handoff_deadline is not None:
                current_deadline = self._pd_handoff_deadlines.get(request_id)
                if (
                    current_deadline != expected_handoff_deadline
                    or current_deadline > time.monotonic()
                ):
                    return False
            self._pd_handoff_deadlines.pop(request_id, None)
            keys = self._pd_request_keys.pop(request_id, set())
            for key in list(keys):
                entry = self._pd_entries.get(key)
                if entry is None:
                    continue

                entry.owners.discard(request_id)
                proxy = entry.proxy_leases.pop(request_id, None)
                if proxy is not None:
                    proxy_contexts.append(proxy.transfer_context)

                entry.pending_delete = True
                if not entry.owners:
                    self._delete_pd_entry_locked(key, entry, release_obj=True)

        self._release_context_leases(proxy_contexts, request_id)
        return bool(keys)

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        with self.data_lock:
            entry = self._pd_entries.get(key)
            if entry is None:
                return False

            entry.pending_delete = True
            if entry.owners:
                return True

            self._delete_pd_entry_locked(key, entry, release_obj=True)
            return True

    def _delete_pd_entry_locked(
        self,
        key: CacheEngineKey,
        entry: PDEntry,
        release_obj: bool,
    ) -> None:
        self.data.pop(key, None)
        self._pd_entries.pop(key, None)
        for owner in list(entry.owners):
            owned_keys = self._pd_request_keys.get(owner)
            if owned_keys is not None:
                owned_keys.discard(key)
                if not owned_keys:
                    self._pd_request_keys.pop(owner, None)
                    self._pd_handoff_deadlines.pop(owner, None)
        entry.owners.clear()
        entry.proxy_leases.clear()

        if release_obj:
            try:
                entry.base_obj.ref_count_down()
            except Exception as e:
                logger.warning("Failed to release PD entry for key %s: %s", key, e)

    def _partition_keys(
        self,
        keys: list[str],
    ) -> tuple[list[int], list[MemoryObj], list[int]]:
        """Partition message keys into already-sent (pinned) and new indexes.

        Iterates over *keys*, calling :meth:`_contains_and_pin` for each.
        Keys that already exist in ``self.data`` are pinned and collected
        as "already sent"; the rest are collected as "new".

        Returns
        -------
        already_sent_indexes : list[int]
            Indexes (into *keys*) of chunks that were already present.
        already_sent_objs : list[MemoryObj]
            The pinned MemoryObj for each already-sent key.  The caller
            **must** call :meth:`_release_pinned` when done.
        new_indexes : list[int]
            Indexes (into *keys*) of chunks that need to be fetched.
        """
        already_sent_indexes: list[int] = []
        already_sent_objs: list[MemoryObj] = []
        new_indexes: list[int] = []
        for idx, key_str in enumerate(keys):
            key = CacheEngineKey.from_string(key_str)
            pinned = self._contains_and_pin(key)
            if pinned is not None:
                already_sent_indexes.append(idx)
                already_sent_objs.append(pinned)
            else:
                new_indexes.append(idx)
        return already_sent_indexes, already_sent_objs, new_indexes

    def _partition_keys_with_handoff(
        self,
        keys: list[str],
        handoff_id: str,
    ) -> tuple[list[int], list[MemoryObj], list[int]]:
        """Partition keys and reserve every existing hit before acknowledgement."""
        lease_id = make_pd_handoff_lease_id(handoff_id)
        already_sent_indexes: list[int] = []
        already_sent_objs: list[MemoryObj] = []
        new_indexes: list[int] = []

        try:
            with self.data_lock:
                for idx, key_str in enumerate(keys):
                    key = CacheEngineKey.from_string(key_str)
                    entry = self._pd_entries.get(key)
                    if entry is None:
                        new_indexes.append(idx)
                        continue

                    base_obj = entry.base_obj
                    if isinstance(base_obj, ProxyMemoryObj) and base_obj.consumed:
                        self._delete_pd_entry_locked(key, entry, release_obj=False)
                        new_indexes.append(idx)
                        continue

                    self._ensure_request_lease_locked(key, entry, lease_id)
                    base_obj.ref_count_up()
                    already_sent_indexes.append(idx)
                    already_sent_objs.append(base_obj)

                if already_sent_indexes:
                    self._refresh_handoff_deadline_locked(lease_id)
        except Exception:
            for mem_obj in already_sent_objs:
                mem_obj.ref_count_down()
            self.release_request_lease(lease_id)
            raise

        return already_sent_indexes, already_sent_objs, new_indexes
