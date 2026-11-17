# SPDX-License-Identifier: Apache-2.0
"""Explicitly assembled PD transfer engine for the MP+PD connector.

Key design decision 2: the PD leg is NOT a storage backend registered
through ``CreateStorageBackends`` — that path belongs to the in-process
line and would double-account storage with the MP server (storage is
owned by the standalone lmcache server in MP mode). Instead the
``AscendPDBackend`` sender/receiver capability surface is instantiated
directly here and driven from the connector hooks:

* P (``kv_producer``) side: ``wait_for_save`` copies the chunk-aligned
  token range computed this step out of vLLM's paged pool into sender
  MemObjs and submits them via ``batched_submit_put_task`` (push mode:
  remote allocation -> hccl/hixl batched_write; no proxy notification
  — an external router gates the decode request on the prefill
  response itself).
* D (``kv_consumer``) side: the receiver mixin accepts incoming writes
  into its NPU buffer autonomously (its alloc handler and channel
  threads); the connector walks the chunk keys to observe how much of a
  request's prefix has arrived and reports the number to the scheduler
  (worker -> scheduler metadata), where it lands in the ledger's
  ``pd_tokens`` — the observation feed for the Step-5 merge.

The delivery plan (发货单: receiver host/ports, req id) is self-built
on the producer from the static ``pd_peer_*`` config (1P1D static
topology); no per-request transfer params are required.

All PD configuration comes from ``kv_connector_extra_config`` using the
same key names as the in-process YAML (``pd_buffer_size``,
``pd_peer_host``, ...); ``pd_role`` is derived from ``kv_role``
(``kv_producer`` -> sender, ``kv_consumer`` -> receiver; ``kv_both``
needs the Step-6 "both" assembly and is rejected for now).
"""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional
import threading
import time

# Third Party
from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPWorkerMetadata
from lmcache.integration.vllm.utils import mla_enabled
from lmcache.logging import init_logger as _lmcache_init_logger
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import ChunkedTokenDatabase
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger
import torch

# First Party
from lmcache_ascend.v1.npu_connector.npu_connectors import VLLMPagedMemNPUConnectorV2
from lmcache_ascend.v1.storage_backend.pd.backend import AscendPDBackend
from lmcache_ascend.v1.storage_backend.utils import release_memory_objects

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)
_pd_logger = _lmcache_init_logger(__name__)


@dataclass
class PDTransferSpec:
    """Scheduler-side 发货单 bookkeeping for one request.

    Self-built on the producer from the static peer config
    (``pd_peer_host`` / ``pd_peer_init_port`` / ``pd_peer_alloc_port``)
    -- the router-orchestrated flow carries no per-request transfer
    params. ``sent_end`` tracks how much of the prompt has been handed
    to the worker for PD delivery.
    """

    req_id: str
    receiver_host: str
    receiver_init_port: list
    receiver_alloc_port: list
    num_prompt_tokens: int
    sent_end: int = 0


@dataclass
class PDSendInstruction:
    """Scheduler -> worker send order for one request, one step.

    Doubles as the sender's ``transfer_spec``: the attribute names below
    are exactly those read by
    ``AscendPDSenderMixin.batched_submit_put_task``.
    """

    req_id: str
    receiver_host: str
    receiver_init_port: list
    receiver_alloc_port: list
    token_ids: list
    block_ids: list
    block_size: int
    start: int
    end: int
    is_last_prefill: bool = False


@dataclass
class MPPDWorkerMetadata(LMCacheMPWorkerMetadata):
    """Worker -> scheduler report extended with PD receive observation.

    ``pd_received``: per-request arrived prefix lengths (Step 4
    observation, mostly logging/verification).

    ``pd_arrived_keys``: incrementally reported string forms of the
    chunk keys newly present in the receiver buffer (Step 5). The
    scheduler derives request chunk keys itself and walks this set, so
    the worker never needs to know about waiting requests.

    ``pd_loaded``: per-request prefix length that the worker has
    CONSECUTIVELY landed into vLLM's paged pool without error (the
    store-through feature stores exactly this range -- anything less
    would persist blocks whose D2D load failed). ``aggregate()`` keeps
    the max per request: TP workers land the same instruction in
    parallel, and the smallest safe value across ranks is bounded by
    each rank's own ledger-driven instruction anyway.
    """

    pd_received: dict[str, int] = field(default_factory=dict)
    pd_arrived_keys: list = field(default_factory=list)
    pd_loaded: dict[str, int] = field(default_factory=dict)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "MPPDWorkerMetadata":
        merged = super().aggregate(other)
        out = MPPDWorkerMetadata(
            completed_store_requests=merged.completed_store_requests,
            failed_store_requests=merged.failed_store_requests,
        )
        out.pd_received.update(self.pd_received)
        out.pd_received.update(getattr(other, "pd_received", {}))
        out.pd_arrived_keys = self.pd_arrived_keys + getattr(
            other, "pd_arrived_keys", []
        )
        for src in (self.pd_loaded, getattr(other, "pd_loaded", {})):
            for req_id, end in src.items():
                out.pd_loaded[req_id] = max(out.pd_loaded.get(req_id, 0), end)
        return out


@dataclass
class PDLoadInstruction:
    """Scheduler -> worker order to land arrived PD chunks into vLLM's
    paged pool (D side, Step 5). Ranges beyond the L1 reach only."""

    request_id: str
    token_ids: list
    block_ids: list
    block_size: int
    start: int
    end: int


def _key_str(key) -> str:
    return key if isinstance(key, str) else key.to_string()


def build_pd_chunk_db(vllm_config: "VllmConfig", extra: dict) -> tuple:
    """Chunk-key database shared by all three derivation points.

    The P worker (sender keys), the D worker (arrival bookkeeping), and
    the D scheduler (union-prefix walk) must derive byte-identical
    ``CacheEngineKey`` objects for the same tokens, so they all build
    the database through this one function from the same config
    surface. Returns ``(token_db, chunk_size)``.
    """
    model_cfg = vllm_config.model_config
    parallel_cfg = vllm_config.parallel_config
    chunk_size = int(extra.get("chunk_size", 256))
    try:
        # Third Party
        from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    except ImportError:
        # Third Party
        from vllm.utils import get_kv_cache_torch_dtype

    kv_dtype = get_kv_cache_torch_dtype(
        vllm_config.cache_config.cache_dtype, model_cfg.dtype
    )
    use_mla = mla_enabled(model_cfg)
    db_config = SimpleNamespace(
        chunk_size=chunk_size,
        save_unfull_chunk=bool(extra.get("save_unfull_chunk", True)),
        remote_url=None,
        enable_pd=True,
        pre_caching_hash_algorithm="builtin",
        get_extra_config_value=lambda key, default=None: extra.get(
            key, default
        ),
    )
    db_metadata = SimpleNamespace(
        model_name=model_cfg.model,
        world_size=parallel_cfg.world_size,
        worker_id=parallel_cfg.rank,
        kv_dtype=kv_dtype,
        use_mla=use_mla,
    )
    return ChunkedTokenDatabase(db_config, db_metadata), chunk_size


class L1QueryOracle:
    """Receiver-side batched "is this chunk in L1" probe.

    Answers, for a token range of one request, which LMCache server
    chunks (in the server's own chunk size / hash space) are already
    resident in the decoder's local L1. Used by the PD alloc handler to
    skip re-transmission of chunks the decoder can serve from its own
    cache: the alloc thread and the scheduler's lookup pipeline query
    the same server, so "alloc says present" implies "scheduler lookup
    hits" and the ledger marks the range L1 -- no P-not-sent /
    D-not-hit hole exists.

    Transport: the stock v0.5.6 MP server RPCs ``p2p_lookup_and_lock``
    -> ``p2p_query_lookup_results`` -> ``p2p_unlock_objects``
    (P2PController is registered unconditionally; ``skip_l2`` makes the
    lookup a pure L1-residency check that completes immediately, gaps
    allowed, per key). Keys are derived with the fleet's TokenHasher
    exactly the way the server derives them from a store's token ids,
    so a mismatched hash/chunk configuration degrades to "not present"
    (the sender re-transmits) rather than misbehaving.

    Every failure path fails open: an oracle that cannot answer is an
    oracle that says "no".
    """

    def __init__(
        self,
        server_url: str,
        zmq_context: Any,
        *,
        model_name: str,
        kv_rank: int,
        chunk_size: int,
        hash_algorithm: str,
        kv_shape: Any,
        kv_dtype: Any,
        cache_salt: str = "",
        timeout: float = 2.0,
    ) -> None:
        # Third Party
        from lmcache.v1.multiprocess.token_hasher import TokenHasher

        self._server_url = server_url
        self._zmq_context = zmq_context
        self._model_name = model_name
        self._kv_rank = kv_rank
        self._cache_salt = cache_salt
        self._timeout = timeout
        self._hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self._kv_shape = kv_shape
        self._kv_dtype = kv_dtype
        self._req_client = None
        # The oracle runs on the receiver's alloc REP thread, but keep a
        # lock anyway: the req_client's ZMQ socket must never be shared
        # concurrently, and close() may come from another thread.
        self._lock = threading.Lock()
        self.chunk_size = chunk_size
        logger.info(
            "L1 query oracle ready: server=%s chunk=%d hash=%s",
            server_url,
            chunk_size,
            hash_algorithm,
        )

    # ------------------------------------------------------------------

    def _client(self):
        if self._req_client is None:
            # Third Party
            from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

            self._req_client = RequestClientFactory.create(
                self._server_url, context=self._zmq_context
            )
        return self._req_client

    def present_server_chunks(
        self, token_ids: list, start: int, end: int
    ) -> frozenset:
        """Indexes of server chunks covering [start, end) that are
        L1-resident (absolute chunk indexes from token 0)."""
        cs = self.chunk_size
        first = start // cs
        # A trailing partial chunk can never be an L1 hit (the server
        # only serves complete chunks), so floor the end.
        last = end // cs
        if last <= first or not token_ids:
            return frozenset()
        try:
            with self._lock:
                return self._query_locked(token_ids, first, last)
        except Exception as e:  # noqa: BLE001 - fail open by contract
            logger.debug("L1 oracle: query failed, treating as miss (%r)", e)
            return frozenset()

    def _query_locked(
        self, token_ids: list, first: int, last: int
    ) -> frozenset:
        # Third Party
        from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey

        hashes = self._hasher.compute_chunk_hashes(
            token_ids, start=first * self.chunk_size, end=last * self.chunk_size
        )
        if not hashes:
            return frozenset()
        keys = [
            ObjectKey(
                chunk_hash=chunk_hash,
                model_name=self._model_name,
                kv_rank=self._kv_rank,
                object_group_id=0,
                cache_salt=self._cache_salt,
            )
            for chunk_hash in hashes
        ]
        client = self._client()
        layout = {
            0: MemoryLayoutDesc(
                shapes=[self._kv_shape], dtypes=[self._kv_dtype]
            )
        }
        task_id = client.p2p_lookup_and_lock(keys, layout).result(
            timeout=self._timeout
        )
        # skip_l2 lookups complete inline; the poll loop only guards a
        # busy server. The job is single-consumption: stop at the first
        # non-None reply.
        addresses = None
        deadline = time.monotonic() + self._timeout
        while addresses is None and time.monotonic() < deadline:
            addresses = client.p2p_query_lookup_results(task_id).result(
                timeout=self._timeout
            )
            if addresses is None:
                time.sleep(0.001)
        if addresses is None:
            return frozenset()
        found = {
            i
            for i, addr in enumerate(addresses)
            if addr is not None and addr.is_valid()
        }
        if found:
            try:
                client.p2p_unlock_objects(
                    [keys[i] for i in sorted(found)]
                ).result(timeout=self._timeout)
            except Exception:  # noqa: BLE001 - locks age out server-side
                logger.debug("L1 oracle: unlock best-effort failed")
        return frozenset(first + i for i in found)

    def close(self) -> None:
        with self._lock:
            client, self._req_client = self._req_client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


class PDEngine:
    """Explicit assembly of the AscendPDBackend capability surface.

    Worker-role only. Constructed lazily in ``register_kv_caches`` (where
    the model config is resolvable) and never through
    ``CreateStorageBackends``.
    """

    VALID_SAVE_POLICIES = ("both", "pd_only", "l1_only")

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        self.vllm_config = vllm_config
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self._extra: dict = kv_transfer_config.kv_connector_extra_config or {}
        kv_role = kv_transfer_config.kv_role
        self.pd_role = {
            "kv_producer": "sender",
            "kv_consumer": "receiver",
        }.get(kv_role)
        self.chunk_size = int(self._extra.get("chunk_size", 256))
        self.save_unfull_chunk = bool(self._extra.get("save_unfull_chunk", True))
        self.save_policy = str(self._extra.get("pd_save_policy", "both")).lower()
        if self.save_policy not in self.VALID_SAVE_POLICIES:
            raise ValueError(
                f"pd_save_policy must be one of {self.VALID_SAVE_POLICIES}, "
                f"got {self._extra.get('pd_save_policy')!r}"
            )
        self._config = self._build_config_shim()
        self.backend: Optional[AscendPDBackend] = None
        self.metadata: Optional[LMCacheMetadata] = None
        self.token_db: Optional[ChunkedTokenDatabase] = None
        self.npu_connector: Optional[Any] = None
        self.fmt: Optional[Any] = None
        self.assembled = False
        self._l1_oracle: Optional[L1QueryOracle] = None

    @classmethod
    def maybe_create(
        cls, vllm_config: "VllmConfig", role: KVConnectorRole
    ) -> Optional["PDEngine"]:
        """Return an engine for worker roles when PD is configured."""
        kv_transfer_config = vllm_config.kv_transfer_config
        extra = (kv_transfer_config.kv_connector_extra_config or {}) if (
            kv_transfer_config is not None
        ) else {}
        if "pd_buffer_size" not in extra:
            return None
        if role != KVConnectorRole.WORKER:
            return None
        engine = cls(vllm_config, role)
        if engine.pd_role is None:
            logger.warning(
                "PD leg requested (pd_buffer_size set) but kv_role=%s is "
                "not supported yet (needs kv_producer/kv_consumer; "
                "kv_both arrives with Step 6); PD leg disabled.",
                kv_transfer_config.kv_role,
            )
            return None
        return engine

    def _build_config_shim(self) -> SimpleNamespace:
        """Attribute surface read by AscendPDBackend / PDConfig / the
        chunk token database, filled from kv_connector_extra_config."""
        extra = self._extra

        def get_extra_config_value(key, default=None):
            return extra.get(key, default)

        default_peer_host = None
        if self.pd_role == "sender" and not extra.get(
            "pd_pull_mode", False
        ):
            default_peer_host = "localhost"

        shim = SimpleNamespace(
            pd_role=self.pd_role,
            pd_buffer_size=int(extra["pd_buffer_size"]),
            pd_buffer_device=extra.get("pd_buffer_device", "npu"),
            pd_peer_host=extra.get("pd_peer_host", default_peer_host),
            pd_peer_init_port=extra.get("pd_peer_init_port"),
            pd_peer_alloc_port=extra.get("pd_peer_alloc_port"),
            pd_peer_query_port=extra.get("pd_peer_query_port"),
            # Inert None values only: upstream v0.5.6's
            # PDConfig.from_cache_engine_config reads these attributes
            # unconditionally. The skip flag below keeps the base
            # class from connecting any proxy (router-orchestrated
            # deployment; no disagg proxy exists).
            pd_proxy_host=None,
            pd_proxy_port=None,
            pd_skip_proxy_notification=True,
            pd_use_cpu_offload=bool(extra.get("pd_use_cpu_offload", False)),
            pd_pull_mode=bool(extra.get("pd_pull_mode", False)),
            pd_delay_pull=bool(extra.get("pd_delay_pull", False)),
            pd_alloc_fail_backoff_ttl=float(
                extra.get("pd_alloc_fail_backoff_ttl", 2.0)
            ),
            pd_pull_done_port=extra.get("pd_pull_done_port"),
            pd_cpu_buffer_size=extra.get("pd_cpu_buffer_size"),
            transfer_channel=extra.get("transfer_channel", "hccl"),
            chunk_size=self.chunk_size,
            save_unfull_chunk=self.save_unfull_chunk,
            remote_url=None,
            enable_pd=True,
            pre_caching_hash_algorithm="builtin",
        )
        shim.get_extra_config_value = get_extra_config_value
        return shim

    def assemble(self, kv_caches: dict) -> None:
        """Build the backend, chunk key database, and NPU copy connector."""
        if self.assembled:
            return
        vllm_config = self.vllm_config
        model_cfg = vllm_config.model_config
        parallel_cfg = vllm_config.parallel_config
        try:
            # Third Party
            from vllm.utils.torch_utils import get_kv_cache_torch_dtype
        except ImportError:
            # Third Party
            from vllm.utils import get_kv_cache_torch_dtype

        kv_dtype = get_kv_cache_torch_dtype(
            vllm_config.cache_config.cache_dtype, model_cfg.dtype
        )
        num_layers = model_cfg.get_num_layers(parallel_cfg)
        num_kv_heads = model_cfg.get_num_kv_heads(parallel_cfg)
        head_size = model_cfg.get_head_size()
        use_mla = mla_enabled(model_cfg)
        kv_shape = (
            num_layers,
            1 if use_mla else 2,
            self.chunk_size,
            num_kv_heads,
            head_size,
        )
        metadata = LMCacheMetadata(
            model_name=model_cfg.model,
            world_size=parallel_cfg.world_size,
            local_world_size=parallel_cfg.world_size,
            worker_id=parallel_cfg.rank,
            local_worker_id=parallel_cfg.rank,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
            role=self.pd_role,
            served_model_name=model_cfg.served_model_name,
            engine_id=getattr(
                vllm_config.kv_transfer_config, "engine_id", None
            ),
            kv_connector_extra_config=self._extra,
        )
        self.metadata = metadata
        self.backend = AscendPDBackend(self._config, metadata)
        if self._l1_oracle is not None:
            self.backend.l1_oracle = self._l1_oracle
        self.token_db, self.chunk_size = build_pd_chunk_db(
            vllm_config, self._extra
        )
        self.save_unfull_chunk = bool(
            self._extra.get("save_unfull_chunk", True)
        )
        self.fmt = getattr(self.backend, "_fmt", None)
        self._reported_keys: set = set()
        device = torch.device(
            f"npu:{parallel_cfg.rank % torch.npu.device_count()}"
        )
        self.npu_connector = VLLMPagedMemNPUConnectorV2.from_metadata(
            metadata, False, device
        )
        self.assembled = True
        logger.info(
            "PD engine assembled: role=%s chunk=%d channel=%s",
            self.pd_role,
            self.chunk_size,
            self._config.transfer_channel,
        )

    # ------------------------------------------------------------------
    # Sender (P side)
    # ------------------------------------------------------------------

    def execute_send(
        self, instr: PDSendInstruction, kvcaches: dict
    ) -> None:
        """Copy [instr.start, instr.end) out of vLLM's paged pool and
        submit the chunks to the receiver named on the instruction."""
        if self.backend is None or self.token_db is None:
            logger.error("PD engine not assembled; dropping send for %s", instr.req_id)
            return
        keys: list = []
        starts: list = []
        ends: list = []
        for start, end, key in self.token_db.process_tokens(
            tokens=instr.token_ids
        ):
            if start >= instr.end:
                break
            if end <= instr.start:
                continue
            keys.append(key)
            starts.append(max(start, instr.start))
            ends.append(min(end, instr.end))
        if not keys:
            return

        objs: list = []
        for s, e in zip(starts, ends, strict=False):
            shape = list(self.metadata.kv_shape)
            token_dim = self.fmt.token_dim() if self.fmt is not None else 2
            shape[token_dim] = self.chunk_size if e - s >= self.chunk_size else e - s
            obj = self.backend.allocate(
                torch.Size(shape), self.metadata.kv_dtype, self.fmt
            )
            if obj is None:
                logger.error(
                    "PD send: allocation failed for req %s chunk [%d,%d); "
                    "sending the %d chunks allocated so far",
                    instr.req_id,
                    s,
                    e,
                    len(objs),
                )
                break
            objs.append(obj)
        if not objs:
            return
        if len(objs) < len(keys):
            keys = keys[: len(objs)]
            starts = starts[: len(objs)]
            ends = ends[: len(objs)]

        slots = []
        block_ids = instr.block_ids
        block_size = instr.block_size
        for pos in range(instr.end):
            slots.append(
                block_ids[pos // block_size] * block_size + pos % block_size
            )
        slot_mapping = torch.tensor(slots, dtype=torch.long)
        try:
            slot_mapping = slot_mapping.to("npu", non_blocking=True)
        except Exception:
            pass

        result = self.npu_connector.batched_from_gpu(
            objs,
            starts,
            ends,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
        )
        if isinstance(result, float):
            logger.debug(
                "PD copy-out for req %s: %d chunks in %.3fs",
                instr.req_id,
                len(objs),
                result,
            )
        # The upstream batched_submit_put_task contract has the CALLER
        # release the allocation reference when the call returns (the
        # storage_manager does it on the in-process path). PDEngine
        # bypasses the storage manager, so it must act as that caller:
        # the push sub-method's own up/down pair is net zero, and this
        # final release retires the allocate() reference on every path
        # (sent, already-sent, alloc-failed, backoff-skip, exception).
        try:
            self.backend.batched_submit_put_task(
                keys, objs, transfer_spec=instr
            )
        finally:
            release_memory_objects(objs)
        logger.info(
            "PD SEND req=%s range=[%d,%d) chunks=%d tokens=%d",
            instr.req_id,
            instr.start,
            instr.end,
            len(objs),
            instr.end - instr.start,
        )

    # ------------------------------------------------------------------
    # Receiver (D side)
    # ------------------------------------------------------------------

    def received_prefix_tokens(self, token_ids: list) -> int:
        """Chunk-aligned prefix length of ``token_ids`` whose KV has
        already arrived in the receiver buffer."""
        if self.backend is None or self.token_db is None:
            return 0
        total = 0
        for start, end, key in self.token_db.process_tokens(tokens=token_ids):
            if not self.backend.contains(key):
                break
            total = end
        return total

    def expected_prefix_tokens(self, num_tokens: int) -> int:
        """Prefix length the receiver expects once every sent chunk of a
        ``num_tokens`` prompt has arrived."""
        if self.save_unfull_chunk:
            return num_tokens
        return num_tokens // self.chunk_size * self.chunk_size

    def drain_arrived_keys(self) -> list:
        """String forms of chunk keys newly present since the last call.

        Content-addressed, so the scheduler can map them onto waiting
        requests without the worker knowing about those requests.
        """
        if self.backend is None:
            return []
        current = {_key_str(k) for k in self.backend.data.keys()}
        new = current - self._reported_keys
        self._reported_keys |= new
        return sorted(new)

    def execute_pd_load(self, instr: PDLoadInstruction, kvcaches: dict):
        """Copy arrived PD chunks in [instr.start, instr.end) from the
        receiver buffer into vLLM's paged pool (D2D).

        Chunks whose key is missing (e.g. evicted after the report)
        are not loaded; the block ids covering them are returned so the
        connector can surface load errors and vLLM recomputes them.

        Returns ``(failed_blocks, landed_end)`` where ``landed_end`` is
        the end of the leading run of chunks that landed successfully
        (starting at ``instr.start``). Store-through persists exactly
        ``[0, landed_end)`` -- a gap must stop the storable prefix even
        when chunks beyond it load fine, or blocks holding unloaded
        (garbage) data would be persisted to L1.
        """
        if self.backend is None or self.token_db is None:
            logger.error(
                "PD engine not assembled; dropping load for %s",
                instr.request_id,
            )
            return list(instr.block_ids), instr.start
        keys: list = []
        starts: list = []
        ends: list = []
        for start, end, key in self.token_db.process_tokens(
            tokens=instr.token_ids
        ):
            if start >= instr.end:
                break
            if end <= instr.start:
                continue
            keys.append(key)
            starts.append(max(start, instr.start))
            ends.append(min(end, instr.end))
        objs: list = []
        copy_starts: list = []
        copy_ends: list = []
        failed_blocks: set = set()
        landed_end = instr.start
        for key, s, e in zip(keys, starts, ends, strict=False):
            obj = self.backend._lookup(key, pin=True)
            if obj is None:
                logger.warning(
                    "PD load: chunk [%d,%d) of req %s missing at load "
                    "time; vLLM will recompute it",
                    s,
                    e,
                    instr.request_id,
                )
                first = max(s, instr.start) // instr.block_size
                last = (min(e, instr.end) - 1) // instr.block_size
                for bi in range(first, last + 1):
                    if 0 <= bi < len(instr.block_ids):
                        failed_blocks.add(instr.block_ids[bi])
                continue
            if s == landed_end:
                landed_end = e
            objs.append(obj)
            copy_starts.append(s)
            copy_ends.append(e)
        if objs:
            slots = []
            for pos in range(instr.end):
                slots.append(
                    instr.block_ids[pos // instr.block_size]
                    * instr.block_size
                    + pos % instr.block_size
                )
            slot_mapping = torch.tensor(slots, dtype=torch.long)
            try:
                slot_mapping = slot_mapping.to("npu", non_blocking=True)
            except Exception:
                pass
            result = self.npu_connector.batched_to_gpu(
                objs,
                copy_starts,
                copy_ends,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
            )
            if result is not None:
                for _ in result:
                    pass
            try:
                torch.npu.synchronize()
            except Exception:
                pass
            for obj in objs:
                obj.ref_count_down()
        logger.info(
            "PD LOAD req=%s landed=[%d,%d) landed_tokens=%d "
            "chunks_loaded=%d chunks_failed=%d",
            instr.request_id,
            instr.start,
            landed_end,
            landed_end - instr.start,
            len(objs),
            len(failed_blocks) > 0 and 1 or 0,
        )
        return sorted(failed_blocks), landed_end

    def attach_l1_oracle(self, oracle: Optional["L1QueryOracle"]) -> None:
        """Inject the L1 residency oracle (receiver only).

        Called from the connector's ``register_kv_caches`` once the MP
        worker adapter exists; the oracle is forwarded to the assembled
        backend (or kept pending until ``assemble`` runs).
        """
        self._l1_oracle = oracle
        if self.backend is not None:
            self.backend.l1_oracle = oracle

    def close(self) -> None:
        """Best-effort teardown of the assembled backend."""
        oracle = getattr(self, "_l1_oracle", None)
        if oracle is not None:
            oracle.close()
        backend = self.backend
        if backend is None:
            return
        try:
            backend.running = False
            for sock in getattr(backend, "side_channels", []) or []:
                try:
                    sock.close()
                except Exception:
                    pass
            for thread in getattr(backend, "running_threads", []) or []:
                thread.join(timeout=2.0)
            channel = getattr(backend, "transfer_channel", None)
            channel_close = getattr(channel, "close", None)
            if channel_close is not None:
                channel_close()
        except Exception:
            _pd_logger.exception("PD engine close failed")
