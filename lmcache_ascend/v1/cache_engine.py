# SPDX-License-Identifier: Apache-2.0
"""
LMCacheEngine for Ascend NPU.

"""

# Standard
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
import queue
import threading
import time

# Third Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    convert_tokens_to_list,
)
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import TokenDatabase
import torch

logger = init_logger(__name__)

ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]


class ThreadSafeEventList:
    """queue.Queue-backed, list-compatible thread-safe buffer for
    ``CacheStoreEvent`` objects.

    Upstream ``LMCacheEngine`` treats ``self.kv_events`` as a list that
    callers ``.append(...)`` to and ``get_kv_events`` snapshots-and-resets.
    When ``store_async`` is active, appends happen on the background
    worker thread while drains happen on the main thread — a data race
    on a plain ``list``.  This wrapper funnels all appends through a
    ``queue.Queue`` so we inherit its producer/consumer thread safety,
    while preserving ``.append(...)`` / truthiness semantics that
    upstream code paths rely on.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[CacheStoreEvent]" = queue.Queue()

    def append(self, event: CacheStoreEvent) -> None:
        self._q.put(event)

    def __bool__(self) -> bool:
        return not self._q.empty()

    def __len__(self) -> int:
        return self._q.qsize()

    def drain(self) -> List[CacheStoreEvent]:
        out: List[CacheStoreEvent] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


class AscendLMCacheEngine(LMCacheEngine):
    """Ascend NPU variant of ``LMCacheEngine`` with an async store path."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        super().__init__(
            config,
            metadata,
            token_database,
            gpu_connector,
            broadcast_fn,
            broadcast_object_fn,
        )
        self.is_store_async = self.config.store_async
        self._store_queue_maxsize = max(0, int(self.config.store_async_max_queue_size))
        if self.is_store_async:
            self._store_queue: Optional[queue.Queue] = None
            self._store_worker_thread: Optional[threading.Thread] = None
            self._store_lock = threading.Lock()
            self._store_cv = threading.Condition(self._store_lock)

            # req_id -> number of in-flight background stores.  Entry
            # removed when count hits 0.
            self._pending_store_reqs: Dict[str, int] = {}
            # req_ids whose generation finished while stores were still
            # draining.  Re-checked on each ``get_finished_stores`` call.
            self._deferred_finished_req_ids: set = set()
            # req_ids already reported as finished_sending, to prevent
            # duplicate reports after the scheduler frees blocks.
            self._reported_finished_store_ids: set = set()

        # Serialize between store and lookup.
        self._engine_state_lock = threading.RLock()

        self._device_id: Optional[int] = None

        # None means auto-estimate in post_init based on available NPU memory.
        self._broadcast_shard_size = self.config.get_extra_config_value(
            "broadcast_shard_size", None
        )

        if self.kv_events_enabled and self.is_store_async:
            self.kv_events = ThreadSafeEventList()

    def _ensure_store_worker(self) -> None:
        if self._store_queue is not None:
            return
        self._store_queue = queue.Queue(maxsize=self._store_queue_maxsize)
        self._store_worker_thread = threading.Thread(
            target=self._store_worker_loop,
            daemon=True,
            name="lmcache-ascend-store-worker",
        )
        self._store_worker_thread.start()

    def post_init(self, **kwargs) -> None:
        super().post_init(**kwargs)
        if self.is_store_async:
            self._device_id = torch.npu.current_device()
            self._ensure_store_worker()
            queue_mode = "unbounded" if self._store_queue_maxsize == 0 else "bounded"
            logger.info(
                "Ascend async store queue initialized: mode=%s maxsize=%d",
                queue_mode,
                self._store_queue_maxsize,
            )

        # Override upstream broadcast_stream with a dedicated NPU stream
        # so broadcast and to_gpu can execute on separate streams.
        if self.save_only_first_rank:
            if not hasattr(self.gpu_connector, "load_stream"):
                raise RuntimeError(
                    "gpu_connector must have 'load_stream' when "
                    "save_only_first_rank is True"
                )
            # Auto-estimate or validate shard_size now that model weights
            # are loaded and NPU memory state is stable.
            if self._broadcast_shard_size is None:
                self._broadcast_shard_size = self._estimate_shard_size()
            else:
                estimated = self._estimate_shard_size()
                if self._broadcast_shard_size > estimated:
                    logger.warning(
                        "broadcast_shard_size=%d (user-set) may cause NPU OOM "
                        "(estimated safe max=%d). Suggestions: "
                        "(1) reduce broadcast_shard_size to <= %d; "
                        "(2) or increase --gpu-memory-utilization appropriately "
                        "(note: KV cache also uses free memory, do not set too high)",
                        self._broadcast_shard_size,
                        estimated,
                        estimated,
                    )
                    self._broadcast_shard_size = estimated

            self.broadcast_stream = torch.npu.Stream()
            logger.info(
                "Ascend broadcast stream initialized: shard_size=%d",
                self._broadcast_shard_size,
            )

    def _estimate_shard_size(self) -> int:
        """Estimate a safe ``broadcast_shard_size`` from available NPU memory.

        Uses 1/4 of the free memory as the pool budget so the remainder
        is reserved for KV-cache growth and temporary buffers.

        Returns the estimated shard size, clamped to [1, 16].
        """
        chunk_size = self.metadata.chunk_size
        shapes = self.metadata.get_shapes(chunk_size)
        dtypes = self.metadata.get_dtypes()
        per_chunk_bytes = get_size_bytes(shapes, dtypes)

        device = self.metadata.worker_id
        props = torch.npu.get_device_properties(device)
        total_mem = props.total_memory
        allocated = torch.npu.memory_allocated(device)
        available = total_mem - allocated
        pool_budget = available // 4

        if per_chunk_bytes <= 0:
            raise RuntimeError(
                f"Invalid per-chunk size {per_chunk_bytes} bytes; "
                "cannot estimate broadcast_shard_size. "
                "Check model config (shapes/dtypes)."
            )
        max_shard = int(pool_budget // (2 * per_chunk_bytes))
        if max_shard <= 0:
            raise RuntimeError(
                f"NPU free memory insufficient for broadcast pool "
                f"(available={available / 1024**3:.2f} GB, "
                f"pool_budget={pool_budget / 1024**3:.2f} GB, "
                f"per_chunk={per_chunk_bytes / 1024**2:.1f} MB; "
                f"need at least {2 * per_chunk_bytes / 1024**2:.1f} MB "
                f"pool budget for shard_size=1). "
                "Consider reducing --gpu-memory-utilization or chunk_size."
            )
        recommended = min(max_shard, 16)

        logger.info(
            "Estimated broadcast_shard_size=%d "
            "(per_chunk=%.1f MB, available=%.2f GB, "
            "pool_budget=%.2f GB)",
            recommended,
            per_chunk_bytes / 1024**2,
            available / 1024**3,
            pool_budget / 1024**3,
        )
        return recommended

    def _build_shard_plan(
        self,
        total: int,
        shard_size: int,
    ) -> List[Tuple[int, int]]:
        """Split ``range(total)`` into contiguous shards.

        Returns a list of ``(offset, count)`` tuples such that
        ``offset + count <= total`` and the shards tile the full range
        with at most ``shard_size`` elements each.  The last shard may
        be smaller than ``shard_size`` when ``total`` is not evenly
        divisible.

        :param total: Number of chunks to tile.
        :param shard_size: Maximum width of each shard.
        :return: List of ``(offset, count)`` pairs in increasing offset.
        """
        if total <= 0:
            return []
        step = max(1, shard_size)
        return [(i, min(step, total - i)) for i in range(0, total, step)]

    def _try_release_pending(
        self,
        pending: List[Tuple[List[MemoryObj], torch.npu.Event]],
    ) -> None:
        """Release ref-counts of shards whose ``to_gpu`` Event has fired."""
        remaining: List[Tuple[List[MemoryObj], torch.npu.Event]] = []
        for objs, ev in pending:
            if ev.query():
                for obj in objs:
                    obj.ref_count_down()
            else:
                remaining.append((objs, ev))
        pending[:] = remaining

    def _submit_togpu(
        self,
        ctx: Tuple[List[MemoryObj], List[int], List[int], "torch.npu.Event", int, int],
        load_stream: "torch.npu.Stream",
        pending: List[Tuple[List[MemoryObj], "torch.npu.Event"]],
        **kwargs,
    ) -> None:
        """Enqueue ctx's to_gpu on load_stream and release its slot."""
        objs, starts, ends, ev_bc, idx, slot = ctx
        load_stream.wait_event(ev_bc)
        with torch.npu.stream(load_stream):
            for obj, s, e in zip(objs, starts, ends, strict=False):
                self.gpu_connector.to_gpu(obj, s, e, **kwargs)
            ev_togpu = torch.npu.Event()
            ev_togpu.record()
        self._pool_scatter_ev[slot].record(load_stream)
        pending.append((objs, ev_togpu))
        self._try_release_pending(pending)
        logger.debug(
            "rank=%d shard[%d] cnt=%d submitted",
            self.metadata.worker_id,
            idx,
            len(objs),
        )

    # Ping-pong merged-buffer pool for batched broadcast.  Two slots
    # suffice: while slot A's to_gpu runs on load_stream, slot B
    # receives the next shard's broadcast on broadcast_stream.
    _merged_pool: List["torch.Tensor"] = []
    _pool_scatter_ev: List["torch.npu.Event"] = []
    _pool_bytes: int = 0

    def _ensure_merged_pool(self, max_bytes: int, device: str) -> bool:
        """Allocate the 2-slot ping-pong pool if not yet created or if a
        larger shard demands it.  Returns False on OOM."""
        if self._merged_pool and self._pool_bytes >= max_bytes:
            return True
        try:
            self._merged_pool = [
                torch.empty(max_bytes, dtype=torch.uint8, device=device),
                torch.empty(max_bytes, dtype=torch.uint8, device=device),
            ]
        except RuntimeError as e:
            logger.warning(
                "rank=%d NPU OOM allocating merged-buffer pool "
                "(%d bytes/slot); batched broadcast disabled: %s",
                self.metadata.worker_id,
                max_bytes,
                e,
            )
            self._merged_pool = []
            self._pool_bytes = 0
            return False
        self._pool_scatter_ev = [torch.npu.Event(), torch.npu.Event()]
        self._pool_bytes = max_bytes
        return True

    def _broadcast_metadata_table(
        self,
        reordered_chunks: list,
        shard_size: int,
        first_rank: int,
    ) -> Optional[Dict[str, Any]]:
        """Exchange the chunk metadata table + shard plan in one collective.

        Rank 0 builds the plan; other ranks receive it.  The plan
        contains ``total``, ``shard_plan``, ``meta`` (per-chunk
        metadata), ``shard_layouts`` (per-chunk byte offsets within
        each shard's merged buffer), and ``max_shard_bytes``.
        """
        if self.metadata.is_first_rank():
            total = len(reordered_chunks)
            meta_table: List[Tuple[int, int, Dict[str, Any]]] = [
                (
                    start,
                    end_pos,
                    mem_obj.metadata.to_dict(),
                )
                for (_, mem_obj, start, end_pos) in reordered_chunks
            ]
            shard_plan = self._build_shard_plan(total, shard_size)

            # Pre-compute per-chunk byte offsets within each shard's
            # merged buffer.
            #
            # ``shard_layouts`` parallels ``shard_plan``: each entry is
            # a list of (chunk_global_idx, byte_offset, byte_size)
            # tuples describing where the chunk lives inside the
            # shard's merged buffer.
            shard_layouts: List[List[Tuple[int, int, int]]] = []
            max_shard_bytes = 0
            for offset, count in shard_plan:
                layout: List[Tuple[int, int, int]] = []
                byte_off = 0
                for i in range(offset, offset + count):
                    # Use the logical data size (not phy_size, which
                    # may include alignment padding) so that the
                    # receiver can faithfully reconstruct the chunk
                    # tensor via dtype/shape views.
                    chunk_meta = MemoryObjMetadata.from_dict(meta_table[i][2])
                    chunk_bytes = chunk_meta.get_size()
                    layout.append((i, byte_off, chunk_bytes))
                    byte_off += chunk_bytes
                shard_layouts.append(layout)
                if byte_off > max_shard_bytes:
                    max_shard_bytes = byte_off

            plan: Optional[Dict[str, Any]] = {
                "total": total,
                "shard_plan": shard_plan,
                "meta": meta_table,
                "shard_layouts": shard_layouts,
                "max_shard_bytes": max_shard_bytes,
            }
        else:
            plan = None
        return self.broadcast_object_fn(plan, first_rank)

    def _pipelined_sharded_broadcast_and_load(
        self,
        reordered_chunks: list,
        ret_mask: torch.Tensor,
        **kwargs,
    ) -> None:
        """Shard-level pipelined broadcast + ``to_gpu`` on two NPU streams.

        Phase 1: rank 0 broadcasts the metadata table + shard plan in a
        single collective.  Phase 2: per-shard loop batch-broadcasts on
        ``broadcast_stream`` and defers ``to_gpu`` to ``load_stream``
        one shard behind, so the two overlap on the NPU.
        """
        shard_size = self._broadcast_shard_size
        first_rank = self.metadata.first_rank
        load_stream = self.gpu_connector.load_stream

        plan = self._broadcast_metadata_table(reordered_chunks, shard_size, first_rank)
        if plan is None or plan.get("total", 0) == 0:
            # Sender's CPU mem_objs won't enter the pipeline; release
            # them now to avoid leaking.
            if self.metadata.is_first_rank():
                for _, mem_obj, _, _ in reordered_chunks:
                    mem_obj.ref_count_down()
            return

        if self.metadata.is_first_rank():
            # sender
            self._pipeline_broadcast_and_load(
                plan,
                load_stream,
                reordered_chunks=reordered_chunks,
                **kwargs,
            )
        else:
            # receiver
            self._pipeline_broadcast_and_load(
                plan,
                load_stream,
                ret_mask=ret_mask,
                **kwargs,
            )

    def _fill_shard_sender(
        self,
        merged: "torch.Tensor",
        layout: List[Tuple[int, int, int]],
        meta_table: List[Tuple[int, int, Dict[str, Any]]],
        reordered_chunks: list,
    ) -> Tuple[List[TensorMemoryObj], List[int], List[int]]:
        """Sender: H2D-copy each chunk into ``merged`` and build views.

        Returns ``(objs, starts, ends)``.  Broadcast is issued by the
        caller after this returns.
        """
        objs, starts, ends = [], [], []
        for ci, byte_off, byte_size in layout:
            _, mem_obj, _, _ = reordered_chunks[ci]
            start, end_pos, _ = meta_table[ci]
            raw = mem_obj.raw_tensor
            if raw is None:
                raise ValueError(f"rank=0 chunk [{start}:{end_pos}] raw_tensor is None")

            dst = merged[byte_off : byte_off + byte_size]
            dst.copy_(
                raw.contiguous().view(torch.uint8).flatten(),
                non_blocking=True,
            )

            meta = mem_obj.metadata
            objs.append(
                TensorMemoryObj(
                    raw_data=dst.view(meta.dtype).view(meta.shape),
                    metadata=MemoryObjMetadata(
                        shape=meta.shape,
                        dtype=meta.dtype,
                        address=meta.address,
                        phy_size=meta.phy_size,
                        ref_count=1,
                        fmt=meta.fmt,
                        shapes=meta.shapes,
                        dtypes=meta.dtypes,
                    ),
                    parent_allocator=None,
                )
            )
            starts.append(start)
            ends.append(end_pos)
        return objs, starts, ends

    def _fill_shard_receiver(
        self,
        merged: "torch.Tensor",
        layout: List[Tuple[int, int, int]],
        meta_table: List[Tuple[int, int, Dict[str, Any]]],
        ret_mask: torch.Tensor,
    ) -> Tuple[List[TensorMemoryObj], List[int], List[int]]:
        """Receiver: split ``merged`` into per-chunk views.

        Broadcast must have already been issued into ``merged``.
        Returns ``(objs, starts, ends)`` and updates ``ret_mask``.
        """
        objs, starts, ends = [], [], []
        for ci, byte_off, byte_size in layout:
            start, end_pos, meta_dict = meta_table[ci]
            metadata = MemoryObjMetadata.from_dict(meta_dict)
            dst = merged[byte_off : byte_off + byte_size]
            objs.append(
                TensorMemoryObj(
                    raw_data=dst.view(metadata.dtype).view(metadata.shape),
                    metadata=metadata,
                    parent_allocator=None,
                )
            )
            starts.append(start)
            ends.append(end_pos)
            ret_mask[start:end_pos] = True
        return objs, starts, ends

    def _pipeline_broadcast_and_load(
        self,
        plan: Dict[str, Any],
        load_stream: "torch.npu.Stream",
        *,
        reordered_chunks: Optional[list] = None,
        ret_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Unified sharded broadcast + to_gpu pipeline for both ranks.

        For each shard, on ``broadcast_stream``: sender H2D-copies
        chunks into a ping-pong ``merged_buffer`` slot then broadcasts;
        receiver broadcasts into the slot then splits it back.  In both
        cases ``to_gpu[N]`` is deferred until ``broadcast[N+1]`` is
        enqueued so the two overlap on the NPU.
        """
        is_sender = reordered_chunks is not None
        meta_table = plan["meta"]
        shard_plan = plan["shard_plan"]
        shard_layouts = plan["shard_layouts"]
        device = f"npu:{self.metadata.worker_id}"
        if not self._ensure_merged_pool(plan["max_shard_bytes"], device):
            raise RuntimeError(
                f"Failed to allocate merged broadcast pool on {device} "
                f"({plan['max_shard_bytes']} bytes/slot). "
                "Consider reducing broadcast_shard_size."
            )

        pending: List[Tuple[List[MemoryObj], torch.npu.Event]] = []

        prev_ctx = None
        try:
            for shard_idx, _ in enumerate(shard_plan):
                layout = shard_layouts[shard_idx]
                slot = shard_idx % 2

                # Slot was last used by shard_idx-2; wait for its
                # to_gpu before overwriting.
                if shard_idx >= 2:
                    self.broadcast_stream.wait_event(self._pool_scatter_ev[slot])

                merged = self._merged_pool[slot]
                shard_bytes = sum(s for _, _, s in layout)

                with torch.npu.stream(self.broadcast_stream):
                    if is_sender:
                        objs, starts, ends = self._fill_shard_sender(
                            merged,
                            layout,
                            meta_table,
                            reordered_chunks,
                        )
                        self.broadcast_fn(
                            merged[:shard_bytes], self.metadata.first_rank
                        )
                    else:
                        self.broadcast_fn(
                            merged[:shard_bytes], self.metadata.first_rank
                        )
                        objs, starts, ends = self._fill_shard_receiver(
                            merged,
                            layout,
                            meta_table,
                            ret_mask,
                        )

                    ev_bc = torch.npu.Event()
                    ev_bc.record()

                # Now that broadcast[N+1] is enqueued, submit to_gpu[N].
                if prev_ctx is not None:
                    self._submit_togpu(prev_ctx, load_stream, pending, **kwargs)
                prev_ctx = (objs, starts, ends, ev_bc, shard_idx, slot)

            if prev_ctx is not None:
                self._submit_togpu(prev_ctx, load_stream, pending, **kwargs)

        finally:
            load_stream.synchronize()
            for objs, _ in pending:
                for obj in objs:
                    try:
                        obj.ref_count_down()
                    except Exception:
                        pass
            pending.clear()

            # Sender-only: release the original CPU mem_objs whose
            # raw_tensor was H2D-copied into the merged buffer.
            if is_sender:
                for _, mem_obj, _, _ in reordered_chunks:
                    try:
                        mem_obj.ref_count_down()
                    except Exception:
                        pass

    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        Overrides upstream to use sharded broadcast when
        ``broadcast_shard_size != 0``, reducing peak NPU memory usage by
        interleaving broadcast receive and ``batched_to_gpu`` per shard.

        :param tokens: The tokens of the corresponding KV caches.
        :param mask: The mask for the tokens. Should have the same length as
            tokens. The mask should ALWAYS be like FFFFFTTTTTTT, where True
            means the tokens needs to be matched, and the Falses will ALWAYS
            be at the PREFIX of the tensor.
        :param **kwargs: Forwarded to ``batched_to_gpu``. Should include KV
            cache specific information (e.g., paged KV buffer and the page
            tables).
        :return: Boolean mask indicating which tokens are retrieved. Same
            length as *tokens*. On CPU.
        :raises ValueError: If the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            with retrieve_stats.profile_process_tokens():
                if self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )

        # NOTE(niming) --- Sharded broadcast ---
        # The sharded pipeline handles both broadcast and to_gpu internally,
        # so we do not invoke batched_to_gpu below for either rank.
        # Rank 0's to_gpu reuses the NPU tensors produced by the broadcast
        # and non-rank-0's runs on the broadcast receive buffers, avoiding
        # a second CPU->NPU PCIe transfer.
        if self.save_only_first_rank:
            with retrieve_stats.profile_broadcast():
                self._pipelined_sharded_broadcast_and_load(
                    reordered_chunks, ret_mask, **kwargs
                )
        elif len(reordered_chunks) > 0:
            with retrieve_stats.profile_to_gpu():
                _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                self.gpu_connector.batched_to_gpu(
                    list(memory_objs), list(starts), list(ends), **kwargs
                )

        # --- Cleanup ---
        # When save_only_first_rank is set, the sharded-broadcast pipeline
        # takes ownership of the sender's CPU mem_objs (it releases them in
        # its finally block after the H2D copies complete).  So we skip the
        # ref_count_down here for the sender to avoid double-free.
        skip_refcnt = self.save_only_first_rank and self.metadata.is_first_rank()
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                if self.storage_manager is None:
                    raise ValueError("storage_manager is required for remove")
                self.storage_manager.remove(key, self.retrieve_locations)
                if self._is_sync_pd_backend() and not skip_refcnt:
                    memory_obj.ref_count_down()
            elif not self.async_loading and not skip_refcnt:
                memory_obj.ref_count_down()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(
            retrieve_stats,
            retrieved_tokens,
        )
        onload_time = retrieve_stats.time_to_retrieve()
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f gb, "
                "cost %.4f ms, throughput: %.4f GB/s;",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                onload_time * 1000,
                tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )
        return ret_mask

    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """Retrieve a KV cache layer, optionally honoring a per-call connector override.

        Upstream ``LMCacheEngine.retrieve_layer`` ignores any
        ``gpu_connector_override`` kwarg and always uses ``self.gpu_connector``.
        The hole connector's legacy fallback path (``LMCBlender`` with
        ``legacy_gpu_connector``) relies on the override to retrieve via the
        legacy connector when the engine's installed connector is the hole one.
        This Ascend-local override honors the override; non-hole callers that
        don't pass it see identical upstream behavior.
        """
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        assert self.storage_manager is not None
        gpu_connector = kwargs.pop("gpu_connector_override", None) or self.gpu_connector
        assert gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        location = None
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)

            if current_location := self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
            ):
                if location is None:
                    location = current_location
                else:
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )
            else:
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        if keys:
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

            get_generator = self.storage_manager.layerwise_batched_get(
                keys_layer_major,
                location=location,
            )

            assert_layerwise_gpu_connector(gpu_connector)

            mem_obj_consumer = gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)

            to_count_down = []
            for layer_id in range(self.num_layers):
                task = next(get_generator)

                assert task is not None

                if layer_id == 0:
                    yield torch.sum(ret_mask)
                else:
                    yield None

                mem_objs_layer = task.result()
                mem_obj_consumer.send(mem_objs_layer)
                to_count_down.extend(mem_objs_layer)

            for mem_obj in to_count_down:
                mem_obj.ref_count_down()
        else:
            for layer_id in range(self.num_layers):
                yield None

        yield None

        if keys:
            next(mem_obj_consumer)

            for mem_obj in to_count_down:
                if mem_obj.is_pinned:
                    mem_obj.unpin()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d out of total %d tokens",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
            )

        yield ret_mask

    def _store_worker_loop(self) -> None:
        if not self.is_store_async:
            return
        if self._device_id is not None:
            torch.npu.set_device(self._device_id)
        while True:
            work = self._store_queue.get()
            if work is None:  # poison pill
                self._store_queue.task_done()
                break

            (
                req_id,
                tokens,
                hashes,
                offsets,
                mask,
                num_to_store_tokens,
                kwargs,
            ) = work
            try:
                self._run_store_pipeline(
                    req_id, tokens, hashes, offsets, mask, num_to_store_tokens, kwargs
                )
            except Exception:
                logger.exception("Background store failed for req %s", req_id)
            finally:
                with self._store_lock:
                    cnt = self._pending_store_reqs.get(req_id, 1) - 1
                    if cnt <= 0:
                        self._pending_store_reqs.pop(req_id, None)
                    else:
                        self._pending_store_reqs[req_id] = cnt
                    logger.debug(
                        "Async store done for req %s; remaining=%d",
                        req_id,
                        max(cnt, 0),
                    )
                    self._store_cv.notify_all()
                self._store_queue.task_done()

    @torch.inference_mode()
    def _run_store_pipeline(
        self,
        req_id: str,
        tokens: Optional[Union[torch.Tensor, list]],
        hashes: Optional[List[int]],
        offsets: Optional[List[int]],
        mask: Optional[torch.Tensor],
        num_to_store_tokens: int,
        kwargs: dict,
    ) -> None:
        """Shared implementation for sync and async store.
        From upstream store function.
        """
        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                # Allocate the memory object
                num_tokens = end - start
                kv_shapes = self.metadata.get_shapes(num_tokens)
                kv_dtypes = self.metadata.get_dtypes()

                # TODO (Jiayi): should be batched in the future
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if memory_obj is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to store only "
                        f" {len(memory_objs)}"
                        " total chunks of KV cache."
                    )
                    break

                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                tot_kv_size += memory_obj.get_size()
                tot_token_num += num_tokens

                # Create KV event
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return

        put_submitted = False
        try:
            result = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )
            store_stats.from_gpu_time = result if isinstance(result, float) else 0.0

            with self._engine_state_lock:
                with store_stats.profile_put():
                    transfer_spec = kwargs.get("transfer_spec", None)
                    # TODO: we implicitly rely on batched_put to call ref_count_down
                    # this management should be done in a cleaner way
                    self.storage_manager.batched_put(
                        keys,
                        memory_objs,
                        transfer_spec=transfer_spec,
                        location=self.store_location,
                    )
                    put_submitted = True

                self.stats_monitor.on_store_finished(
                    store_stats,
                    tot_token_num,
                )
        except Exception:
            if not put_submitted:
                for mem_obj in memory_objs:
                    mem_obj.ref_count_down()
            raise

        tot_time = store_stats.time_to_store()

        # NOTE(#233): `from_gpu_time` includes a `wait_for_forward` stall since
        # PR #221 (slot_mapping/ordering async copy), so it is NOT pure device
        # copy time. Report the breakdown explicitly to avoid misleading
        # "offload_time" semantics; a dedicated device-copy metric should be
        # added at the connector layer (see Issue #233 fix plan).
        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_total_time: %.4f ms "
            "(process_tokens: %.4f ms, from_gpu: %.4f ms), "
            "put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.process_tokens_time * 1000,
            store_stats.from_gpu_time * 1000,
            store_stats.put_time * 1000,
        )

    def get_finished_stores(self, finished_req_ids: set) -> set:
        if not self.is_store_async:
            return None
        result: set = set()
        with self._store_lock:
            # Forget req_ids the scheduler no longer asks about.
            # This bounds the set to at most |finished_req_ids|.
            self._reported_finished_store_ids &= finished_req_ids

            for req_id in list(self._deferred_finished_req_ids):
                if req_id not in self._pending_store_reqs:
                    result.add(req_id)
                    self._deferred_finished_req_ids.discard(req_id)

            for req_id in finished_req_ids:
                if req_id in self._reported_finished_store_ids:
                    # Already reported — skip to avoid scheduler seeing
                    # a duplicate finished_sending for blocks it has
                    # already freed.
                    continue
                if req_id in self._pending_store_reqs:
                    self._deferred_finished_req_ids.add(req_id)
                else:
                    result.add(req_id)

            self._reported_finished_store_ids.update(result)
        return result

    def wait_for_pending_stores(self, req_ids: Iterable[str]) -> set[str]:
        """Wait until async stores for the given requests have drained.

        vLLM reports preempted request ids to workers before the next forward can
        overwrite their freed KV blocks.  If one of those ids still has a
        background store reading paged KV, drain it here to avoid store-after-free.
        """
        if not self.is_store_async:
            return set()

        req_id_set = set(req_ids)
        if not req_id_set:
            return set()

        with self._store_cv:
            pending_at_start = {
                req_id for req_id in req_id_set if req_id in self._pending_store_reqs
            }
            if not pending_at_start:
                return set()

            pending_counts = {
                req_id: self._pending_store_reqs[req_id] for req_id in pending_at_start
            }
            logger.info(
                "Waiting for pending async stores before preemption: "
                "req_ids=%s pending_counts=%s",
                sorted(pending_at_start),
                pending_counts,
            )
            start_time = time.monotonic()
            self._store_cv.wait_for(
                lambda: not any(
                    req_id in self._pending_store_reqs for req_id in req_id_set
                )
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Pending async stores drained before preemption: req_ids=%s "
            "elapsed=%.4f ms",
            sorted(pending_at_start),
            elapsed_ms,
        )
        return pending_at_start

    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and self.kv_events:
            if self.is_store_async:
                return self.kv_events.drain()
            events = list(self.kv_events)
            self.kv_events.clear()
            return events
        return []

    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        # Serialize against the store-worker thread's
        with self._engine_state_lock:
            return super().lookup(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                search_range=search_range,
                lookup_id=lookup_id,
                pin=pin,
                request_configs=request_configs,
            )

    def lookup_unpin(self, lookup_id: str) -> None:
        with self._engine_state_lock:
            super().lookup_unpin(lookup_id)

    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        assert self.storage_manager is not None

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="npu"
            )

        # lmcache-ascend start ---------------------
        if not self.is_store_async:
            self._run_store_pipeline(
                req_id, tokens, hashes, offsets, mask, num_to_store_tokens, kwargs
            )
        else:
            self._ensure_store_worker()
            with self._store_lock:
                self._pending_store_reqs[req_id] = (
                    self._pending_store_reqs.get(req_id, 0) + 1
                )
                pending = self._pending_store_reqs[req_id]

            enqueued = False
            try:
                self._store_queue.put(
                    (
                        req_id,
                        tokens,
                        hashes,
                        offsets,
                        mask,
                        num_to_store_tokens,
                        kwargs,
                    )
                )
                enqueued = True
            finally:
                if not enqueued:
                    with self._store_lock:
                        cnt = self._pending_store_reqs.get(req_id, 1) - 1
                        if cnt <= 0:
                            self._pending_store_reqs.pop(req_id, None)
                        else:
                            self._pending_store_reqs[req_id] = cnt
                        self._store_cv.notify_all()

            logger.debug(
                "Enqueued async store for req %s; pending=%d tokens=%d",
                req_id,
                pending,
                num_to_store_tokens,
            )
        # lmcache-ascend end ---------------------

    def close(self) -> None:
        """Stop the bg worker gracefully, then close the base engine."""
        # Push poison pill first so any in-flight work drains before
        # ``storage_manager.close()`` runs inside ``super().close()``.
        if self._store_queue is not None:
            try:
                # Bounded queues can be full here; avoid blocking forever
                # while still preferring graceful drain.
                while True:
                    try:
                        self._store_queue.put_nowait(None)
                        break
                    except queue.Full:
                        if (
                            self._store_worker_thread is None
                            or not self._store_worker_thread.is_alive()
                        ):
                            logger.warning(
                                "Ascend store queue is full and worker is not alive; "
                                "cannot enqueue shutdown signal cleanly."
                            )
                            break
                        time.sleep(0.01)
                if self._store_worker_thread is not None:
                    self._store_worker_thread.join(timeout=10)
                    if self._store_worker_thread.is_alive():
                        logger.warning(
                            "Ascend store worker did not stop within 10s; "
                            "proceeding with engine shutdown."
                        )
            except Exception:
                logger.exception("Error stopping Ascend store worker")

        super().close()
