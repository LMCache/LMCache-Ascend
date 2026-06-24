# SPDX-License-Identifier: Apache-2.0
"""
LMCacheEngine for Ascend NPU.

"""

# Standard
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import queue
import threading
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    convert_tokens_to_list,
)
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import TokenDatabase
import torch

logger = init_logger(__name__)


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

        self._broadcast_shard_size = self.config.get_extra_config_value(
            "broadcast_shard_size", 16
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
        if self.save_only_first_rank and hasattr(self.gpu_connector, "load_stream"):
            self.broadcast_stream = torch.npu.Stream()
            logger.info(
                "Ascend broadcast stream initialized: shard_size=%d",
                self._broadcast_shard_size,
            )

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
        return [
            (i, min(step, total - i)) for i in range(0, total, step)
        ]

    def _alloc_broadcast_buffer(
        self,
        metadata: MemoryObjMetadata,
        local_rank: int,
        first_rank: int,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Allocate NPU broadcast receive buffer, with CPU fallback on OOM."""
        try:
            return torch.empty(
                torch.Size([metadata.get_size()]),
                dtype=torch.uint8,
                device=f"npu:{local_rank}",
            )
        except RuntimeError as e:
            logger.warning(
                "rank=%d NPU OOM allocating broadcast buffer for chunk "
                "[%d:%d] (size=%d bytes), falling back to CPU pinned memory. "
                "This chunk will be treated as cache miss. Error: %s",
                self.metadata.worker_id,
                start,
                end,
                metadata.get_size(),
                e,
            )
            return torch.empty(
                torch.Size([metadata.get_size()]),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )

    def _try_release_pending(
        self,
        pending: List[Tuple[List[MemoryObj], torch.npu.Event]],
    ) -> None:
        """Release NPU tensors of shards whose ``to_gpu`` has already finished.

        Iterates through ``pending`` and, for any shard whose completion
        ``Event`` has been recorded on ``load_stream``, calls
        ``ref_count_down`` on each ``MemoryObj``.  The NPU tensor's lifetime
        is bound to the ``MemoryObj``, so the explicit ``ref_count_down``
        replaces the previous reliance on Python garbage collection.

        ``Event.query()`` is a non-blocking poll, so this function does not
        stall the CPU dispatcher.
        """
        remaining: List[Tuple[List[MemoryObj], torch.npu.Event]] = []
        for objs, ev in pending:
            if ev.query():
                for obj in objs:
                    obj.ref_count_down()
            else:
                remaining.append((objs, ev))
        pending[:] = remaining

    def _broadcast_metadata_table(
        self,
        reordered_chunks: list,
        shard_size: int,
        first_rank: int,
    ) -> Optional[Dict[str, Any]]:
        """Exchange the full chunk metadata table in a single collective.

        Builds (on rank 0) or receives (on other ranks) a dict containing:

        * ``total``: number of chunks to broadcast.
        * ``shard_plan``: list of ``(offset, count)`` tuples tiling the
          range ``[0, total)`` with at most ``shard_size`` chunks each.
        * ``meta``: list of ``(start, end, meta_dict)`` tuples, one per
          chunk, where ``meta_dict`` is the serialized
          :class:`MemoryObjMetadata`.

        This collapses what used to be ``(1 + N)`` per-shard collective
        barriers into a single barrier, so the Phase 3 pipeline loop can
        run with no further ``broadcast_object_fn`` calls and the CPU
        dispatcher stays ahead of NPU execution.

        :param reordered_chunks: Sender-side chunks (ignored on
            non-first-rank workers, where ``reordered_chunks`` is empty).
        :param shard_size: Maximum width of each shard.
        :param first_rank: Source rank for the broadcast.
        :return: The exchanged plan dict. ``None`` if the broadcast
            itself returned ``None`` (defensive — should not happen in
            normal operation).
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
            plan: Optional[Dict[str, Any]] = {
                "total": total,
                "shard_plan": shard_plan,
                "meta": meta_table,
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

        Phase 1 (``_broadcast_metadata_table``): rank 0 broadcasts the
        full chunk metadata table + shard plan in a single collective.
        All ranks now hold the same plan with no further object-level
        barriers needed.

        Phase 3 (``_pipeline_sender`` / ``_pipeline_receiver``): the
        per-shard loop submits broadcast work to ``broadcast_stream``
        and ``to_gpu`` work to ``load_stream`` with a single recorded
        ``Event`` per shard providing the cross-stream dependency.
        Receiver-side NPU receive buffers are allocated per shard
        (rather than up front for the whole plan) so that
        ``_try_release_pending`` can reclaim each shard's buffers as
        soon as its ``to_gpu`` event fires, keeping peak NPU memory
        bounded at roughly ``2 * shard_size * chunk_size``.

        The CPU dispatcher returns to the next shard immediately after
        recording the event, so broadcast and to_gpu overlap on the NPU.

        :param reordered_chunks: Sender-side chunk list (empty on
            non-first-rank workers).
        :param ret_mask: Boolean mask updated by the receiver to mark
            retrieved token positions.
        :param kwargs: Forwarded to ``gpu_connector.to_gpu``.
        """
        shard_size = self._broadcast_shard_size
        first_rank = self.metadata.first_rank
        load_stream = self.gpu_connector.load_stream

        plan = self._broadcast_metadata_table(
            reordered_chunks, shard_size, first_rank
        )
        if plan is None or plan.get("total", 0) == 0:
            return

        if self.metadata.is_first_rank():
            self._pipeline_sender(
                plan,
                reordered_chunks,
                load_stream,
                **kwargs,
            )
        else:
            self._pipeline_receiver(
                plan,
                ret_mask,
                load_stream,
                **kwargs,
            )

    def _pipeline_sender(
        self,
        plan: Dict[str, Any],
        reordered_chunks: list,
        load_stream: "torch.npu.Stream",
        **kwargs,
    ) -> None:
        """Rank 0 side of the sharded broadcast pipeline (Phase 3).

        Iterates ``plan["shard_plan"]`` and for each shard submits:

        * On ``broadcast_stream``: pinned CPU→NPU H2D via
          :meth:`to` with ``non_blocking=True``, then ``broadcast_fn`` (HCCL).  One
          ``Event`` is recorded after the shard's last broadcast.
        * On ``load_stream``: ``gpu_connector.to_gpu`` per chunk, gated
          on the broadcast ``Event`` via ``wait_event``.

        The loop body contains no collective barriers — the only
        barrier happened in :meth:`_broadcast_metadata_table`.  CPU
        returns to the next shard immediately after submitting to_gpu.
        """
        meta_table: List[Tuple[int, int, Dict[str, Any]]] = plan["meta"]
        shard_plan: List[Tuple[int, int]] = plan["shard_plan"]

        pending: List[Tuple[List[MemoryObj], torch.npu.Event]] = []

        try:
            for shard_idx, (offset, count) in enumerate(shard_plan):
                t_bc_start = time.perf_counter()

                sub_objs: List[TensorMemoryObj] = []
                sub_starts: List[int] = []
                sub_ends: List[int] = []
                t_h2d_total = 0.0

                with torch.npu.stream(self.broadcast_stream):
                    for i in range(offset, offset + count):
                        _, mem_obj, _, _ = reordered_chunks[i]
                        start, end_pos, _ = meta_table[i]

                        raw = mem_obj.raw_tensor
                        if raw is None:
                            raise ValueError(
                                "rank=0 _pipeline_sender: chunk [%d:%d] "
                                "raw_tensor is None "
                                "(is_valid=%s, ref_count=%d)." %
                                (
                                    start,
                                    end_pos,
                                    mem_obj.is_valid(),
                                    mem_obj.get_ref_count(),
                                )
                            )
                        t_h2d = time.perf_counter()
                        gpu_tensor = raw.to(
                            f"npu:{self.metadata.worker_id}",
                            non_blocking=True,
                        )
                        t_h2d_total += time.perf_counter() - t_h2d
                        self.broadcast_fn(gpu_tensor, self.metadata.first_rank)

                        meta = mem_obj.metadata
                        meta_copy = MemoryObjMetadata(
                            shape=meta.shape,
                            dtype=meta.dtype,
                            address=meta.address,
                            phy_size=meta.phy_size,
                            ref_count=1,
                            fmt=meta.fmt,
                            shapes=meta.shapes,
                            dtypes=meta.dtypes,
                        )
                        sub_objs.append(
                            TensorMemoryObj(
                                raw_data=gpu_tensor,
                                metadata=meta_copy,
                                parent_allocator=None,
                            )
                        )
                        sub_starts.append(start)
                        sub_ends.append(end_pos)

                    ev_bc = torch.npu.Event()
                    ev_bc.record()

                t_bc_end = time.perf_counter()

                load_stream.wait_event(ev_bc)
                with torch.npu.stream(load_stream):
                    for obj, start, end_pos in zip(
                        sub_objs, sub_starts, sub_ends
                    ):
                        self.gpu_connector.to_gpu(
                            obj, start, end_pos, **kwargs
                        )
                    ev_togpu = torch.npu.Event()
                    ev_togpu.record()

                t_togpu_submit = time.perf_counter()

                logger.debug(
                    "rank=0 shard[%d] cnt=%d bc=%.2fms h2d=%.2fms enqueue=%.2fms",
                    shard_idx,
                    count,
                    (t_bc_end - t_bc_start) * 1000,
                    t_h2d_total * 1000,
                    (t_togpu_submit - t_bc_end) * 1000,
                )

                pending.append((sub_objs, ev_togpu))
                self._try_release_pending(pending)

        finally:
            for objs, ev in pending:
                try:
                    ev.synchronize()
                except Exception:
                    pass
                for obj in objs:
                    try:
                        obj.ref_count_down()
                    except Exception:
                        pass
            pending.clear()

    def _pipeline_receiver(
        self,
        plan: Dict[str, Any],
        ret_mask: torch.Tensor,
        load_stream: "torch.npu.Stream",
        **kwargs,
    ) -> None:
        """Non-rank-0 side of the sharded broadcast pipeline (Phase 3).

        Iterates ``plan["shard_plan"]`` and for each shard submits:

        * On ``broadcast_stream``: ``broadcast_fn`` (HCCL) into
          per-shard receive buffers.  One ``Event`` recorded after the
          last receive of the shard.
        * On ``load_stream``: ``gpu_connector.to_gpu`` per chunk, gated
          on the broadcast ``Event`` via ``wait_event``.

        The loop body contains no collective barriers — the only
        barrier happened in :meth:`_broadcast_metadata_table`.  CPU
        returns to the next shard immediately after submitting to_gpu.
        Chunks that fell back to CPU buffers (NPU OOM during
        allocation) are skipped on ``to_gpu`` and logged once per
        occurrence.
        """
        meta_table: List[Tuple[int, int, Dict[str, Any]]] = plan["meta"]
        shard_plan: List[Tuple[int, int]] = plan["shard_plan"]
        local_rank = self.metadata.worker_id % torch.npu.device_count()

        pending: List[Tuple[List[MemoryObj], torch.npu.Event]] = []

        try:
            for shard_idx, (offset, count) in enumerate(shard_plan):
                t_bc_start = time.perf_counter()

                shard_objs: List[TensorMemoryObj] = []
                shard_starts: List[int] = []
                shard_ends: List[int] = []

                with torch.npu.stream(self.broadcast_stream):
                    for j in range(count):
                        start, end_pos, meta_dict = meta_table[offset + j]
                        metadata = MemoryObjMetadata.from_dict(meta_dict)
                        raw = self._alloc_broadcast_buffer(
                            metadata, local_rank, self.metadata.first_rank,
                            start, end_pos,
                        )
                        self.broadcast_fn(raw, self.metadata.first_rank)

                        if raw.device.type == "cpu":
                            logger.warning(
                                "rank=%d chunk [%d:%d] received on CPU due to "
                                "NPU OOM, skipping to_gpu (cache miss)",
                                self.metadata.worker_id,
                                start,
                                end_pos,
                            )
                            continue

                        shard_objs.append(
                            TensorMemoryObj(
                                raw_data=raw,
                                metadata=metadata,
                                parent_allocator=None,
                            )
                        )
                        shard_starts.append(start)
                        shard_ends.append(end_pos)
                        ret_mask[start:end_pos] = True

                    ev_bc = torch.npu.Event()
                    ev_bc.record()

                t_bc_end = time.perf_counter()

                if not shard_objs:
                    continue

                load_stream.wait_event(ev_bc)
                with torch.npu.stream(load_stream):
                    for obj, start, end_pos in zip(
                        shard_objs, shard_starts, shard_ends
                    ):
                        self.gpu_connector.to_gpu(
                            obj, start, end_pos, **kwargs
                        )
                    ev_togpu = torch.npu.Event()
                    ev_togpu.record()

                t_togpu_submit = time.perf_counter()

                logger.debug(
                    "rank=%d shard[%d] cnt=%d bc=%.2fms enqueue=%.2fms",
                    self.metadata.worker_id,
                    shard_idx,
                    len(shard_objs),
                    (t_bc_end - t_bc_start) * 1000,
                    (t_togpu_submit - t_bc_end) * 1000,
                )

                pending.append((shard_objs, ev_togpu))
                self._try_release_pending(pending)

        finally:
            for objs, ev in pending:
                try:
                    ev.synchronize()
                except Exception:
                    pass
                for obj in objs:
                    try:
                        obj.ref_count_down()
                    except Exception:
                        pass
            pending.clear()

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

        # --- Cleanup ---
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                if self.storage_manager is None:
                    raise ValueError("storage_manager is required for remove")
                self.storage_manager.remove(key, self.retrieve_locations)
                if self._is_sync_pd_backend():
                    memory_obj.ref_count_down()
            elif not self.async_loading:
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
                tot_kv_size / onload_time / 1024**3
                if onload_time > 0
                else 0,
            )
        return ret_mask

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

        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
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
