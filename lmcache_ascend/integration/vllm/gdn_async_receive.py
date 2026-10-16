# SPDX-License-Identifier: Apache-2.0
"""GDN receive-only metadata, fenced transfer worker and completion ownership.

Only an opted-in GDN connector calls these helpers. A receive completion is
reported after Attention and recurrent state copies and source cleanup finish.
"""

# Standard
from copy import deepcopy
from dataclasses import dataclass
import json
import queue
import threading
import time

# Third Party
from lmcache.logging import init_logger
import torch

# First Party
from lmcache_ascend.v1.async_control import CompletionLedger
from lmcache_ascend.v1.gdn_async_config import gdn_async_restore_enabled

logger = init_logger(__name__)


@dataclass
class Receive:
    request: object
    execution: object
    ready: object = None


def acquire_load(impl, receive):
    """Detach the timed lease; acquire owned Attention gets before any H2D."""
    # First Party
    from lmcache_ascend.v1.state_cache import StateCache, StateLoadError

    engine, request, execution = impl.lmcache_engine, receive.request, receive.execution
    local, boundary = (
        request.load_spec.vllm_cached_tokens,
        request.load_spec.lmcache_cached_tokens,
    )
    chunks, selection = [], None
    with engine._engine_state_lock:
        try:
            selection = getattr(engine, "_state_lookup_selections", {}).pop(
                request.req_id, None
            )
            engine.lookup_pins.pop(request.req_id, None)
            if selection is None or selection.boundary != boundary:
                raise StateLoadError("Selected GDN lease expired before receive")
            if selection.timer is not None:
                selection.timer.cancel()
                selection.timer = None
            if not 0 <= local < boundary <= len(request.token_ids):
                raise StateLoadError("Invalid full-async boundary")
            if tuple(request.token_ids[:boundary]) != execution.token_ids[:boundary]:
                raise StateLoadError("Full-async prefix identity mismatch")
            if request.request_configs != execution.request_configs:
                raise StateLoadError("Full-async request config mismatch")
            operations = StateCache(
                engine.storage_manager, engine.token_database, impl._lmcache_chunk_size
            ).prepare_load(
                execution, boundary, impl.state_layouts, impl.state_kv_caches, selection
            )
            locations = {
                key: loc
                for loc, keys in selection.attention_pins.items()
                for key in keys
            }
            mask = torch.ones(boundary, dtype=torch.bool)
            mask[: local // impl._lmcache_chunk_size * impl._lmcache_chunk_size] = False
            covered = torch.zeros(boundary, dtype=torch.bool)
            for start, end, key in engine.token_database.process_tokens(
                tokens=request.token_ids[:boundary],
                mask=mask,
                request_configs=request.request_configs,
            ):
                location = locations.get(key)
                obj = (
                    None
                    if location is None
                    else engine.storage_manager.get(key, location=location)
                )
                if obj is None:
                    raise StateLoadError("Selected Attention chunk missing")
                chunks.append((obj, start, end))
                covered[start:end] = True
            if not bool(covered[local:boundary].all()):
                raise StateLoadError("Incomplete selected Attention [C,R)")
            return selection, chunks, operations
        except BaseException:
            for obj, _, _ in chunks:
                obj.ref_count_down()
            if selection is not None:
                selection.close(engine.storage_manager)
            raise


def perform_load(impl, receive):
    """All host/device waits here run on the load worker, never forward thread."""
    # First Party
    from lmcache_ascend.v1.state_transfer import transfer_state

    engine, request = impl.lmcache_engine, receive.request
    selection = None
    chunks = []
    safe_to_release = True
    started = time.perf_counter()
    try:
        selection, chunks, operations = acquire_load(impl, receive)
        stream = engine.gpu_connector.load_stream
        with (
            engine._gdn_transfer_lock,
            torch.inference_mode(),
            torch.npu.stream(stream),
        ):
            stream.wait_event(receive.ready)  # Allocated-block zeroing before H2D.
            kwargs = impl._prepare_hybrid_attention(request, selection.boundary)
            try:
                engine.gpu_connector.batched_to_gpu(
                    [obj for obj, _, _ in chunks],
                    [s for _, s, _ in chunks],
                    [e for _, _, e in chunks],
                    **kwargs,
                )
                for operation in operations:
                    transfer_state(operation)
            finally:
                # Keep shared scratch locked through the completion fence,
                # including partially submitted transfer failures.
                stream.synchronize()
        # Previously preempted requests can still own async stores. Drain those
        # before a possible abort receive ACK permits physical block recycling.
        engine.wait_for_pending_stores({request.req_id})
        groups = [op.checkpoint.group_index for op in operations]
        elapsed = time.perf_counter() - started
        logger.info(
            "GDN_FULL_ASYNC_LOAD_FENCED req_id=%s rank=%s boundary=%d "
            "groups=%s cost_ms=%.3f",
            request.req_id,
            engine.metadata.worker_id,
            selection.boundary,
            groups,
            elapsed * 1000,
        )
    except BaseException:
        try:
            engine.gpu_connector.load_stream.synchronize()
        except BaseException:
            safe_to_release = False
            impl._full_quarantine.append((receive, selection, chunks))
        raise
    finally:
        if safe_to_release:
            for obj, _, _ in chunks:
                obj.ref_count_down()
            if selection is not None:
                selection.close(engine.storage_manager)


def load_loop(impl):
    engine = impl.lmcache_engine
    try:
        torch.npu.set_device(engine._device_id)
        while True:
            receive = impl._full_queue.get()
            try:
                if receive is None:
                    break
                impl._full_ledger.check()
                perform_load(impl, receive)
                impl._full_completed += 1
                impl._full_ledger.complete(receive.request.req_id)
                logger.info(
                    "GDN_FULL_ASYNC_LOAD_DONE req_id=%s rank=%s; "
                    "KV AND GDN restored; receive ACK ready",
                    receive.request.req_id,
                    engine.metadata.worker_id,
                )
                del receive
            finally:
                impl._full_queue.task_done()
    except BaseException as error:
        impl._full_ledger.fail(
            getattr(
                getattr(locals().get("receive"), "request", None), "req_id", "worker"
            ),
            error,
        )
        logger.exception("GDN_FULL_ASYNC_LOAD_ERROR; fail closed; no successful ACK")


def make_receive(impl, request, blocks):
    # Third Party
    from vllm.v1.core.sched.output import NewRequestData

    # First Party
    from lmcache_ascend.integration.vllm import multi_group_vllm_adapter as mg

    spec = deepcopy(impl.load_specs[request.request_id])
    boundary = spec.lmcache_cached_tokens
    tables = mg._normalize_block_ids(blocks.get_block_ids(), impl._num_kv_groups)
    # Async receive is NOT present in scheduled_new_reqs (0 forward tokens).
    # Build standalone connector metadata from the actual allocation callback.
    new = NewRequestData.from_request(request, tables)
    new.prompt_token_ids = list(request.all_token_ids[:boundary])
    new.num_computed_tokens = boundary
    tracker = mg.RequestTracker.from_new_request(
        impl.config,
        new,
        boundary,
        boundary,
        True,
        expected_num_groups=impl._num_kv_groups,
    )
    tracker.save_in_current_step = False
    req_meta = mg.ReqMeta.from_request_tracker(
        tracker,
        impl._block_sizes_by_group,
        impl._lmcache_chunk_size,
        load_spec=spec,
        discard_partial_chunks=True,
        save_decode_cache=False,
        compress_ratios=impl._compress_ratios_by_group,
        sliding_window_size_by_group=impl._sliding_window_size_by_group,
        primary_kv_group_idx=impl._state_primary_kv_group_idx,
    )
    if req_meta is None or len(req_meta.token_ids) != boundary:
        raise RuntimeError("Receive metadata does not cover selected boundary")
    sizes = tuple(
        (i, impl._block_sizes_by_group[i])
        for i, group in enumerate(impl._kv_cache_config.kv_cache_groups)
        if any(
            mg.state_group_index(impl._kv_cache_config, name) == i
            for name in group.layer_names
        )
    )
    execution = mg.StateExecution(
        request.request_id,
        tuple(req_meta.token_ids),
        boundary,
        boundary,
        boundary,
        tuple(tuple(ids) for ids in tables),
        sizes,
        True,
        False,
        deepcopy(req_meta.request_configs),
    )
    if not execution.load_blocks(boundary):
        raise RuntimeError("Receive allocation lacks required GDN checkpoint target")
    return Receive(req_meta, execution)


class GDNAsyncReceiveMixin:
    """Private protocol helpers explicitly called by the Ascend adapter."""

    def _init_gdn_receive(self):
        self._gdn_restore_enabled = gdn_async_restore_enabled(self.config)
        if not self._gdn_restore_enabled:
            return
        if self._state_primary_kv_group_idx is None:
            raise ValueError("gdn_async_restore is restricted to GDN models")
        if not self.config.enable_async_loading or not self.config.store_async:
            raise ValueError("GDN async restore requires asynchronous save and load")
        if not self._vllm_config.scheduler_config.async_scheduling:
            raise ValueError("GDN async restore requires vLLM asynchronous scheduling")
        self._full_receives = {}
        self._full_resume_boundaries = {}
        self._full_ledger = CompletionLedger()
        self._full_queue = queue.Queue(maxsize=128)
        self._full_thread = None
        self._full_quarantine = []
        self._full_enqueued = self._full_completed = 0
        self._full_abort_receives = set()

    def _gdn_lookup(self, request, num_computed_tokens):
        # First Party
        from lmcache_ascend.integration.vllm import multi_group_vllm_adapter as mg
        from lmcache_ascend.v1 import state_lookup as sl

        if self._bypass_multimodal_state_request(
            request
        ) or request.request_id.startswith("mock_req"):
            return 0
        req_id = request.request_id
        self._requests_priority[req_id] = getattr(request, "priority", 0)
        tokens = tuple(request.all_token_ids)
        configs = deepcopy(mg.extract_request_configs(request.sampling_params))
        upper = max(
            0, min(request.num_tokens - 1, len(tokens) - self.skip_last_n_tokens)
        )
        if upper <= num_computed_tokens:
            return 0
        signature = (
            tokens,
            num_computed_tokens,
            upper,
            json.dumps(configs, sort_keys=True),
        )
        world = self.worker_count
        boundary = self.lookup_client.poll(
            req_id,
            signature,
            lambda client: sl.lookup_state(
                client, tokens, req_id, num_computed_tokens, upper, world, configs
            ),
        )
        if boundary is None:
            return None
        external = max(0, boundary - num_computed_tokens)
        if external and external < self.config.min_retrieve_tokens:
            self.lookup_client.cancel_lookup(req_id)
            boundary = external = 0
        self.load_specs[req_id] = mg.LoadSpec(
            num_computed_tokens, boundary if external else 0, False
        )
        logger.info(
            "Hybrid lookup: Reqid: %s, Total tokens %d, "
            "Inference Engine computed tokens: %d, "
            "LMCache hit tokens: %d, need to load: %d; async_receive=%s",
            req_id,
            request.num_tokens,
            num_computed_tokens,
            boundary if external else 0,
            external,
            bool(external),
        )
        return external

    def _gdn_allocated(self, request, num_external_tokens, blocks=None):
        # First Party
        from lmcache_ascend.integration.vllm import multi_group_vllm_adapter as mg

        resume = (
            self._full_resume_boundaries.pop(request.request_id, None)
            if not num_external_tokens
            else None
        )
        if resume is not None:
            if request.num_computed_tokens < resume:
                raise RuntimeError(
                    "Async resumed request lost its completed receive boundary"
                )
            self.load_specs[request.request_id] = mg.LoadSpec(
                request.num_computed_tokens, 0, False
            )
        self._update_state_after_alloc_sync(request, num_external_tokens, blocks=blocks)
        if num_external_tokens:
            if blocks is None:
                raise RuntimeError("Full async allocation callback requires all groups")
            receive = make_receive(self, request, blocks)
            if request.request_id in self._full_receives:
                raise RuntimeError("Duplicate pending receive allocation")
            self._full_receives[request.request_id] = receive
            self.lookup_client.mark_admitted(request.request_id)
            self._full_resume_boundaries[request.request_id] = receive.execution.start
            # No second load when the request is later admitted for forward.
            self.load_specs[request.request_id].can_load = False

    def _gdn_metadata(self, output):
        result = self._build_connector_meta_sync(output)
        result.gdn_async_receives = list(self._full_receives.values())
        self._full_receives = {}
        for req_id in output.finished_req_ids:
            self._full_resume_boundaries.pop(req_id, None)
        return result

    def _gdn_start_load(self, context, **kwargs):
        self._state_mm_native_finished_req_ids = getattr(
            self._parent._get_connector_metadata(), "native_mm_finished_req_ids", ()
        )
        self._full_ledger.check()
        self.current_layer = 0
        self._wait_for_save_done = False
        meta = self._parent._get_connector_metadata()
        receives = getattr(meta, "gdn_async_receives", ())
        # Allocator zeroing submitted before start_load must precede H2D.
        for received in receives:
            receive = deepcopy(received)
            ready = torch.npu.Event()
            ready.record(torch.npu.current_stream())
            receive.ready = ready
            self._full_ledger.begin(receive.request.req_id)
            if self._full_thread is None:
                self._full_thread = threading.Thread(
                    target=load_loop,
                    args=(self,),
                    daemon=True,
                    name="gdn-full-async-load",
                )
                self._full_thread.start()
            self._full_queue.put_nowait(receive)
            self._full_enqueued += 1
            logger.info(
                "GDN_FULL_ASYNC_LOAD_ENQUEUED req_id=%s rank=%s boundary=%d; "
                "no forward until ACK",
                receive.request.req_id,
                self.lmcache_engine.metadata.worker_id,
                receive.execution.start,
            )

    def _gdn_finished(self, finished_req_ids):
        self._full_ledger.check()
        owned = self._full_ledger.owned()
        self._full_abort_receives.update(set(finished_req_ids) & owned)
        # Abort of an async receive frees via receive ACK, NOT sending ACK.
        sending, unused = self._get_finished_sync(
            set(finished_req_ids) - self._full_abort_receives
        )
        if unused:
            raise RuntimeError("Unexpected native receiver completion")
        receiving = self._full_ledger.drain()
        self._full_abort_receives.difference_update(receiving)
        return sending, receiving or None

    def _gdn_shutdown(self):
        if self._full_thread is not None:
            self._full_queue.put(None, timeout=120)
            self._full_thread.join(120)
            if (
                self._full_thread.is_alive()
                or self._full_quarantine
                or self._full_ledger.active
            ):
                raise RuntimeError(
                    "Cannot close allocator with outstanding asynchronous receive"
                )
