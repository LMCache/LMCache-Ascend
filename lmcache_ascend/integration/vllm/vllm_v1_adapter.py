# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Optional
import sys

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.token_database import ChunkedTokenDatabase
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.request import RequestStatus
import torch

# First Party
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
    LMCacheConnectorV1ImplMultiGroup,
)
from lmcache_ascend.integration.vllm.multi_spec_flatten import (
    build_flat_kv_caches,
    has_multiple_scheduler_groups,
)
from lmcache_ascend.integration.vllm.skip_state_groups import (
    apply_skip_policy_from_env_to_flattened,
)
from lmcache_ascend.integration.vllm.state_groups import (
    build_state_layouts,
    request_primary,
    select_state_primary,
    validate_state_config,
)
from lmcache_ascend.v1.state_cache import StateCache, StateLoadError
from lmcache_ascend.v1.state_lookup import cancel_state_lookup
from lmcache_ascend.v1.state_transfer import transfer_state
from lmcache_ascend.v1.storage_backend.storage_manager import state_store_locations

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Impl(LMCacheConnectorV1ImplMultiGroup):
    # Type declarations for upstream-inherited attributes (mypy has-type fix)
    kv_caches: dict[str, Any]

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
        kv_cache_config: Optional[Any] = None,
    ):
        logger.debug("Initializing LMCacheAscendConnectorV1Impl")
        super().__init__(vllm_config, role, parent, kv_cache_config=kv_cache_config)
        if self._num_kv_groups > 1:
            assert self._discard_partial_chunks, (
                "Multi-group KV cache requires discard_partial_chunks=True; "
                "partial-chunk store/load is not supported across KV cache groups."
            )
        self.store_async = self.config.store_async
        self._wait_for_save_done = True
        self._finished_req_ids_waiting_for_save: set[str] = set()
        self._late_finished_sending: set[str] = set()
        self._failed_state_loads: set[str] = set()
        self._finished_state_loads: set[str] = set()
        logger.debug("store_async: %s", self.store_async)

    @_lmcache_nvtx_annotate
    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register KV caches (upstream) with Ascend multi-group preprocessing."""
        primary = (
            select_state_primary(self._kv_cache_config)
            if self._kv_cache_config is not None
            else None
        )
        engine = self.lmcache_engine
        if primary is not None:
            validate_state_config(self.config, self._vllm_config)
            if (
                engine is None
                or type(engine.token_database) is not ChunkedTokenDatabase
            ):
                raise ValueError("GDN checkpoints require ChunkedTokenDatabase")
            if engine.save_only_first_rank or engine.remove_after_retrieve:
                raise ValueError(
                    "GDN checkpoints reject save_only_first_rank/remove_after_retrieve"
                )
            locations = state_store_locations(engine.storage_manager)
            if not locations:
                raise ValueError("GDN checkpoints require a writable CPU/disk tier")
            requested = set(self.config.retrieve_locations or ())
            if self.config.store_location:
                requested.add(self.config.store_location)
            if not requested <= set(locations):
                raise ValueError("GDN checkpoint locations must be available")
        self.state_layouts = (
            build_state_layouts(self._kv_cache_config, kv_caches)
            if self._kv_cache_config is not None
            else ()
        )
        self.state_kv_caches = {
            name: tuple(kv_caches[name])
            for layout in self.state_layouts
            for name in layout.layer_names
        }
        attention_kv = (
            {
                name: entry
                for name, entry in kv_caches.items()
                if name not in self.state_kv_caches
            }
            if self.state_layouts
            else kv_caches
        )
        flat_kv = attention_kv
        sched_by_layer: tuple[int, ...] | None = None
        layer_to_groups: dict[str, list[int]] | None = None
        bundled = False
        multi_group = has_multiple_scheduler_groups(self._kv_cache_config)

        if multi_group:
            flat_kv, sched_by_layer, layer_to_groups, bundled = build_flat_kv_caches(
                attention_kv,
                self._kv_cache_config,
            )
            flat_kv, sched_by_layer, layer_to_groups = (
                apply_skip_policy_from_env_to_flattened(
                    self._kv_cache_config,
                    flat_kv,
                    sched_by_layer,
                    layer_to_groups,
                    bundled=bundled,
                )
            )
            logger.info(
                "Preprocessed multi-spec KV caches: %d model layers -> "
                "%d logical layers (bundled=%s)",
                len(kv_caches),
                len(flat_kv),
                bundled,
            )

        engine = getattr(self, "lmcache_engine", None)
        connector = getattr(engine, "gpu_connector", None) if engine else None
        if connector is not None and hasattr(connector, "layout_hints"):
            hints = connector.layout_hints or {}
            hints["vllm_block_size"] = self._block_size
            if multi_group:
                hints["block_sizes_by_group"] = self._block_sizes_by_group
                hints["compress_ratios_by_group"] = self._compress_ratios_by_group
                hints["sliding_window_size_by_group"] = getattr(
                    self, "_sliding_window_size_by_group", None
                )
                hints["scheduler_group_by_flat_layer"] = sched_by_layer
                hints["layer_to_scheduler_groups"] = layer_to_groups
                hints["model_kv_caches"] = attention_kv
                hints["flat_layer_names"] = list(flat_kv.keys())
                hints["bundle_multi_spec"] = bundled
            connector.layout_hints = hints

        if self.state_layouts:
            if connector is None or not hasattr(connector, "ensure_kv_layer_groups"):
                raise ValueError(
                    "GDN checkpoints require the Ascend grouped KV connector"
                )
            connector.num_layers = len(flat_kv)
            self.num_layers = len(flat_kv)

        # Build kv_layer_groups_manager before post_init() so
        # metadata.get_shapes() allocates one MemoryObj slot per NPU group.
        if connector is not None and hasattr(connector, "ensure_kv_layer_groups"):
            try:
                connector.ensure_kv_layer_groups(list(flat_kv.values()))
                if (
                    self.state_layouts
                    and len(engine.metadata.kv_layer_groups_manager.kv_layer_groups)
                    != 1
                ):
                    raise ValueError("Hybrid requires one full Attention payload group")
                logger.info(
                    "Registered KV layer groups during register_kv_caches "
                    "(%d layers, kv_layer_groups_manager=%s)",
                    len(flat_kv),
                    getattr(
                        getattr(engine, "metadata", None),
                        "kv_layer_groups_manager",
                        "N/A",
                    ),
                )
            except Exception:
                if multi_group:
                    logger.error(
                        "Failed to register KV layer groups after multi-spec "
                        "preprocessing",
                        exc_info=True,
                    )
                    raise
                logger.warning(
                    "Failed to register KV layer groups; "
                    "will fall back to legacy single-group allocation",
                    exc_info=True,
                )

        logger.info("Registering KV caches")
        assert len(self.kv_caches) == 0 and len(flat_kv) > 0
        self.kv_caches = flat_kv
        if self.state_layouts:
            engine.state_layouts = self.state_layouts
        self._manager.post_init()

    # Upstream start_load_kv only transfers the primary group's slot_mapping.
    # Multi-group retrieve needs ALL per-group slot mappings on NPU so the
    # connector can DMA each spec's KV plane to the correct paged blocks.
    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self.current_layer = 0
        self._wait_for_save_done = False

        if self._num_kv_groups <= 1:
            super().start_load_kv(forward_context, **kwargs)
            self._mark_failed_p2p_loads_for_recompute()
            return

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        if getattr(self, "state_layouts", ()):
            executions = {item.req_id: item for item in metadata.state_executions}
            try:
                for request in metadata.requests:
                    spec = request.load_spec
                    if spec is None or not spec.can_load:
                        continue
                    if forward_context.attn_metadata is None:
                        self._record_state_load_failure(
                            request, "all", "Missing Attention forward metadata"
                        )
                        continue
                    self._load_hybrid_request(request, executions.get(request.req_id))
            finally:
                # start_load exceptions bypass the runner's normal save finally.
                # Release even the selections of requests not reached in this batch.
                self._unpin_save_requests(metadata)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None
        gpu_connector = self.lmcache_engine.gpu_connector
        self.layerwise_retrievers = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                continue

            tokens = request.token_ids
            slot_mappings_cpu: list[torch.Tensor] = []
            for group_idx in range(request.num_kv_groups):
                group_slot_mapping = request.get_slot_mapping(group_idx)
                assert isinstance(group_slot_mapping, torch.Tensor)
                slot_mappings_cpu.append(group_slot_mapping.pin_memory())

            pg = request.primary_kv_group_idx
            slot_mapping_cpu = slot_mappings_cpu[pg]
            assert len(slot_mapping_cpu) <= len(tokens)

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens

            slot_mappings_npu: list[torch.Tensor] = []
            filtered_slot_mappings_npu: tuple[torch.Tensor, ...] | None = None
            with torch.npu.stream(gpu_connector.load_stream):
                for sm_cpu in slot_mappings_cpu:
                    slot_mappings_npu.append(
                        sm_cpu.to(device="npu", dtype=torch.long, non_blocking=True)
                    )
                slot_mapping_npu = slot_mappings_npu[pg]
                if request.filtered_slot_by_group is not None:
                    filtered_slot_mappings_npu = tuple(
                        sm_cpu.to(device="npu", dtype=torch.long, non_blocking=True)
                        for sm_cpu in request.filtered_slot_by_group
                    )

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            retrieve_kwargs: dict = {
                "kvcaches": kvcaches,
                "slot_mapping": slot_mapping_npu,
                "vllm_cached_tokens": request.load_spec.vllm_cached_tokens,
                "request_configs": request.request_configs,
                "req_id": request.req_id,
            }
            if request.num_kv_groups > 1:
                retrieve_kwargs["slot_mappings_by_group"] = tuple(slot_mappings_cpu)
                retrieve_kwargs["slot_mappings_npu_by_group"] = tuple(slot_mappings_npu)
            if filtered_slot_mappings_npu is not None:
                retrieve_kwargs["filtered_slot_mappings_npu"] = (
                    filtered_slot_mappings_npu
                )
            if request.slot_valid_prefix_by_group is not None:
                retrieve_kwargs["slot_valid_prefix_by_group"] = (
                    request.slot_valid_prefix_by_group
                )

            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                if self.enable_blending:
                    logger.warning(
                        "enable_blending is unsupported with multi-group KV; "
                        "using layerwise retrieve instead"
                    )
                layerwise_retriever = self.lmcache_engine.retrieve_layer(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    **retrieve_kwargs,
                    sync=sync,
                )
                next(layerwise_retriever)
                next(layerwise_retriever)
                self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    **retrieve_kwargs,
                )

                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping_npu[:lmcache_cached_tokens],
                        block_size=self._block_sizes_by_group[pg],
                    )
                    self._invalid_block_ids.update(missing_blocks)

        self._mark_failed_p2p_loads_for_recompute()

    def _record_state_load_failure(self, request, group, reason, *, exc_info=False):
        self._failed_state_loads.add(request.req_id)
        logger.error(
            "Hybrid load failed: request=%s rank=%s group=%s R=%s reason=%s",
            request.req_id,
            self.lmcache_engine.metadata.worker_id,
            group,
            request.load_spec.lmcache_cached_tokens,
            reason,
            exc_info=exc_info,
        )

    def _prepare_hybrid_attention(self, request, boundary):
        """Validate required Attention slots before allocating transfer mappings."""
        state_groups = {layout.group_index for layout in self.state_layouts}
        local = request.load_spec.vllm_cached_tokens
        slots = []
        for group in range(request.num_kv_groups):
            if group in state_groups:
                slots.append(torch.empty(0, dtype=torch.long))
                continue
            mapping = request.get_slot_mapping(group)
            if (
                not isinstance(mapping, torch.Tensor)
                or mapping.ndim != 1
                or len(mapping) < boundary
                or bool((mapping[local:boundary] < 0).any())
            ):
                raise StateLoadError("Missing Attention target mapping", group)
            slots.append(mapping)
        cpu_slots = tuple(slot.pin_memory() for slot in slots)
        with torch.npu.stream(self.lmcache_engine.gpu_connector.load_stream):
            npu_slots = tuple(
                slot.to(device="npu", dtype=torch.long, non_blocking=True)
                for slot in cpu_slots
            )
            result = {
                "kvcaches": list(self.kv_caches.values()),
                "slot_mapping": npu_slots[request.primary_kv_group_idx],
                "slot_mappings_by_group": cpu_slots,
                "slot_mappings_npu_by_group": npu_slots,
                "vllm_cached_tokens": local,
                "request_configs": request.request_configs,
                "req_id": request.req_id,
            }
            if request.filtered_slot_by_group is not None:
                result["filtered_slot_mappings_npu"] = tuple(
                    slot.to(device="npu", dtype=torch.long, non_blocking=True)
                    for slot in request.filtered_slot_by_group
                )
            if request.slot_valid_prefix_by_group is not None:
                result["slot_valid_prefix_by_group"] = (
                    request.slot_valid_prefix_by_group
                )
        return result

    def _load_hybrid_request(self, request, execution):
        """Restore one selected R locally; detected failure is logged, not recovery."""
        spec = request.load_spec
        if spec is None or not spec.can_load:
            return False
        engine = self.lmcache_engine
        boundary, local = spec.lmcache_cached_tokens, spec.vllm_cached_tokens
        group = "all"
        with engine._engine_state_lock:
            try:
                selection = engine.get_state_lookup(request.req_id, boundary)
                if not 0 <= local < boundary <= len(request.token_ids):
                    raise StateLoadError("Inconsistent scheduler load boundary")
                if execution is not None and (
                    tuple(request.token_ids[:boundary])
                    != execution.token_ids[:boundary]
                    or request.request_configs != execution.request_configs
                ):
                    raise StateLoadError("Mismatched execution prefix identity")
                operations = StateCache(
                    engine.storage_manager,
                    engine.token_database,
                    self._lmcache_chunk_size,
                ).prepare_load(
                    execution,
                    boundary,
                    self.state_layouts,
                    self.state_kv_caches,
                    selection,
                )
                group = "attention"
                retrieve_kwargs = self._prepare_hybrid_attention(request, boundary)
                mask = torch.ones(boundary, dtype=torch.bool)
                mask[: local // self._lmcache_chunk_size * self._lmcache_chunk_size] = (
                    False
                )
                ret_mask = engine.retrieve(
                    request.token_ids[:boundary], mask, **retrieve_kwargs
                )
                engine.gpu_connector.load_stream.synchronize()
                # The engine may reload the chunk overlapping C. A total count
                # can hide missing tokens above C behind copied ones below C.
                if (
                    ret_mask.ndim != 1
                    or len(ret_mask) != boundary
                    or not bool(ret_mask[local:boundary].all())
                ):
                    raise StateLoadError(
                        "Incomplete Attention coverage of [C,R)", group
                    )
                with torch.npu.stream(engine.gpu_connector.load_stream):
                    for operation in operations:
                        group = operation.checkpoint.group_index
                        transfer_state(operation)
                engine.gpu_connector.load_stream.synchronize()
                logger.info(
                    "Hybrid load complete: request=%s rank=%s groups=%s "
                    "R=%s C=%s targets=%s",
                    request.req_id,
                    engine.metadata.worker_id,
                    [op.checkpoint.group_index for op in operations],
                    boundary,
                    local,
                    [
                        (op.checkpoint.group_index, op.runtime.block_id)
                        for op in operations
                    ],
                )
                return True
            except StateLoadError as error:
                self._record_state_load_failure(request, error.group, str(error))
                return False
            except Exception as error:
                self._record_state_load_failure(
                    request, group, str(error), exc_info=True
                )
                raise
            finally:
                engine.lookup_unpin(request.req_id)

    def _mark_failed_p2p_loads_for_recompute(self) -> None:
        gpu_connector = getattr(self.lmcache_engine, "gpu_connector", None)
        drain = getattr(gpu_connector, "drain_failed_load_req_ids", None)
        if drain is None:
            return
        failed_req_ids = drain()
        if not failed_req_ids:
            return

        metadata = self._parent._get_connector_metadata()
        if not isinstance(metadata, LMCacheConnectorMetadata):
            return

        for request in metadata.requests:
            if request.req_id not in failed_req_ids:
                continue
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:
                continue

            if getattr(self, "state_layouts", ()):
                self._record_state_load_failure(
                    request, "attention", "P2P pull failure"
                )
                continue

            tokens = request.token_ids
            slot_mapping = request.slot_mapping
            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            expected_mask = token_mask[:lmcache_cached_tokens]
            ret_mask = torch.zeros(lmcache_cached_tokens, dtype=torch.bool)

            missing_blocks = self.record_failed_blocks(
                request.req_id,
                expected_mask,
                ret_mask,
                slot_mapping[:lmcache_cached_tokens],
            )
            self._invalid_block_ids.update(missing_blocks)
            logger.error(
                "Marked %d KV blocks invalid for req %s after P2P pull "
                "failure; vLLM will recompute them locally.",
                len(missing_blocks),
                request.req_id,
            )

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        # vLLM invokes this method from a generator finally, even if forward fails.
        forward_failed = sys.exc_info()[0] is not None
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if forward_failed and getattr(self, "state_layouts", ()):
            self._unpin_save_requests(connector_metadata)
            self._wait_for_save_done = True
            self._retire_finished_state_loads()
            return

        if self.kv_role == "kv_consumer":
            if self.lmcache_engine is not None:
                self._unpin_save_requests(connector_metadata)
            self._wait_for_save_done = True
            self._retire_finished_state_loads()
            return

        # lmcache-ascend start: skip save on passive ranks ---------------------
        # Under save_only_first_rank (default for MLA/DSA), only the first rank
        # owns a storage_manager; the other ranks are "passive" and neither
        # store nor look up locally. The base store() already no-ops for them,
        # and _local_persist_skip's local lookup would assert on the missing
        # storage_manager. Short-circuit the whole save path for these ranks.
        if self.lmcache_engine is not None and self.lmcache_engine._is_passive():
            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)
            self._wait_for_save_done = True
            self._replay_finished_stores_after_save()
            return
        # lmcache-ascend end --------------------------------------------------

        if self.use_layerwise:
            assert not self.store_async, (
                "Layerwise storing is not supported with async store"
            )
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                self.lmcache_engine.lookup_unpin(request.req_id)
            self._wait_for_save_done = True
            self._replay_finished_stores_after_save()
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        # lmcache-ascend start ---------------------
        ordering_event = torch.npu.Event()
        ordering_event.record()
        # lmcache-ascend end ---------------------

        try:
            self._save_state_executions(connector_metadata, ordering_event)
        finally:
            if getattr(self, "state_layouts", ()):
                self._unpin_save_requests(connector_metadata)

        for request in connector_metadata.requests:
            self.lmcache_engine.lookup_unpin(request.req_id)

            if request.req_id in getattr(self, "_failed_state_loads", set()):
                # A suffix computed from failed recurrent state is not publishable
                # as either Attention KV or a new state checkpoint.
                continue

            try:
                save_spec = request.save_spec
                token_ids = request.token_ids

                # lmcache-ascend start: local-vs-remote hit distinction ------
                # ``save_spec.skip_leading_tokens`` is seeded from the *total*
                # LMCache hit (local + remote). When the matched prefix was
                # pulled from a remote peer, the local CPU backend is still
                # cold for those chunks, so skipping them here makes every
                # subsequent request re-pull the same KV from the peer.
                # Re-derive how many leading tokens are *already local* and, if
                # a remote-loaded prefix is missing locally, persist it into the
                # local backend so later hits stay local.
                persist_remote_skip = self._local_persist_skip(request, token_ids)
                # lmcache-ascend end ----------------------------------------

                if (
                    (save_spec is None or not save_spec.can_save)
                    and self.kv_role != "kv_producer"
                    and persist_remote_skip is None
                ):
                    continue

                pg = request.primary_kv_group_idx
                slot_mappings_cpu: list[torch.Tensor] = []
                for group_idx in range(request.num_kv_groups):
                    group_slot_mapping = request.get_slot_mapping(group_idx)
                    assert isinstance(group_slot_mapping, torch.Tensor)
                    assert len(group_slot_mapping) <= len(token_ids)
                    slot_mappings_cpu.append(group_slot_mapping.pin_memory())

                slot_mapping = slot_mappings_cpu[pg]
                if request.num_kv_groups > 1:
                    logger.info(
                        "Multi-group wait_for_save: multi-group slot_mapping "
                        "(%d groups); primary group %d has %d slots for "
                        "%d tokens",
                        request.num_kv_groups,
                        pg,
                        len(slot_mapping),
                        len(token_ids),
                    )
                elif len(slot_mapping) != len(token_ids):
                    logger.debug(
                        "slot_mapping length %d != token_ids length %d "
                        "(primary group %d, compress_ratio %d)",
                        len(slot_mapping),
                        len(token_ids),
                        pg,
                        self._compress_ratios_by_group[pg],
                    )

                # lmcache-ascend start ---------------------
                slot_mappings_npu: list[torch.Tensor] = []
                filtered_slot_mappings_npu: tuple[torch.Tensor, ...] | None = None
                with torch.npu.stream(self.lmcache_engine.gpu_connector.store_stream):
                    for sm_cpu in slot_mappings_cpu:
                        slot_mappings_npu.append(
                            sm_cpu.to(device="npu", dtype=torch.long, non_blocking=True)
                        )
                    slot_mapping_npu = slot_mappings_npu[pg]
                    if request.filtered_slot_by_group is not None:
                        filtered_slot_mappings_npu = tuple(
                            sm_cpu.to(device="npu", dtype=torch.long, non_blocking=True)
                            for sm_cpu in request.filtered_slot_by_group
                        )
                # lmcache-ascend end ---------------------

                if persist_remote_skip is not None:
                    skip_leading_tokens = persist_remote_skip
                elif save_spec is not None:
                    skip_leading_tokens = save_spec.skip_leading_tokens
                else:
                    skip_leading_tokens = 0

                if skip_leading_tokens == len(token_ids):
                    continue
                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                is_last_prefill = request.is_last_prefill
                if is_last_prefill:
                    if request.disagg_spec:
                        request.disagg_spec.is_last_prefill = True
                else:
                    if not self.enable_blending:
                        token_len = len(token_ids)
                        aligned_token_len = (
                            token_len
                            // self._lmcache_chunk_size
                            * self._lmcache_chunk_size
                        )
                        token_ids = token_ids[:aligned_token_len]
                        store_mask = store_mask[:aligned_token_len]
                        slot_mappings_cpu = [
                            sm[:aligned_token_len] for sm in slot_mappings_cpu
                        ]
                        slot_mapping = slot_mappings_cpu[pg]
                        slot_mappings_npu = [
                            sm[:aligned_token_len] for sm in slot_mappings_npu
                        ]
                        slot_mapping_npu = slot_mappings_npu[pg]

                store_kwargs: dict = {
                    "kvcaches": kvcaches,
                    "slot_mapping": slot_mapping,
                    "offset": skip_leading_tokens,
                    "transfer_spec": request.disagg_spec,
                    "request_configs": request.request_configs,
                    "req_id": request.req_id,
                    "ordering_event": ordering_event,
                    "slot_mapping_npu": slot_mapping_npu,
                }
                if request.num_kv_groups > 1:
                    store_kwargs["slot_mappings_by_group"] = tuple(slot_mappings_cpu)
                    store_kwargs["slot_mappings_npu_by_group"] = tuple(
                        slot_mappings_npu
                    )
                if filtered_slot_mappings_npu is not None:
                    store_kwargs["filtered_slot_mappings_npu"] = (
                        filtered_slot_mappings_npu
                    )
                if request.slot_valid_prefix_by_group is not None:
                    store_kwargs["slot_valid_prefix_by_group"] = (
                        request.slot_valid_prefix_by_group
                    )

                self.lmcache_engine.store(
                    token_ids,
                    mask=store_mask,
                    **store_kwargs,
                )

                if get_pp_group().is_last_rank:
                    save_spec.skip_leading_tokens = len(token_ids)
                    if request.disagg_spec:
                        request.disagg_spec.num_transferred_tokens = len(token_ids)
            except Exception:
                # Do not let one failing request abort the save loop
                logger.exception(
                    "wait_for_save failed for request %s; skipping save",
                    request.req_id,
                )
                continue

        self._wait_for_save_done = True
        self._replay_finished_stores_after_save()

    def _local_persist_skip(self, request, token_ids) -> Optional[int]:
        """Decide whether a remote-loaded prefix must be persisted locally.

        The base save path skips every token LMCache reported as a hit
        (``save_spec.skip_leading_tokens`` == total local + remote hit). For a
        prefix pulled from a remote peer, the local CPU backend is still cold,
        so skipping it forces a re-pull on every subsequent request.

        Returns the chunk-aligned number of leading tokens to skip when the
        request must back-fill the local cache (i.e. some matched-and-loaded
        prefix is not yet local), or ``None`` to keep the base save behavior
        unchanged.
        """
        if self.kv_role == "kv_consumer":
            return None
        # Only meaningful when a local CPU backend exists to back-fill into.
        if not getattr(self.config, "local_cpu", False):
            return None
        save_spec = request.save_spec
        if save_spec is None:
            return None
        load_spec = getattr(request, "load_spec", None)
        if load_spec is None or not load_spec.can_load:
            return None
        loaded_prefix = load_spec.lmcache_cached_tokens
        if loaded_prefix <= 0:
            return None

        # Contiguous prefix already resident in the local CPU backend.
        local_present = self.lmcache_engine.lookup(
            token_ids,
            search_range=["LocalCPUBackend"],
            pin=False,
            request_configs=request.request_configs,
        )
        local_present = (
            local_present // self._lmcache_chunk_size * self._lmcache_chunk_size
        )
        if local_present >= loaded_prefix:
            # Whole matched prefix is already local; nothing to back-fill.
            return None

        logger.info(
            "Persisting remote-loaded KV into local cache for request %s: "
            "local_prefix=%d loaded_prefix=%d (storing %d trailing tokens)",
            request.req_id,
            local_present,
            loaded_prefix,
            len(token_ids) - local_present,
        )
        return local_present

    def _unpin_save_requests(self, metadata):
        req_ids = {request.req_id for request in metadata.requests}
        req_ids.update(
            execution.req_id for execution in getattr(metadata, "state_executions", ())
        )
        for req_id in req_ids:
            self.lmcache_engine.lookup_unpin(req_id)

    def _save_state_executions(self, metadata, ordering_event):
        """Save state independently of Attention requests and save watermarks."""
        layouts = getattr(self, "state_layouts", ())
        if not layouts:
            return
        for execution in getattr(metadata, "state_executions", ()):
            if execution.req_id in getattr(self, "_failed_state_loads", set()):
                continue
            try:
                self.lmcache_engine.store_state(
                    execution, layouts, self.state_kv_caches, ordering_event
                )
            except Exception:
                logger.exception(
                    "State save failed: request=%s rank=%s boundary=%s",
                    execution.req_id,
                    self.lmcache_engine.metadata.worker_id,
                    execution.end,
                )
                raise

    def _may_register_store_after_wait_for_save(self, request: "Request") -> bool:
        if request.req_id in getattr(self, "_failed_state_loads", set()):
            return False
        if self.kv_role == "kv_consumer":
            return False
        save_spec = request.save_spec
        if save_spec is None:
            return False
        if not save_spec.can_save and self.kv_role != "kv_producer":
            return False
        return save_spec.skip_leading_tokens != len(request.token_ids)

    def _retire_finished_state_loads(self) -> None:
        if getattr(self, "_finished_state_loads", None):
            self._failed_state_loads.difference_update(self._finished_state_loads)
            self._finished_state_loads.clear()

    def _replay_finished_stores_after_save(self) -> None:
        self._retire_finished_state_loads()
        if not self._finished_req_ids_waiting_for_save or self.lmcache_engine is None:
            return

        finished_sending = self.lmcache_engine.get_finished_stores(
            self._finished_req_ids_waiting_for_save
        )
        if finished_sending:
            self._late_finished_sending |= finished_sending
        self._finished_req_ids_waiting_for_save = set()

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if self.lmcache_engine is None:
            return None, None
        if getattr(self, "state_layouts", ()):
            for req_id in finished_req_ids:
                self.lmcache_engine.lookup_unpin(req_id)
            if self._wait_for_save_done:
                self._failed_state_loads.difference_update(finished_req_ids)
            else:
                # Some runner paths report finished before wait_for_save; keep
                # failed IDs until that final save opportunity has been skipped.
                self._finished_state_loads.update(finished_req_ids)
        query_req_ids = set(finished_req_ids)
        if not self._wait_for_save_done:
            # NOTE (gingfung): The is a workaround logic for the case
            # where the requests is deferred (i.e. spec_decode or MTP)
            # and the model_runner call get_finished before wait_for_save.
            connector_metadata = self._parent._get_connector_metadata()
            assert isinstance(connector_metadata, LMCacheConnectorMetadata)

            waiting_for_save = {
                request.req_id
                for request in connector_metadata.requests
                if request.req_id in finished_req_ids
                and self._may_register_store_after_wait_for_save(request)
            }
            if waiting_for_save:
                self._finished_req_ids_waiting_for_save |= waiting_for_save
                query_req_ids -= waiting_for_save

        finished_sending = self.lmcache_engine.get_finished_stores(query_req_ids)
        if self._late_finished_sending:
            finished_sending |= self._late_finished_sending
            self._late_finished_sending = set()
        return (
            finished_sending if finished_sending else None,
            None,
        )

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if self.lmcache_engine is None:
            return

        logger.debug(
            "LMCache-Ascend handling preemptions: req_ids=%s",
            sorted(preempted_req_ids),
        )

        # Lookup pins are request-scoped and normally released in wait_for_save().
        # A preempted request may leave that path before its metadata is replayed.
        for req_id in preempted_req_ids:
            self.lmcache_engine.lookup_unpin(req_id)

        if not self.store_async or self.kv_role == "kv_consumer":
            return

        waited_req_ids = self.lmcache_engine.wait_for_pending_stores(preempted_req_ids)
        if waited_req_ids:
            logger.info(
                "Handled preemptions after draining async stores: req_ids=%s",
                sorted(waited_req_ids),
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if getattr(self, "_state_primary_kv_group_idx", None) is not None:
            try:
                if self.lookup_client is not None:
                    cancel_state_lookup(self.lookup_client, request.request_id)
            finally:
                self.load_specs.pop(request.request_id, None)
                self._allocated_blocks.pop(request.request_id, None)
        # Add patch from upstream LMCache#3340 (regression LMCache#3337)
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._layerwise_save_storers.pop(request.request_id, None)

        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED:
            # ``request_finished`` is a Scheduler-side connector API.
            # The Scheduler typically does not initialize the storage
            # engine (unless ``enable_scheduler_bypass_lookup`` is set);
            # only the Worker role builds it by default. The Scheduler
            # *does* own the lookup_client though, so the async lookup
            # cancel below must run independently of the engine check
            # to avoid leaking in-flight async lookups on Scheduler-side
            # aborts. See LMCache#3337.
            if self.lmcache_engine is None:
                logger.warning(
                    "Skipping abort-time backend cleanup for request %s: "
                    "lmcache_engine is not initialized (Scheduler role "
                    "without enable_scheduler_bypass_lookup).",
                    request.request_id,
                )
            else:
                # Notify storage backends of aborted requests
                sm = self.lmcache_engine.storage_manager
                if sm is not None:
                    sm.cancel_request(request.request_id)

            if self.async_loading:
                # Cancel any ongoing async lookup and prefetch tasks on
                # workers. Independent of ``lmcache_engine`` because the
                # Scheduler owns ``lookup_client`` even when it does not
                # build an engine.
                lookup_id = request.request_id
                if self.lookup_client is None:
                    logger.warning(
                        "Skipping abort-time async lookup cancel for "
                        "request %s: lookup_client is not initialized "
                        "while async_loading is enabled. Engine stays "
                        "alive; this request's lookup is dropped.",
                        request.request_id,
                    )
                else:
                    self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        # chunk_hashes return start ---------------------
        if getattr(self.config, "enable_chunk_hashes_return", False):
            inner = self.lookup_client
            while hasattr(inner, "actual_lookup_client"):
                inner = inner.actual_lookup_client
            new_hashes = inner.get_cached_hashes(request.request_id)
            return_params = return_params or {}
            return_params["chunk_hashes"] = new_hashes
        # chunk_hashes return end ---------------------

        if (
            request.status == RequestStatus.FINISHED_ABORTED
            and self.lmcache_engine is not None
        ):
            self.lmcache_engine.lookup_unpin(request.request_id)

            if self.store_async and self.kv_role != "kv_consumer":
                try:
                    self.lmcache_engine.wait_for_pending_stores({request.request_id})
                except Exception:
                    logger.warning(
                        "wait_for_pending_stores failed for aborted request %s",
                        request.request_id,
                        exc_info=True,
                    )

        delay_free = self.store_async and self.kv_role != "kv_consumer"
        return delay_free, return_params

    @_lmcache_nvtx_annotate
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """vLLM HMA hook; delegates to :meth:`request_finished` (upstream LMCache)."""
        if not block_ids:
            if getattr(self, "_state_primary_kv_group_idx", None) is not None:
                return self.request_finished(request, [])
            return False, None
        state_primary = self._state_primary_kv_group_idx
        if state_primary is not None:
            primary = request_primary(
                block_ids, self._block_sizes_by_group, state_primary
            )
            return self.request_finished(request, block_ids[primary])
        if len(block_ids) > 1:
            if len(block_ids) == len(self._block_sizes_by_group):
                primary = max(
                    range(len(block_ids)),
                    key=lambda i: len(block_ids[i]) * self._block_sizes_by_group[i],
                )
            else:
                primary = 0
            logger.debug(
                "LMCache-Ascend: request_finished_all_groups: %d KV groups; "
                "using primary group %d (%d blocks)",
                len(block_ids),
                primary,
                len(block_ids[primary]),
            )
            return self.request_finished(request, block_ids[primary])
        return self.request_finished(request, block_ids[0])
