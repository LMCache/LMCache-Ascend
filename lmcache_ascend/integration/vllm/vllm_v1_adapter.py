# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
    LMCacheConnectorV1ImplMultiGroup,
)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

# First Party
from lmcache_ascend.integration.vllm.multi_spec_flatten import (
    build_flat_kv_caches,
    should_flatten_kv_caches,
)
from lmcache_ascend.integration.vllm.skip_state_groups import (
    apply_skip_policy_from_env_to_flattened,
)
from lmcache_ascend.v1.npu_connector.npu_connectors import build_mp_launch_meta
from lmcache_ascend.v1.slot_mapping_utils import (
    iter_lmcache_chunk_ranges,
    iter_store_chunk_ranges,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.request import RequestStatus
import torch

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Impl(LMCacheConnectorV1ImplMultiGroup):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
        kv_cache_config: Optional[Any] = None,
    ):
        logger.debug("Initializing LMCacheAscendConnectorV1Impl")
        super().__init__(vllm_config, role, parent, kv_cache_config=kv_cache_config)
        self.store_async = self.config.store_async
        self._wait_for_save_done = True
        self._finished_req_ids_waiting_for_save: set[str] = set()
        self._late_finished_sending: set[str] = set()
        logger.debug("store_async: %s", self.store_async)

    @_lmcache_nvtx_annotate
    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Preprocess multi-spec layers and push upstream layout hints.

        The model runner passes one tuple entry per layer name, but multi-spec
        layers pack multiple 4-D sub-tensors (one per spec/scheduler-group).
        With bundling (default), the tuple is kept as-is and the NPU connector
        derives per-group parameters (block size, hidden bytes, DMA stride)
        dimensionality-agnostically via build_kv_layer_groups.
        """
        flat_kv = kv_caches
        sched_by_layer: tuple[int, ...] | None = None
        layer_to_groups: dict[str, list[int]] | None = None
        cache_cfg = getattr(self._vllm_config, "cache_config", None)
        block_size = int(getattr(cache_cfg, "block_size", 0) or 0) if cache_cfg else 0
        ie_logical_block_size = block_size
        if self._kv_cache_config is not None and getattr(
            self._kv_cache_config, "kv_cache_groups", None
        ):
            ie_logical_block_size = max(
                int(g.kv_cache_spec.block_size)
                for g in self._kv_cache_config.kv_cache_groups
            )

        bundled = False
        if should_flatten_kv_caches(self._kv_cache_config):
            flat_kv, sched_by_layer, layer_to_groups, bundled = build_flat_kv_caches(
                kv_caches,
                self._kv_cache_config,
                ie_logical_block_size=ie_logical_block_size or None,
            )
            flat_kv, sched_by_layer, layer_to_groups = apply_skip_policy_from_env_to_flattened(
                self._kv_cache_config,
                flat_kv,
                sched_by_layer,
                layer_to_groups,
                bundled=bundled,
            )
            logger.info(
                "Preprocessed multi-spec KV caches: %d model layers -> "
                "%d logical layers (bundled=%s)",
                len(kv_caches),
                len(flat_kv),
                bundled,
            )

        try:
            # Inject extracted layout hints into the NPU connector
            engine = getattr(self, "lmcache_engine", None)
            connector = getattr(engine, "gpu_connector", None) if engine else None
            if ie_logical_block_size and connector is not None and hasattr(
                connector, "layout_hints"
            ):
                hints = connector.layout_hints or {}
                if block_size:
                    hints["vllm_block_size"] = block_size
                if self._num_kv_groups > 1:
                    hints["block_sizes_by_group"] = self._block_sizes_by_group
                hints["inference_engine_logical_block_size"] = ie_logical_block_size
                hints["compress_ratios_by_group"] = self._compress_ratios_by_group
                hints["sliding_window_size_by_group"] = getattr(
                    self, "_sliding_window_size_by_group", None
                )
                if sched_by_layer is not None:
                    hints["scheduler_group_by_flat_layer"] = sched_by_layer
                if layer_to_groups is not None:
                    hints["layer_to_scheduler_groups"] = layer_to_groups
                    hints["model_kv_caches"] = kv_caches
                    hints["flat_layer_names"] = list(flat_kv.keys())
                if should_flatten_kv_caches(self._kv_cache_config):
                    hints["bundle_multi_spec"] = bundled
                connector.layout_hints = hints
        except Exception:
            logger.warning(
                "Failed to push layout hints into NPU connector",
                exc_info=True,
            )

        # Build kv_layer_groups_manager before post_init() so
        # metadata.get_shapes() allocates one MemoryObj slot per NPU group.
        # Pointer tables stay lazy in _initialize_pointers on first store.
        flatten_active = should_flatten_kv_caches(self._kv_cache_config)
        try:
            engine = getattr(self, "lmcache_engine", None)
            connector = getattr(engine, "gpu_connector", None) if engine else None
            if connector is not None and hasattr(connector, "ensure_kv_layer_groups"):
                connector.ensure_kv_layer_groups(list(flat_kv.values()))
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
            if flatten_active:
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

        super().register_kv_caches(flat_kv, *args, **kwargs)

    # Upstream start_load_kv only transfers the primary group's slot_mapping.
    # Multi-group retrieve needs ALL per-group slot mappings on NPU so the
    # connector can DMA each spec's KV plane to the correct paged blocks.
    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self.current_layer = 0
        self._wait_for_save_done = False

        if self._num_kv_groups <= 1:
            super().start_load_kv(forward_context, **kwargs)
            return

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        if len(self.kv_caches) == 0:
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None
        kvcaches = list(self.kv_caches.values())
        gpu_connector = self.lmcache_engine.gpu_connector
        self.layerwise_retrievers = []

        last_idx = None
        for idx, request in enumerate(metadata.requests):
            if request.load_spec is not None and request.load_spec.can_load:
                last_idx = idx

        for idx, request in enumerate(metadata.requests):
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
            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens

            slot_mappings_npu: list[torch.Tensor] = []
            filtered_slot_mappings_npu: tuple[torch.Tensor, ...] | None = None
            mp_launch_meta = None
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
                if (
                    filtered_slot_mappings_npu is not None
                    and request.slot_valid_prefix_by_group is not None
                ):
                    gpu_connector._initialize_pointers(kvcaches)
                    chunk_ranges = iter_lmcache_chunk_ranges(
                        lmcache_cached_tokens,
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        lmcache_chunk_size=self._lmcache_chunk_size,
                    )
                    if chunk_ranges:
                        mp_launch_meta = build_mp_launch_meta(
                            gpu_connector,
                            chunk_ranges=chunk_ranges,
                            slot_mappings_by_group=tuple(slot_mappings_cpu),
                            prefixes_by_group=request.slot_valid_prefix_by_group,
                            filtered_slot_mappings_npu=filtered_slot_mappings_npu,
                            compress_ratios=tuple(self._compress_ratios_by_group),
                            stream=gpu_connector.load_stream,
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
            if mp_launch_meta is not None:
                retrieve_kwargs["mp_launch_meta"] = mp_launch_meta

            if self.use_layerwise:
                sync = idx == last_idx
                if self.enable_blending:
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping_npu[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
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

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            if self.lmcache_engine is not None:
                for request in connector_metadata.requests:
                    self.lmcache_engine.lookup_unpin(request.req_id)
            self._wait_for_save_done = True
            return

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

        for request in connector_metadata.requests:
            self.lmcache_engine.lookup_unpin(request.req_id)

            try:
                save_spec = request.save_spec
                if (
                    save_spec is None or not save_spec.can_save
                ) and self.kv_role != "kv_producer":
                    continue

                token_ids = request.token_ids

                pg = request.primary_kv_group_idx
                slot_mappings_cpu: list[torch.Tensor] = []
                for group_idx in range(request.num_kv_groups):
                    group_slot_mapping = request.get_slot_mapping(group_idx)
                    assert isinstance(group_slot_mapping, torch.Tensor)
                    assert len(group_slot_mapping) <= len(token_ids)
                    slot_mappings_cpu.append(group_slot_mapping)

                slot_mapping = slot_mappings_cpu[pg]
                if request.num_kv_groups > 1:
                    logger.info(
                        "Ascend wait_for_save: multi-group slot_mapping "
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
                            sm_cpu.to(device="npu", dtype=torch.long)
                        )
                    slot_mapping_npu = slot_mappings_npu[pg]
                    if request.filtered_slot_by_group is not None:
                        filtered_slot_mappings_npu = tuple(
                            sm_cpu.to(device="npu", dtype=torch.long)
                            for sm_cpu in request.filtered_slot_by_group
                        )
                # lmcache-ascend end ---------------------

                skip_leading_tokens = save_spec.skip_leading_tokens

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

                if (
                    filtered_slot_mappings_npu is not None
                    and request.slot_valid_prefix_by_group is not None
                ):
                    gpu_connector = self.lmcache_engine.gpu_connector
                    gpu_connector._initialize_pointers(kvcaches)
                    chunk_ranges = iter_store_chunk_ranges(
                        len(token_ids),
                        skip_leading_tokens,
                        self._lmcache_chunk_size,
                    )
                    if chunk_ranges:
                        store_kwargs["mp_launch_meta"] = build_mp_launch_meta(
                            gpu_connector,
                            chunk_ranges=chunk_ranges,
                            slot_mappings_by_group=tuple(slot_mappings_cpu),
                            prefixes_by_group=request.slot_valid_prefix_by_group,
                            filtered_slot_mappings_npu=filtered_slot_mappings_npu,
                            compress_ratios=tuple(self._compress_ratios_by_group),
                            stream=gpu_connector.store_stream,
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

    def _may_register_store_after_wait_for_save(self, request: "Request") -> bool:
        if self.kv_role == "kv_consumer":
            return False
        save_spec = request.save_spec
        if save_spec is None:
            return False
        if not save_spec.can_save and self.kv_role != "kv_producer":
            return False
        return save_spec.skip_leading_tokens != len(request.token_ids)

    def _replay_finished_stores_after_save(self) -> None:
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
        _, return_params = super().request_finished(request, block_ids)

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
        """vLLM HMA hook; delegates to :meth:`request_finished` (see upstream LMCache)."""
        if not block_ids:
            return False, None
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
