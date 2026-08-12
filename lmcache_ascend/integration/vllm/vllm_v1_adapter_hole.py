# SPDX-License-Identifier: Apache-2.0
"""
Hole-mode LMCache connector implementation for non-contiguous CacheBlend reuse.

This module subclasses the upstream LMCache v1 connector flow to support
requests whose reusable prefix contains uncached middle segments ("holes").
Instead of reducing the lookup result to a single contiguous matched-prefix
length, the connector carries hole-aware scheduler and worker state through
lookup, load, blend, and save paths.

See `docs/hole-feature-overview.md` for the maintainer-facing description of
the request flow, type surface, and integration strategy.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional
import time

# Third Party
from lmcache import utils
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    create_lmcache_metadata,
    extract_mm_features,
    mla_enabled,
)
from lmcache.integration.vllm.vllm_v1_adapter import (
    DisaggSpec,
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
    extract_request_configs,
    tmp_disagg_tracker,
)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import need_gpu_interm_buffer
from lmcache.v1.lookup_client.lmcache_lookup_client import (
    LMCacheLookupClient,
    LMCacheLookupServer,
)
from lmcache.v1.rpc.zmq_transport import (
    SocketParams,
    ZmqReqRepClientTransport,
    ZmqRouterServerTransport,
)
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache
from lmcache.v1.trace_utils import (
    advance_layerwise_storers_with_timing,
    emit_request_timer,
    mask_to_string,
    summarize_ranges,
    summarize_slot_mapping,
    trace_flow,
    trace_flow_enabled,
)
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.core.sched.output import SchedulerOutput
import torch

# First Party
from lmcache_ascend import _build_info
from lmcache_ascend.v1.blend.blender import LMCBlender
from lmcache_ascend.v1.blend.hole_blender import LMCBlenderHole
from lmcache_ascend.v1.hole_segment_utils import HoleSegmentHelper
from lmcache_ascend.v1.hole_types import HoleLoadSpec, HoleSaveSpec

if _build_info.__framework_name__ == "pytorch":
    # First Party
    from lmcache_ascend.v1.npu_connector import (
        VLLMBufferLayerwiseNPUConnector,
    )
    from lmcache_ascend.v1.npu_hole_connector import (
        VLLMBufferLayerwiseNPUHoleConnector,
    )
else:
    raise ValueError("Hole connector is only supported in LMCache-Ascend pytorch.")

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _merge_request_output_params(
    base_params: Optional[dict[str, Any]],
    extra_params: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not base_params and not extra_params:
        return None
    merged: dict[str, Any] = {}
    if base_params:
        merged.update(base_params)
    if extra_params:
        merged.update(extra_params)
    return merged


def _set_request_output_metrics(
    request,
    *,
    prompt_tokens: int,
    hit_tokens: int,
    mode: str,
    covered_tokens: Optional[int] = None,
    prefix_miss_tokens: Optional[int] = None,
) -> None:
    prompt_tokens = max(int(prompt_tokens), 0)
    hit_tokens = max(int(hit_tokens), 0)
    req_hit_rate = (
        float(hit_tokens) / float(prompt_tokens) if prompt_tokens > 0 else 0.0
    )
    request._lmcache_prompt_tokens = prompt_tokens
    request._lmcache_hit_tokens = hit_tokens
    request._lmcache_req_hit_rate = req_hit_rate
    request._lmcache_mode = str(mode)
    if covered_tokens is not None:
        request._lmcache_covered_tokens = max(int(covered_tokens), 0)
    if prefix_miss_tokens is not None:
        request._lmcache_prefix_miss_tokens = max(int(prefix_miss_tokens), 0)


def _collect_request_output_metrics(request) -> Optional[dict[str, Any]]:
    prompt_tokens = getattr(request, "_lmcache_prompt_tokens", None)
    hit_tokens = getattr(request, "_lmcache_hit_tokens", None)
    req_hit_rate = getattr(request, "_lmcache_req_hit_rate", None)
    mode = getattr(request, "_lmcache_mode", None)
    if prompt_tokens is None or hit_tokens is None or req_hit_rate is None:
        return None

    params: dict[str, Any] = {
        "req_hit_rate": float(req_hit_rate),
        "lmcache_req_hit_rate": float(req_hit_rate),
        "lmcache_hit_tokens": int(hit_tokens),
        "lmcache_prompt_tokens": int(prompt_tokens),
    }
    if mode is not None:
        params["lmcache_mode"] = str(mode)
    covered_tokens = getattr(request, "_lmcache_covered_tokens", None)
    if covered_tokens is not None:
        params["lmcache_covered_tokens"] = int(covered_tokens)
    prefix_miss_tokens = getattr(request, "_lmcache_prefix_miss_tokens", None)
    if prefix_miss_tokens is not None:
        params["lmcache_prefix_miss_tokens"] = int(prefix_miss_tokens)
    return params


def init_lmcache_engine_hole(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
    role: str,
) -> LMCacheEngine:
    curr_engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if curr_engine:
        return curr_engine

    if lmcache_config.enable_async_loading:
        raise ValueError(
            "Hole CacheBlend only supports synchronous lookup/load. "
            "Set enable_async_loading=False."
        )
    if not lmcache_config.use_layerwise:
        raise ValueError("Hole CacheBlend requires use_layerwise=True.")
    if not lmcache_config.enable_blending:
        raise ValueError("Hole CacheBlend requires enable_blending=True.")

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    assert isinstance(lmcache_config, LMCacheEngineConfig), (
        "LMCache v1 configuration should be passed."
    )

    use_mla = mla_enabled(model_config)
    if use_mla and (
        lmcache_config.remote_serde != "naive"
        and lmcache_config.remote_serde is not None
    ):
        raise ValueError("MLA only works with naive serde mode.")

    if use_mla and lmcache_config.enable_blending:
        raise ValueError(
            "We haven't supported MLA with hole CacheBlend yet. Please disable it."
        )

    if use_mla and not lmcache_config.save_unfull_chunk:
        lmcache_config.save_unfull_chunk = True

    num_gpus = torch.npu.device_count()
    local_rank = parallel_config.rank % num_gpus
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    metadata, _ = create_lmcache_metadata(vllm_config, role=role)

    use_gpu = need_gpu_interm_buffer(lmcache_config)
    vllm_gpu_connector: Optional[GPUConnectorInterface]

    if role == "scheduler":
        vllm_gpu_connector = None
        tpg = SimpleNamespace()
        tpg.broadcast = lambda tensor, src: tensor
        tpg.broadcast_object = lambda obj, src: obj
    else:
        vllm_gpu_connector = VLLMBufferLayerwiseNPUHoleConnector.from_metadata(
            metadata, use_gpu, device
        )
        tpg = get_tp_group()

    return LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        lmcache_config,
        metadata,
        vllm_gpu_connector,
        tpg.broadcast,
        tpg.broadcast_object,
    )


@dataclass
class HoleRequestTracker(RequestTracker):
    prefix_misses_saved: bool = True
    hole_load_spec: Optional[HoleLoadSpec] = None

    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        load_spec: Optional[HoleLoadSpec],
        skip_save: bool,
    ) -> "HoleRequestTracker":
        unfolded_block_ids = []
        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            unfolded_block_ids = new_request.block_ids[0].copy()

        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)
        request_configs = extract_request_configs(new_request.sampling_params)
        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        initial_saved_tokens = 0
        prefix_misses_saved = True
        if load_spec is not None:
            initial_saved_tokens = load_spec.covered_tokens
            prefix_misses_saved = len(load_spec.prefix_miss_ranges) == 0

        return HoleRequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=initial_saved_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            request_configs=request_configs,
            skip_save=skip_save,
            prefix_misses_saved=prefix_misses_saved,
            hole_load_spec=load_spec,
        )

    def set_hole_load_spec(self, load_spec: Optional[HoleLoadSpec]) -> None:
        if load_spec is None:
            return
        self.hole_load_spec = load_spec
        if len(load_spec.prefix_miss_ranges) == 0:
            self.prefix_misses_saved = True


@dataclass
class HoleReqMeta:
    req_id: str
    token_ids: list[int]
    slot_mapping: torch.Tensor
    is_last_prefill: bool = False
    save_spec: Optional[HoleSaveSpec] = None
    load_spec: Optional[HoleLoadSpec] = None
    disagg_spec: Optional[DisaggSpec] = None
    request_configs: Optional[dict] = None

    @staticmethod
    def from_request_tracker(
        tracker: HoleRequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[HoleLoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["HoleReqMeta"]:
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)
        hole_load_spec = tracker.hole_load_spec

        is_last_prefill = input_token_len >= tracker.prompt_len

        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        save_frontier = tracker.num_saved_tokens
        chunk_boundary = (
            (tracker.num_saved_tokens + lmcache_chunk_size) // lmcache_chunk_size
        ) * lmcache_chunk_size
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        pending_prefix_miss_ranges = []
        covered_tokens = 0
        if hole_load_spec is not None:
            covered_tokens = hole_load_spec.covered_tokens
            if not tracker.prefix_misses_saved:
                for start, end in hole_load_spec.prefix_miss_ranges:
                    if end <= num_tokens_to_save:
                        pending_prefix_miss_ranges.append((start, end))

        has_pending_prefix_misses = len(pending_prefix_miss_ranges) > 0
        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or request_skip
            or (
                tracker.num_saved_tokens > 0
                and input_token_len < chunk_boundary
                and not has_pending_prefix_misses
            )
            or (
                tracker.is_decode_phase
                and not save_decode_cache
                and not has_pending_prefix_misses
            )
        )

        effective_load_spec = load_spec
        if effective_load_spec is not None and not effective_load_spec.can_load:
            effective_load_spec = None

        if skip_save and effective_load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = max(tracker.num_saved_tokens, num_tokens_to_save)

        token_ids = input_token_ids.copy()
        if tracker.mm_hashes:
            token_ids_tensor = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids_tensor, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids_tensor.tolist()

        num_blocks = len(tracker.allocated_block_ids)
        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks for %s",
                tracker.req_id,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[: len(token_ids)]

        save_spec = HoleSaveSpec(
            num_saved_tokens=save_frontier,
            prefix_misses_saved=tracker.prefix_misses_saved,
            prefix_miss_ranges=pending_prefix_miss_ranges,
            covered_tokens=covered_tokens,
            num_tokens_to_save=num_tokens_to_save,
            can_save=not skip_save,
        )

        if (
            not skip_save
            and hole_load_spec is not None
            and len(pending_prefix_miss_ranges)
            == len(hole_load_spec.prefix_miss_ranges)
        ):
            # Mark prefix misses as emitted once the save spec contains every
            # eligible hole range for this request. This prevents the next decode
            # forward from rebuilding the same prefix-miss save spec again.
            tracker.prefix_misses_saved = True

        return HoleReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=effective_load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
        )


class LMCacheConnectorV1ImplHole(LMCacheConnectorV1Impl):
    """
    Hole-aware connector subclass for non-contiguous CacheBlend reuse.

    This connector replaces the upstream contiguous-prefix matched-tokens model
    with hole-aware request state (`HoleLoadSpec`, `HoleReqMeta`,
    `HoleLookupResult`). Scheduler-side lookup computes a covered region with
    hit and prefix-miss ranges; worker-side `start_load_kv()` hands that
    covered region (`R1 + H + R2`) to the hole-specific NPU connector and
    blender; `save_kv_layer()` persists the recomputed prefix-miss regions so
    they can become reusable on later requests.

    One non-obvious detail is the duplicated `async_loading` validation guard
    between `init_lmcache_engine_hole()` and `_init_connector_state()`. That
    duplication is intentional because some factory paths can reach connector
    initialization without going through engine creation first. See the comment
    in `_init_connector_state()` and `docs/hole-feature-overview.md` section 4
    for the full request-flow description.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        super().__init__(vllm_config, role, parent)

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        # Base sets up generic connector state and manager-owned services.
        super()._init_connector_state(role, vllm_config, config)

        # NOTE: this guard duplicates the validation in
        # init_lmcache_engine_hole() intentionally. Two factory paths can
        # reach _init_connector_state without the engine-level guard having
        # executed:
        #   (1) scheduler role with enable_scheduler_bypass_lookup=False -
        #       the factory returns None before init_lmcache_engine_hole is
        #       called (__init__.py:696-707)
        #   (2) existing ENGINE_NAME instance reused from
        #       LMCacheEngineBuilder - the factory returns the cached engine
        #       before init_lmcache_engine_hole (__init__.py:710-715)
        # This guard is the authoritative safety net for those paths.
        if config.enable_async_loading:
            raise ValueError(
                "Hole CacheBlend only supports synchronous lookup/load. "
                "Set enable_async_loading=False."
            )

        # Hole-specific list; base uses _layerwise_save_storers for the standard path.
        self.layerwise_storers = []
        self._pending_prefix_miss_save_req_ids: set[str] = set()
        self._save_decode_cache = config.save_decode_cache

        self._lookup_segment_helper = HoleSegmentHelper(
            self.config,
            self.lmcache_engine_metadata,
        )

        if role == KVConnectorRole.SCHEDULER:
            # TODO(scheduler-bypass): OLD behavior — when
            # config.enable_scheduler_bypass_lookup was True, the scheduler built
            # a FULL hole engine (init_lmcache_engine_hole role="scheduler");
            # otherwise metadata-only. Under the manager model the engine comes
            # from the factory. MUST verify the factory patch makes the manager
            # build a scheduler-side hole engine when bypass is enabled, else
            # this behavior is LOST. Resolve when implementing the factory patch.
            # TODO(path-B-cleanup): pure lookup services are constructed inline,
            # duplicating LookupClientFactory's transport setup. A factory hook
            # should eventually expose these services through the manager.
            # This keeps the connector thin and consistent with the post-refactor
            # architecture. Deferred to keep the current rerun-unblocking change small.
            kv_extra = self.lmcache_engine_metadata.kv_connector_extra_config or {}
            rpc_port = kv_extra.get("lmcache_rpc_port", 0)
            assert self.lmcache_engine_metadata.engine_id is not None, (
                "engine_id is required for RPC communication"
            )
            lookup_ids = self.config.get_lookup_server_worker_ids(
                self.lmcache_engine_metadata.use_mla,
                self.lmcache_engine_metadata.world_size,
            )
            ranks = (
                lookup_ids
                if len(lookup_ids) > 0
                else list(range(self.lmcache_engine_metadata.world_size))
            )
            socket_params = [
                SocketParams(
                    socket_path=get_zmq_rpc_path_lmcache(
                        self.lmcache_engine_metadata.engine_id,
                        "lookup_pure",
                        rpc_port,
                        rank,
                    ),
                    rank=rank,
                )
                for rank in ranks
            ]
            transport = ZmqReqRepClientTransport(
                socket_params=socket_params,
                timeout_ms=self.config.lookup_timeout_ms,
            )
            self._pure_lookup_client = LMCacheLookupClient(
                self.config,
                self.lmcache_engine_metadata,
                transport,
            )
        else:
            assert self.use_layerwise and self.enable_blending
            use_gpu = need_gpu_interm_buffer(self.config)
            assert isinstance(
                self.lmcache_engine.gpu_connector, VLLMBufferLayerwiseNPUHoleConnector
            )
            self.legacy_gpu_connector = VLLMBufferLayerwiseNPUConnector.from_metadata(
                self.lmcache_engine.metadata,
                use_gpu,
                self.device,
            )
            self.legacy_blender = LMCBlender(
                self.lmcache_engine,
                self.legacy_gpu_connector,
                VLLMModelTracker.get_model(ENGINE_NAME),
                self.config,
            )
            self.legacy_gpu_connector.fused_rotary_emb = (
                self.legacy_blender.layerwise_model.fused_rotary_emb
            )
            self.blender = LMCBlenderHole(
                self.lmcache_engine,
                self.lmcache_engine.gpu_connector,
                VLLMModelTracker.get_model(ENGINE_NAME),
                self.config,
            )
            # TODO(path-B-cleanup): pure lookup services are constructed inline,
            # duplicating LookupClientFactory's transport setup. A factory hook
            # should eventually expose these services through the manager.
            # This keeps the connector thin and consistent with the post-refactor
            # architecture. Deferred to keep the current rerun-unblocking change small.
            kv_extra = self.lmcache_engine.metadata.kv_connector_extra_config or {}
            rpc_port = kv_extra.get("lmcache_rpc_port", 0)
            assert self.lmcache_engine.metadata.engine_id is not None, (
                "engine_id is required for RPC communication"
            )
            socket_path = get_zmq_rpc_path_lmcache(
                self.lmcache_engine.metadata.engine_id,
                "lookup_pure",
                rpc_port,
                self.lmcache_engine.metadata.worker_id,
            )
            transport = ZmqRouterServerTransport(
                socket_path=socket_path,
            )
            self._pure_lookup_server = LMCacheLookupServer(
                self.lmcache_engine,
                self.lmcache_engine.metadata,
                transport,
            )

        logger.info(
            "LMCache hole connector initialized for role %s with version %s",
            role,
            utils.get_version(),
        )

    def shutdown(self):
        for name in ("_pure_lookup_server", "_pure_lookup_client"):
            resource = getattr(self, name, None)
            if resource is None:
                continue
            try:
                resource.close()
            except Exception:
                logger.exception("Failed to close %s cleanly.", name)
        super().shutdown()

    def _get_lookup_token_ids(self, request) -> list[int]:
        token_ids = list(request.all_token_ids)
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            token_tensor = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(token_tensor, mm_hashes, mm_positions)
            token_ids = token_tensor.tolist()
        if self.skip_last_n_tokens > 0:
            token_ids = token_ids[: -self.skip_last_n_tokens]
        return token_ids

    def _remaining_segment_count_after_prefix(
        self,
        token_ids: list[int],
        cached_prefix_tokens: int,
    ) -> int:
        remaining = 0
        for start, end in self._lookup_segment_helper.split_ranges(token_ids):
            if end <= cached_prefix_tokens:
                continue
            remaining += 1
        return remaining

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        if request.request_id.startswith("mock_req"):
            return 0
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        lookup_total_start = time.perf_counter()
        req_id = request.request_id
        token_ids = self._get_lookup_token_ids(request)
        request_configs = extract_request_configs(request.sampling_params)

        if req_id not in self._requests_priority:
            self._requests_priority[req_id] = getattr(request, "priority", 0)

        lookup_cache_start = time.perf_counter()
        nohole_hit_tokens = self._pure_lookup_client.lookup_cache(lookup_id=req_id)
        emit_request_timer(
            "lookup_cache",
            req_id=req_id,
            duration_ms=(time.perf_counter() - lookup_cache_start) * 1000.0,
            path="hole",
        )

        if nohole_hit_tokens == -1:
            lookup_rpc_start = time.perf_counter()
            nohole_hit_tokens = self._pure_lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )
            emit_request_timer(
                "lookup_rpc",
                req_id=req_id,
                duration_ms=(time.perf_counter() - lookup_rpc_start) * 1000.0,
                path="hole",
            )

        assert nohole_hit_tokens is not None

        remaining_tail_segments = self._remaining_segment_count_after_prefix(
            token_ids,
            nohole_hit_tokens,
        )
        if nohole_hit_tokens == request.num_tokens or remaining_tail_segments < 2:
            lookup_postprocess_start = time.perf_counter()
            need_to_allocate = nohole_hit_tokens - num_computed_tokens
            if nohole_hit_tokens == request.num_tokens and nohole_hit_tokens > 0:
                need_to_allocate -= 1

            mode = "pure_hit" if nohole_hit_tokens == request.num_tokens else "legacy"
            self.load_specs[req_id] = LoadSpec(
                vllm_cached_tokens=num_computed_tokens,
                lmcache_cached_tokens=nohole_hit_tokens,
                can_load=False,
            )
            logger.info(
                "Reqid: %s, Hole adapter fast-path resolved mode=%s with LMCache hit "
                "tokens=%d, need to load=%d",
                req_id,
                mode,
                nohole_hit_tokens,
                need_to_allocate,
            )
            if trace_flow_enabled():
                trace_flow(
                    "vllm_adapter.hole",
                    "lookup_result",
                    req_id=req_id,
                    total_tokens=request.num_tokens,
                    num_computed_tokens=num_computed_tokens,
                    mode=mode,
                    covered_tokens=nohole_hit_tokens,
                    hit_tokens=nohole_hit_tokens,
                    prefix_miss_tokens=0,
                    need_to_allocate=need_to_allocate,
                    pure_lookup_remaining_segments=remaining_tail_segments,
                )
            _set_request_output_metrics(
                request,
                prompt_tokens=len(token_ids),
                hit_tokens=nohole_hit_tokens,
                mode=mode,
                covered_tokens=nohole_hit_tokens,
                prefix_miss_tokens=0,
            )
            emit_request_timer(
                "lookup_postprocess",
                req_id=req_id,
                duration_ms=(time.perf_counter() - lookup_postprocess_start) * 1000.0,
                path="hole",
                load_mode=mode,
            )
            emit_request_timer(
                "lookup_total",
                req_id=req_id,
                duration_ms=(time.perf_counter() - lookup_total_start) * 1000.0,
                path="hole",
                load_mode=mode,
            )
            if need_to_allocate <= 0:
                return 0
            return need_to_allocate

        lookup_hole_cache_start = time.perf_counter()
        hole_result = self.lookup_client.lookup_cache(lookup_id=req_id)
        emit_request_timer(
            "lookup_hole_cache",
            req_id=req_id,
            duration_ms=(time.perf_counter() - lookup_hole_cache_start) * 1000.0,
            path="hole",
        )
        if hole_result is None:
            lookup_hole_rpc_start = time.perf_counter()
            hole_result = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )
            emit_request_timer(
                "lookup_hole_rpc",
                req_id=req_id,
                duration_ms=(time.perf_counter() - lookup_hole_rpc_start) * 1000.0,
                path="hole",
            )

        lookup_postprocess_start = time.perf_counter()
        covered_tokens = hole_result.covered_tokens
        hit_tokens = sum(end - start for start, end in hole_result.hit_ranges)
        prefix_miss_tokens = sum(
            end - start for start, end in hole_result.prefix_miss_ranges
        )
        need_to_allocate = covered_tokens - num_computed_tokens
        if covered_tokens == request.num_tokens and covered_tokens > 0:
            need_to_allocate -= 1

        if hole_result.mode == "pure_hit":
            self.load_specs[req_id] = LoadSpec(
                vllm_cached_tokens=num_computed_tokens,
                lmcache_cached_tokens=covered_tokens,
                can_load=False,
            )
        else:
            self.load_specs[req_id] = HoleLoadSpec(
                mode=hole_result.mode,
                covered_tokens=covered_tokens,
                tail_start=hole_result.tail_start,
                hit_ranges=list(hole_result.hit_ranges),
                prefix_miss_ranges=list(hole_result.prefix_miss_ranges),
                vllm_cached_tokens=num_computed_tokens,
                can_load=False,
                location=hole_result.location,
            )

        logger.info(
            "Reqid: %s, Total tokens %d, LMCache hit tokens: %d, need to load: %d",
            req_id,
            request.num_tokens,
            hit_tokens,
            need_to_allocate,
        )
        logger.info(
            "Reqid: %s, Hole mode=%s, covered tokens=%d, prefix miss tokens=%d",
            req_id,
            hole_result.mode,
            covered_tokens,
            prefix_miss_tokens,
        )
        if trace_flow_enabled():
            trace_flow(
                "vllm_adapter.hole",
                "lookup_result",
                req_id=req_id,
                total_tokens=request.num_tokens,
                num_computed_tokens=num_computed_tokens,
                mode=hole_result.mode,
                covered_tokens=covered_tokens,
                hit_ranges=summarize_ranges(hole_result.hit_ranges),
                prefix_miss_ranges=summarize_ranges(hole_result.prefix_miss_ranges),
                hit_tokens=hit_tokens,
                prefix_miss_tokens=prefix_miss_tokens,
                need_to_allocate=need_to_allocate,
            )
        _set_request_output_metrics(
            request,
            prompt_tokens=len(token_ids),
            hit_tokens=hit_tokens,
            mode=hole_result.mode,
            covered_tokens=covered_tokens,
            prefix_miss_tokens=prefix_miss_tokens,
        )
        emit_request_timer(
            "lookup_postprocess",
            req_id=req_id,
            duration_ms=(time.perf_counter() - lookup_postprocess_start) * 1000.0,
            path="hole",
            load_mode=hole_result.mode,
        )
        emit_request_timer(
            "lookup_total",
            req_id=req_id,
            duration_ms=(time.perf_counter() - lookup_total_start) * 1000.0,
            path="hole",
            load_mode=hole_result.mode,
        )

        if need_to_allocate <= 0:
            return 0
        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        self.lookup_client.clear_lookup_status(request.request_id)
        if self._pure_lookup_client is not None:
            self._pure_lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]
            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )
            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
            )
            tmp_disagg_tracker[request.request_id] = disagg_spec

        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            return

        load_spec = self.load_specs[request.request_id]
        if num_external_tokens == 0:
            load_spec.can_load = False
            return

        if isinstance(load_spec, LoadSpec):
            recalc_last = (
                1 if load_spec.lmcache_cached_tokens == request.num_tokens else 0
            )
            assert (
                num_external_tokens
                == load_spec.lmcache_cached_tokens
                - load_spec.vllm_cached_tokens
                - recalc_last
            ), (
                f"Mismatch in tokens to load for request {request.request_id}: "
                f"{num_external_tokens} vs {load_spec.lmcache_cached_tokens} - "
                f"{load_spec.vllm_cached_tokens} - {recalc_last}"
            )
            load_spec.can_load = True
            return

        recalc_last = 1 if load_spec.covered_tokens == request.num_tokens else 0
        assert (
            num_external_tokens
            == load_spec.covered_tokens - load_spec.vllm_cached_tokens - recalc_last
        ), (
            f"Mismatch in tokens to load for request {request.request_id}: "
            f"{num_external_tokens} vs {load_spec.covered_tokens} - "
            f"{load_spec.vllm_cached_tokens} - {recalc_last}"
        )
        load_spec.can_load = True

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> LMCacheConnectorMetadata:
        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save
        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            request_priority = self._requests_priority.pop(request.req_id, 0)
            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            if isinstance(load_spec, LoadSpec):
                request_tracker = RequestTracker.from_new_request(
                    self.config,
                    request,
                    num_tokens_to_compute,
                    load_spec.lmcache_cached_tokens,
                    skip_save,
                )
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self._save_decode_cache,
                )
            else:
                request_tracker = HoleRequestTracker.from_new_request(
                    self.config,
                    request,
                    num_tokens_to_compute,
                    load_spec,
                    skip_save,
                )
                req_meta = HoleReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self._save_decode_cache,
                )
            self._request_trackers[request.req_id] = request_tracker
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if isinstance(cached_reqs, list):
            for req in cached_reqs:
                load_spec = self.load_specs.pop(req.req_id, None)
                request_tracker = self._request_trackers[req.req_id]
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                    if isinstance(load_spec, LoadSpec):
                        lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    else:
                        lmcache_cached_tokens = load_spec.covered_tokens
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None
                    all_token_ids = list(vllm_request.all_token_ids)
                if isinstance(request_tracker, HoleRequestTracker):
                    request_tracker.set_hole_load_spec(
                        load_spec if isinstance(load_spec, HoleLoadSpec) else None
                    )
                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )
                if isinstance(request_tracker, HoleRequestTracker):
                    req_meta = HoleReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        self._lmcache_chunk_size,
                        load_spec=load_spec
                        if isinstance(load_spec, HoleLoadSpec)
                        else None,
                        discard_partial_chunks=self._discard_partial_chunks,
                        save_decode_cache=self._save_decode_cache,
                    )
                else:
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        self._lmcache_chunk_size,
                        load_spec=load_spec
                        if isinstance(load_spec, LoadSpec)
                        else None,
                        discard_partial_chunks=self._discard_partial_chunks,
                        save_decode_cache=self._save_decode_cache,
                    )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            request = self._unfinished_requests.get(req_id)
            if request is None:
                raise ValueError(f"Request {req_id} missing from unfinished requests")
            num_current_tokens = request.num_computed_tokens
            new_token_ids = request.all_token_ids[
                num_current_tokens : num_current_tokens + num_new_tokens
            ]
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                vllm_cached_tokens = load_spec.vllm_cached_tokens
                if isinstance(load_spec, LoadSpec):
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                else:
                    lmcache_cached_tokens = load_spec.covered_tokens
            if isinstance(request_tracker, HoleRequestTracker):
                request_tracker.set_hole_load_spec(
                    load_spec if isinstance(load_spec, HoleLoadSpec) else None
                )

            if hasattr(cached_reqs, "resumed_req_ids"):
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}."
                )

            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                assert request.num_computed_tokens == max(
                    lmcache_cached_tokens, vllm_cached_tokens
                ), (
                    f"Preempted request {req_id} has unexpected num_computed_tokens "
                    f"{request.num_computed_tokens}"
                )

            all_token_ids = list(request.all_token_ids) if preempted else None
            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )
            if isinstance(request_tracker, HoleRequestTracker):
                req_meta = HoleReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec
                    if isinstance(load_spec, HoleLoadSpec)
                    else None,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self._save_decode_cache,
                )
            else:
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec if isinstance(load_spec, LoadSpec) else None,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self._save_decode_cache,
                )
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self.current_layer = 0
        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector; use register_kv_caches "
                "to initialize kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)
        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but attn_metadata is None")
            return

        assert self.lmcache_engine is not None
        self.layerwise_retrievers = []

        for request in metadata.requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue

            load_spec = request.load_spec
            if isinstance(load_spec, LoadSpec):
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                if lmcache_cached_tokens <= load_spec.vllm_cached_tokens:
                    continue

                tokens = request.token_ids
                slot_mapping = request.slot_mapping.to(self.device)
                token_mask = torch.ones(len(tokens), dtype=torch.bool)
                masked_token_count = (
                    load_spec.vllm_cached_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                token_mask[:masked_token_count] = False
                if trace_flow_enabled():
                    trace_flow(
                        "vllm_adapter.nohole",
                        "start_load_request",
                        req_id=request.req_id,
                        token_count=len(tokens),
                        lmcache_cached_tokens=lmcache_cached_tokens,
                        vllm_cached_tokens=load_spec.vllm_cached_tokens,
                        masked_token_count=masked_token_count,
                        token_mask=mask_to_string(token_mask[:lmcache_cached_tokens]),
                        slot_mapping=summarize_slot_mapping(
                            slot_mapping[:lmcache_cached_tokens]
                        ),
                    )
                logger.info(
                    "Reqid: %s, Hole lookup resolved to pure_hit; using legacy/nohole "
                    "load path",
                    request.req_id,
                )
                self.legacy_blender.blend(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                    timer_path="nohole",
                    timer_load_mode="pure_hit",
                )
            else:
                if load_spec.covered_tokens <= load_spec.vllm_cached_tokens:
                    continue

                if load_spec.mode == "pure_hit":
                    tokens = request.token_ids[: load_spec.covered_tokens]
                    slot_mapping = request.slot_mapping[: load_spec.covered_tokens].to(
                        self.device
                    )
                    token_mask = torch.ones(len(tokens), dtype=torch.bool)
                    token_mask[: load_spec.vllm_cached_tokens] = False
                    if trace_flow_enabled():
                        trace_flow(
                            "vllm_adapter.nohole",
                            "start_load_request",
                            req_id=request.req_id,
                            token_count=len(tokens),
                            lmcache_cached_tokens=load_spec.covered_tokens,
                            vllm_cached_tokens=load_spec.vllm_cached_tokens,
                            masked_token_count=load_spec.vllm_cached_tokens,
                            token_mask=mask_to_string(token_mask),
                            slot_mapping=summarize_slot_mapping(slot_mapping),
                        )
                    logger.info(
                        "Reqid: %s, Hole lookup resolved to pure_hit; "
                        "using legacy/nohole load path",
                        request.req_id,
                    )
                    self.legacy_blender.blend(
                        tokens,
                        token_mask,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        request_configs=request.request_configs,
                        req_id=request.req_id,
                        timer_path="nohole",
                        timer_load_mode="pure_hit",
                    )
                else:
                    tokens = request.token_ids[: load_spec.covered_tokens]
                    slot_mapping = request.slot_mapping[: load_spec.covered_tokens].to(
                        self.device
                    )
                    token_mask = torch.ones(len(tokens), dtype=torch.bool)
                    token_mask[: load_spec.vllm_cached_tokens] = False
                    if trace_flow_enabled():
                        trace_flow(
                            "vllm_adapter.hole",
                            "start_load_request",
                            req_id=request.req_id,
                            mode=load_spec.mode,
                            covered_tokens=load_spec.covered_tokens,
                            vllm_cached_tokens=load_spec.vllm_cached_tokens,
                            hit_ranges=summarize_ranges(load_spec.hit_ranges),
                            prefix_miss_ranges=summarize_ranges(
                                load_spec.prefix_miss_ranges
                            ),
                            token_mask=mask_to_string(token_mask),
                            slot_mapping=summarize_slot_mapping(slot_mapping),
                        )
                    self.lmcache_engine.gpu_connector.set_gap_buffer_semantics(
                        "hole",
                        request.req_id,
                        load_spec.mode,
                    )
                    try:
                        self.blender.blend(
                            tokens,
                            load_spec,
                            token_mask,
                            kvcaches=kvcaches,
                            slot_mapping=slot_mapping,
                            prefix_start=load_spec.vllm_cached_tokens,
                            request_configs=request.request_configs,
                            req_id=request.req_id,
                            timer_path="hole",
                            timer_load_mode=load_spec.mode,
                        )
                    finally:
                        self.lmcache_engine.gpu_connector.set_gap_buffer_semantics(
                            "hole"
                        )

            self._stats_monitor.update_interval_vllm_hit_tokens(
                load_spec.vllm_cached_tokens
            )
            self._stats_monitor.update_interval_prompt_tokens(len(request.token_ids))

    @_lmcache_nvtx_annotate
    def _advance_layerwise_storers(
        self,
        *,
        layer_id: int,
        timer_path: str,
    ) -> None:
        advance_layerwise_storers_with_timing(
            self.layerwise_storers,
            layer_id=layer_id,
            timer_path=timer_path,
        )

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return
        if self.kv_role == "kv_consumer":
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)
        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        if self.current_layer == 0:
            self.layerwise_storers = []
            self._pending_prefix_miss_save_req_ids.clear()
            is_first = True

            for request in connector_metadata.requests:
                save_spec = request.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue

                token_ids = request.token_ids
                slot_mapping = request.slot_mapping.to(self.device)
                if isinstance(save_spec, SaveSpec):
                    if self.kv_role == "kv_producer":
                        skip_leading_tokens = 0
                    else:
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
                    if trace_flow_enabled():
                        trace_flow(
                            "vllm_adapter.nohole",
                            "save_request",
                            req_id=request.req_id,
                            token_count=len(token_ids),
                            skip_leading_tokens=skip_leading_tokens,
                            store_mask=mask_to_string(store_mask),
                            slot_mapping=summarize_slot_mapping(slot_mapping),
                        )
                    logger.info(
                        "Storing KV cache for %d out of %d tokens "
                        "(skip_leading_tokens=%d) for request %s",
                        len(token_ids) - skip_leading_tokens,
                        len(token_ids),
                        skip_leading_tokens,
                        request.req_id,
                    )
                    layerwise_storer = self.lmcache_engine.store_layer(
                        token_ids,
                        mask=store_mask,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        offset=skip_leading_tokens,
                        sync=is_first,
                        req_id=request.req_id,
                        request_configs=request.request_configs,
                    )
                    load_mode = (
                        None
                        if request.load_spec is None
                        else getattr(request.load_spec, "mode", None)
                    )
                    self.layerwise_storers.append(
                        (request.req_id, layerwise_storer, load_mode)
                    )
                    if is_first:
                        is_first = False
                    continue

                save_limit = save_spec.num_tokens_to_save
                if save_limit == 0 and len(save_spec.prefix_miss_ranges) == 0:
                    continue

                if not save_spec.prefix_misses_saved:
                    for start, end in save_spec.prefix_miss_ranges:
                        miss_tokens = token_ids[start:end]
                        miss_slots = slot_mapping[start:end]
                        if len(miss_tokens) == 0:
                            continue
                        # Future optimization: batch all prefix-miss ranges in one
                        # sparse store path to amortize per-hole Python/storage cost.
                        logger.info(
                            "Storing hole prefix-miss KV cache for %d tokens "
                            "(request_len=%d, global_range=[%d,%d), "
                            "local_skip_leading_tokens=%d) "
                            "for request %s",
                            len(miss_tokens),
                            len(token_ids),
                            start,
                            end,
                            0,
                            request.req_id,
                        )
                        if trace_flow_enabled():
                            trace_flow(
                                "vllm_adapter.hole",
                                "save_prefix_miss",
                                req_id=request.req_id,
                                global_start=start,
                                global_end=end,
                                miss_len=len(miss_tokens),
                                slot_mapping=summarize_slot_mapping(miss_slots),
                                position_offset=start,
                            )
                        layerwise_storer = self.lmcache_engine.store_layer(
                            miss_tokens,
                            mask=torch.ones(len(miss_tokens), dtype=torch.bool),
                            kvcaches=kvcaches,
                            slot_mapping=miss_slots,
                            offset=0,
                            position_offset=start,
                            sync=is_first,
                            req_id=request.req_id,
                            request_configs=request.request_configs,
                        )
                        load_mode = (
                            None
                            if request.load_spec is None
                            else getattr(request.load_spec, "mode", None)
                        )
                        self.layerwise_storers.append(
                            (request.req_id, layerwise_storer, load_mode)
                        )
                        self._pending_prefix_miss_save_req_ids.add(request.req_id)
                        if is_first:
                            is_first = False

                    tracker = self._request_trackers.get(request.req_id)
                    if (
                        tracker is not None
                        and isinstance(tracker, HoleRequestTracker)
                        and tracker.hole_load_spec is not None
                    ):
                        all_prefix_misses_enqueued = True
                        for _, end in tracker.hole_load_spec.prefix_miss_ranges:
                            if end > save_limit:
                                all_prefix_misses_enqueued = False
                                break
                        if all_prefix_misses_enqueued:
                            # Mark prefix misses as saved as soon as they are
                            # enqueued. Otherwise each later decode step rebuilds
                            # a save spec with the same hole range and re-saves
                            # the exact same prefix-miss chunk over and over.
                            tracker.prefix_misses_saved = True

                if save_limit <= save_spec.covered_tokens:
                    continue

                store_token_ids = token_ids[:save_limit]
                store_slot_mapping = slot_mapping[:save_limit]
                store_mask = torch.ones(len(store_token_ids), dtype=torch.bool)
                store_mask[: save_spec.covered_tokens] = False
                if trace_flow_enabled():
                    trace_flow(
                        "vllm_adapter.hole",
                        "save_tail_request",
                        req_id=request.req_id,
                        save_limit=save_limit,
                        covered_tokens=save_spec.covered_tokens,
                        store_mask=mask_to_string(store_mask),
                        slot_mapping=summarize_slot_mapping(store_slot_mapping),
                    )
                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(store_token_ids) - save_spec.covered_tokens,
                    len(store_token_ids),
                    save_spec.covered_tokens,
                    request.req_id,
                )

                layerwise_storer = self.lmcache_engine.store_layer(
                    store_token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=store_slot_mapping,
                    offset=save_spec.covered_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                    request_configs=request.request_configs,
                )
                load_mode = (
                    None
                    if request.load_spec is None
                    else getattr(request.load_spec, "mode", None)
                )
                self.layerwise_storers.append(
                    (request.req_id, layerwise_storer, load_mode)
                )
                if is_first:
                    is_first = False

        self._advance_layerwise_storers(
            layer_id=self.current_layer,
            timer_path="hole",
        )

        self.current_layer += 1

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            return

        if self.use_layerwise:
            for _, layerwise_storer, _load_mode in self.layerwise_storers:
                next(layerwise_storer)

            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)

            for req_id in self._pending_prefix_miss_save_req_ids:
                tracker = self._request_trackers.get(req_id)
                if (
                    tracker is None
                    or not isinstance(tracker, HoleRequestTracker)
                    or tracker.hole_load_spec is None
                ):
                    continue
                last_prefix_miss_end = 0
                for _, end in tracker.hole_load_spec.prefix_miss_ranges:
                    last_prefix_miss_end = max(last_prefix_miss_end, end)
                if tracker.num_saved_tokens >= last_prefix_miss_end:
                    tracker.prefix_misses_saved = True

            self._pending_prefix_miss_save_req_ids.clear()
            return

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        should_wait, return_params = super().request_finished(request, block_ids)
        metric_params = _collect_request_output_metrics(request)
        return should_wait, _merge_request_output_params(return_params, metric_params)
