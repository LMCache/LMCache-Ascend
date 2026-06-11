# SPDX-License-Identifier: Apache-2.0
"""Multi-group vLLM adapter layer for LMCache-Ascend (DSv4 / HMA).

vllm adapter overrides for multi-group KV cache without modificaiton to upstream LMCache.
Ascend-specific overrides remain in ``vllm_v1_adapter.LMCacheAscendConnectorV1Impl``.
"""
# Standard
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Optional, Union

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta as UpstreamReqMeta,
    RequestTracker as UpstreamRequestTracker,
    logger,
)
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig

from lmcache_ascend.v1.slot_mapping_utils import build_filtered_slot_mappings

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.sched.output import NewRequestData

BlockIdsLike = Optional[Union[list[int], list[list[int]], "tuple[list[int], ...]"]]


def _empty_block_ids_by_group(num_groups: int) -> "tuple[list[int], ...]":
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    return tuple([] for _ in range(num_groups))


def _normalize_block_ids(
    block_ids: BlockIdsLike,
    expected_num_groups: int,
) -> "tuple[list[int], ...]":
    """Normalize vLLM block ids into one list per KV cache group.

    vLLM has used both flat single-group block id lists and grouped
    tuple/list layouts across releases.  Keep the normalized form grouped
    so mixed KV-cache models can preserve each group's allocation.
    """
    if expected_num_groups < 1:
        raise ValueError(
            f"expected_num_groups must be >= 1, got {expected_num_groups}"
        )

    if block_ids is None:
        return _empty_block_ids_by_group(expected_num_groups)

    if isinstance(block_ids, list):
        if len(block_ids) == 0:
            return _empty_block_ids_by_group(expected_num_groups)

        if all(
            isinstance(group_block_ids, (list, tuple))
            for group_block_ids in block_ids
        ):
            if len(block_ids) != expected_num_groups:
                raise ValueError(
                    "Block group count mismatch: "
                    f"expected {expected_num_groups}, got {len(block_ids)}"
                )
            return tuple(list(group_block_ids) for group_block_ids in block_ids)

        if expected_num_groups != 1:
            raise ValueError(
                "Received single-group block_ids for a multi-group request: "
                f"expected_num_groups={expected_num_groups}"
            )
        return (block_ids.copy(),)

    if isinstance(block_ids, tuple):
        if len(block_ids) != expected_num_groups:
            raise ValueError(
                "Block group count mismatch: "
                f"expected {expected_num_groups}, got {len(block_ids)}"
            )
        return tuple(list(group_block_ids) for group_block_ids in block_ids)

    raise ValueError(f"Unsupported block_ids type: {type(block_ids)}")


def _normalize_block_sizes(
    block_sizes: Union[int, "tuple[int, ...]", list[int]],
    expected_num_groups: int,
) -> "tuple[int, ...]":
    if isinstance(block_sizes, int):
        block_sizes_by_group: "tuple[int, ...]" = (block_sizes,)
    else:
        block_sizes_by_group = tuple(block_sizes)

    if len(block_sizes_by_group) != expected_num_groups:
        raise ValueError(
            "Block size group count mismatch: "
            f"expected {expected_num_groups}, got {len(block_sizes_by_group)}"
        )
    return block_sizes_by_group


def _sliding_window_store_mask(
    tokens_uncompressed: torch.Tensor,
    sliding_win_size: int,
    lmcache_chunk_size: int,
) -> torch.Tensor:
    """True for token positions in the trailing sliding-window region of each chunk.

    Within each LMCache chunk, only the last ``sliding_win_size`` tokens are
    live in a sliding-window KV group.
    """
    chunk_start = (tokens_uncompressed // lmcache_chunk_size) * lmcache_chunk_size
    chunk_end = chunk_start + lmcache_chunk_size
    chunk_window_start = chunk_end - sliding_win_size
    return tokens_uncompressed >= chunk_window_start


def _build_slot_mapping_for_group(
    block_ids: list[int],
    block_size: int,
    num_tokens: int,
    is_store: bool,
    lmcache_chunk_size: int = 256,
    compress_ratio: int = 1,
    sliding_win_size: int | None = None,
) -> torch.Tensor:
    """Map compressed rows for tokens in [window_start_token, num_tokens).

    Returns num_tokens/compress_ratio slot: a slot for each row (i.e., token or 
    compressed token) in the sequence. The slots for the rows that must not be copied are -1.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be >= 1, got {compress_ratio}")
    if not block_ids or num_tokens == 0:
        return torch.empty(0, dtype=torch.long)

    tokens_uncompressed = torch.arange(num_tokens, dtype=torch.long)
    tokens_compressed = (tokens_uncompressed // compress_ratio)[::compress_ratio]
    block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
    block_idx = tokens_compressed // block_size
    block_ids = block_ids_tensor[block_idx]
    valid_mask = block_ids != 0
    if (is_store and sliding_win_size is not None and sliding_win_size > 0):
        valid_mask &= _sliding_window_store_mask(
            tokens_uncompressed,
            sliding_win_size=sliding_win_size,
            lmcache_chunk_size=lmcache_chunk_size,
        )[::compress_ratio]
    if not bool(valid_mask.any()):
        return torch.empty(0, dtype=torch.long)
    slots = block_ids * block_size + (tokens_compressed % block_size)
    if bool(valid_mask.all()):
        return slots
    # Mixed valid/null rows: keep index alignment; callers slice away -1 before kernels.
    return slots.masked_fill(~valid_mask, -1)


def _build_slot_mappings_by_group(
    block_ids_by_group: "tuple[list[int], ...]",
    block_sizes_by_group: "tuple[int, ...]",
    num_tokens: int,
    is_store: bool,
    lmcache_chunk_size: int = 256,
    compress_ratios: "tuple[int, ...] | None" = None,
    sliding_window_size_by_group: "tuple[int | None, ...] | None" = None,
) -> "tuple[torch.Tensor, ...]":
    if len(block_ids_by_group) != len(block_sizes_by_group):
        raise ValueError(
            "Block ids and block sizes group count mismatch: "
            f"{len(block_ids_by_group)} vs {len(block_sizes_by_group)}"
        )
    num_groups = len(block_ids_by_group)
    swsbg = sliding_window_size_by_group
    if compress_ratios is not None and len(compress_ratios) != num_groups:
        raise ValueError(
            "Compress ratios and block ids group count mismatch: "
            f"{len(compress_ratios)} vs {num_groups}"
        )
    if swsbg is not None and len(swsbg) != num_groups:
        raise ValueError(
            "Sliding window sizes and block ids group count mismatch: "
            f"{len(sliding_window_size_by_group)} vs {num_groups}"
        )
    mappings: list[torch.Tensor] = []
    for g, (group_block_ids, block_size) in enumerate(
        zip(block_ids_by_group, block_sizes_by_group)
    ):
        ratio = compress_ratios[g] if compress_ratios else 1
        sw_size = swsbg[g] if swsbg is not None else None
        mappings.append(
            _build_slot_mapping_for_group(
                list(group_block_ids),
                block_size,
                num_tokens,
                is_store,
                lmcache_chunk_size,
                ratio,
                sw_size,
            )
        )
    return tuple(mappings)


@dataclass
class RequestTracker(UpstreamRequestTracker):
    # Block ids grouped by KV cache group (multi-group path).
    allocated_block_ids_by_group: "tuple[list[int], ...]" = field(
        default_factory=tuple
    )

    @property
    def num_kv_groups(self) -> int:
        return len(self.allocated_block_ids_by_group)

    def get_allocated_block_ids(self, group_idx: int) -> list[int]:
        return self.allocated_block_ids_by_group[group_idx]

    def _sync_primary_allocated_block_ids(self) -> None:
        if self.allocated_block_ids_by_group:
            self.allocated_block_ids = list(self.allocated_block_ids_by_group[0])

    @_lmcache_nvtx_annotate
    @classmethod
    def from_new_request(
        cls,
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
        expected_num_groups: int = 1,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Delegates to upstream for shared fields, then attaches per-group
        block ids normalized for multi-group KV cache layouts.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            skip_save (bool): whether the request cache should be saved
            expected_num_groups (int): number of KV cache groups.
        """
        base = super().from_new_request(
            lmcache_config,
            new_request,
            num_tokens_to_compute,
            lmcache_cached_tokens,
            skip_save,
        )
        allocated_block_ids_by_group = _normalize_block_ids(
            new_request.block_ids,
            expected_num_groups,
        )
        return cls(
            **{
                f.name: getattr(base, f.name)
                for f in fields(UpstreamRequestTracker)
                if f.name != "allocated_block_ids"
            },
            allocated_block_ids=list(allocated_block_ids_by_group[0]),
            allocated_block_ids_by_group=allocated_block_ids_by_group,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is scheduled again.

        Delegates token/saved-state/decode logic to upstream, then applies
        multi-group block-id normalization and per-group merging.
        """
        if self.num_kv_groups == 1:
            super().update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )
            self.allocated_block_ids_by_group = (list(self.allocated_block_ids),)
            return

        # vLLM may pass None, a flat single-group list, or a grouped tuple/list
        # depending on scheduler version and model cache layout.
        new_block_ids_by_group = _normalize_block_ids(
            new_block_ids,
            self.num_kv_groups,
        )
        old_block_ids_by_group = self.allocated_block_ids_by_group

        super().update(
            new_token_ids,
            new_block_ids_by_group[0],
            preempted=preempted,
            lmcache_cached_tokens=lmcache_cached_tokens,
            vllm_cached_tokens=vllm_cached_tokens,
            all_token_ids=all_token_ids,
        )

        if preempted:
            self.allocated_block_ids_by_group = new_block_ids_by_group
        else:
            self.allocated_block_ids_by_group = tuple(
                old_group_ids + new_group_ids
                for old_group_ids, new_group_ids in zip(
                    old_block_ids_by_group,
                    new_block_ids_by_group,
                )
            )
        self._sync_primary_allocated_block_ids()


@dataclass
class ReqMeta(UpstreamReqMeta):
    # Slot mappings grouped by KV cache group.
    slot_mappings_by_group: "tuple[torch.Tensor, ...]" = field(default_factory=tuple)
    # Allocated block ids grouped by KV cache group.
    allocated_block_ids_by_group: "tuple[list[int], ...]" = field(
        default_factory=tuple
    )
    # Index of the KV group whose block table covers the most logical tokens
    # (dense / full-sequence path). Used for store/retrieve when only one
    # group's slot_mapping can drive the LMCache engine (Phase 2: all groups).
    primary_kv_group_idx: int = 0
    # Per-sched-group dense slot mappings (no -1) and valid-slot prefix arrays.
    filtered_slot_by_group: Optional[tuple[torch.Tensor, ...]] = None
    slot_valid_prefix_by_group: Optional[tuple[torch.Tensor, ...]] = None

    @property
    def num_kv_groups(self) -> int:
        return len(self.slot_mappings_by_group)

    def get_slot_mapping(self, group_idx: int) -> torch.Tensor:
        return self.slot_mappings_by_group[group_idx]

    def get_allocated_block_ids(self, group_idx: int) -> list[int]:
        return self.allocated_block_ids_by_group[group_idx]

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_sizes_by_group: Union[int, "tuple[int, ...]", list[int]],
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
        compress_ratios: "tuple[int, ...] | None" = None,
        sliding_window_size_by_group: "tuple[int | None, ...] | None" = None,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Reuses upstream save/load policy via ``UpstreamReqMeta.from_request_tracker``,
        then attaches multi-group slot mappings and filtered slot metadata.
        """
        block_sizes = _normalize_block_sizes(
            block_sizes_by_group,
            tracker.num_kv_groups,
        )
        if tracker.num_kv_groups > 1:
            assert discard_partial_chunks, (
                "Multi-group KV cache requires discard_partial_chunks=True; "
                "partial-chunk store/load is not supported for state and sliding windowgroups."
            )
        if tracker.num_kv_groups == 1:
            primary_kv_group_idx = 0
        else:
            primary_kv_group_idx = max(
                range(tracker.num_kv_groups),
                key=lambda i: len(tracker.allocated_block_ids_by_group[i])
                * block_sizes[i],
            )

        saved_allocated_block_ids = tracker.allocated_block_ids
        tracker.allocated_block_ids = list(
            tracker.allocated_block_ids_by_group[primary_kv_group_idx]
        )
        try:
            base = UpstreamReqMeta.from_request_tracker(
                tracker,
                block_sizes[primary_kv_group_idx],
                lmcache_chunk_size=lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=discard_partial_chunks,
                save_decode_cache=save_decode_cache,
            )
        finally:
            tracker.allocated_block_ids = saved_allocated_block_ids

        if base is None:
            return None

        token_ids = list(base.token_ids)
        num_tokens = len(token_ids)

        assert base.save_spec is not None
        slot_mappings_by_group = _build_slot_mappings_by_group(
            tracker.allocated_block_ids_by_group,
            block_sizes,
            num_tokens,
            is_store=base.save_spec.can_save,
            lmcache_chunk_size=lmcache_chunk_size,
            compress_ratios=compress_ratios,
            sliding_window_size_by_group=sliding_window_size_by_group,
        )

        filtered_slot_by_group: Optional[tuple[torch.Tensor, ...]] = None
        slot_valid_prefix_by_group: Optional[tuple[torch.Tensor, ...]] = None
        needs_filtered = (
            load_spec is not None and load_spec.can_load
        ) or base.save_spec.can_save
        if needs_filtered:
            ratios = compress_ratios or tuple(1 for _ in block_sizes)
            filtered_slot_by_group, slot_valid_prefix_by_group = (
                build_filtered_slot_mappings(
                    slot_mappings_by_group,
                    compress_ratios=ratios,
                )
            )

        return ReqMeta(
            req_id=base.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mappings_by_group[primary_kv_group_idx],
            slot_mappings_by_group=slot_mappings_by_group,
            allocated_block_ids_by_group=tuple(
                list(group_block_ids)
                for group_block_ids in tracker.allocated_block_ids_by_group
            ),
            is_last_prefill=base.is_last_prefill,
            save_spec=base.save_spec,
            load_spec=base.load_spec,
            disagg_spec=base.disagg_spec,
            request_configs=base.request_configs,
            primary_kv_group_idx=primary_kv_group_idx,
            filtered_slot_by_group=filtered_slot_by_group,
            slot_valid_prefix_by_group=slot_valid_prefix_by_group,
        )


class LMCacheConnectorV1ImplMultiGroup(LMCacheConnectorV1Impl):
    """Multi-group scheduler/worker plumbing layered on upstream LMCache."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
        kv_cache_config: Optional[Any] = None,
    ):
        self._kv_cache_config: Optional[Any] = (
            kv_cache_config
            if kv_cache_config is not None
            else getattr(parent, "_kv_cache_config", None)
        )
        super().__init__(vllm_config, role, parent)

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        super()._init_connector_state(role, vllm_config, config)
        if (
            self._kv_cache_config is not None
            and getattr(self._kv_cache_config, "kv_cache_groups", None)
        ):
            self._num_kv_groups = len(self._kv_cache_config.kv_cache_groups)
            self._block_sizes_by_group: "tuple[int, ...]" = tuple(
                group.kv_cache_spec.block_size
                for group in self._kv_cache_config.kv_cache_groups
            )
            try:
                groups = self._kv_cache_config.kv_cache_groups
                mems = [
                    g.kv_cache_spec.max_memory_usage_bytes(vllm_config)
                    for g in groups
                ]
                max_mem_hint_idx = int(mems.index(max(mems)))
            except Exception:
                max_mem_hint_idx = 0
            logger.info(
                "LMCache KV cache groups: count=%d, block_sizes_by_group=%s, "
                "max_memory_usage_hint_idx=%d (per-request primary_kv_group_idx "
                "uses argmax(len(block_ids)*block_size))",
                self._num_kv_groups,
                self._block_sizes_by_group,
                max_mem_hint_idx,
            )
        else:
            self._num_kv_groups = 1
            self._block_sizes_by_group = (self._block_size,)
            logger.info(
                "LMCache KV cache groups: count=1, block_size=%d "
                "(no kv_cache_groups on connector)",
                self._block_size,
            )

        # Compression ratios from vLLM kv_cache_spec (one entry per scheduler
        # group). Used here to size slot_mappings_by_group: compressed groups store one
        # slot per ``compress_ratio`` tokens, not one slot per token.
        if (
            self._num_kv_groups > 1
            and self._kv_cache_config is not None
            and getattr(self._kv_cache_config, "kv_cache_groups", None)
        ):
            groups = self._kv_cache_config.kv_cache_groups
            max_bs = max(self._block_sizes_by_group)
            for g_idx, bs in enumerate(self._block_sizes_by_group):
                if max_bs % bs != 0:
                    raise ValueError(
                        f"Max block size {max_bs} is not a multiple of "
                        f"group {g_idx} block size {bs}"
                    )
            self._compress_ratios_by_group = tuple(
                int(getattr(g.kv_cache_spec, "compress_ratio", 1)) for g in groups
            )
            self._sliding_window_size_by_group = tuple(
                int(sw) if (sw := getattr(g.kv_cache_spec, "sliding_window", None))
                is not None else None
                for g in groups
            )
        else:
            self._compress_ratios_by_group = (1,)
            self._sliding_window_size_by_group = None

        if self._num_kv_groups > 1:
            assert self._discard_partial_chunks, (
                "Multi-group KV cache requires discard_partial_chunks=True; "
                "partial-chunk store/load is not supported for state and sliding windowgroups."
            )

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: Optional[int] = None,
    ) -> set[int]:
        """Per-group block_size override; delegates to upstream implementation."""
        if block_size is None or block_size == self._block_size:
            return super().record_failed_blocks(
                request_id,
                expected_mask,
                ret_mask,
                slot_mapping,
            )
        saved_block_size = self._block_size
        self._block_size = block_size
        try:
            return super().record_failed_blocks(
                request_id,
                expected_mask,
                ret_mask,
                slot_mapping,
            )
        finally:
            self._block_size = saved_block_size

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
                expected_num_groups=self._num_kv_groups,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_sizes_by_group,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
                compress_ratios=self._compress_ratios_by_group,
                sliding_window_size_by_group=self._sliding_window_size_by_group,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                load_spec = self.load_specs.pop(req.req_id, None)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_sizes_by_group,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                    compress_ratios=self._compress_ratios_by_group,
                    sliding_window_size_by_group=self._sliding_window_size_by_group,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                # On full cache hit, get_num_new_matched_tokens subtracts 1
                # to force last-token recomputation. This only affects
                # num_computed_tokens when lmcache has all tokens AND
                # provides more than vLLM's local cache.
                expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                full_hit_adj = (
                    lmcache_cached_tokens == len(request.all_token_ids)
                    and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                )
                if full_hit_adj:
                    expected -= 1
                assert request.num_computed_tokens == expected, (
                    f"Preempted request {req_id} has "
                    f"num_computed_tokens {request.num_computed_tokens} "
                    f"but expected {expected} "
                    f"(full_hit_adj={full_hit_adj})"
                )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = min(
                    len(group_block_ids) * bs
                    for group_block_ids, bs in zip(
                        request_tracker.allocated_block_ids_by_group,
                        self._block_sizes_by_group,
                    )
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_sizes_by_group,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
                compress_ratios=self._compress_ratios_by_group,
                sliding_window_size_by_group=self._sliding_window_size_by_group,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta