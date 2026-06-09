# SPDX-License-Identifier: Apache-2.0
"""Multi-group vLLM adapter layer for LMCache-Ascend (DSv4 / HMA).

vllm adapter overrides for multi-group KV cache without modificaiton to upstream LMCache.
Ascend-specific overrides remain in ``vllm_v1_adapter.LMCacheAscendConnectorV1Impl``.
"""
# Standard
from dataclasses import dataclass, field
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
from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
)
from lmcache.integration.vllm.vllm_v1_adapter import (
    DisaggSpec,
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    SaveSpec,
    extract_request_configs,
    logger,
    tmp_disagg_tracker,
)
from lmcache.utils import _lmcache_nvtx_annotate, cdiv
from lmcache.v1.config import LMCacheEngineConfig

from lmcache_ascend.v1.slot_mapping_utils import build_filtered_slot_mappings

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
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
    """Returns a boolean mask of tokens to NOT be stored in sliding-window groups."""
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
    if not block_ids:
        return torch.empty(0, dtype=torch.long)

    tokens_uncompressed = torch.arange(num_tokens, dtype=torch.long)
    if tokens_uncompressed.numel() == 0:
        return torch.empty(0, dtype=torch.long)
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
    mappings: list[torch.Tensor] = []
    for g, (group_block_ids, block_size) in enumerate(
        zip(block_ids_by_group, block_sizes_by_group)
    ):
        ratio = compress_ratios[g] if compress_ratios else 1
        swsbg = sliding_window_size_by_group
        sw_size = (swsbg[g] if swsbg and g < len(swsbg) else None)
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
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far, grouped by KV cache group.
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids_by_group: "tuple[list[int], ...]"

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    @property
    def num_kv_groups(self) -> int:
        return len(self.allocated_block_ids_by_group)

    def get_allocated_block_ids(self, group_idx: int) -> list[int]:
        return self.allocated_block_ids_by_group[group_idx]

    @property
    def allocated_block_ids(self) -> list[int]:
        assert self.num_kv_groups == 1, (
            "allocated_block_ids is only valid for single-group requests"
        )
        return self.allocated_block_ids_by_group[0]

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,  # noqa: ARG004
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
        expected_num_groups: int = 1,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

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
        allocated_block_ids_by_group = _normalize_block_ids(
            new_request.block_ids,
            expected_num_groups,
        )

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        token_ids_slice = new_request.prompt_token_ids[:num_tokens_to_compute].copy()
        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=token_ids_slice,
            allocated_block_ids_by_group=allocated_block_ids_by_group,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
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
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        # vLLM may pass None, a flat single-group list, or a grouped tuple/list
        # depending on scheduler version and model cache layout.
        new_block_ids_by_group = _normalize_block_ids(
            new_block_ids,
            self.num_kv_groups,
        )

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            # the block ids will change after preemption
            self.allocated_block_ids_by_group = new_block_ids_by_group
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            merged: list[list[int]] = []
            for old_group_ids, new_group_ids in zip(
                self.allocated_block_ids_by_group,
                new_block_ids_by_group,
            ):
                merged.append(old_group_ids + new_group_ids)
            self.allocated_block_ids_by_group = tuple(merged)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mappings grouped by KV cache group.
    slot_mappings_by_group: "tuple[torch.Tensor, ...]"
    # Allocated block ids grouped by KV cache group.
    allocated_block_ids_by_group: "tuple[list[int], ...]" = field(
        default_factory=tuple
    )

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
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

    @property
    def slot_mapping(self) -> torch.Tensor:
        assert self.num_kv_groups == 1, (
            "slot_mapping is only valid for single-group requests"
        )
        return self.slot_mappings_by_group[0]

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

        Args:
            tracker (RequestTracker): the request tracker.
            block_sizes_by_group: vLLM block size for each KV cache group.
                Accepts a single int (single-group), list, or tuple.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.
            compress_ratios: per-group semantic compression ratios from
                ``kv_cache_spec.compress_ratio`` (default 1).  Compressed groups
                store one slot per ``ratio`` tokens in their slot mappings.
            sliding_window_size_by_group: optional per-group SW size in tokens
                (from vLLM ``kv_cache_spec.sliding_window``).

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        block_sizes = _normalize_block_sizes(
            block_sizes_by_group,
            tracker.num_kv_groups,
        )
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        num_tokens = num_tokens_to_save
        if load_spec is not None and load_spec.can_load:
            num_tokens = max(num_tokens_to_save, load_spec.lmcache_cached_tokens)

        if num_tokens <= input_token_len:
            token_ids = input_token_ids[:num_tokens]
        else:
            token_ids = list(input_token_ids) + [0] * (num_tokens - input_token_len)

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        slot_mappings_by_group = _build_slot_mappings_by_group(
            tracker.allocated_block_ids_by_group,
            block_sizes,
            num_tokens,
            is_store=save_spec.can_save,
            lmcache_chunk_size=lmcache_chunk_size,
            compress_ratios=compress_ratios,
            sliding_window_size_by_group=sliding_window_size_by_group,
        )

        # For load operation: log if the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        # For disagg requests, compute total_chunks for sender admission control.
        if tracker.disagg_spec is not None and tracker.disagg_spec.total_chunks == 0:
            # Only compute once (on first batch)
            total_chunks_for_req = (
                tracker.prompt_len + lmcache_chunk_size - 1
            ) // lmcache_chunk_size
            tracker.disagg_spec.total_chunks = total_chunks_for_req

        # Pick the group whose allocated blocks cover the most logical tokens
        # (typically dense SWA / full-attention), not group 0 (often compressed).
        if tracker.num_kv_groups == 1:
            primary_kv_group_idx = 0
        else:
            primary_kv_group_idx = max(
                range(tracker.num_kv_groups),
                key=lambda i: len(tracker.allocated_block_ids_by_group[i])
                * block_sizes[i],
            )

        filtered_slot_by_group: Optional[tuple[torch.Tensor, ...]] = None
        slot_valid_prefix_by_group: Optional[tuple[torch.Tensor, ...]] = None
        needs_filtered = (load_spec is not None and load_spec.can_load) or save_spec.can_save
        if needs_filtered:
            ratios = compress_ratios or tuple(1 for _ in block_sizes)
            filtered_slot_by_group, slot_valid_prefix_by_group = (
                build_filtered_slot_mappings(
                    slot_mappings_by_group,
                    compress_ratios=ratios,
                )
            )

        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mappings_by_group=slot_mappings_by_group,
            allocated_block_ids_by_group=tuple(
                list(group_block_ids)
                for group_block_ids in tracker.allocated_block_ids_by_group
            ),
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
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

    def start_load_kv(self, forward_context: "ForwardContext", **_kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None

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
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            # Multi-group: use primary_kv_group_idx (dense / max token slots), not 0.
            # Full multi-group retrieve still requires Phase 2 completion.
            pg = request.primary_kv_group_idx
            if request.num_kv_groups == 1:
                slot_mapping = request.slot_mappings_by_group[pg].to(self.device)
            else:
                logger.warning_once(
                    "start_load_kv: multi-group slot_mapping (%d groups); "
                    "using primary group %d only. Full multi-group retrieve "
                    "requires Phase 2 completion.",
                    request.num_kv_groups,
                    pg,
                )
                slot_mapping = request.slot_mappings_by_group[pg].to(self.device)
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

                # Check the result
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
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                        block_size=self._block_sizes_by_group[pg],
                    )
                    self._invalid_block_ids.update(missing_blocks)

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: Optional[int] = None,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.
            block_size: vLLM block size for the slot_mapping's KV group. Defaults
                to ``self._block_size`` when omitted.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        bs = block_size if block_size is not None else self._block_size
        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // bs
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, _layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

        if self.layerwise_retrievers:
            self.current_layer += 1

        return

    @_lmcache_nvtx_annotate

    def save_kv_layer(
        self,
        _layer_name: str,
        _kv_layer: torch.Tensor,
        _attn_metadata: "AttentionMetadata",
        **_kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
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
        is_first = True

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is None:
                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                # Multi-group: use primary_kv_group_idx (dense path), not group 0.
                pg = request.primary_kv_group_idx
                if request.num_kv_groups > 1:
                    logger.warning_once(
                        "save_kv_layer (layerwise=None): multi-group "
                        "slot_mapping (%d groups); using primary group %d only. "
                        "Full support requires Phase 2.",
                        request.num_kv_groups,
                        pg,
                    )
                slot_mapping = request.slot_mappings_by_group[pg]
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.to(self.device)

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    assert save_spec is not None
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                )
                self._layerwise_save_storers[request.req_id] = layerwise_storer
                if is_first:
                    is_first = False

            next(layerwise_storer)

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