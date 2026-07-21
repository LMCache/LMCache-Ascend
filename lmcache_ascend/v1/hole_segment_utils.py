# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Iterable, Optional, Union

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import SegmentTokenDatabase
import torch

# First Party
from lmcache_ascend.v1.hole_types import HoleLookupResult, Range

logger = init_logger(__name__)


class HoleSegmentHelper:
    def __init__(self, config: LMCacheEngineConfig, metadata) -> None:
        self.token_database = SegmentTokenDatabase(config, metadata)
        self.sep_len = self.token_database.sep_len

    def _to_tensor(
        self, token_ids: Union[torch.Tensor, Iterable[int], list[int]]
    ) -> torch.Tensor:
        if isinstance(token_ids, torch.Tensor):
            return token_ids.to(device="cpu", dtype=torch.long)
        return torch.tensor(list(token_ids), dtype=torch.long, device="cpu")

    def split_ranges(
        self, token_ids: Union[torch.Tensor, Iterable[int], list[int]]
    ) -> list[Range]:
        tokens = self._to_tensor(token_ids)
        if len(tokens) == 0:
            return []

        ranges: list[Range] = []
        start_idx = 0
        token_chunks = self.token_database._fast_split_by_subtensor(tokens)
        for idx, token_chunk in enumerate(token_chunks):
            token_chunk_len = len(token_chunk)
            end_idx = start_idx + token_chunk_len
            if idx > 0:
                start_idx += self.sep_len
                end_idx += self.sep_len
            ranges.append((start_idx, end_idx))
            start_idx = end_idx
        return ranges

    def tokens_for_range(
        self,
        token_ids: Union[torch.Tensor, Iterable[int], list[int]],
        token_range: Range,
    ) -> torch.Tensor:
        tokens = self._to_tensor(token_ids)
        start, end = token_range
        return tokens[start:end]

    def make_cache_key(
        self,
        token_ids: Union[torch.Tensor, Iterable[int], list[int]],
        token_range: Range,
        request_configs: Optional[dict] = None,
    ):
        chunk_tokens = self.tokens_for_range(token_ids, token_range)
        chunk_hash = self.token_database._hash_tokens(chunk_tokens)
        return self.token_database._make_key_by_hash(chunk_hash, request_configs)


def leading_hit_end(segment_ranges: list[Range], hit_flags: list[bool]) -> int:
    end = 0
    for hit, token_range in zip(hit_flags, segment_ranges, strict=False):
        if not hit:
            break
        end = token_range[1]
    return end


def merge_hit_flags(rank_results: list[HoleLookupResult]) -> HoleLookupResult:
    if not rank_results:
        return HoleLookupResult(mode="legacy", covered_tokens=0, tail_start=0)

    base = rank_results[0]
    if len(rank_results) == 1:
        return base

    merged_hit_flags = list(base.hit_flags)
    for result in rank_results[1:]:
        if result.segment_ranges != base.segment_ranges:
            raise ValueError("Segment ranges differ across ranks in hole lookup.")
        merged_hit_flags = [
            lhs and rhs
            for lhs, rhs in zip(merged_hit_flags, result.hit_flags, strict=False)
        ]
    merged_location = base.location
    for result in rank_results[1:]:
        if result.location != merged_location:
            merged_location = None
            break
    return derive_lookup_result(
        base.segment_ranges,
        merged_hit_flags,
        location=merged_location,
    )


def derive_lookup_result(
    segment_ranges: list[Range],
    hit_flags: list[bool],
    location: Optional[str] = None,
) -> HoleLookupResult:
    if len(segment_ranges) != len(hit_flags):
        raise ValueError("segment_ranges and hit_flags must have the same length.")

    if not segment_ranges:
        return HoleLookupResult(mode="legacy", covered_tokens=0, tail_start=0)

    contiguous_prefix_end = leading_hit_end(segment_ranges, hit_flags)

    if not hit_flags[0]:
        return HoleLookupResult(
            mode="legacy",
            covered_tokens=0,
            tail_start=0,
            location=None,
            segment_ranges=segment_ranges,
            hit_flags=hit_flags,
            hit_ranges=[],
            prefix_miss_ranges=[],
        )

    last_hit_idx = -1
    for idx, is_hit in enumerate(hit_flags):
        if is_hit:
            last_hit_idx = idx

    if last_hit_idx < 0:
        return HoleLookupResult(
            mode="legacy",
            covered_tokens=0,
            tail_start=0,
            location=None,
            segment_ranges=segment_ranges,
            hit_flags=hit_flags,
            hit_ranges=[],
            prefix_miss_ranges=[],
        )

    covered_tokens = segment_ranges[last_hit_idx][1]
    hit_ranges: list[Range] = []
    prefix_miss_ranges: list[Range] = []
    for token_range, is_hit in zip(
        segment_ranges[: last_hit_idx + 1],
        hit_flags[: last_hit_idx + 1],
        strict=False,
    ):
        if is_hit:
            hit_ranges.append(token_range)
        else:
            prefix_miss_ranges.append(token_range)

    mode = "hole" if prefix_miss_ranges else "pure_hit"
    return HoleLookupResult(
        mode=mode,
        covered_tokens=covered_tokens,
        tail_start=covered_tokens,
        location=location,
        segment_ranges=segment_ranges,
        hit_flags=hit_flags,
        hit_ranges=hit_ranges,
        prefix_miss_ranges=prefix_miss_ranges,
    )
