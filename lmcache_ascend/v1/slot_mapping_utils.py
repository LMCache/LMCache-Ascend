# SPDX-License-Identifier: Apache-2.0
"""Slot-mapping helpers for multi-plane KV transfer (slice bounds, compaction, precompute)."""

from __future__ import annotations

from typing import Sequence

import torch


def multi_plane_slot_slice_bounds(
    token_start: int,
    token_end: int,
    sched_g: int,
    compress_ratios: Sequence[int],
    sm_len: int,
) -> tuple[int, int]:
    """Map token range ``[token_start, token_end)`` to ``sm`` slice bounds."""
    if token_end <= token_start:
        return 0, 0
    ratio = int(compress_ratios[sched_g]) if sched_g < len(compress_ratios) else 1
    if ratio <= 1:
        s0, s1 = int(token_start), min(int(token_end), sm_len)
    else:
        s0 = int(token_start) // ratio
        s1 = min((int(token_end) + ratio - 1) // ratio, sm_len)
    return s0, max(s0, s1)


def compact_slot_mapping_chunk(
    sm: torch.Tensor,
    g_start: int,
    g_end: int,
    sched_g: int,
    compress_ratios: Sequence[int],
) -> torch.Tensor:
    """Slice one logical chunk from global ``sm`` and drop dead rows (``-1``)."""
    slot_start, slot_end = multi_plane_slot_slice_bounds(
        g_start,
        g_end,
        sched_g,
        compress_ratios,
        int(sm.shape[0]),
    )
    slot_slice = sm[slot_start:slot_end]
    if slot_slice.numel() == 0:
        return slot_slice
    return slot_slice[slot_slice != -1]


def iter_lmcache_chunk_ranges(
    lmcache_cached_tokens: int,
    *,
    vllm_cached_tokens: int,
    lmcache_chunk_size: int,
) -> list[tuple[int, int]]:
    """Chunk-aligned ``[start, end)`` token ranges in the LMCache load window."""
    if lmcache_chunk_size <= 0:
        raise ValueError(f"lmcache_chunk_size must be positive, got {lmcache_chunk_size}")
    if lmcache_cached_tokens <= 0:
        return []

    load_start = (
        int(vllm_cached_tokens) // lmcache_chunk_size * lmcache_chunk_size
    )
    load_end = int(lmcache_cached_tokens)
    if load_end <= load_start:
        return []

    ranges: list[tuple[int, int]] = []
    chunk_start = load_start
    while chunk_start < load_end:
        chunk_end = min(chunk_start + lmcache_chunk_size, load_end)
        ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return ranges


def dense_bounds_from_prefix(
    prefix: torch.Tensor,
    s0: int,
    s1: int,
) -> tuple[int, int]:
    """Map full-``sm`` row bounds ``[s0, s1)`` to dense filtered offsets via prefix."""
    if s1 <= s0:
        return 0, 0
    dense_start = int(prefix[s0])
    dense_count = int(prefix[s1]) - dense_start
    return dense_start, dense_count


def iter_store_chunk_ranges(
    token_len: int,
    skip_leading: int,
    chunk_size: int,
) -> list[tuple[int, int]]:
    """Chunk-aligned ``[start, end)`` token ranges in the LMCache store window."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    skip = int(skip_leading) // chunk_size * chunk_size
    if skip >= token_len:
        return []
    ranges: list[tuple[int, int]] = []
    chunk_start = skip
    while chunk_start < token_len:
        chunk_end = min(chunk_start + chunk_size, token_len)
        ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return ranges


def compute_mp_plane_launch_row(
    g_start: int,
    g_end: int,
    sched_groups: Sequence[int],
    *,
    slot_mappings_by_group: Sequence[torch.Tensor],
    prefixes_by_group: Sequence[torch.Tensor],
    filtered_slot_mappings_npu: Sequence[torch.Tensor],
    compress_ratios: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    """Per-plane ``ptrs`` / ``starts`` / ``counts`` for one multi-plane chunk."""
    ptrs: list[int] = []
    starts: list[int] = []
    counts: list[int] = []
    for sched_g in sched_groups:
        if sched_g >= len(slot_mappings_by_group):
            raise IndexError(
                f"scheduler group {sched_g} out of range "
                f"(num={len(slot_mappings_by_group)})"
            )
        sm_len = int(slot_mappings_by_group[sched_g].shape[0])
        s0, s1 = multi_plane_slot_slice_bounds(
            g_start, g_end, sched_g, compress_ratios, sm_len
        )
        dense_start, dense_count = dense_bounds_from_prefix(
            prefixes_by_group[sched_g], s0, s1
        )
        ptrs.append(int(filtered_slot_mappings_npu[sched_g].data_ptr()))
        starts.append(dense_start)
        counts.append(dense_count)
    return ptrs, starts, counts


def build_filtered_slot_mappings(
    slot_mappings_by_group: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    compress_ratios: tuple[int, ...] | Sequence[int],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Build per-sched-group dense filtered mappings and valid-slot prefix arrays.

    ``prefix[g][k]`` counts valid (non ``-1``) slots in ``sm[g][0:k)``.
    """
    ratios = tuple(int(r) for r in compress_ratios)
    if not ratios:
        ratios = (1,) * len(slot_mappings_by_group)

    filtered: list[torch.Tensor] = []
    prefixes: list[torch.Tensor] = []
    for sched_g, sm in enumerate(slot_mappings_by_group):
        if sm.numel() == 0:
            filtered.append(sm)
            prefixes.append(torch.zeros(1, dtype=torch.int32))
            continue
        valid = (sm != -1).to(torch.int32)
        prefix = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(valid, dim=0)]
        )
        filtered.append(sm[sm != -1])
        prefixes.append(prefix)
    return tuple(filtered), tuple(prefixes)
