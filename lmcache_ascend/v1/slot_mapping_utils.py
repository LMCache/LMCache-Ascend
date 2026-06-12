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
    """Map a global token span to compressed slot bounds in the full slot-mapping tensor.

    Each scheduler group ``sched_g`` has its own compression ratio. Given logical token
    indices ``[token_start, token_end)`` returns ``(s0, s1)`` such that
    ``sm[s0:s1]`` covers exactly those tokens for that group. ``sm_len`` is the
    length of that group's full slot-mapping tensor (``len(sm)``).

    Raises:
        ValueError: If ``sched_g`` is out of range for ``compress_ratios``, if
            ``ratio < 1``, or if the computed slice ``[s0, s1)`` falls outside
            ``[0, sm_len)``.

    Returns ``(0, 0)`` when the token span is empty.
    """
    if token_end <= token_start:
        return 0, 0
    if sched_g < 0 or sched_g >= len(compress_ratios):
        raise ValueError(
            f"scheduler group {sched_g} out of range for compress_ratios "
            f"(len={len(compress_ratios)})"
        )
    ratio = int(compress_ratios[sched_g])
    if ratio < 1:
        raise ValueError(f"compress_ratios[{sched_g}] must be >= 1, got {ratio}")
    if ratio <= 1:
        s0, s1 = token_start, token_end
    else:
        s0 = token_start // ratio
        s1 = (token_end + ratio - 1) // ratio
    if s0 < 0 or s1 > sm_len:
        raise ValueError(
            f"slot slice [{s0}, {s1}) out of range for sm length {sm_len} "
            f"(range [{token_start}, {token_end}), sched_g={sched_g}, ratio={ratio})"
        )
    return s0, s1


def dense_bounds_from_prefix(
    prefix: torch.Tensor,
    s0: int,
    s1: int,
) -> tuple[int, int]:
    """Map full-``sm`` row bounds ``[s0, s1)`` to a dense filtered slice.

    ``prefix`` comes from :func:`build_filtered_slot_mappings`; ``prefix[k]`` is
    the count of non-``-1`` rows in ``sm[0:k)``. Returns
    ``(dense_start, dense_count)`` so that
    ``filtered[dense_start : dense_start + dense_count]`` equals the valid
    entries in ``sm[s0:s1]``.
    """
    if s1 <= s0:
        return 0, 0
    dense_start = int(prefix[s0])
    dense_count = int(prefix[s1]) - dense_start
    return dense_start, dense_count


def compute_mp_plane_launch_ptrs(
    sched_groups: Sequence[int],
    filtered_slot_mappings_npu: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Per-plane NPU pointers into dense filtered slot-mapping tensors.

    One int64 pointer per entry in ``sched_groups``, indexing
    ``filtered_slot_mappings_npu[g]``. Passed to the multi-plane KV transfer
    kernel so each plane reads its own compacted slot list.
    """
    return torch.tensor(
        [int(filtered_slot_mappings_npu[g].data_ptr()) for g in sched_groups],
        dtype=torch.int64,
        pin_memory=True,
    )


def compute_mp_plane_launch_row(
    g_start: int,
    g_end: int,
    sched_groups: Sequence[int],
    *,
    slot_mappings_by_group: Sequence[torch.Tensor],
    prefixes_by_group: Sequence[torch.Tensor],
    compress_ratios: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Per-plane dense offsets for one token chunk in a multi-plane transfer.

    For each plane's scheduler group, maps a logical range to physical slot range 
    and produces pinned CPU tensors ``starts`` and ``counts`` that are used by
    transfer kernel to index into each plane's filtered slot-mapping buffer.

    Returns:
        ``starts``, ``counts``, and ``has_work`` — the last is ``True`` when at
        least one plane has a non-zero dense slot count for this chunk.
    """
    starts: list[int] = []
    counts: list[int] = []
    has_work = False
    for sched_g in sched_groups:
        sm_len = int(slot_mappings_by_group[sched_g].shape[0])
        s0, s1 = multi_plane_slot_slice_bounds(
            g_start, g_end, sched_g, compress_ratios, sm_len
        )
        dense_start, dense_count = dense_bounds_from_prefix(
            prefixes_by_group[sched_g], s0, s1
        )
        starts.append(dense_start)
        counts.append(dense_count)
        if dense_count > 0:
            has_work = True
    return (
        torch.tensor(starts, dtype=torch.int32, pin_memory=True),
        torch.tensor(counts, dtype=torch.int32, pin_memory=True),
        has_work,
    )


def build_filtered_slot_mappings(
    slot_mappings_by_group: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    compress_ratios: tuple[int, ...] | Sequence[int],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Precompute per-group compacted slot mappings and prefix lookup tables.

    For each scheduler group, strips ``-1`` (invalid) rows from ``sm`` and
    builds a prefix array where ``prefix[g][k]`` counts valid slots in
    ``sm[g][0:k)``. Enables O(1) per-chunk dense slicing via
    :func:`dense_bounds_from_prefix` instead of scanning ``-1`` on every
    transfer.
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
