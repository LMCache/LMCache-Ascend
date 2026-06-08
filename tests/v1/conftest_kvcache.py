# SPDX-License-Identifier: Apache-2.0
"""Generic KV cache / NPU kernel test helpers (format-agnostic)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Sequence

import pytest
import torch

from lmcache.v1.memory_management import PinMemoryAllocator, TensorMemoryObj
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemNPUConnectorV2,
    _pointers_for_entry,
)
from lmcache_ascend.v1.slot_mapping_utils import (
    build_filtered_slot_mappings,
    multi_plane_slot_slice_bounds,
)


def _filtered_slot_invoke_kwargs(
    slot_mappings: tuple[torch.Tensor, ...],
    compress_ratios: tuple[int, ...],
) -> dict:
    filtered, prefixes = build_filtered_slot_mappings(
        tuple(sm.cpu() for sm in slot_mappings),
        compress_ratios=compress_ratios,
    )
    dev = slot_mappings[0].device
    return {
        "filtered_slot_mappings_npu": tuple(f.to(dev) for f in filtered),
        "slot_valid_prefix_by_group": prefixes,
    }


def npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def device() -> torch.device:
    if npu_available():
        return torch.device("npu:0")
    return torch.device("cpu")


@contextmanager
def pinned_lmc_chunk(
    shape: tuple[int, ...] | torch.Size,
    dtype: torch.dtype = torch.uint8,
    *,
    pool_bytes: int = 512 * 1024 * 1024,
) -> Iterator[tuple[TensorMemoryObj, torch.Tensor]]:
    allocator = PinMemoryAllocator(pool_bytes)
    mem_obj = allocator.allocate(torch.Size(shape), dtype)
    assert mem_obj is not None
    mem_tensor = mem_obj.tensor
    assert mem_tensor is not None
    assert mem_tensor.device.type == "cpu"
    mem_tensor.zero_()
    try:
        yield mem_obj, mem_tensor
    finally:
        mem_obj.ref_count_down()
        allocator.close()


def entry_as_plane_list(entry: torch.Tensor | tuple | list) -> list[torch.Tensor]:
    if isinstance(entry, torch.Tensor):
        return [entry]
    return list(entry)


def first_layer_tensor(entry: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(entry, torch.Tensor):
        return entry
    return entry[0]


def flatten_paged_slots(paged: torch.Tensor) -> torch.Tensor:
    if paged.ndim == 4:
        return paged.reshape(paged.shape[0] * paged.shape[1], -1)
    if paged.ndim == 3:
        return paged.reshape(paged.shape[0] * paged.shape[1], -1)
    return paged.reshape(-1, paged.shape[-1])


def fill_paged_slots(
    paged: torch.Tensor,
    slot_indices: torch.Tensor,
    *,
    base: float,
) -> None:
    flat = flatten_paged_slots(paged)
    for i, slot in enumerate(slot_indices.tolist()):
        flat[int(slot)] = base + float(i) * 0.001


def slot_concat_and_offsets(
    sched_groups: list[int],
    slot_mappings: tuple[torch.Tensor, ...],
    g_start: int,
    g_end: int,
    compress_ratios: tuple[int, ...],
) -> tuple[torch.Tensor, list[int]]:
    sm_parts: list[torch.Tensor] = []
    for sched_g in sched_groups:
        sm = slot_mappings[sched_g]
        s0, s1 = multi_plane_slot_slice_bounds(
            g_start, g_end, sched_g, compress_ratios, int(sm.shape[0])
        )
        sm_parts.append(sm[s0:s1])
    slot_concat = torch.cat(sm_parts, dim=0)
    offsets = [0]
    for part in sm_parts:
        offsets.append(offsets[-1] + int(part.shape[0]))
    return slot_concat, offsets


def fill_multi_plane_pattern(
    planes: list[torch.Tensor],
    sched_groups: list[int],
    slot_mappings: tuple[torch.Tensor, ...],
    chunk: int,
    compress_ratios: tuple[int, ...],
) -> None:
    for pi, sched_g in enumerate(sched_groups):
        sm = slot_mappings[sched_g]
        s0, s1 = multi_plane_slot_slice_bounds(
            0, chunk, sched_g, compress_ratios, int(sm.shape[0])
        )
        fill_paged_slots(planes[pi], sm[s0:s1], base=1.0 + pi * 10.0)


def assert_multi_plane_round_trip_parity(
    planes_src: list[torch.Tensor],
    planes_dst: list[torch.Tensor],
    sched_groups: list[int],
    slot_mappings: tuple[torch.Tensor, ...],
    chunk: int,
    compress_ratios: tuple[int, ...],
    *,
    label: str,
) -> None:
    slot_concat, offsets = slot_concat_and_offsets(
        sched_groups, slot_mappings, 0, chunk, compress_ratios
    )
    for pi, (src, dst) in enumerate(zip(planes_src, planes_dst)):
        slots = slot_concat[offsets[pi] : offsets[pi + 1]]
        flat_s = flatten_paged_slots(src)
        flat_d = flatten_paged_slots(dst)
        if not torch.equal(flat_s[slots], flat_d[slots]):
            raise AssertionError(
                f"{label}: plane {pi} round-trip mismatch "
                f"(sched_g={sched_groups[pi]}, num_slots={slots.numel()})"
            )


def _compressed_slot_index(
    token: int,
    sched_g: int,
    compress_ratios: Sequence[int],
    *,
    window_start: int = 0,
) -> int:
    """Map a logical token to an index in a windowed or full slot_mapping tensor."""
    if token < window_start:
        return -1
    ratio = int(compress_ratios[sched_g]) if sched_g < len(compress_ratios) else 1
    if ratio <= 1:
        return token - window_start
    return (token - window_start) // ratio


def assert_token_aligned_plane_parity(
    plane_store: torch.Tensor,
    plane_load: torch.Tensor,
    sched_g: int,
    store_sm: torch.Tensor,
    load_sm: torch.Tensor,
    token_start: int,
    token_end: int,
    compress_ratios: Sequence[int],
    window_starts: Sequence[int],
    *,
    label: str,
) -> None:
    """Compare KV at store vs load paged slots for the same logical tokens."""
    win = int(window_starts[sched_g]) if sched_g < len(window_starts) else 0
    flat_s = flatten_paged_slots(plane_store)
    flat_d = flatten_paged_slots(plane_load)
    lo = max(token_start, win)
    hi = token_end
    if lo >= hi:
        return
    ratio = int(compress_ratios[sched_g]) if sched_g < len(compress_ratios) else 1
    step = 1 if ratio <= 1 else ratio
    for t in range(lo, hi, step):
        store_idx = _compressed_slot_index(t, sched_g, compress_ratios, window_start=0)
        load_idx = _compressed_slot_index(
            t, sched_g, compress_ratios, window_start=win
        )
        if store_idx < 0 or store_idx >= int(store_sm.shape[0]):
            continue
        if load_idx < 0 or load_idx >= int(load_sm.shape[0]):
            continue
        store_slot = int(store_sm[store_idx].item())
        load_slot = int(load_sm[load_idx].item())
        if not torch.equal(flat_s[store_slot], flat_d[load_slot]):
            raise AssertionError(
                f"{label}: token {t} sched_g={sched_g} mismatch "
                f"(store_slot={store_slot}, load_slot={load_slot})"
            )


def multi_plane_round_trip_via_connector(
    connector: VLLMPagedMemNPUConnectorV2,
    gi: int,
    chunk: int,
    slot_mappings: tuple[torch.Tensor, ...],
    compress_ratios: tuple[int, ...],
    *,
    label: str,
) -> None:
    if chunk <= 0:
        return
    dev = connector.kvcaches_device
    g_params = connector.per_group_params[gi]
    sched_groups = list(g_params.get("scheduler_groups_per_plane") or [])
    layer_idx = int(g_params["layer_indices"][0])
    entry = connector.kvcaches[layer_idx]
    planes_template = entry_as_plane_list(entry)

    planes_src = [p.clone() for p in planes_template]
    fill_multi_plane_pattern(
        planes_src, sched_groups, slot_mappings, chunk, compress_ratios
    )
    planes_work = [p.clone() for p in planes_src]
    planes_dst = [torch.zeros_like(p) for p in planes_template]

    lmc_chunk_row_bytes = int(g_params["k_extra"])
    pool_bytes = max(512 * 1024 * 1024, lmc_chunk_row_bytes * chunk * 2)
    ptrs_store = torch.tensor(
        [p.data_ptr() for p in planes_work], dtype=torch.int64, device=dev
    )
    ptrs_load = torch.tensor(
        [p.data_ptr() for p in planes_dst], dtype=torch.int64, device=dev
    )

    with pinned_lmc_chunk(
        (1, 1, chunk, lmc_chunk_row_bytes), torch.uint8, pool_bytes=pool_bytes
    ) as (_mem_obj, lmc_chunk):
        slot_invoke = _filtered_slot_invoke_kwargs(slot_mappings, compress_ratios)
        connector._invoke_multi_plane_kv_transfer(
            mem_tensor=lmc_chunk,
            group_ptrs=ptrs_store,
            group_params=g_params,
            slot_mappings_by_group=slot_mappings,
            compress_ratios=compress_ratios,
            g_start=0,
            g_end=chunk,
            is_store=True,
            npu_group_idx=gi,
            **slot_invoke,
        )
        torch.npu.synchronize()

        connector._invoke_multi_plane_kv_transfer(
            mem_tensor=lmc_chunk,
            group_ptrs=ptrs_load,
            group_params=g_params,
            slot_mappings_by_group=slot_mappings,
            compress_ratios=compress_ratios,
            g_start=0,
            g_end=chunk,
            is_store=False,
            npu_group_idx=gi,
            **slot_invoke,
        )
        torch.npu.synchronize()

    assert_multi_plane_round_trip_parity(
        planes_src,
        planes_dst,
        sched_groups,
        slot_mappings,
        chunk,
        compress_ratios,
        label=label,
    )


def separate_kv_round_trip_via_connector(
    connector: VLLMPagedMemNPUConnectorV2,
    gi: int,
    chunk: int,
    slot_mappings: tuple[torch.Tensor, ...],
    compress_ratios: tuple[int, ...],
    *,
    label: str,
) -> None:
    if chunk <= 0:
        return
    from lmcache.v1.memory_management import MemoryFormat

    from .conftest_ds4 import _get_multi_group_pinned_allocator

    g_params = connector.per_group_params[gi]
    num_planes = int(g_params.get("num_planes", 0))
    if num_planes <= 0:
        raise AssertionError(
            f"{label}: expected num_planes>=1 for SEPARATE_KV multi-plane routing, "
            f"got params={g_params}"
        )

    slot_g = int(g_params.get("scheduler_slot_group", 0))
    sm = slot_mappings[slot_g]
    s0, s1 = multi_plane_slot_slice_bounds(
        0, chunk, slot_g, compress_ratios, int(sm.shape[0])
    )
    sm_slice = sm[s0:s1]

    layer_indices = tuple(g_params.get("layer_indices", (gi,)))
    assert connector.metadata is not None
    klg = connector.metadata.kv_layer_groups_manager.kv_layer_groups[gi]
    expected_shape = torch.Size(
        [
            int(klg.shape_desc.kv_size),
            int(klg.num_layers),
            chunk,
            int(klg.hidden_dim_size),
        ]
    )
    mem_obj = _get_multi_group_pinned_allocator().allocate(
        [expected_shape],
        [connector.metadata.get_dtypes()[gi]],
        fmt=MemoryFormat.KV_2LTD,
    )
    assert mem_obj is not None
    try:
        lmc_tensor = mem_obj.get_tensor(0)
        assert lmc_tensor is not None
        assert tuple(lmc_tensor.shape) == tuple(expected_shape), (
            f"{label}: LMCache shape {lmc_tensor.shape} != {expected_shape}"
        )

        paged_src_list: list[torch.Tensor] = []
        paged_dst_list: list[torch.Tensor] = []
        ptrs_store = connector.group_kv_cache_pointers[gi].clone()
        ptrs_load = connector.group_kv_cache_pointers[gi].clone()
        for li, layer_idx in enumerate(layer_indices):
            tmpl = first_layer_tensor(connector.kvcaches[int(layer_idx)])
            src = tmpl.clone()
            fill_paged_slots(src, sm_slice, base=float(gi) + li * 0.25)
            dst = torch.zeros_like(tmpl)
            paged_src_list.append(src)
            paged_dst_list.append(dst)
            ptrs_store[li] = src.data_ptr()
            ptrs_load[li] = dst.data_ptr()

        slot_invoke = _filtered_slot_invoke_kwargs(slot_mappings, compress_ratios)
        connector._invoke_multi_plane_kv_transfer(
            mem_tensor=lmc_tensor,
            group_ptrs=ptrs_store,
            group_params=g_params,
            slot_mappings_by_group=slot_mappings,
            compress_ratios=compress_ratios,
            g_start=0,
            g_end=chunk,
            is_store=True,
            npu_group_idx=gi,
            **slot_invoke,
        )
        torch.npu.synchronize()

        connector._invoke_multi_plane_kv_transfer(
            mem_tensor=lmc_tensor,
            group_ptrs=ptrs_load,
            group_params=g_params,
            slot_mappings_by_group=slot_mappings,
            compress_ratios=compress_ratios,
            g_start=0,
            g_end=chunk,
            is_store=False,
            npu_group_idx=gi,
            **slot_invoke,
        )
        torch.npu.synchronize()
    finally:
        mem_obj.ref_count_down()

    for li, (src, dst) in enumerate(zip(paged_src_list, paged_dst_list)):
        flat_s = flatten_paged_slots(src)
        flat_d = flatten_paged_slots(dst)
        if not torch.equal(flat_s[sm_slice], flat_d[sm_slice]):
            raise AssertionError(
                f"{label}: SEPARATE_KV round-trip mismatch layer {li}"
            )


LARGE_TOKEN_COPY_SIZE = 40960
MULTI_PLANE_C8_TOKEN_ALIGN = 32

def power_of_two_boundary_triplet(exp: int) -> tuple[int, int, int]:
    """Return (2^exp - 1, 2^exp, 2^exp + 1)."""
    base = 1 << exp
    return (base - 1, base, base + 1)


def tokens_32b_plane_stride_aligned(
    num_tokens: int, *, min_plane_hd_bytes: int = 2
) -> bool:
    """True when min-plane stride (hd * num_tokens) is 32-byte aligned."""
    return (min_plane_hd_bytes * num_tokens) % 32 == 0


def aligned_near_power_of_two_triplet(
    exp: int, *, align: int = MULTI_PLANE_C8_TOKEN_ALIGN
) -> tuple[int, int, int]:
    """(2^exp - align, 2^exp, 2^exp + align) with 32B plane-stride alignment."""
    base = 1 << exp
    lo, hi = base - align, base + align
    assert lo > 0, f"aligned triplet for exp={exp} underflow"
    for ntok in (lo, base, hi):
        assert tokens_32b_plane_stride_aligned(ntok), (
            f"token count {ntok} misaligned for hd=2 scale plane"
        )
    return (lo, base, hi)


def kernel_transfer_boundary_token_cases() -> tuple[tuple[int, int], ...]:
    """(num_tokens, chunk_size) pairs for direct multi-plane DSA-C8 kernel tests.

    Literal 2^N±1 token counts are invalid for DSA-C8 layout (2 B scale plane);
    use aligned_near_power_of_two_triplet (±16 around 256/512) plus uneven splits.
    """
    cases: list[tuple[int, int]] = []
    for exp in (8, 9):
        for ntok in aligned_near_power_of_two_triplet(exp):
            cases.append((ntok, ntok))
    cases.append((1024, 256))
    cases.append((512, 240))
    chunk = 512
    for ntok in (
        LARGE_TOKEN_COPY_SIZE - MULTI_PLANE_C8_TOKEN_ALIGN,
        LARGE_TOKEN_COPY_SIZE,
        LARGE_TOKEN_COPY_SIZE + MULTI_PLANE_C8_TOKEN_ALIGN,
    ):
        assert tokens_32b_plane_stride_aligned(ntok), (
            f"token count {ntok} misaligned for hd=2 scale plane"
        )
        cases.append((ntok, chunk))
    return tuple(cases)


def ds4_smoke_chunk_sizes() -> tuple[int, ...]:
    """Default DS4 round-trip chunk sizes (smoke tier)."""
    sizes = {
        0,
        1,
        *power_of_two_boundary_triplet(8),
        *power_of_two_boundary_triplet(9),
        768,
        1024,
        LARGE_TOKEN_COPY_SIZE - 1,
        LARGE_TOKEN_COPY_SIZE,
        LARGE_TOKEN_COPY_SIZE + 1,
        32768,
    }
    return tuple(sorted(sizes))


def npu_group_index_with_num_planes(
    connector: VLLMPagedMemNPUConnectorV2, num_planes: int
) -> int:
    params = connector.per_group_params or []
    for i, p in enumerate(params):
        if int(p.get("num_planes", 0)) == num_planes:
            return i
    raise AssertionError(
        f"no NPU group with num_planes={num_planes}; "
        f"formats={[(x.get('kv_format'), x.get('num_planes')) for x in params]}"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "extended: optional full DS4 chunk matrix (also set DS4_CHUNK_TIER=full)",
    )
