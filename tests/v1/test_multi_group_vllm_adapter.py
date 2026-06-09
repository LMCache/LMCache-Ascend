# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multi-group vLLM adapter helpers and dataclasses."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
    LMCacheConnectorV1ImplMultiGroup,
    ReqMeta,
    RequestTracker,
    _build_slot_mapping_for_group,
    _build_slot_mappings_by_group,
    _normalize_block_ids,
    _normalize_block_sizes,
)
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    _uses_multi_plane_kv_transfer,
    build_mp_launch_meta,
)
from lmcache_ascend.v1.slot_mapping_utils import (
    build_filtered_slot_mappings,
    compact_slot_mapping_chunk,
    compute_mp_plane_launch_row,
    dense_bounds_from_prefix,
    iter_lmcache_chunk_ranges,
    iter_store_chunk_ranges,
    multi_plane_slot_slice_bounds,
)

from .conftest_ds4 import DS4_CHUNK_SIZE, DS4_COMPRESS_RATIOS, make_slot_mappings


def _make_tracker(
    *,
    token_ids: list[int] | None = None,
    allocated_block_ids_by_group: tuple[list[int], ...] = ([0, 1], [10]),
    prompt_len: int = 100,
    num_saved_tokens: int = 0,
    skip_save: bool = False,
) -> RequestTracker:
    return RequestTracker(
        req_id="r1",
        prompt_len=prompt_len,
        token_ids=list(token_ids if token_ids is not None else [0, 1]),
        allocated_block_ids_by_group=allocated_block_ids_by_group,
        num_saved_tokens=num_saved_tokens,
        skip_save=skip_save,
    )


@pytest.mark.parametrize(
    ("block_ids", "expected_num_groups", "expected"),
    [
        pytest.param(None, 3, ([], [], []), id="none"),
        pytest.param([], 2, ([], []), id="empty_list"),
        pytest.param([1, 2, 3], 1, ([1, 2, 3],), id="flat_single"),
        pytest.param([[1], [2, 3]], 2, ([1], [2, 3]), id="nested_list"),
        pytest.param(([1], [2]), 2, ([1], [2]), id="nested_tuple"),
        pytest.param(([10], [20, 21]), 2, ([10], [20, 21]), id="tuple_groups"),
    ],
)
def test_normalize_block_ids(block_ids, expected_num_groups, expected) -> None:
    result = _normalize_block_ids(block_ids, expected_num_groups)
    assert result == expected
    if (
        block_ids is not None
        and block_ids != []
        and expected_num_groups == 1
        and isinstance(block_ids, list)
        and not all(isinstance(x, (list, tuple)) for x in block_ids)
    ):
        assert result[0] is not block_ids


@pytest.mark.parametrize(
    ("block_ids", "expected_num_groups", "match"),
    [
        pytest.param([1], 0, "expected_num_groups must be >= 1", id="zero_groups"),
        pytest.param([1, 2], 2, "single-group", id="flat_multi_group"),
        pytest.param([[1], [2]], 3, "Block group count mismatch", id="group_len_mismatch"),
        pytest.param("bad", 1, "Unsupported block_ids", id="unsupported_type"),
    ],
)
def test_normalize_block_ids_errors(block_ids, expected_num_groups, match) -> None:
    with pytest.raises(ValueError, match=match):
        _normalize_block_ids(block_ids, expected_num_groups)


@pytest.mark.parametrize(
    ("block_sizes", "expected_num_groups", "expected"),
    [
        pytest.param(128, 1, (128,), id="int"),
        pytest.param([128, 1024], 2, (128, 1024), id="list"),
        pytest.param((32, 64), 2, (32, 64), id="tuple"),
    ],
)
def test_normalize_block_sizes(block_sizes, expected_num_groups, expected) -> None:
    assert _normalize_block_sizes(block_sizes, expected_num_groups) == expected


def test_normalize_block_sizes_mismatch() -> None:
    with pytest.raises(ValueError, match="Block size group count mismatch"):
        _normalize_block_sizes((128, 1024), 3)


@pytest.mark.parametrize(
    ("block_ids", "block_size", "num_tokens", "expected_len", "expected_values"),
    [
        pytest.param([0], 4, 0, 0, None, id="zero_tokens"),
        pytest.param([1, 2], 4, 5, 5, [4, 5, 6, 7, 8], id="basic"),
        pytest.param([1] * 25, 4, 100, 100, None, id="many_tokens"),
    ],
)
def test_build_slot_mapping_for_group(
    block_ids, block_size, num_tokens, expected_len, expected_values
) -> None:
    sm = _build_slot_mapping_for_group(
        block_ids, block_size, num_tokens, is_store=False
    )
    assert sm.dtype == torch.long
    assert len(sm) == expected_len
    if expected_values is not None:
        assert sm.tolist() == expected_values


@pytest.mark.parametrize(
    ("block_ids_by_group", "block_sizes_by_group", "num_tokens", "compress_ratios", "expected_lens"),
    [
        pytest.param(
            ([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]),
            (128, 1024),
            512,
            (8, 1),
            (64, 512),
            id="ds4_two_groups",
        ),
        pytest.param(
            ([1, 2], [1, 2]),
            (16, 16),
            32,
            None,
            (32, 32),
            id="no_compress",
        ),
    ],
)
def test_build_slot_mappings_by_group(
    block_ids_by_group, block_sizes_by_group, num_tokens, compress_ratios, expected_lens
) -> None:
    mappings = _build_slot_mappings_by_group(
        block_ids_by_group,
        block_sizes_by_group,
        num_tokens,
        is_store=False,
        compress_ratios=compress_ratios,
    )
    assert tuple(len(m) for m in mappings) == expected_lens


def test_build_slot_mappings_global_index_for_chunk_slice() -> None:
    """Global sm: token 384 is row 48; connector slices sm[32:64] for chunk [256,512)."""
    mappings = _build_slot_mappings_by_group(
        ([5],),
        (128,),
        512,
        is_store=False,
        compress_ratios=(8,),
    )
    assert len(mappings[0]) == 64
    assert mappings[0][48].item() == 5 * 128 + 48
    s0, s1 = multi_plane_slot_slice_bounds(256, 512, 0, (8,), len(mappings[0]))
    assert (s0, s1) == (32, 64)


def test_build_slot_mapping_for_group_store_masks_sliding_window() -> None:
    sm = _build_slot_mapping_for_group(
        [1, 2, 3, 4, 5, 6, 7, 8],
        4,
        32,
        is_store=True,
        compress_ratio=1,
        sliding_win_size=8,
        lmcache_chunk_size=16,
    )

    assert len(sm) == 32
    assert sm[:8].tolist() == [-1] * 8
    assert sm[8:16].tolist() == list(range(12, 20))
    assert sm[16:24].tolist() == [-1] * 8
    assert sm[24:].tolist() == list(range(28, 36))


def test_build_slot_mappings_by_group_per_chunk_store_window() -> None:
    mappings = _build_slot_mappings_by_group(
        ([1, 2, 3, 4, 5, 6, 7, 8],),
        (4,),
        32,
        is_store=True,
        compress_ratios=(1,),
        sliding_window_size_by_group=(8,),
        lmcache_chunk_size=16,
    )

    assert len(mappings[0]) == 32
    # chunk [0,16): live [8,16); chunk [16,32): live [24,32)
    assert mappings[0][:8].tolist() == [-1] * 8
    assert mappings[0][8:16].tolist() == list(range(12, 20))
    assert mappings[0][16:24].tolist() == [-1] * 8
    assert mappings[0][24:].tolist() == list(range(28, 36))


def test_build_slot_mapping_store_per_chunk_two_lmcache_chunks() -> None:
    """512 tokens, ratio 4, W=128, chunk 256: live rows in both LMCache chunks."""
    block_ids = list(range(1, 5))
    sm = _build_slot_mapping_for_group(
        block_ids,
        128,
        512,
        is_store=True,
        compress_ratio=4,
        sliding_win_size=128,
        lmcache_chunk_size=256,
    )
    live = (sm >= 0).nonzero(as_tuple=True)[0]
    live_tokens = live * 4
    assert live.numel() == 64
    assert live_tokens.min().item() == 128
    assert live_tokens.max().item() == 508
    assert set((live_tokens // 256).tolist()) == {0, 1}


def test_build_slot_mapping_load_uses_block_id_zeros() -> None:
    """On load, vLLM null block_ids mark non-resident rows (no store SW mask)."""
    sm = _build_slot_mapping_for_group(
        [0, 0, 1, 1, 1, 1, 1, 1],
        4,
        32,
        is_store=False,
        compress_ratio=1,
    )
    assert sm[:8].tolist() == [-1] * 8
    assert (sm[8:] >= 0).all()


def test_build_slot_mapping_two_block_ids_first_covers_full_prefill() -> None:
    """Second block id is reserved past 512 tokens; prefill rows all use block_ids[0]."""
    sm = _build_slot_mapping_for_group(
        [1, 2],
        128,
        512,
        is_store=False,
        compress_ratio=4,
    )
    assert len(sm) == 128
    assert (sm >= 0).all()
    assert (sm // 128).unique().tolist() == [1]


def test_build_slot_mapping_single_block_no_index_error() -> None:
    sm = _build_slot_mapping_for_group(
        [10],
        128,
        512,
        is_store=False,
        compress_ratio=4,
    )
    assert len(sm) == 128
    assert (sm >= 0).all()


def test_multi_plane_slot_slice_bounds_global_sm() -> None:
    ratios = (8,)
    sm_len = 64

    assert multi_plane_slot_slice_bounds(0, 256, 0, ratios, sm_len) == (0, 32)
    assert multi_plane_slot_slice_bounds(256, 512, 0, ratios, sm_len) == (32, 64)


def test_compact_slot_mapping_chunk_per_lmcache_chunk() -> None:
    sm = torch.full((128,), -1, dtype=torch.long)
    sm[32:64] = torch.arange(32, 64, dtype=torch.long)
    sm[96:128] = torch.arange(96, 128, dtype=torch.long)
    chunk0 = compact_slot_mapping_chunk(sm, 0, 256, 0, (4,))
    chunk1 = compact_slot_mapping_chunk(sm, 256, 512, 0, (4,))
    assert chunk0.tolist() == list(range(32, 64))
    assert chunk1.tolist() == list(range(96, 128))


def test_filtered_slot_mappings_match_runtime_compact() -> None:
    sm = torch.full((128,), -1, dtype=torch.long)
    sm[32:64] = torch.arange(32, 64, dtype=torch.long)
    sm[96:128] = torch.arange(96, 128, dtype=torch.long)
    mappings = (sm,)
    ratios = (4,)
    lmcache_cached_tokens = 512
    lmcache_chunk_size = 256

    filtered, prefixes = build_filtered_slot_mappings(
        mappings,
        compress_ratios=ratios,
    )
    for token_start, token_end in iter_lmcache_chunk_ranges(
        lmcache_cached_tokens,
        vllm_cached_tokens=0,
        lmcache_chunk_size=lmcache_chunk_size,
    ):
        for sched_g, sm_g in enumerate(mappings):
            expected = compact_slot_mapping_chunk(
                sm_g, token_start, token_end, sched_g, ratios
            )
            s0, s1 = multi_plane_slot_slice_bounds(
                token_start, token_end, sched_g, ratios, int(sm_g.shape[0])
            )
            dense_start, dense_count = dense_bounds_from_prefix(
                prefixes[sched_g], s0, s1
            )
            actual = filtered[sched_g][dense_start : dense_start + dense_count]
            assert actual.equal(expected)


def test_req_meta_populates_filtered_slot_mappings_on_load() -> None:
    tracker = _make_tracker(
        token_ids=list(range(256)),
        prompt_len=256,
        allocated_block_ids_by_group=([0, 1], [10, 11]),
        skip_save=True,
    )
    meta = ReqMeta.from_request_tracker(
        tracker,
        block_sizes_by_group=(128, 1024),
        lmcache_chunk_size=DS4_CHUNK_SIZE,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=256,
            can_load=True,
        ),
        compress_ratios=(8, 1),
    )
    assert meta is not None
    assert meta.filtered_slot_by_group is not None
    assert meta.slot_valid_prefix_by_group is not None
    assert len(meta.filtered_slot_by_group) == 2
    assert len(meta.slot_valid_prefix_by_group) == 2


def test_req_meta_populates_filtered_slot_mappings_on_save() -> None:
    tracker = _make_tracker(
        token_ids=list(range(32)),
        prompt_len=32,
        allocated_block_ids_by_group=([0],),
    )
    meta = ReqMeta.from_request_tracker(
        tracker,
        block_sizes_by_group=128,
        lmcache_chunk_size=DS4_CHUNK_SIZE,
    )
    assert meta is not None
    assert meta.filtered_slot_by_group is not None
    assert meta.slot_valid_prefix_by_group is not None


def test_req_meta_skips_filtered_slot_mappings_without_load_or_save() -> None:
    tracker = _make_tracker(
        token_ids=list(range(32)),
        prompt_len=32,
        allocated_block_ids_by_group=([0],),
        skip_save=True,
    )
    meta = ReqMeta.from_request_tracker(
        tracker,
        block_sizes_by_group=128,
        lmcache_chunk_size=DS4_CHUNK_SIZE,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=32,
            can_load=False,
        ),
    )
    assert meta is not None
    assert meta.filtered_slot_by_group is None
    assert meta.slot_valid_prefix_by_group is None


def test_build_slot_mappings_by_group_mismatch() -> None:
    with pytest.raises(ValueError, match="Block ids and block sizes group count mismatch"):
        _build_slot_mappings_by_group(
            ([0],), (128, 1024), 64, is_store=False
        )


@pytest.mark.parametrize(
    ("initial_groups", "new_token_ids", "new_block_ids", "preempted", "all_token_ids", "check"),
    [
        pytest.param(
            ([1],),
            [2],
            [3],
            False,
            None,
            lambda t: (t.allocated_block_ids_by_group == ([1, 3],) and t.token_ids == [0, 1, 2]),
            id="merge_flat",
        ),
        pytest.param(
            ([1], [10]),
            [2],
            ([20], [30]),
            False,
            None,
            lambda t: t.allocated_block_ids_by_group == ([1, 20], [10, 30]),
            id="merge_grouped",
        ),
        pytest.param(
            ([1], [10]),
            [2],
            None,
            False,
            None,
            lambda t: t.allocated_block_ids_by_group == ([1], [10]),
            id="merge_none_new_blocks",
        ),
        pytest.param(
            ([1],),
            [99],
            [4],
            False,
            None,
            lambda t: t.is_decode_phase is True,
            id="decode_flag",
        ),
        pytest.param(
            ([1], [10]),
            [5],
            ([20], [30]),
            True,
            list(range(50)),
            lambda t: (
                t.allocated_block_ids_by_group == ([20], [30])
                and t.num_saved_tokens == 8
                and t.token_ids == list(range(9))
            ),
            id="preempted",
        ),
    ],
)
def test_request_tracker_update(
    initial_groups, new_token_ids, new_block_ids, preempted, all_token_ids, check
) -> None:
    tracker = _make_tracker(allocated_block_ids_by_group=tuple(initial_groups))
    if preempted:
        tracker.update(
            new_token_ids,
            new_block_ids,
            preempted=True,
            lmcache_cached_tokens=8,
            vllm_cached_tokens=4,
            all_token_ids=all_token_ids,
        )
    else:
        tracker.update(new_token_ids, new_block_ids)
    assert check(tracker)


def test_request_tracker_update_preempted_requires_all_token_ids() -> None:
    tracker = _make_tracker(allocated_block_ids_by_group=([1], [10]))
    with pytest.raises(AssertionError, match="no all_token_ids"):
        tracker.update([5], ([20], [30]), preempted=True)


@pytest.mark.parametrize(
    ("tracker_kwargs", "from_kwargs", "check"),
    [
        pytest.param(
            {
                "token_ids": list(range(300)),
                "prompt_len": 300,
                "num_saved_tokens": 256,
                "allocated_block_ids_by_group": ([0],),
            },
            {"block_sizes_by_group": 128, "lmcache_chunk_size": DS4_CHUNK_SIZE},
            lambda m: m is None,
            id="returns_none",
        ),
        pytest.param(
            {
                "token_ids": list(range(512)),
                "prompt_len": 512,
                "num_saved_tokens": 0,
                "allocated_block_ids_by_group": ([1, 2, 3, 4], [1, 2, 3, 4]),
            },
            {"block_sizes_by_group": (128, 1024), "lmcache_chunk_size": DS4_CHUNK_SIZE},
            lambda m: (
                m is not None
                and m.save_spec is not None
                and m.save_spec.can_save
                and len(m.token_ids) == 512
                and tuple(len(s) for s in m.slot_mappings_by_group) == (512, 512)
            ),
            id="basic_save",
        ),
        pytest.param(
            {
                "token_ids": list(range(32)),
                "prompt_len": 32,
                "skip_save": True,
                "allocated_block_ids_by_group": ([0],),
            },
            {
                "block_sizes_by_group": 128,
                "load_spec": LoadSpec(
                    vllm_cached_tokens=0,
                    lmcache_cached_tokens=32,
                    can_load=True,
                ),
            },
            lambda m: m is not None and m.load_spec is not None and m.load_spec.can_load,
            id="load_only",
        ),
        pytest.param(
            {
                "token_ids": list(range(64)),
                "prompt_len": 64,
                "allocated_block_ids_by_group": ([1], [1, 2, 3, 4, 5, 6, 7, 8]),
            },
            {"block_sizes_by_group": (128, 1024), "lmcache_chunk_size": DS4_CHUNK_SIZE},
            lambda m: m is not None and m.primary_kv_group_idx == 1,
            id="primary_dense_group",
        ),
        pytest.param(
            {
                "token_ids": list(range(512)),
                "prompt_len": 512,
                "allocated_block_ids_by_group": ([1, 2, 3, 4, 5, 6, 7, 8], [1, 2]),
            },
            {
                "block_sizes_by_group": (128, 1024),
                "compress_ratios": (8, 1),
                "lmcache_chunk_size": DS4_CHUNK_SIZE,
            },
            lambda m: (
                m is not None
                and tuple(len(s) for s in m.slot_mappings_by_group) == (64, 512)
            ),
            id="compress_ratios",
        ),
    ],
)
def test_req_meta_from_request_tracker(tracker_kwargs, from_kwargs, check) -> None:
    tracker = _make_tracker(**tracker_kwargs)
    meta = ReqMeta.from_request_tracker(tracker, **from_kwargs)
    assert check(meta)


@patch(
    "lmcache_ascend.integration.vllm.multi_group_vllm_adapter.extract_mm_features",
    return_value=(None, None),
)
@patch(
    "lmcache_ascend.integration.vllm.multi_group_vllm_adapter.extract_request_configs",
    return_value=None,
)
@pytest.mark.parametrize(
    ("block_ids", "expected_num_groups", "expected_groups"),
    [
        pytest.param([10, 20], 1, ([10, 20],), id="flat"),
        pytest.param([[1], [2, 3]], 2, ([1], [2, 3]), id="nested"),
    ],
)
def test_request_tracker_from_new_request(
    _mock_configs,
    _mock_mm,
    block_ids,
    expected_num_groups,
    expected_groups,
) -> None:
    new_request = SimpleNamespace(
        req_id="req-1",
        block_ids=block_ids,
        prompt_token_ids=list(range(48)),
        sampling_params=None,
    )
    tracker = RequestTracker.from_new_request(
        lmcache_config=SimpleNamespace(),
        new_request=new_request,
        num_tokens_to_compute=32,
        lmcache_cached_tokens=0,
        skip_save=False,
        expected_num_groups=expected_num_groups,
    )
    assert tracker.req_id == "req-1"
    assert tracker.allocated_block_ids_by_group == expected_groups
    assert len(tracker.token_ids) == 32


def _make_record_failed_blocks_adapter(*, block_size: int = 128):
    adapter = object.__new__(LMCacheConnectorV1ImplMultiGroup)
    adapter._block_size = block_size
    return adapter


def test_record_failed_blocks_uses_group_block_size() -> None:
    """Block IDs must use the KV group's block size, not cache_config.block_size."""
    adapter = _make_record_failed_blocks_adapter(block_size=128)
    expected_mask = torch.tensor([True, True])
    ret_mask = torch.tensor([True, False])
    slot_mapping = torch.tensor([0, 1024], dtype=torch.long)

    with_group_bs = adapter.record_failed_blocks(
        "req",
        expected_mask,
        ret_mask,
        slot_mapping,
        block_size=1024,
    )
    with_default_bs = adapter.record_failed_blocks(
        "req",
        expected_mask,
        ret_mask,
        slot_mapping,
    )

    assert with_group_bs == {1}
    assert with_default_bs == {8}
    assert with_group_bs != with_default_bs


def test_record_failed_blocks_respects_expected_mask() -> None:
    """vLLM-cached tokens must not contribute to failed block IDs."""
    adapter = _make_record_failed_blocks_adapter(block_size=128)
    expected_mask = torch.tensor([False, True, True])
    ret_mask = torch.tensor([False, True, False])
    slot_mapping = torch.tensor([999, 1024, 2048], dtype=torch.long)

    result = adapter.record_failed_blocks(
        "req",
        expected_mask,
        ret_mask,
        slot_mapping,
        block_size=1024,
    )

    assert result == {2}


def test_record_failed_blocks_no_missing_tokens() -> None:
    adapter = _make_record_failed_blocks_adapter(block_size=128)
    mask = torch.tensor([True, True])

    result = adapter.record_failed_blocks(
        "req",
        mask,
        mask,
        torch.tensor([0, 128], dtype=torch.long),
    )

    assert result == set()


def test_mp_launch_meta_matches_runtime_row(ds4_setup) -> None:
    """Precomputed launch rows must match runtime compute_mp_plane_launch_row."""
    from unittest.mock import patch

    connector, _, kv_caches, dev = ds4_setup
    num_tokens = DS4_CHUNK_SIZE * 2
    slot_mappings = make_slot_mappings(num_tokens, dev)
    cpu_mappings = tuple(sm.cpu() for sm in slot_mappings)
    filtered_cpu, prefixes = build_filtered_slot_mappings(
        cpu_mappings,
        compress_ratios=DS4_COMPRESS_RATIOS[: len(slot_mappings)],
    )
    filtered_npu = tuple(f.to(dev) for f in filtered_cpu)
    ratios = DS4_COMPRESS_RATIOS[: len(slot_mappings)]

    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        connector._initialize_pointers(kv_caches)

    load_ranges = iter_lmcache_chunk_ranges(
        num_tokens,
        vllm_cached_tokens=0,
        lmcache_chunk_size=DS4_CHUNK_SIZE,
    )
    store_ranges = iter_store_chunk_ranges(num_tokens, 0, DS4_CHUNK_SIZE)
    chunk_ranges = list(dict.fromkeys(load_ranges + store_ranges))

    meta = build_mp_launch_meta(
        connector,
        chunk_ranges=chunk_ranges,
        slot_mappings_by_group=cpu_mappings,
        prefixes_by_group=prefixes,
        filtered_slot_mappings_npu=filtered_npu,
        compress_ratios=ratios,
    )
    assert connector.per_group_params is not None
    for npu_g, group_params in enumerate(connector.per_group_params):
        if not _uses_multi_plane_kv_transfer(group_params):
            continue
        num_planes = int(group_params["num_planes"])
        sched_groups = group_params.get("scheduler_groups_per_plane") or []
        if len(sched_groups) != num_planes:
            sched_groups = [
                int(group_params.get("scheduler_slot_group", 0))
            ] * num_planes
        for g_start, g_end in chunk_ranges:
            _, exp_starts, exp_counts = compute_mp_plane_launch_row(
                g_start,
                g_end,
                sched_groups,
                slot_mappings_by_group=cpu_mappings,
                prefixes_by_group=prefixes,
                filtered_slot_mappings_npu=filtered_npu,
                compress_ratios=ratios,
            )
            if sum(exp_counts) == 0:
                assert (g_start, g_end, npu_g) not in meta
                continue
            starts_npu, counts_npu = meta[(g_start, g_end, npu_g)]
            assert starts_npu.cpu().tolist() == exp_starts
            assert counts_npu.cpu().tolist() == exp_counts
