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

from .conftest_ds4 import DS4_CHUNK_SIZE


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
    if block_ids is not None and block_ids != [] and expected_num_groups == 1:
        if isinstance(block_ids, list) and not all(
            isinstance(x, (list, tuple)) for x in block_ids
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
        pytest.param([0, 1], 4, 5, 5, [0, 1, 2, 3, 4], id="basic"),
        pytest.param([0], 4, 100, 4, None, id="cap_overflow"),
    ],
)
def test_build_slot_mapping_for_group(
    block_ids, block_size, num_tokens, expected_len, expected_values
) -> None:
    sm = _build_slot_mapping_for_group(block_ids, block_size, num_tokens)
    assert sm.dtype == torch.long
    assert len(sm) == expected_len
    if expected_values is not None:
        assert sm.tolist() == expected_values


@pytest.mark.parametrize(
    ("block_ids_by_group", "block_sizes_by_group", "num_tokens", "compress_ratios", "expected_lens"),
    [
        pytest.param(
            ([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]),
            (128, 1024),
            512,
            (8, 1),
            (64, 512),
            id="ds4_two_groups",
        ),
        pytest.param(
            ([0, 1], [0, 1]),
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
        compress_ratios=compress_ratios,
    )
    assert tuple(len(m) for m in mappings) == expected_lens


def test_build_slot_mappings_by_group_mismatch() -> None:
    with pytest.raises(ValueError, match="Block ids and block sizes group count mismatch"):
        _build_slot_mappings_by_group(([0],), (128, 1024), 64)


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
                "allocated_block_ids_by_group": ([0, 1, 2, 3], [0, 1, 2, 3]),
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
                "allocated_block_ids_by_group": ([0], [0, 1, 2, 3, 4, 5, 6, 7]),
            },
            {"block_sizes_by_group": (128, 1024), "lmcache_chunk_size": DS4_CHUNK_SIZE},
            lambda m: m is not None and m.primary_kv_group_idx == 1,
            id="primary_dense_group",
        ),
        pytest.param(
            {
                "token_ids": list(range(512)),
                "prompt_len": 512,
                "allocated_block_ids_by_group": ([0, 1, 2, 3, 4, 5, 6, 7], [0, 1]),
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
