# SPDX-License-Identifier: Apache-2.0
"""DS4 production KV cache round-trip tests (bundled connector, real NPU kernels)."""

from __future__ import annotations

import pytest
import torch

from lmcache_ascend.integration.vllm.multi_spec_flatten import build_flat_kv_caches
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.npu_connectors import multi_plane_slot_slice_bounds

from .conftest_ds4 import (
    DS4_IE_LOGICAL_BLOCK_SIZE,
    DS4_PRODUCTION_CHUNK_TOKENS,
    build_bundled_ds4_connector,
    compress_ratios_from_block_sizes,
    ds4_multi_plane_round_trip,
    ds4_roundtrip_chunk_sizes,
    ds4_separate_kv_round_trip,
    ds4_slot_concat_and_offsets,
    make_ds4_kv_caches_dict,
    make_production_slot_mappings,
    npu_available,
    prefill_tokens_for_chunk,
)
from .conftest_kvcache import npu_group_index_with_num_planes
from .test_layer_to_scheduler_group_mapping import L2, _make_ds4_kv_cache_config


def test_ds4_multi_plane_slot_slice_bounds_768_tokens_chunk256() -> None:
    """768-token prefill, chunk ``[0, 256)``: per-plane slot counts for bundled L2."""
    # Unique scope: pure CPU expectation test for slot slicing math before any kernel launch.
    # Other DS4 tests execute kernels; this one guards index math regressions deterministically.
    ratios = compress_ratios_from_block_sizes()
    _, _, layer_to_groups, _ = build_flat_kv_caches(
        make_ds4_kv_caches_dict(torch.device("cpu"), num_blocks=4),
        _make_ds4_kv_cache_config(),
        ie_logical_block_size=DS4_IE_LOGICAL_BLOCK_SIZE,
    )
    sched_groups = layer_to_groups[L2]
    slot_mappings = make_production_slot_mappings(768, torch.device("cpu"))
    plane_slot_lens = []
    for sched_g in sched_groups:
        sm_len = int(slot_mappings[sched_g].shape[0])
        s0, s1 = multi_plane_slot_slice_bounds(0, 256, sched_g, ratios, sm_len)
        plane_slot_lens.append(s1 - s0)
    assert plane_slot_lens == [32, 32, 256, 256, 8, 8, 32, 32]
    assert sum(plane_slot_lens) == 656


@pytest.mark.skipif(
    not npu_available(),
    reason="NPU required for DS4 L2 bundled multi-plane round-trip",
)
@pytest.mark.parametrize(
    "chunk",
    ds4_roundtrip_chunk_sizes(),
    ids=[str(c) for c in ds4_roundtrip_chunk_sizes()],
)
def test_ds4_l2_multi_plane_round_trip_various_chunk_sizes(
    monkeypatch: pytest.MonkeyPatch,
    chunk: int,
) -> None:
    """Bundled L2 eight-tuple: real multi-plane kernel round-trip at ``g_end=chunk``."""
    # Unique scope: end-to-end kernel parity for L2 bundled 8-plane topology across chunk matrix.
    # L3 test covers 4-plane topology; this one also enforces the production 656-slot concat check.
    connector, _, _, dev = build_bundled_ds4_connector(monkeypatch)
    gi = npu_group_index_with_num_planes(connector, 8)
    g_params = connector.per_group_params[gi]
    sched_groups = list(g_params.get("scheduler_groups_per_plane") or [])
    prefill = prefill_tokens_for_chunk(chunk)
    slot_mappings = make_production_slot_mappings(prefill, dev)
    if chunk == DS4_PRODUCTION_CHUNK_TOKENS:
        slot_concat, _ = ds4_slot_concat_and_offsets(
            sched_groups, slot_mappings, 0, chunk
        )
        assert int(slot_concat.shape[0]) == 656, (
            f"expected slot_concat_len=656 for chunk {chunk}, "
            f"got {slot_concat.shape[0]}, sched_groups={sched_groups}"
        )
    ds4_multi_plane_round_trip(
        connector,
        gi,
        chunk,
        slot_mappings,
        label=f"L2_bundled_8plane_chunk{chunk}",
    )


@pytest.mark.skipif(
    not npu_available(),
    reason="NPU required for DS4 L3 bundled multi-plane round-trip",
)
@pytest.mark.parametrize(
    "chunk",
    ds4_roundtrip_chunk_sizes(),
    ids=[str(c) for c in ds4_roundtrip_chunk_sizes()],
)
def test_ds4_l3_multi_plane_round_trip_various_chunk_sizes(
    monkeypatch: pytest.MonkeyPatch,
    chunk: int,
) -> None:
    """Bundled L3 four-tuple: real multi-plane kernel round-trip at ``g_end=chunk``."""
    # Unique scope: end-to-end kernel parity for L3 4-plane topology across the same chunk matrix.
    # L2 test has different scheduler-to-plane fan-out; this catches topology-specific routing/copies.
    connector, _, _, dev = build_bundled_ds4_connector(monkeypatch)
    gi = npu_group_index_with_num_planes(connector, 4)
    prefill = prefill_tokens_for_chunk(chunk)
    slot_mappings = make_production_slot_mappings(prefill, dev)
    ds4_multi_plane_round_trip(
        connector,
        gi,
        chunk,
        slot_mappings,
        label=f"L3_bundled_4plane_chunk{chunk}",
    )


@pytest.mark.skipif(
    not npu_available(),
    reason="NPU required for DS4 production SEPARATE_KV round-trip",
)
@pytest.mark.parametrize(
    "chunk",
    ds4_roundtrip_chunk_sizes(),
    ids=[str(c) for c in ds4_roundtrip_chunk_sizes()],
)
def test_ds4_separate_kv_round_trip_various_chunk_sizes(
    monkeypatch: pytest.MonkeyPatch,
    chunk: int,
) -> None:
    """Round-trip each production SEPARATE_KV NPU group at ``g_end=chunk``."""
    # Validates SEPARATE_KV groups (including num_planes=1 routed via multi-plane kernel).
    # Bundled MULTI_PLANE_KV / DSA_C8 tuples are covered by L2/L3 tests.
    connector, _, _, dev = build_bundled_ds4_connector(monkeypatch)
    prefill = prefill_tokens_for_chunk(chunk)
    slot_mappings = make_production_slot_mappings(prefill, dev)
    params = connector.per_group_params or []
    tested = 0
    for gi, g_params in enumerate(params):
        kv_fmt = int(g_params.get("kv_format", -1))
        if kv_fmt in (
            KVCacheFormat.MULTI_PLANE_KV.value,
            KVCacheFormat.DSA_C8_KV.value,
        ):
            continue
        if kv_fmt != KVCacheFormat.SEPARATE_KV.value:
            continue
        ds4_separate_kv_round_trip(
            connector,
            gi,
            chunk,
            slot_mappings,
            label=f"SEPARATE_npu_group_{gi}_chunk{chunk}",
        )
        tested += 1
    assert tested >= 1, (
        f"expected at least one SEPARATE_KV group; formats="
        f"{[(p.get('kv_format'), p.get('num_planes')) for p in params]}"
    )
