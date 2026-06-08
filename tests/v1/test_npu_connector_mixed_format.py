# SPDX-License-Identifier: Apache-2.0
"""Tests for mixed-format KV cache handling in the NPU connector."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemNPUConnectorV2,
    _build_dsa_c8_multi_plane_group_params,
    _derive_group_params,
    _is_kernel_compatible_entry,
    _materialize_mp_device_params,
)

from .conftest_ds4 import (
    DS4_CHUNK_SIZE,
    allocate_multi_group_memory_obj,
    build_bundled_ds4_connector,
    ds4_setup,
    make_slot_mappings,
    make_slot_transfer_kwargs,
    npu_available,
)


class _FakeShapeDesc:
    def __init__(self, nb: int, bs: int, block_stride_elems: int = 0) -> None:
        self.nb = nb
        self.bs = bs
        self.block_stride_elems = block_stride_elems


def test_is_kernel_compatible_entry_mixed_lengths() -> None:
    swa = [torch.zeros(2, 4, 8, device="cpu")]
    dsa = (
        torch.zeros(2, 4, 8, device="cpu"),
        torch.zeros(2, 4, 4, device="cpu"),
        torch.zeros(2, 4, 16, device="cpu"),
        torch.zeros(2, 4, 1, device="cpu"),
    )
    state = [torch.zeros(1, device="cpu") for _ in range(8)]
    assert _is_kernel_compatible_entry(swa)
    assert _is_kernel_compatible_entry(dsa)
    assert not _is_kernel_compatible_entry(state)


def test_scheduler_slot_group_for_npu_group() -> None:
    connector = VLLMPagedMemNPUConnectorV2(
        hidden_dim_size=512,
        num_layers=3,
        use_mla=True,
    )
    connector.layout_hints = {
        "primary_kv_group_idx": 1,
        "scheduler_group_by_flat_layer": [3, 1, 4, 1, 1],
    }
    assert connector._scheduler_slot_group_for_npu_group(0, [0]) == 3
    assert connector._scheduler_slot_group_for_npu_group(1, [1]) == 1
    assert connector._scheduler_slot_group_for_npu_group(2, [2]) == 4
    connector.layout_hints = {"primary_kv_group_idx": 1}
    assert connector._scheduler_slot_group_for_npu_group(0, [0]) == 1


def test_derive_group_params_dsa_c8() -> None:
    k = torch.zeros(10, 128, 512, dtype=torch.bfloat16)
    v = torch.zeros(10, 128, 64, dtype=torch.bfloat16)
    dsa_k = torch.zeros(10, 128, 128, dtype=torch.int8)
    dsa_scale = torch.zeros(10, 128, 1, dtype=torch.float16)
    entry = (k, v, dsa_k, dsa_scale)
    params = _derive_group_params(
        entry,
        KVCacheFormat.DSA_C8_KV,
        _FakeShapeDesc(nb=10, bs=128),
    )
    assert params["kv_format"] == KVCacheFormat.DSA_C8_KV.value
    assert params["block_size"] == 128
    assert "dsa_c8_plane_bytes" in params
    kb, vb, db, sb = params["dsa_c8_plane_bytes"]
    assert kb > 0 and vb > 0 and db > 0 and sb > 0


def test_initialize_pointers_mixed_format_no_unpack_crash() -> None:
    """Mixed 1- vs 4-element entries use per-group pointers (no flat DSA_C8 table)."""
    num_blocks = 4
    block_size = 8

    npu = torch.device("npu:0") if hasattr(torch, "npu") and torch.npu.is_available() else None
    if npu is None:
        pytest.skip("NPU not available")

    swa_layer = torch.zeros(
        num_blocks, block_size, 512, dtype=torch.bfloat16, device=npu
    )
    dsa_layer = (
        torch.zeros(num_blocks, block_size, 512, dtype=torch.bfloat16, device=npu),
        torch.zeros(num_blocks, block_size, 64, dtype=torch.bfloat16, device=npu),
        torch.zeros(num_blocks, block_size, 128, dtype=torch.int8, device=npu),
        torch.zeros(num_blocks, block_size, 1, dtype=torch.float16, device=npu),
    )

    kv_caches = [swa_layer, swa_layer, dsa_layer]

    connector = VLLMPagedMemNPUConnectorV2(
        hidden_dim_size=512,
        num_layers=len(kv_caches),
        use_mla=True,
    )
    connector.metadata = MagicMock()
    connector.metadata.kv_layer_groups_manager = None
    connector.metadata.chunk_size = 256
    connector.layout_hints = {
        "vllm_block_size": block_size,
        "primary_kv_group_idx": 1,
        "block_sizes_by_group": (128, 128, 128),
    }
    connector.kvcaches = kv_caches
    connector.num_layers = len(kv_caches)

    mock_sd = MagicMock()
    mock_sd.nb = num_blocks
    mock_sd.bs = block_size
    mock_sd.block_stride_elems = 0

    mock_group_swa = MagicMock()
    mock_group_swa.layer_indices = [0, 1]
    mock_group_swa.shape_desc = mock_sd

    mock_group_dsa = MagicMock()
    mock_group_dsa.layer_indices = [2]
    mock_group_dsa.shape_desc = mock_sd

    mock_manager = MagicMock()
    mock_manager.kv_layer_groups = [
        mock_group_swa,
        mock_group_dsa,
    ]
    connector.metadata.kv_layer_groups_manager = mock_manager

    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        ptrs = connector._initialize_pointers(kv_caches)

    assert connector._is_mixed_format
    assert connector.page_buffer_size == 0
    assert connector.group_kv_cache_pointers is not None
    assert connector.per_group_params is not None
    assert ptrs is not None


def test_ds4_pointer_init_matches_scheduler_map(ds4_setup) -> None:
    """Per-group pointers and scheduler_slot_group align with flatten map."""
    connector, metadata, kv_caches, _ = ds4_setup
    sched_map = connector.layout_hints["scheduler_group_by_flat_layer"]
    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        connector._initialize_pointers(kv_caches)

    assert connector.group_kv_cache_pointers is not None
    assert connector.per_group_params is not None
    assert len(connector.per_group_params) >= 2

    for params in connector.per_group_params:
        assert params["scheduler_slot_group"] >= 0

    mgr = metadata.kv_layer_groups_manager
    assert mgr is not None
    for group_idx, group in enumerate(mgr.kv_layer_groups):
        expected = {sched_map[i] for i in group.layer_indices}
        assert len(expected) == 1
        assert (
            connector.per_group_params[group_idx]["scheduler_slot_group"]
            == next(iter(expected))
        )

    sched_groups = {
        p["scheduler_slot_group"] for p in connector.per_group_params
    }
    assert 1 in sched_groups


def test_multi_group_store_dispatches_uint8_kernels(ds4_setup) -> None:
    """int8/float32 memory groups must pass uint8 tensors to transfer kernels."""
    connector, metadata, kv_caches, dev = ds4_setup
    num_tokens = 64
    mem_obj = allocate_multi_group_memory_obj(metadata, num_tokens)
    assert metadata.get_dtypes().count(torch.uint8) >= 2

    slot_mappings = make_slot_mappings(num_tokens, dev)
    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        connector._initialize_pointers(kv_caches)
    kwargs = {
        "kvcaches": kv_caches,
        "slot_mapping": slot_mappings[1],
        "slot_mapping_npu": slot_mappings[1],
        "slot_mappings_npu_by_group": slot_mappings,
        "no_sync": True,
        **make_slot_transfer_kwargs(
            slot_mappings,
            connector=connector,
            chunk_ranges=[(0, num_tokens)],
        ),
    }

    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        with (
            patch(
                "lmcache_ascend.v1.npu_connector.npu_connectors.lmc_ops.multi_layer_kv_transfer"
            ) as mock_xfer,
            patch(
                "lmcache_ascend.v1.npu_connector.npu_connectors.lmc_ops.multi_layer_kv_transfer_multi_plane"
            ) as mock_plane_xfer,
        ):
            connector.from_gpu(mem_obj, 0, num_tokens, **kwargs)

    kernel_tensors = [
        call.args[0]
        for call in (*mock_xfer.call_args_list, *mock_plane_xfer.call_args_list)
    ]
    assert kernel_tensors
    for t in kernel_tensors:
        assert t.dtype != torch.int8
        assert t.dtype != torch.float32
    assert any(t.dtype == torch.uint8 for t in kernel_tensors)


def test_bundled_multi_plane_kernel_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundled groups route through multi-plane: SEPARATE kv_size=1 (num_planes=1) and L2/L3."""
    # Routing-only: validates dispatch split and call cardinality (not data parity).
    if not npu_available():
        pytest.skip("NPU not available")
    import lmcache_ascend.c_ops as lmc_ops

    connector, metadata, _, dev = build_bundled_ds4_connector(monkeypatch)
    mem = allocate_multi_group_memory_obj(metadata, DS4_CHUNK_SIZE)
    slot_mappings = make_slot_mappings(DS4_CHUNK_SIZE, dev)

    transfer_kwargs = make_slot_transfer_kwargs(
        slot_mappings,
        connector=connector,
        chunk_ranges=[(0, DS4_CHUNK_SIZE)],
    )
    with patch.object(
        connector, "_invoke_multi_plane_kv_transfer"
    ) as mock_mp, patch.object(lmc_ops, "multi_layer_kv_transfer") as mock_single:
        connector._multi_group_kv_transfer(
            mem,
            0,
            DS4_CHUNK_SIZE,
            slot_mappings,
            is_store=True,
            stream=connector.store_stream,
            **transfer_kwargs,
        )
        total = mock_mp.call_count * 4 + mock_single.call_count

    assert mock_mp.call_count >= 1
    l2_calls = [
        c for c in mock_mp.call_args_list
        if c.kwargs["group_params"].get("num_planes") == 8
    ]
    assert len(l2_calls) == 1
    assert any(
        c.kwargs["group_params"].get("num_planes") == 4
        for c in mock_mp.call_args_list
    )
    separate_calls = [
        c for c in mock_mp.call_args_list
        if c.kwargs["group_params"].get("num_planes") == 1
        and c.kwargs["group_params"].get("kv_format")
        == KVCacheFormat.SEPARATE_KV.value
    ]
    assert len(separate_calls) == 2
    assert mock_mp.call_count == 4
    assert mock_single.call_count == 0
    assert total <= 20


def test_initialize_pointers_skips_detect_on_warm_path() -> None:
    """Second _initialize_pointers call must not invoke KVCacheFormat.detect again."""
    num_blocks = 4
    block_size = 8

    npu = (
        torch.device("npu:0")
        if hasattr(torch, "npu") and torch.npu.is_available()
        else None
    )
    if npu is None:
        pytest.skip("NPU not available")

    swa_layer = torch.zeros(
        num_blocks, block_size, 512, dtype=torch.bfloat16, device=npu
    )
    dsa_layer = (
        torch.zeros(num_blocks, block_size, 512, dtype=torch.bfloat16, device=npu),
        torch.zeros(num_blocks, block_size, 64, dtype=torch.bfloat16, device=npu),
        torch.zeros(num_blocks, block_size, 128, dtype=torch.int8, device=npu),
        torch.zeros(num_blocks, block_size, 1, dtype=torch.float16, device=npu),
    )
    kv_caches = [swa_layer, swa_layer, dsa_layer]

    connector = VLLMPagedMemNPUConnectorV2(
        hidden_dim_size=512,
        num_layers=len(kv_caches),
        use_mla=True,
    )
    connector.metadata = MagicMock()
    connector.metadata.chunk_size = 256
    connector.layout_hints = {
        "vllm_block_size": block_size,
        "primary_kv_group_idx": 1,
        "block_sizes_by_group": (128, 128, 128),
    }
    connector.kvcaches = kv_caches
    connector.num_layers = len(kv_caches)

    mock_sd = MagicMock()
    mock_sd.nb = num_blocks
    mock_sd.bs = block_size
    mock_sd.block_stride_elems = 0

    mock_group_swa = MagicMock()
    mock_group_swa.layer_indices = [0, 1]
    mock_group_swa.shape_desc = mock_sd

    mock_group_dsa = MagicMock()
    mock_group_dsa.layer_indices = [2]
    mock_group_dsa.shape_desc = mock_sd

    mock_manager = MagicMock()
    mock_manager.kv_layer_groups = [mock_group_swa, mock_group_dsa]
    connector.metadata.kv_layer_groups_manager = mock_manager

    detect_calls = 0
    real_detect = KVCacheFormat.detect

    def counting_detect(*args, **kwargs):
        nonlocal detect_calls
        detect_calls += 1
        return real_detect(*args, **kwargs)

    with (
        patch(
            "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
            return_value=False,
        ),
        patch.object(KVCacheFormat, "detect", side_effect=counting_detect),
    ):
        connector._initialize_pointers(kv_caches)
        after_first = detect_calls
        connector._initialize_pointers(kv_caches)

    assert after_first >= 1
    assert detect_calls == after_first


def test_materialize_mp_device_params_idempotent() -> None:
    """Repeated materialize must reuse the same mp_device tensors."""
    params = _build_dsa_c8_multi_plane_group_params(
        (512, 64, 128, 1),
        block_size=128,
        page_buffer_size=1280,
        num_tokens=256,
    )
    device = torch.device("cpu")
    _materialize_mp_device_params(params, device)
    assert params.get("mp_device") is not None
    pbs_first = params["mp_device"]["pbs"]

    _materialize_mp_device_params(params, device)
    assert params["mp_device"]["pbs"] is pbs_first


def test_mp_device_lazy_materialized_on_first_transfer(ds4_setup) -> None:
    """mp_device tensors are created on first transfer, not at pointer init."""
    connector, metadata, kv_caches, dev = ds4_setup
    num_tokens = 64
    mem_obj = allocate_multi_group_memory_obj(metadata, num_tokens)
    slot_mappings = make_slot_mappings(num_tokens, dev)

    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        connector._initialize_pointers(kv_caches)

    for params in connector.per_group_params:
        assert params.get("mp_device") is None

    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.lmc_ops."
        "multi_layer_kv_transfer_multi_plane"
    ):
        connector._multi_group_kv_transfer(
            mem_obj,
            0,
            num_tokens,
            slot_mappings,
            is_store=True,
            stream=connector.store_stream,
            **make_slot_transfer_kwargs(
                slot_mappings,
                connector=connector,
                chunk_ranges=[(0, num_tokens)],
            ),
        )

    materialized = [
        p for p in connector.per_group_params if p.get("mp_device") is not None
    ]
    assert materialized
