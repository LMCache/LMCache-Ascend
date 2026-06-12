# SPDX-License-Identifier: Apache-2.0
"""NPU connector multi-group ``start_load_kv`` adapter tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata

from .conftest_ds4 import (
    DS4_NUM_SCHEDULER_GROUPS,
    DS4_VLLM_BLOCK_SIZE,
    make_ascend_adapter_for_load,
    make_forward_context,
    make_load_req_meta,
    make_partial_fail_load_req_meta,
    npu_available,
)


def test_start_load_kv_multi_group() -> None:
    """Multi-group load invokes retrieve with per-group slot mappings."""
    if not npu_available():
        pytest.skip("NPU not available")
    adapter = make_ascend_adapter_for_load()
    req = make_load_req_meta()
    pg = req.primary_kv_group_idx
    primary_cpu = req.slot_mappings_by_group[pg]
    metadata = LMCacheConnectorMetadata(requests=[req])
    adapter._parent._get_connector_metadata.return_value = metadata

    adapter.start_load_kv(make_forward_context())

    adapter.lmcache_engine.retrieve.assert_called_once()
    _args, kwargs = adapter.lmcache_engine.retrieve.call_args
    assert "slot_mappings_npu_by_group" in kwargs
    assert len(kwargs["slot_mappings_npu_by_group"]) == DS4_NUM_SCHEDULER_GROUPS
    assert "slot_mappings_by_group" in kwargs
    assert len(kwargs["slot_mappings_by_group"]) == DS4_NUM_SCHEDULER_GROUPS
    primary = kwargs["slot_mapping"]
    primary_npu = kwargs["slot_mappings_npu_by_group"][pg]
    assert len(primary) == len(primary_cpu)
    assert primary.data_ptr() == primary_npu.data_ptr()
    assert "mp_launch_meta" not in kwargs


def test_start_load_kv_single_group_delegates_to_super() -> None:
    pytest.importorskip("vllm")
    from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
        LMCacheConnectorV1ImplMultiGroup,
    )

    adapter = make_ascend_adapter_for_load(num_kv_groups=1)
    adapter._compress_ratios_by_group = (1,)
    adapter._block_sizes_by_group = (DS4_VLLM_BLOCK_SIZE,)

    with patch.object(
        LMCacheConnectorV1ImplMultiGroup,
        "start_load_kv",
    ) as mock_super:
        adapter.start_load_kv(make_forward_context())
        mock_super.assert_called_once()


def test_start_load_kv_skips_requests_without_load_spec() -> None:
    if not npu_available():
        pytest.skip("NPU not available")
    adapter = make_ascend_adapter_for_load()
    no_spec = make_load_req_meta(omit_load_spec=True)
    cannot_load = make_load_req_meta(can_load=False)
    metadata = LMCacheConnectorMetadata(requests=[no_spec, cannot_load])
    adapter._parent._get_connector_metadata.return_value = metadata

    adapter.start_load_kv(make_forward_context())

    adapter.lmcache_engine.retrieve.assert_not_called()


def test_start_load_kv_partial_fail_uses_primary_group_block_size() -> None:
    """Partial retrieve must record block IDs using the primary group's block size.

    Slot 1024 is block 1 at bs=1024 but block 8 at bs=128 (cache_config default).
    """
    if not npu_available():
        pytest.skip("NPU not available")

    adapter = make_ascend_adapter_for_load(num_kv_groups=2)
    adapter._block_sizes_by_group = (128, 1024)
    adapter._compress_ratios_by_group = (8, 1)
    adapter._block_size = DS4_VLLM_BLOCK_SIZE
    adapter.lmcache_engine.retrieve.return_value = torch.tensor([True, False])

    req = make_partial_fail_load_req_meta()
    metadata = LMCacheConnectorMetadata(requests=[req])
    adapter._parent._get_connector_metadata.return_value = metadata

    adapter.start_load_kv(make_forward_context())

    adapter.lmcache_engine.retrieve.assert_called_once()
    assert adapter._invalid_block_ids == {1}
    assert 1024 // 128 == 8
    assert {8} != {1}
