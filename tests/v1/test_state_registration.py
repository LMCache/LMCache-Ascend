# SPDX-License-Identifier: Apache-2.0
"""Adapter integration checks for the Ascend host acceptance suite."""

# Standard
from types import SimpleNamespace

# Third Party
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec
import pytest
import torch

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl


def test_registration_builds_layout_before_rejecting_hybrid_transfer():
    connector = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    spec = MambaSpec(
        block_size=16,
        shapes=((1, 3), (2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(["gdn.0"], spec)]
    )
    tensors = {
        "gdn.0": [torch.empty(5, 1, 3, dtype=torch.bfloat16), torch.empty(5, 2, 2)]
    }
    with pytest.raises(NotImplementedError, match="hybrid"):
        connector.register_kv_caches(tensors)
    assert connector.state_layouts[0].nbytes == 24


def test_request_finished_uses_the_same_spec_primary(monkeypatch):
    connector = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    connector._state_primary_kv_group_idx = 1
    connector._block_sizes_by_group = (16, 16)
    monkeypatch.setattr(
        connector,
        "request_finished",
        lambda request, blocks: (False, {"blocks": blocks}),
    )
    assert connector.request_finished_all_groups(None, ([1] * 32, [9, 10])) == (
        False,
        {"blocks": [9, 10]},
    )
