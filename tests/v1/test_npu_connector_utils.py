# SPDX-License-Identifier: Apache-2.0
"""Tests for Ascend KV cache permute helpers."""

from __future__ import annotations

import torch

from lmcache_ascend.v1.npu_connector.npu_connectors import VLLMPagedMemNPUConnectorV2
from lmcache_ascend.v1.npu_connector.utils import (
    _maybe_permute,
    permute_kv_caches_to_contiguous,
)


def test_maybe_permute_passes_through_nonzero_storage_offset() -> None:
    blob = torch.zeros(5000, dtype=torch.int8)
    view = blob[128 : 128 + 512].view(2, 4, 8, 8)
    assert view.storage_offset() != 0
    out = _maybe_permute(view)
    assert out.data_ptr() == view.data_ptr()
    assert out.storage_offset() == view.storage_offset()


def test_permute_kv_caches_tuple_with_sliced_planes() -> None:
    blob = torch.zeros(10000, dtype=torch.int8)
    k = blob[256 : 256 + 4096].view(4, 8, 1, 128)
    v = blob[512 : 512 + 32].view(4, 8, 1, 1)
    assert k.storage_offset() != 0
    result = permute_kv_caches_to_contiguous([(k, v)])
    rk, rv = result[0]
    assert rk.data_ptr() == k.data_ptr()
    assert rv.data_ptr() == v.data_ptr()


def test_initialize_kvcaches_ptr_uses_ascend_permute() -> None:
    connector = VLLMPagedMemNPUConnectorV2(
        hidden_dim_size=64,
        num_layers=1,
        use_gpu=False,
    )
    blob = torch.zeros(1024, dtype=torch.int8)
    t = blob[64 : 64 + 256].view(2, 4, 1, 32)
    kv = [(t, t)]
    connector.initialize_kvcaches_ptr(kvcaches=kv)
    assert connector.kvcaches is not None
    assert len(connector.kvcaches) == 1
