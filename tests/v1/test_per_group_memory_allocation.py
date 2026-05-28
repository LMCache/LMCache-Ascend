# SPDX-License-Identifier: Apache-2.0
"""Test that per-NPU-group memory allocation produces a MemoryObj whose
``group_prefix_sum`` matches the number of KV layer groups, so that
``get_tensor(i)`` works for every active (and skipped) group index.

This is the regression test for the DSv4 ``IndexError`` in
``_multi_group_kv_transfer`` where ``memory_obj.get_tensor(2)``
crashed because the MemoryObj was allocated with a single flat shape.
"""

from unittest.mock import patch

import pytest
import torch
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
    get_size_bytes,
)
from lmcache.v1.metadata import LMCacheMetadata

from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.kv_layer_groups import build_kv_layer_groups
from lmcache_ascend.v1.npu_connector.npu_connectors import VLLMPagedMemNPUConnectorV2

from .conftest_ds4 import DS4_CHUNK_SIZE, allocate_multi_group_memory_obj, ds4_setup


def _make_ascend_format_manager(
    kv_caches,
    kv_format: KVCacheFormat,
    num_blocks: int,
    **kwargs,
) -> KVLayerGroupsManager:
    mgr = KVLayerGroupsManager.__new__(KVLayerGroupsManager)
    build_kv_layer_groups(
        mgr,
        kv_caches,
        kv_format=kv_format,
        num_blocks=num_blocks,
        **kwargs,
    )
    return mgr


def _make_metadata(
    manager: KVLayerGroupsManager,
    *,
    use_mla: bool = False,
    chunk_size: int = 256,
    kv_dtype: torch.dtype = torch.bfloat16,
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test-per-group-alloc",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=(1, 1 if use_mla else 2, chunk_size, 1, 1),
        use_mla=use_mla,
        chunk_size=chunk_size,
        kv_layer_groups_manager=manager,
    )


def _allocate_memory_obj(
    shapes: list[torch.Size],
    dtypes: list[torch.dtype],
) -> TensorMemoryObj:
    """Simulate what StorageManager.allocate does: create a flat uint8
    buffer and wrap it in a TensorMemoryObj with per-group metadata."""
    raw_size = get_size_bytes(shapes, dtypes)
    raw_data = torch.zeros(raw_size, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=shapes[0] if len(shapes) == 1 else shapes[0],
        dtype=dtypes[0],
        address=0,
        phy_size=raw_size,
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
        shapes=shapes,
        dtypes=dtypes,
    )
    return TensorMemoryObj(raw_data, meta, parent_allocator=None)


def test_heterogeneous_groups_per_group_get_tensor():
    """3 groups (state/skip, SWA-attention, DSA-attention) produce a MemoryObj
    with group_prefix_sum length 4, allowing get_tensor(0..2)."""
    num_blocks, block_size, num_heads, head_size = 8, 16, 4, 64

    state_layer = (
        torch.empty(num_blocks, 32, dtype=torch.float16),
        torch.empty(num_blocks, 16, dtype=torch.float16),
    )
    attn_layer = (
        torch.empty(num_blocks, block_size, num_heads, head_size, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, num_heads, head_size, dtype=torch.bfloat16),
    )
    dsa_layer = (
        torch.empty(num_blocks, block_size, 1, 512, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, 64, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, 128, dtype=torch.bfloat16),
    )

    kv_caches = [state_layer, attn_layer, attn_layer, dsa_layer]
    mgr = _make_ascend_format_manager(
        kv_caches,
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
    )

    assert len(mgr.kv_layer_groups) >= 2

    md = _make_metadata(mgr)
    num_tokens = 64
    shapes = md.get_shapes(num_tokens)
    dtypes = md.get_dtypes()

    assert len(shapes) == len(mgr.kv_layer_groups)
    assert len(dtypes) == len(mgr.kv_layer_groups)

    mem_obj = _allocate_memory_obj(shapes, dtypes)

    assert len(mem_obj.group_prefix_sum) == len(shapes) + 1

    for i in range(len(shapes)):
        tensor = mem_obj.get_tensor(i)
        assert tensor is not None, f"get_tensor({i}) returned None"
        assert tensor.shape == shapes[i], (
            f"get_tensor({i}) shape mismatch: {tensor.shape} != {shapes[i]}"
        )


def test_single_group_backward_compat():
    """Without kv_layer_groups_manager, metadata returns a single flat shape
    and get_tensor(0) works while get_tensor(1) would be out of range."""
    md = LMCacheMetadata(
        model_name="test-single-group",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(5, 1, 256, 1, 512),
        use_mla=True,
        chunk_size=256,
        kv_layer_groups_manager=None,
    )
    shapes = md.get_shapes(64)
    dtypes = md.get_dtypes()

    assert len(shapes) == 1
    assert len(dtypes) == 1

    mem_obj = _allocate_memory_obj(shapes, dtypes)
    assert len(mem_obj.group_prefix_sum) == 2
    assert mem_obj.get_tensor(0) is not None


def test_mixed_block_size_two_groups_allocation():
    """Dense + compressor layers with different block sizes produce
    two groups, each independently accessible via get_tensor."""
    num_blocks, num_heads, head_size = 32, 8, 64
    dense = (
        torch.empty(num_blocks, 16, num_heads, head_size, dtype=torch.float16),
        torch.empty(num_blocks, 16, num_heads, head_size, dtype=torch.float16),
    )
    compressor = (
        torch.empty(num_blocks, 64, num_heads, head_size, dtype=torch.float16),
        torch.empty(num_blocks, 64, num_heads, head_size, dtype=torch.float16),
    )

    mgr = _make_ascend_format_manager(
        [dense, dense, compressor],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
        layout_hints={"inference_engine_logical_block_size": 128},
    )
    assert len(mgr.kv_layer_groups) == 2

    md = _make_metadata(mgr)
    num_tokens = 128
    shapes = md.get_shapes(num_tokens)
    dtypes = md.get_dtypes()

    assert len(shapes) == 2
    mem_obj = _allocate_memory_obj(shapes, dtypes)
    assert len(mem_obj.group_prefix_sum) == 3

    for i in range(2):
        t = mem_obj.get_tensor(i)
        assert t is not None
        assert t.shape == shapes[i]


def test_multi_group_memory_obj_tensor_view_fails(ds4_setup) -> None:
    """Document why single-group ``memory_obj.tensor`` must not be used on multi-group."""
    _, metadata, _, _ = ds4_setup
    mem_obj = allocate_multi_group_memory_obj(metadata, DS4_CHUNK_SIZE)
    assert len(mem_obj.group_prefix_sum) >= 3
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        _ = mem_obj.tensor


def test_single_group_connector_from_gpu_uses_tensor() -> None:
    """Single-group MLA connector path still uses ``memory_obj.tensor`` unchanged."""
    from .conftest_kvcache import device, npu_available

    if not npu_available():
        pytest.skip("NPU not available")
    dev = device()
    num_blocks, block_size = 8, 16
    kv_lora_rank, qk_rope_head_dim = 512, 64
    layer = (
        torch.randn(num_blocks, block_size, 1, kv_lora_rank, device=dev),
        torch.randn(num_blocks, block_size, 1, qk_rope_head_dim, device=dev),
    )
    kv_caches = [layer, layer]
    connector = VLLMPagedMemNPUConnectorV2(
        hidden_dim_size=kv_lora_rank + qk_rope_head_dim,
        num_layers=2,
        use_mla=True,
    )
    connector.layout_hints = {"vllm_block_size": block_size}
    connector.kvcaches = kv_caches

    mgr = KVLayerGroupsManager.__new__(KVLayerGroupsManager)
    build_kv_layer_groups(
        mgr,
        kv_caches,
        kv_format=KVCacheFormat.MLA_KV,
        num_blocks=num_blocks,
    )
    metadata = LMCacheMetadata(
        model_name="mla-single-group",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 1, 256, 1, kv_lora_rank + qk_rope_head_dim),
        use_mla=True,
        chunk_size=256,
        kv_layer_groups_manager=mgr,
    )
    connector.metadata = metadata

    num_tokens = 16
    shapes = metadata.get_shapes(num_tokens)
    dtypes = metadata.get_dtypes()
    raw_size = get_size_bytes(shapes, dtypes)
    raw_data = torch.zeros(raw_size, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=shapes[0],
        dtype=dtypes[0],
        address=0,
        phy_size=raw_size,
        ref_count=1,
        fmt=MemoryFormat.KV_MLA_FMT,
        shapes=shapes,
        dtypes=dtypes,
    )
    mem_obj = TensorMemoryObj(raw_data, meta, parent_allocator=None)
    assert mem_obj.tensor is not None

    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=dev) % num_blocks
    kwargs = {
        "kvcaches": kv_caches,
        "slot_mapping": slot_mapping,
        "slot_mapping_npu": slot_mapping,
        "no_sync": True,
    }
    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        with patch(
            "lmcache_ascend.v1.npu_connector.npu_connectors.lmc_ops.multi_layer_kv_transfer"
        ) as mock_xfer:
            connector.from_gpu(mem_obj, 0, num_tokens, **kwargs)
    assert mock_xfer.called
