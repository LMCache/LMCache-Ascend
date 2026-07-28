# SPDX-License-Identifier: Apache-2.0
"""Test that per-NPU-group memory allocation produces a MemoryObj whose
``group_prefix_sum`` matches the number of KV layer groups, so that
``get_tensor(i)`` works for every active (and skipped) group index.

This is the regression test for the DSv4 ``IndexError`` in
``_multi_group_kv_transfer`` where ``memory_obj.get_tensor(2)``
crashed because the MemoryObj was allocated with a single flat shape.
"""

from unittest.mock import patch

import lmcache_ascend  # noqa: F401
import pytest
import torch
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
    get_size_bytes,
)
from lmcache.v1.metadata import LMCacheMetadata

from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.kv_layer_groups import build_kv_layer_groups
from lmcache_ascend.v1.memory_management import is_multi_group_memory_obj
from lmcache_ascend.v1.npu_connector.npu_connectors import VLLMPagedMemNPUConnectorV2

from .conftest_ds4 import (
    DS4_CHUNK_SIZE,
    DS4_PRODUCTION_CHUNK_TOKENS,
    allocate_multi_group_memory_obj,
    build_bundled_ds4_connector,
    make_ds4_setup,
)


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


def test_bundled_ds4_sw_group_uses_physical_token_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CR128 sliding-window NPU group allocates ``physical_chunk_size`` rows."""
    _, metadata, _, _ = build_bundled_ds4_connector(monkeypatch)
    mgr = metadata.kv_layer_groups_manager
    assert mgr is not None
    num = DS4_PRODUCTION_CHUNK_TOKENS
    shapes = metadata.get_shapes(num)
    sw_groups = [
        g
        for g in mgr.kv_layer_groups
        if g.compress_ratio > 1 and g.physical_chunk_size < num
    ]
    if not sw_groups:
        pytest.skip("5-layer DS4 fixture has no compressed SW group")
    for group in sw_groups:
        idx = mgr.kv_layer_groups.index(group)
        assert shapes[idx][2] == group.physical_chunk_size
    for group_idx, group in enumerate(mgr.kv_layer_groups):
        if getattr(group, "multi_plane_hidden_bytes", None) is not None:
            assert shapes[group_idx][2] == num


def test_multi_group_memory_obj_tensor_is_flat_uint8() -> None:
    """Multi-group .tensor is flat uint8; meta.shape stays group-0."""
    _, metadata, _, _ = make_ds4_setup()
    mem_obj = allocate_multi_group_memory_obj(metadata, DS4_CHUNK_SIZE)
    assert len(mem_obj.group_prefix_sum) >= 3
    shapes = metadata.get_shapes(DS4_CHUNK_SIZE)
    assert mem_obj.meta.shape == shapes[0]
    assert mem_obj.meta.dtype == metadata.get_dtypes()[0]
    tensor = mem_obj.tensor
    assert tensor is not None
    assert tensor.shape == (mem_obj.get_size(),)
    assert tensor.dtype == torch.uint8
    for i, shape in enumerate(shapes):
        group_tensor = mem_obj.get_tensor(i)
        assert group_tensor is not None
        assert group_tensor.shape == shape


def test_paged_allocate_syncs_group_prefix_sum_for_partial_chunk() -> None:
    """Paged freelist allocate must refresh prefixes when shapes shrink."""
    _, metadata, _, _ = make_ds4_setup()
    full_shapes = metadata.get_shapes()
    partial_shapes = metadata.get_shapes(num_tokens=max(1, DS4_CHUNK_SIZE // 2))
    dtypes = metadata.get_dtypes()
    assert full_shapes != partial_shapes

    page_bytes = get_size_bytes(full_shapes, dtypes)
    buffer = torch.zeros(page_bytes * 2, dtype=torch.uint8)
    allocator = PagedTensorMemoryAllocator(
        buffer, full_shapes, dtypes, fmt=MemoryFormat.KV_2LTD
    )

    mem_obj = allocator.allocate(partial_shapes, dtypes, fmt=MemoryFormat.KV_2LTD)
    assert mem_obj is not None
    assert mem_obj.meta.shapes == partial_shapes

    expected = [0]
    nbytes = 0
    for shape, dtype in zip(partial_shapes, dtypes, strict=True):
        nbytes += int(shape.numel()) * dtype.itemsize
        expected.append(nbytes)
    assert mem_obj.group_prefix_sum == expected
    assert mem_obj.get_size() == expected[-1]
    for i, shape in enumerate(partial_shapes):
        group_tensor = mem_obj.get_tensor(i)
        assert group_tensor is not None
        assert group_tensor.shape == shape


def test_multi_group_disk_save_load_roundtrip(tmp_path) -> None:
    """Disk tier stores multi-group chunks and reloads per-group structure."""
    # Standard
    import asyncio
    import os

    # Third Party
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import PinMemoryAllocator
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

    _, metadata, _, _ = make_ds4_setup()
    mem_obj = allocate_multi_group_memory_obj(metadata, DS4_CHUNK_SIZE)
    num_bytes = mem_obj.get_size()
    mem_obj.raw_data[:num_bytes] = torch.arange(num_bytes, dtype=torch.uint8)

    key = CacheEngineKey(
        model_name="ds4-multi-group",
        world_size=1,
        worker_id=0,
        chunk_hash=12345,
        dtype=torch.uint8,
    )

    loop = asyncio.new_event_loop()
    try:
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=DS4_CHUNK_SIZE,
            local_disk=str(tmp_path),
            max_local_disk_size=500.0,
            lmcache_instance_id="test_instance",
        )
        allocator = PinMemoryAllocator(512 * 1024 * 1024)
        cpu_backend = LocalCPUBackend(config, memory_allocator=allocator)
        disk_backend = LocalDiskBackend(
            config=config,
            loop=loop,
            local_cpu_backend=cpu_backend,
            dst_device="cpu",
        )

        mem_obj.ref_count_up()
        disk_backend.async_save_bytes_to_disk(key, mem_obj)

        disk_meta = disk_backend.dict[key]
        assert disk_meta.shapes is not None
        assert disk_meta.dtypes is not None
        assert os.path.getsize(disk_meta.path) == num_bytes

        loaded = disk_backend.get_blocking(key)
        assert loaded is not None
        assert loaded.get_size() == num_bytes
        assert loaded.tensor is not None
        assert loaded.tensor.shape == (num_bytes,)
        assert torch.equal(loaded.raw_data[:num_bytes], mem_obj.raw_data[:num_bytes])
        shapes = metadata.get_shapes(DS4_CHUNK_SIZE)
        for i in range(len(shapes)):
            begin = loaded.group_prefix_sum[i]
            end = loaded.group_prefix_sum[i + 1]
            assert torch.equal(
                loaded.raw_data[begin:end],
                mem_obj.raw_data[begin:end],
            )

        allocator.close()
    finally:
        loop.close()


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


def test_fill_shard_sender_receiver_preserve_multi_group_flat_raw_data():
    """Sharded-broadcast fill must not reshape multi-group blobs to group-0.

    Regression for RuntimeError: shape '[..., 129]' invalid for input of
    size <full multi-group nbytes> in ``_fill_shard_sender`` / receiver.
    """
    shapes = [
        torch.Size([1, 2, 256, 64]),
        torch.Size([1, 2, 256, 128]),
    ]
    dtypes = [torch.bfloat16, torch.bfloat16]
    mem_obj = _allocate_memory_obj(shapes, dtypes)
    assert is_multi_group_memory_obj(mem_obj)
    byte_size = mem_obj.get_size()
    mem_obj.raw_data.fill_(7)

    merged = torch.zeros(byte_size, dtype=torch.uint8)
    layout = [(0, 0, byte_size)]
    meta_table = [(0, 256, mem_obj.metadata.to_dict())]
    reordered_chunks = [(None, mem_obj, 0, 256)]

    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)

    objs, starts, ends = AscendLMCacheEngine._fill_shard_sender(
        engine, merged, layout, meta_table, reordered_chunks
    )
    assert starts == [0] and ends == [256]
    assert len(objs) == 1
    assert is_multi_group_memory_obj(objs[0])
    assert objs[0].get_shapes() == shapes
    assert objs[0].get_dtypes() == dtypes
    assert objs[0].raw_data.dtype == torch.uint8
    assert objs[0].raw_data.numel() == byte_size
    assert torch.equal(objs[0].raw_data, merged)

    ret_mask = torch.zeros(256, dtype=torch.bool)
    recv_objs, r_starts, r_ends = AscendLMCacheEngine._fill_shard_receiver(
        engine, merged, layout, meta_table, ret_mask
    )
    assert r_starts == [0] and r_ends == [256]
    assert len(recv_objs) == 1
    assert is_multi_group_memory_obj(recv_objs[0])
    assert recv_objs[0].get_shapes() == shapes
    assert recv_objs[0].get_dtypes() == dtypes
    assert recv_objs[0].raw_data.dtype == torch.uint8
    assert recv_objs[0].raw_data.numel() == byte_size
    assert ret_mask.all()


def test_fill_shard_sender_receiver_single_group_still_reshapes():
    """Single-group path keeps the existing dtype/shape view of the slice."""
    shapes = [torch.Size([1, 2, 256, 64])]
    dtypes = [torch.bfloat16]
    mem_obj = _allocate_memory_obj(shapes, dtypes)
    assert not is_multi_group_memory_obj(mem_obj)
    byte_size = mem_obj.get_size()

    merged = torch.zeros(byte_size, dtype=torch.uint8)
    layout = [(0, 0, byte_size)]
    meta_table = [(0, 256, mem_obj.metadata.to_dict())]
    reordered_chunks = [(None, mem_obj, 0, 256)]

    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)

    objs, _, _ = AscendLMCacheEngine._fill_shard_sender(
        engine, merged, layout, meta_table, reordered_chunks
    )
    assert len(objs) == 1
    assert objs[0].raw_data.dtype == torch.bfloat16
    assert objs[0].raw_data.shape == shapes[0]

    ret_mask = torch.zeros(256, dtype=torch.bool)
    recv_objs, _, _ = AscendLMCacheEngine._fill_shard_receiver(
        engine, merged, layout, meta_table, ret_mask
    )
    assert len(recv_objs) == 1
    assert recv_objs[0].raw_data.dtype == torch.bfloat16
    assert recv_objs[0].raw_data.shape == shapes[0]
    assert ret_mask.all()
