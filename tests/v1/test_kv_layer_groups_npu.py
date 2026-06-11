# SPDX-License-Identifier: Apache-2.0
"""Tests for Ascend KV-layer grouping helpers and upstream manager integration.

Legacy multi-view layers are covered via
:func:`lmcache_ascend.v1.kv_layer_groups.build_kv_layer_groups` (direct call).
Flattened DSv4 paths use upstream :class:`KVLayerGroupsManager` with
``gpu_kv_format`` from :func:`normalize_kv_and_discover_format`.

The tests use **CPU tensors** because grouping is a pure-Python
classification step that never touches kernels — the Ascend NPU
device is only required by downstream transfer kernels, which these
tests don't invoke.
"""

# Third Party
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.metadata import LMCacheMetadata
import pytest
import torch

# First Party
import lmcache_ascend  # noqa: F401  — applies get_shapes patch
from lmcache_ascend.v1.kv_format import (
    KVCacheFormat,
    _get_primary_blob_view,
    _is_shared_storage_blob,
)
from lmcache_ascend.v1.kv_layer_groups import (
    _multi_plane_lmc_row_bytes,
    _get_kv_cache_group_key_and_info,
    build_kv_layer_groups,
)
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    _derive_group_params,
    _split_kv_layer_groups_by_scheduler_slot,
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
    kv_dtype: torch.dtype = torch.float16,
) -> LMCacheMetadata:
    """Wrap a freshly-built manager into an ``LMCacheMetadata`` instance.

    Only the fields actually read by the metadata helpers are
    populated — everything else gets a placeholder so the dataclass
    constructs.
    """
    return LMCacheMetadata(
        model_name="ascend-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=(1, 2 if not use_mla else 1, chunk_size, 1, 1),
        use_mla=use_mla,
        chunk_size=chunk_size,
        kv_layer_groups_manager=manager,
    )


# --------------------------------------------------------------------------- #
# Single-format groupings                                                     #
# --------------------------------------------------------------------------- #


def test_single_layer_separate_kv_groups_as_attention():
    num_blocks, block_size, num_heads, head_size = 32, 16, 8, 64
    layer = (
        torch.empty(num_blocks, block_size, num_heads, head_size, dtype=torch.float16),
        torch.empty(num_blocks, block_size, num_heads, head_size, dtype=torch.float16),
    )
    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
    )
    assert len(mgr.kv_layer_groups) == 1
    g = mgr.kv_layer_groups[0]
    assert g.layer_indices == [0]
    assert g.shape_desc.kv_size == 2
    assert g.shape_desc.bs == block_size
    assert g.shape_desc.nb == num_blocks
    assert g.shape_desc.nh * g.shape_desc.hs == num_heads * head_size
    assert g.dtype == torch.float16
    assert g.compress_ratio == 1
    assert g.physical_chunk_size == 256


def test_mla_2tuple_classified_as_attention_not_gdn():
    """Regression for upstream's ``len==2 + mismatched shapes → GDN``."""
    num_blocks, block_size = 32, 16
    kv_lora_rank, qk_rope_head_dim = 512, 64
    layer = (
        torch.empty(num_blocks, block_size, 1, kv_lora_rank, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, qk_rope_head_dim, dtype=torch.bfloat16),
    )
    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.MLA_KV,
        num_blocks,
    )
    assert len(mgr.kv_layer_groups) == 1
    g = mgr.kv_layer_groups[0]
    assert g.shape_desc.kv_size == 1
    assert g.hidden_dim_size == kv_lora_rank + qk_rope_head_dim
    assert g.dtype == torch.bfloat16
    assert g.compress_ratio == 1
    assert g.physical_chunk_size == 256


def test_dsa_3tuple_classified_as_attention():
    num_blocks, block_size = 32, 16
    kv_lora_rank, qk_rope_head_dim, dsa_head_dim = 512, 64, 128
    layer = (
        torch.empty(num_blocks, block_size, 1, kv_lora_rank, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, qk_rope_head_dim, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, dsa_head_dim, dtype=torch.bfloat16),
    )
    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.DSA_KV,
        num_blocks,
    )
    assert len(mgr.kv_layer_groups) == 1
    g = mgr.kv_layer_groups[0]
    assert g.shape_desc.kv_size == 1
    assert g.hidden_dim_size == kv_lora_rank + qk_rope_head_dim + dsa_head_dim
    assert g.compress_ratio == 1
    assert g.physical_chunk_size == 256


def test_dsa_c8_4tuple_classified_as_attention_with_mixed_dtypes():
    num_blocks, block_size = 32, 16
    kv_lora_rank, qk_rope_head_dim, dsa_head_dim = 512, 64, 128
    layer = (
        torch.empty(num_blocks, block_size, 1, kv_lora_rank, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, qk_rope_head_dim, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, dsa_head_dim, dtype=torch.int8),
        torch.empty(num_blocks, block_size, 1, 1, dtype=torch.float16),
    )
    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.DSA_C8_KV,
        num_blocks,
    )
    assert len(mgr.kv_layer_groups) == 1
    g = mgr.kv_layer_groups[0]
    assert g.shape_desc.kv_size == 1
    assert g.dtype == torch.uint8
    assert g.multi_plane_hidden_bytes is not None
    assert g.shape_desc.hs == _multi_plane_lmc_row_bytes(
        g.multi_plane_hidden_bytes,
        g.physical_chunk_size,
    )
    assert g.shape_desc.element_size == 2  # max itemsize across tuple tensors
    assert g.compress_ratio == 1
    assert g.physical_chunk_size == 256


def test_shared_storage_blob_classified_with_vllm_block_size():
    """DSv4-style multi-view layers share one int8 blob; group as SEPARATE_KV."""
    num_blocks, vllm_bs, hidden = 8, 16, 32
    page_bytes = vllm_bs * hidden
    blob = torch.zeros(num_blocks * page_bytes, dtype=torch.int8)
    full = blob.view(num_blocks, vllm_bs, hidden)
    # Two views with different dtypes/shapes but one storage.
    v_bf16 = full.view(torch.bfloat16)
    v_int8 = full
    layer = [v_bf16, v_int8]
    assert _is_shared_storage_blob(layer)

    mgr = _make_ascend_format_manager(
        [layer, layer],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
        layout_hints={"vllm_block_size": vllm_bs},
    )
    assert len(mgr.kv_layer_groups) == 1
    g = mgr.kv_layer_groups[0]
    assert g.layer_indices == [0, 1]
    assert g.shape_desc.kv_size == 2
    assert g.shape_desc.bs == vllm_bs
  # bfloat16 reinterpretation halves the last dim vs int8 layout.
    assert g.shape_desc.hs == _get_primary_blob_view(layer).shape[-1]
    assert g.shape_desc.element_size == 2  # primary view is bfloat16
    assert g.dtype == torch.bfloat16


def test_shared_storage_blob_padded_storage_uses_primary_view():
    """Blob with per-block padding (133168-style) must not divide storage.nbytes."""
    num_blocks = 2
    page_bytes_padded = 133168
    blob = torch.zeros(num_blocks * page_bytes_padded, dtype=torch.int8)
    small = blob[: num_blocks * 32 * 64 * 2].view(torch.bfloat16).view(
        num_blocks, 32, 64
    )
    large = blob[: num_blocks * 128 * 512 * 2].view(torch.bfloat16).view(
        num_blocks, 128, 512
    )
    layer = [small, large]
    assert _is_shared_storage_blob(layer)
    assert _get_primary_blob_view(layer) is large

    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
        layout_hints={"vllm_block_size": 32},
    )
    g = mgr.kv_layer_groups[0]
    assert g.shape_desc.bs == 128
    assert g.shape_desc.hs == 512
    assert g.dtype == torch.bfloat16

    key = _get_kv_cache_group_key_and_info(layer, vllm_block_size=32)
    kv_size, hidden, bs, dtype_key, num_tensors = key
    assert kv_size == 2
    assert bs == 128
    assert hidden == 512
    assert dtype_key == torch.bfloat16
    assert num_tensors == 0

    class _ShapeDesc:
        nb = num_blocks
        bs = 128
        block_stride_elems = 0

    params = _derive_group_params(
        layer, KVCacheFormat.SEPARATE_KV, _ShapeDesc()
    )
    assert params["block_size"] == 128
    assert params["page_buffer_size"] == num_blocks * 128
    assert params["num_planes"] == 1
    assert params["per_plane_hidden_dim_bytes"] == [512 * 2]


def test_layout_hint_block_stride_elems_on_shape_desc():
    num_blocks, block_size = 8, 16
    layer = (
        torch.empty(num_blocks, block_size, 1, 64, dtype=torch.float16),
        torch.empty(num_blocks, block_size, 1, 64, dtype=torch.float16),
    )
    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
        layout_hints={"block_stride_elems": 128},
    )
    assert mgr.kv_layer_groups[0].shape_desc.block_stride_elems == 128


# --------------------------------------------------------------------------- #
# Mixed-block_size: PR #3171 fix                                              #
# --------------------------------------------------------------------------- #


def test_mixed_block_size_yields_two_groups():
    """Compressor + dense layers share the outer pool but differ in ``bs``."""
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
    )
    assert len(mgr.kv_layer_groups) == 2
    by_bs = {g.shape_desc.bs: g for g in mgr.kv_layer_groups}
    assert sorted(by_bs.keys()) == [16, 64]
    assert by_bs[16].layer_indices == [0, 1]
    assert by_bs[64].layer_indices == [2]
    md = _make_metadata(mgr)
    assert md.get_num_groups() == 2


# --------------------------------------------------------------------------- #
# GDN + attention hybrid: two distinct grouping keys                          #
# --------------------------------------------------------------------------- #


def test_gdn_fallback_for_non_paged_state_pair():
    """Mismatched 2-tuple that doesn't look like paged KV forms its own group."""
    num_blocks, block_size, num_heads, head_size = 32, 16, 8, 64
    attn_layer = (
        torch.empty(num_blocks, block_size, num_heads, head_size, dtype=torch.float16),
        torch.empty(num_blocks, block_size, num_heads, head_size, dtype=torch.float16),
    )
    # SSM/conv state: 2-D, mismatched, NOT 4-D paged-block layout.
    gdn_layer = (
        torch.empty(num_blocks, 32, dtype=torch.float16),
        torch.empty(num_blocks, 16, dtype=torch.float16),
    )
    mgr = _make_ascend_format_manager(
        [attn_layer, gdn_layer],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
    )
    assert len(mgr.kv_layer_groups) == 2
    md = _make_metadata(mgr)
    assert md.get_num_groups() == 2
    assert len(md.get_dtypes()) == 2


# --------------------------------------------------------------------------- #
# Metadata helpers (upstream LMCacheMetadata)                                 #
# --------------------------------------------------------------------------- #


def test_metadata_helpers_for_dsa_c8_pool():
    num_blocks, block_size = 32, 16
    kv_lora_rank, qk_rope_head_dim, dsa_head_dim = 512, 64, 128
    layer = (
        torch.empty(num_blocks, block_size, 1, kv_lora_rank, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, qk_rope_head_dim, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, dsa_head_dim, dtype=torch.int8),
        torch.empty(num_blocks, block_size, 1, 1, dtype=torch.float16),
    )
    mgr = _make_ascend_format_manager(
        [layer, layer],
        KVCacheFormat.DSA_C8_KV,
        num_blocks,
    )
    g = mgr.kv_layer_groups[0]
    assert g.multi_plane_hidden_bytes is not None
    md = _make_metadata(mgr, use_mla=True)
    assert md.get_num_groups() == 1
    assert md.get_dtypes() == [torch.uint8]
    hidden = _multi_plane_lmc_row_bytes(
        g.multi_plane_hidden_bytes,
        g.physical_chunk_size,
    )
    assert md.get_shapes() == [torch.Size([1, 2, 256, hidden])]


def test_block_stride_elems_inferred_from_dim0_stride():
    """``block_stride_elems`` defaults from tensor stride when layout_hints omit it."""
    num_blocks, block_size, num_heads, head_size = 8, 16, 8, 64
    big_k = torch.empty(
        num_blocks * 2, block_size, num_heads, head_size, dtype=torch.float16
    )
    big_v = torch.empty(
        num_blocks * 2, block_size, num_heads, head_size, dtype=torch.float16
    )
    k = big_k[::2]
    v = big_v[::2]
    assert k.shape[0] == num_blocks
    tight = k.numel() // k.shape[0]
    assert k.stride(0) > tight
    layer = (k, v)
    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
        layout_hints={},
    )
    g = mgr.kv_layer_groups[0]
    assert g.shape_desc.block_stride_elems == k.stride(0)


# --------------------------------------------------------------------------- #
# Upstream MP path (GPUKVFormat) still works                                  #
# --------------------------------------------------------------------------- #


def test_upstream_init_with_gpu_kv_format():
    """Upstream manager accepts ``GPUKVFormat`` directly (MP / V3 path)."""
    # First Party
    import lmcache.c_ops as lmc_ops

    if not hasattr(lmc_ops, "GPUKVFormat"):
        pytest.skip("upstream GPUKVFormat not available in this build")

    # NL_X_TWO_NB_BS_NH_HS expects per-layer ``[2, NB, BS, NH, HS]``.
    tensors = [torch.empty(2, 32, 16, 8, 64, dtype=torch.float16)]
    mgr = KVLayerGroupsManager(
        tensors,
        gpu_kv_format=lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        num_blocks=32,
    )
    assert len(mgr.kv_layer_groups) == 1
    g = mgr.kv_layer_groups[0]
    assert g.shape_desc.kv_size == 2
    assert g.shape_desc.bs == 16
    assert g.dtype == torch.float16
    assert g.compress_ratio == 1
    assert g.physical_chunk_size == 256


def test_sliding_window_reduces_physical_chunk_size_and_multi_plane_row_width():
    num_blocks, block_size = 8, 128
    layer = (
        torch.empty(num_blocks, block_size, 1, 512, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, 64, dtype=torch.bfloat16),
        torch.empty(num_blocks, 1024, 1, 32, dtype=torch.int8),
        torch.empty(num_blocks, 32, 1, 64, dtype=torch.float32),
    )
    layout_hints = {
        "compress_ratios_by_group": (8,),
        "sliding_window_size_by_group": (128,),
        "scheduler_group_by_flat_layer": (0,),
    }

    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.MULTI_PLANE_KV,
        num_blocks,
        layout_hints=layout_hints,
    )

    g = mgr.kv_layer_groups[0]
    assert g.compress_ratio == 8
    assert g.physical_chunk_size == 16
    assert g.multi_plane_hidden_bytes is not None
    assert g.shape_desc.hs == _multi_plane_lmc_row_bytes(
        g.multi_plane_hidden_bytes,
        g.physical_chunk_size,
    )


def test_get_shapes_single_plane_sw_uses_physical_token_dim():
    """Single-plane CR128 SW group allocates ``physical_chunk_size`` token rows."""
    num_blocks, block_size, hidden = 8, 128, 1024
    layer = torch.empty(num_blocks, block_size, hidden, dtype=torch.uint8)
    layout_hints = {
        "compress_ratios_by_group": (128,),
        "sliding_window_size_by_group": (128,),
        "scheduler_group_by_flat_layer": (0,),
    }
    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
        layout_hints=layout_hints,
        lmcache_logical_chunk_size=256,
    )
    g = mgr.kv_layer_groups[0]
    assert g.physical_chunk_size == 1
    md = _make_metadata(mgr, chunk_size=256)
    shape = md.get_shapes(256)[0]
    assert shape[1] == 1
    assert shape[2] == 1
    assert shape[3] == hidden


def test_get_shapes_multi_plane_keeps_logical_token_dim():
    """Bundled multi-plane groups keep logical chunk size in the token dimension."""
    num_blocks, block_size = 8, 128
    layer = (
        torch.empty(num_blocks, block_size, 1, 512, dtype=torch.bfloat16),
        torch.empty(num_blocks, block_size, 1, 64, dtype=torch.bfloat16),
        torch.empty(num_blocks, 1024, 1, 32, dtype=torch.int8),
        torch.empty(num_blocks, 32, 1, 64, dtype=torch.float32),
    )
    layout_hints = {
        "compress_ratios_by_group": (4, 1, 4, 4),
        "scheduler_group_by_flat_layer": (0,),
    }
    mgr = _make_ascend_format_manager(
        [layer],
        KVCacheFormat.MULTI_PLANE_KV,
        num_blocks,
        layout_hints=layout_hints,
        lmcache_logical_chunk_size=256,
    )
    md = _make_metadata(mgr, chunk_size=256)
    assert md.get_shapes(256)[0][2] == 256


def test_split_sw_physical_chunk_size():
    """Post-split groups inherit SW-aware ``physical_chunk_size`` (sw // ratio)."""
    num_blocks, block_size, hidden = 8, 128, 1024
    layer_a = torch.empty(num_blocks, block_size, hidden, dtype=torch.uint8)
    layer_b = torch.empty(num_blocks, block_size, hidden, dtype=torch.uint8)
    layout_hints = {
        "compress_ratios_by_group": (1, 128),
        "sliding_window_size_by_group": (None, 128),
        "scheduler_group_by_flat_layer": (0, 1),
    }
    mgr = _make_ascend_format_manager(
        [layer_a, layer_b],
        KVCacheFormat.SEPARATE_KV,
        num_blocks,
        layout_hints=layout_hints,
        lmcache_logical_chunk_size=256,
    )
    assert len(mgr.kv_layer_groups) == 1
    _split_kv_layer_groups_by_scheduler_slot(
        mgr,
        (0, 1),
        layout_hints=layout_hints,
        lmcache_logical_chunk_size=256,
    )
    by_ratio = {g.compress_ratio: g for g in mgr.kv_layer_groups}
    assert by_ratio[1].physical_chunk_size == 256
    assert by_ratio[128].physical_chunk_size == 1
