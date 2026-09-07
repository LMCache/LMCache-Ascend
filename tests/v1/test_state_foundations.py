# SPDX-License-Identifier: Apache-2.0
"""GDN foundation regressions using the normal repository test environment."""

# Standard
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

# Third Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import TensorMemoryAllocator, TensorMemoryObj
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
import pytest
import torch

# First Party
from lmcache_ascend.integration.vllm.multi_spec_flatten import (
    ordered_scheduler_groups_for_layer,
)
from lmcache_ascend.v1.state_memory import allocate_state_checkpoint


def gdn_spec(**overrides):
    values = dict(
        block_size=16,
        shapes=((3, 4), (2, 3, 4)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    values.update(overrides)
    return MambaSpec(**values)


def attention_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=4, dtype=torch.bfloat16
    )


def config_for(state_first=True, spec=None):
    groups = [
        KVCacheGroupSpec(["gdn.0", "gdn.1"], spec or gdn_spec()),
        KVCacheGroupSpec(["attn.0"], attention_spec()),
    ]
    return SimpleNamespace(kv_cache_groups=groups if state_first else groups[::-1])


@pytest.mark.parametrize("state_first, expected", [(True, [0, 0]), (False, [1, 1])])
def test_state_planes_share_their_spec_group(state_first, expected):
    tensors = [torch.empty(5, 3, 4, dtype=torch.bfloat16), torch.empty(5, 2, 3, 4)]
    assert (
        ordered_scheduler_groups_for_layer("gdn.0", tensors, config_for(state_first))
        == expected
    )
    # First Party
    from lmcache_ascend.integration.vllm.state_groups import select_state_primary

    assert select_state_primary(config_for(state_first)) == 1 - expected[0]


def test_attention_like_state_shapes_do_not_change_semantics():
    spec = gdn_spec(shapes=((16, 2, 4), (16, 2, 4)))
    tensors = [torch.empty(5, 16, 2, 4, dtype=dtype) for dtype in spec.dtypes]
    assert ordered_scheduler_groups_for_layer(
        "gdn.0", tensors, config_for(spec=spec)
    ) == [0, 0]


def test_state_tensor_must_match_spec():
    tensors = [torch.empty(5, 7, 4, dtype=torch.bfloat16), torch.empty(5, 2, 3, 4)]
    with pytest.raises(ValueError, match="gdn.0"):
        ordered_scheduler_groups_for_layer("gdn.0", tensors, config_for())


def test_uniform_spec_resolves_each_layer():
    spec = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"gdn.0": gdn_spec(), "gdn.1": gdn_spec()}
    )
    tensors = [torch.empty(5, 3, 4, dtype=torch.bfloat16), torch.empty(5, 2, 3, 4)]
    assert ordered_scheduler_groups_for_layer(
        "gdn.1", tensors, config_for(spec=spec)
    ) == [0, 0]


@pytest.mark.parametrize("widths", [(4, 4), (4, 8), (4, 8, 2)])
def test_existing_attention_plane_mapping(widths):
    tensors = tuple(torch.empty(5, 16, 2, width) for width in widths)
    config = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(["attn.0"], attention_spec())]
    )
    assert ordered_scheduler_groups_for_layer("attn.0", tensors, config) == [0] * len(
        widths
    )


def test_non_hybrid_keeps_existing_primary_policy():
    # First Party
    from lmcache_ascend.integration.vllm.state_groups import select_state_primary

    config = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(["attn.0"], attention_spec())]
    )
    assert select_state_primary(config) is None


def test_missing_full_attention_primary_is_rejected():
    # First Party
    from lmcache_ascend.integration.vllm.state_groups import select_state_primary

    config = config_for()
    config.kv_cache_groups.pop()
    with pytest.raises(ValueError, match="full-attention"):
        select_state_primary(config)


@pytest.mark.parametrize(
    "override", [{"mamba_cache_mode": "all"}, {"num_speculative_blocks": 1}]
)
def test_unsupported_state_mode_is_rejected(override):
    tensors = [torch.empty(5, 3, 4, dtype=torch.bfloat16), torch.empty(5, 2, 3, 4)]
    with pytest.raises(ValueError, match="align|speculative"):
        ordered_scheduler_groups_for_layer(
            "gdn.0", tensors, config_for(spec=gdn_spec(**override))
        )


def test_explicit_primary_ignores_current_block_counts():
    # First Party
    from lmcache_ascend.integration.vllm.state_groups import request_primary

    assert request_primary(([1] * 32, [8]), (16, 16), 1) == 1
    assert request_primary(([1], [8] * 32), (16, 16), 1) == 1
    assert request_primary(([1] * 32, [8]), (16, 16), None) == 0


def test_explicit_primary_cannot_fall_back_to_another_group():
    # First Party
    from lmcache_ascend.integration.vllm.state_groups import request_primary

    with pytest.raises(ValueError):
        request_primary(([1],), (16,), 1)


def layout_inputs(layers=2, padded=False):
    spec = gdn_spec(shapes=((1, 3), (2, 2)), page_size_padded=256)
    config = config_for(spec=spec)
    config.kv_cache_groups[0].layer_names = [f"gdn.{i}" for i in range(layers)]
    tensors = {}
    for name in config.kv_cache_groups[0].layer_names:
        conv = torch.empty(5, 1, 3, dtype=torch.bfloat16)
        if padded:
            conv = torch.empty(25, dtype=torch.bfloat16).as_strided(
                (5, 1, 3), (5, 3, 1)
            )
        tensors[name] = [conv, torch.empty(5, 2, 2, dtype=torch.float32)]
    return config, tensors


def make_layout(layers=2, padded=False):
    # First Party
    from lmcache_ascend.integration.vllm.state_groups import build_state_layouts

    config, tensors = layout_inputs(layers, padded)
    return build_state_layouts(config, tensors)[0]


def test_plane_major_payload_excludes_runtime_page_padding():
    layout = make_layout()
    assert layout.layer_names == ("gdn.0", "gdn.1")
    assert [p.name for p in layout.planes] == ["conv", "ssm"]
    assert [p.shape for p in layout.planes] == [(2, 1, 3), (2, 2, 2)]
    assert [p.dtype for p in layout.planes] == [torch.bfloat16, torch.float32]
    assert [p.offset for p in layout.planes] == [0, 12]
    assert [p.nbytes for p in layout.planes] == [12, 32]
    assert layout.nbytes == 44
    with pytest.raises(FrozenInstanceError):
        layout.nbytes = 10


def test_mixed_dtype_offsets_include_only_required_padding():
    layout = make_layout(layers=1)
    assert [p.offset for p in layout.planes] == [0, 8]
    assert layout.nbytes == 24
    assert sum(p.nbytes for p in layout.planes) == 22


def test_source_stride_is_not_payload_compatibility():
    packed = make_layout()
    padded = make_layout(padded=True)
    assert packed.planes[0].block_stride_bytes == (6, 6)
    assert padded.planes[0].block_stride_bytes == (10, 10)
    assert packed.signature == padded.signature


def test_plane_dtype_changes_compatibility():
    # First Party
    from lmcache_ascend.integration.vllm.state_groups import build_state_layouts

    config, tensors = layout_inputs()
    config.kv_cache_groups[0].kv_cache_spec = gdn_spec(
        shapes=((1, 3), (2, 2)), dtypes=(torch.bfloat16, torch.bfloat16)
    )
    for name in tensors:
        tensors[name][1] = tensors[name][1].to(torch.bfloat16)
    assert build_state_layouts(config, tensors)[0].signature != make_layout().signature


def test_noncontiguous_state_elements_are_explicitly_rejected():
    # First Party
    from lmcache_ascend.integration.vllm.state_groups import build_state_layouts

    config, tensors = layout_inputs()
    tensors["gdn.0"][1] = tensors["gdn.0"][1].transpose(1, 2)
    with pytest.raises(ValueError, match="stride|contiguous"):
        build_state_layouts(config, tensors)


@pytest.fixture
def pool():
    # The real pool reserves a 4 KiB-aligned block per allocation.
    allocator = TensorMemoryAllocator(torch.zeros(16384, dtype=torch.uint8))
    yield allocator
    assert allocator.num_active_allocations == 0


def test_allocate_composite_planes_from_real_pool(pool):
    # First Party
    from lmcache_ascend.v1.state_memory import allocate_state_checkpoint

    layout = make_layout(layers=1)
    with allocate_state_checkpoint(layout, pool) as buffer:
        assert pool.num_active_allocations == 1
        base = buffer.memory_obj.raw_tensor.data_ptr()
        conv, ssm = buffer.planes
        assert conv.shape == (1, 1, 3)
        assert ssm.shape == (1, 2, 2)
        assert conv.dtype == torch.bfloat16
        assert ssm.dtype == torch.float32
        assert [conv.data_ptr() - base, ssm.data_ptr() - base] == [0, 8]
        assert ssm.data_ptr() % 4 == 0
        assert buffer.memory_obj.raw_tensor.numel() == 24


def test_live_checkpoints_are_independent_and_release_to_pool(pool):
    # First Party
    from lmcache_ascend.v1.state_memory import allocate_state_checkpoint

    with allocate_state_checkpoint(make_layout(), pool) as first:
        with allocate_state_checkpoint(make_layout(), pool) as second:
            for plane in first.planes:
                plane.fill_(7)
            for plane in second.planes:
                plane.fill_(9)
            assert all(torch.all(plane == 7) for plane in first.planes)
            assert pool.num_active_allocations == 2
        assert pool.num_active_allocations == 1
        assert all(torch.all(plane == 7) for plane in first.planes)
    first.close()
    with pytest.raises(RuntimeError, match="released"):
        _ = first.planes


def test_allocation_failure_does_not_return_partial_buffer():
    # First Party
    from lmcache_ascend.v1.state_memory import allocate_state_checkpoint

    allocator = TensorMemoryAllocator(torch.zeros(8, dtype=torch.uint8))
    with pytest.raises(MemoryError):
        allocate_state_checkpoint(make_layout(), allocator)
    assert allocator.num_active_allocations == 0


def test_view_failure_returns_allocation_to_pool(pool, monkeypatch):
    # First Party
    from lmcache_ascend.v1.state_memory import allocate_state_checkpoint

    original = TensorMemoryObj.get_tensor

    def fail_ssm(self, index):
        if index == 1:
            raise RuntimeError("injected view failure")
        return original(self, index)

    monkeypatch.setattr(TensorMemoryObj, "get_tensor", fail_ssm)
    with pytest.raises(RuntimeError, match="injected view failure"):
        allocate_state_checkpoint(make_layout(), pool)
    assert pool.num_active_allocations == 0


def test_attention_allocation_does_not_allocate_state(pool):
    # Third Party
    from lmcache.v1.memory_management import MemoryFormat

    obj = pool.allocate(torch.Size([2, 1, 16, 4]), torch.bfloat16, MemoryFormat.KV_2LTD)
    assert obj is not None
    assert pool.num_active_allocations == 1
    assert obj.get_tensor(0).shape == (2, 1, 16, 4)
    obj.ref_count_down()


def make_ref(layout, boundary=16, key=None):
    # First Party
    from lmcache_ascend.v1.state_checkpoint import CheckpointRef

    key = key or CacheEngineKey("qwen-test", 2, 0, 123, torch.bfloat16)
    return CheckpointRef.from_chunk(
        key, chunk_end=boundary, boundary=boundary, chunk_size=16, layout=layout
    )


def test_ref_reuses_key_identity_without_allocating_or_claiming_availability():
    layout = make_layout()
    ref = make_ref(layout)
    assert ref.prefix_hash == 123
    assert ref.boundary == 16
    assert (ref.model_name, ref.world_size, ref.worker_id) == ("qwen-test", 2, 0)
    assert ref != CacheEngineKey("qwen-test", 2, 0, 123, torch.bfloat16)
    assert ref != make_ref(replace(layout, group_index=1))
    assert ref != make_ref(replace(layout, version=2))


@pytest.mark.parametrize("chunk_end, boundary", [(16, 15), (16, 32), (15, 15), (0, 0)])
def test_ref_requires_the_exact_complete_chunk_boundary(chunk_end, boundary):
    # First Party
    from lmcache_ascend.v1.state_checkpoint import CheckpointRef

    with pytest.raises(ValueError, match="boundary|chunk"):
        CheckpointRef.from_chunk(
            CacheEngineKey("qwen-test", 2, 0, 123, torch.bfloat16),
            chunk_end=chunk_end,
            boundary=boundary,
            chunk_size=16,
            layout=make_layout(),
        )


def test_existing_chained_prefix_keys_are_preserved():
    # Third Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.token_database import ChunkedTokenDatabase

    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    metadata = LMCacheMetadata(
        "qwen-test", 2, 2, 0, 0, torch.bfloat16, (1, 2, 16, 2, 4)
    )
    database = ChunkedTokenDatabase(config, metadata)
    keys_a = list(database.process_tokens(tokens=[1] * 16 + [3] * 16))
    keys_b = list(database.process_tokens(tokens=[2] * 16 + [3] * 16))
    ref_a = make_ref(make_layout(), keys_a[-1][1], keys_a[-1][2])
    ref_b = make_ref(make_layout(), keys_b[-1][1], keys_b[-1][2])
    assert ref_a.prefix_hash == keys_a[-1][2].chunk_hash
    assert ref_b.prefix_hash == keys_b[-1][2].chunk_hash
    assert ref_a != ref_b


@pytest.mark.parametrize("load", [False, True])
def test_operation_preserves_r_without_copying_or_inventing_e(load):
    # First Party
    from lmcache_ascend.v1.state_checkpoint import StateBlockBinding, StateOperation

    layout = make_layout()
    _, tensors = layout_inputs()
    binding = StateBlockBinding(
        tuple(tuple(tensors[name]) for name in layout.layer_names), 1
    )
    pool = TensorMemoryAllocator(torch.zeros(8192, dtype=torch.uint8))
    with allocate_state_checkpoint(layout, pool) as buffer:
        for plane in buffer.planes:
            plane.fill_(9)
        operation = StateOperation(
            make_ref(layout),
            layout,
            buffer if load else binding,
            binding if load else buffer,
            attention_end=32,
        )
        assert operation.checkpoint.boundary == 16
        assert operation.executed_end is None
        assert operation.attention_end == 32
        assert all(torch.all(plane == 9) for plane in buffer.planes)
    assert pool.num_active_allocations == 0


def test_operation_rejects_wrong_layout_and_out_of_range_block():
    # First Party
    from lmcache_ascend.v1.state_checkpoint import StateBlockBinding, StateOperation

    layout = make_layout()
    _, tensors = layout_inputs()
    entries = tuple(tuple(tensors[name]) for name in layout.layer_names)
    pool = TensorMemoryAllocator(torch.zeros(8192, dtype=torch.uint8))
    with allocate_state_checkpoint(layout, pool) as buffer:
        with pytest.raises(ValueError, match="layout"):
            StateOperation(
                make_ref(replace(layout, version=2)),
                layout,
                StateBlockBinding(entries, 0),
                buffer,
            )
        with pytest.raises(ValueError, match="block"):
            StateOperation(
                make_ref(layout), layout, StateBlockBinding(entries, 5), buffer
            )
