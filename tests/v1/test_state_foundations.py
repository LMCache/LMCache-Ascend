# SPDX-License-Identifier: Apache-2.0
"""GDN foundation regressions using the normal repository test environment."""

# Standard
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

# Third Party
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
