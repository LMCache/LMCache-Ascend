# SPDX-License-Identifier: Apache-2.0
"""GDN foundation regressions using the normal repository test environment."""

# Standard
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
