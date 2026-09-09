# SPDX-License-Identifier: Apache-2.0
"""Adapter integration checks for the Ascend host acceptance suite."""

# Standard
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec
import pytest
import torch

# First Party
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
    LMCacheConnectorV1ImplMultiGroup,
)
from lmcache_ascend.integration.vllm.state_groups import (
    select_state_primary,
    validate_state_config,
)
from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl
from lmcache_ascend.v1.npu_connector.npu_connectors import VLLMPagedMemNPUConnectorV2


def _registration(monkeypatch, state_first=True, merged=False):
    connector = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    spec = MambaSpec(
        block_size=16,
        shapes=((1, 3), (2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(["gdn.0"], spec),
        KVCacheGroupSpec(
            ["attn.0", "attn.1"],
            FullAttentionSpec(
                block_size=16, num_kv_heads=2, head_size=4, dtype=torch.bfloat16
            ),
        ),
        KVCacheGroupSpec(["gdn.1"], spec),
    ]
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=groups if state_first else groups[::-1]
    )
    # Move Attention across the group order, not just the state groups.
    if not state_first:
        connector._kv_cache_config.kv_cache_groups.insert(
            0, connector._kv_cache_config.kv_cache_groups.pop(1)
        )
    tensors = {
        name: [torch.empty(5, 1, 3, dtype=torch.bfloat16), torch.empty(5, 2, 2)]
        for name in ("gdn.0", "gdn.1")
    }
    for name in ("attn.0", "attn.1"):
        tensors[name] = (
            torch.empty(2, 5, 16, 2, 4, dtype=torch.bfloat16)
            if merged
            else tuple(torch.empty(5, 16, 2, 4, dtype=torch.bfloat16) for _ in range(2))
        )
    connector.config = LMCacheEngineConfig.from_defaults()
    connector._vllm_config = _vllm_config()
    connector._block_size = 16
    connector._block_sizes_by_group = (16, 16, 16)
    connector._compress_ratios_by_group = (1, 1, 1)
    connector.kv_caches = {}
    metadata = SimpleNamespace(kv_layer_groups_manager=None, chunk_size=256)
    gpu = VLLMPagedMemNPUConnectorV2.__new__(VLLMPagedMemNPUConnectorV2)
    gpu.metadata, gpu.layout_hints, gpu.use_mla, gpu.num_layers = metadata, {}, False, 4
    # First Party
    from lmcache_ascend.v1.npu_connector import npu_connectors

    monkeypatch.setattr(npu_connectors, "is_310p", lambda: False)
    allocator = SimpleNamespace(use_hot=True)
    allocator.get_allocator_backend = lambda: allocator
    engine = SimpleNamespace(
        token_database=ChunkedTokenDatabase.__new__(ChunkedTokenDatabase),
        save_only_first_rank=False,
        remove_after_retrieve=False,
        gpu_connector=gpu,
        metadata=metadata,
        storage_manager=None,
    )

    def post_init():
        # Match LMCache 0.4.5: storage is created only after KV groups are ready.
        assert connector.kv_caches
        assert engine.metadata.kv_layer_groups_manager is not None
        assert engine.state_layouts is connector.state_layouts
        engine.storage_manager = SimpleNamespace(
            storage_backends={"LocalCPUBackend": allocator}, allocator_backend=allocator
        )

    connector._manager = SimpleNamespace(
        lmcache_engine=engine, post_init=Mock(side_effect=post_init)
    )
    return connector, tensors


def _vllm_config():
    return SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, tensor_parallel_size=2
        ),
    )


@pytest.mark.parametrize("state_first", [True, False])
@pytest.mark.parametrize("merged", [True, False])
def test_registration_preserves_native_attention_and_all_state_groups(
    monkeypatch, state_first, merged
):
    connector, tensors = _registration(monkeypatch, state_first, merged)
    connector.register_kv_caches(tensors)
    primary = select_state_primary(connector._kv_cache_config)
    assert primary == (1 if state_first else 0)
    assert set(connector.kv_caches) == {"attn.0", "attn.1"}
    assert all(
        connector.kv_caches[name] is tensors[name] for name in connector.kv_caches
    )
    assert set(connector.state_kv_caches) == {"gdn.0", "gdn.1"}
    assert all(
        connector.state_kv_caches[name][0] is tensors[name][0]
        for name in connector.state_kv_caches
    )
    engine = connector.lmcache_engine
    assert engine.state_layouts is connector.state_layouts
    assert {layout.group_index for layout in engine.state_layouts} == set(range(3)) - {
        primary
    }
    assert all(layout.nbytes == 24 for layout in engine.state_layouts)
    hints = engine.gpu_connector.layout_hints
    assert set(hints["model_kv_caches"]) == {"attn.0", "attn.1"}
    assert hints["scheduler_group_by_flat_layer"] == (primary, primary)
    assert hints["layer_to_scheduler_groups"]["attn.0"] == [primary] * (
        1 if merged else 2
    )
    assert engine.gpu_connector.num_layers == connector.num_layers == 2
    (group,) = engine.metadata.kv_layer_groups_manager.kv_layer_groups
    assert group.shape_desc.nb == 5
    assert group.shape_desc.nl == 2
    assert group.shape_desc.kv_size == 2
    assert group.shape_desc.element_size == 2
    connector._manager.post_init.assert_called_once()


@pytest.mark.parametrize(
    "field,value",
    [
        ("store_async", True),
        ("enable_async_loading", True),
        ("use_layerwise", True),
        ("enable_blending", True),
        ("enable_chunk_statistics", True),
        ("enable_scheduler_bypass_lookup", True),
        ("enable_pd", True),
        ("enable_p2p", True),
        ("enable_controller", True),
        ("external_lookup_client", "custom"),
        ("remote_url", "custom"),
        ("storage_plugins", ["custom"]),
        ("remote_storage_plugins", ["custom"]),
        ("gds_path", "disk"),
        ("maru_path", "disk"),
        ("hit_miss_ratio", 0.0),
        ("lookup_server_worker_ids", [0]),
        ("lmcache_worker_ids", [0]),
        ("store_location", "RemoteBackend"),
        ("retrieve_locations", ["RemoteBackend"]),
        ("max_local_cpu_size", 0),
        ("pin_timeout_sec", 0),
    ],
)
def test_hybrid_rejects_unsupported_config(field, value):
    config = LMCacheEngineConfig.from_defaults()
    setattr(config, field, value)
    with pytest.raises(ValueError, match="GDN"):
        validate_state_config(config, _vllm_config())


@pytest.mark.parametrize(
    "field",
    [
        "save_only_first_rank",
        "remove_after_retrieve",
        "enable_nixl_storage",
        "audit_backend_enabled",
    ],
)
def test_hybrid_rejects_extra_config_modes(field):
    config = LMCacheEngineConfig.from_defaults(extra_config={field: True})
    with pytest.raises(ValueError, match=field):
        validate_state_config(config, _vllm_config())


@pytest.mark.parametrize("speculative", [True, False])
def test_hybrid_rejects_speculative_and_pipeline_parallel(speculative):
    config = _vllm_config()
    if speculative:
        config.speculative_config = object()
    else:
        config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="speculative|pipeline"):
        validate_state_config(LMCacheEngineConfig.from_defaults(), config)


@pytest.mark.parametrize("disk_only", [True, False])
def test_supported_local_config_allows_default_skip_policy(monkeypatch, disk_only):
    monkeypatch.setenv("LMCACHE_ASCEND_SKIP_STATE_GROUPS", "true")
    monkeypatch.delenv("LMCACHE_ASCEND_SKIP_STATE_LAYER_SUFFIX", raising=False)
    monkeypatch.delenv("LMCACHE_ASCEND_SKIP_STATE_SPEC_ALLOWLIST", raising=False)
    connector, _ = _registration(monkeypatch)
    config = LMCacheEngineConfig.from_defaults(
        local_cpu=not disk_only, local_disk="/cache", max_local_disk_size=10
    )
    validate_state_config(config, _vllm_config())
    assert select_state_primary(connector._kv_cache_config) == 1


@pytest.mark.parametrize("failure", ["nonchunked", "wrapper", "missing_rank"])
def test_scheduler_initialization_checks_real_lookup_client_before_query(
    monkeypatch, failure
):
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    connector, _ = _registration(monkeypatch)
    connector.worker_count = 2
    client = SimpleNamespace(
        config=connector.config,
        token_database=ChunkedTokenDatabase.__new__(ChunkedTokenDatabase),
        transport=SimpleNamespace(world_size=2),
    )
    if failure == "nonchunked":
        client.token_database = object()
    elif failure == "wrapper":
        del client.transport
    else:
        client.transport.world_size = 1
    connector._manager.lookup_client = client
    with pytest.raises(ValueError, match="ChunkedTokenDatabase|synchronous"):
        LMCacheConnectorV1ImplMultiGroup._init_connector_state(
            connector,
            KVConnectorRole.SCHEDULER,
            connector._vllm_config,
            connector.config,
        )


@pytest.mark.parametrize(
    "suffix,allowlist",
    [(".0", ""), (".unused", "MambaSpec"), (".unused", "FullAttentionSpec")],
)
def test_required_hybrid_layers_cannot_be_skipped(monkeypatch, suffix, allowlist):
    connector, tensors = _registration(monkeypatch)
    monkeypatch.setenv("LMCACHE_ASCEND_SKIP_STATE_GROUPS", "true")
    monkeypatch.setenv("LMCACHE_ASCEND_SKIP_STATE_LAYER_SUFFIX", suffix)
    monkeypatch.setenv("LMCACHE_ASCEND_SKIP_STATE_SPEC_ALLOWLIST", allowlist)
    with pytest.raises(ValueError, match="Cannot skip required"):
        connector.register_kv_caches(tensors)
    connector._manager.post_init.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_attention",
        "missing_state",
        "nonchunked",
        "remove",
        "first_rank",
        "nonuniform",
    ],
)
def test_registration_rejects_incomplete_or_unsupported_runtime(monkeypatch, failure):
    connector, tensors = _registration(monkeypatch)
    engine = connector.lmcache_engine
    if failure == "missing_attention":
        del tensors["attn.1"]
    elif failure == "missing_state":
        del tensors["gdn.1"]
    elif failure == "nonchunked":
        engine.token_database = object()
    elif failure == "remove":
        engine.remove_after_retrieve = True
    elif failure == "first_rank":
        engine.save_only_first_rank = True
    else:
        tensors["attn.1"] = tuple(t[:4] for t in tensors["attn.1"])
    with pytest.raises(ValueError):
        connector.register_kv_caches(tensors)
    assert not hasattr(engine, "state_layouts")
    connector._manager.post_init.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["extra_backend", "no_writable_tier", "missing_tier"]
)
def test_registration_validates_backends_after_post_init(monkeypatch, failure):
    connector, tensors = _registration(monkeypatch)
    initialize = connector._manager.post_init.side_effect

    def post_init():
        initialize()
        backends = connector.lmcache_engine.storage_manager.storage_backends
        if failure == "extra_backend":
            backends["RemoteBackend"] = object()
        elif failure == "no_writable_tier":
            backends["LocalCPUBackend"].use_hot = False

    connector._manager.post_init.side_effect = post_init
    if failure == "missing_tier":
        connector.config.store_location = "LocalDiskBackend"
    with pytest.raises(ValueError, match="local CPU/disk|writable CPU/disk|available"):
        connector.register_kv_caches(tensors)
    connector._manager.post_init.assert_called_once()


@pytest.mark.parametrize("failure", ["swa", "eagle", "state_mode", "extra_attention"])
def test_hybrid_rejects_unsupported_group_semantics(monkeypatch, failure):
    connector, _ = _registration(monkeypatch)
    groups = connector._kv_cache_config.kv_cache_groups
    if failure == "swa":
        groups[1].kv_cache_spec = replace(groups[1].kv_cache_spec, sliding_window=16)
    elif failure == "eagle":
        groups[0].is_eagle_group = True
    elif failure == "state_mode":
        groups[0].kv_cache_spec = replace(
            groups[0].kv_cache_spec, mamba_cache_mode="all"
        )
    else:
        groups.append(KVCacheGroupSpec(["attn.extra"], groups[1].kv_cache_spec))
    with pytest.raises(ValueError):
        select_state_primary(connector._kv_cache_config)


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
