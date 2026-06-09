# SPDX-License-Identifier: Apache-2.0
"""DS4 (DeepSeek V4) fixtures and helpers for multi-group connector tests."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

import lmcache_ascend  # noqa: F401

from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec
from lmcache_ascend.integration.vllm.multi_spec_flatten import build_flat_kv_caches
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
    ReqMeta,
    _build_slot_mapping_for_group,
)
from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemNPUConnectorV2,
    build_mp_launch_meta,
)
from lmcache_ascend.v1.slot_mapping_utils import build_filtered_slot_mappings

from .conftest_kvcache import (
    LARGE_TOKEN_COPY_SIZE,
    device,
    ds4_smoke_chunk_sizes,
    multi_plane_round_trip_via_connector,
    npu_available,
    power_of_two_boundary_triplet,
    separate_kv_round_trip_via_connector,
    slot_concat_and_offsets,
)
from .test_layer_to_scheduler_group_mapping import (
    L0,
    L1,
    L2,
    L3,
    L4,
    _make_ds4_kv_cache_config,
)

DS4_BLOCK_SIZES_BY_GROUP = (
    128,
    128,
    128,
    1024,
    32,
    32,
    128,
    128,
    128,
    128,
    64,
    64,
)
DS4_NUM_SCHEDULER_GROUPS = len(DS4_BLOCK_SIZES_BY_GROUP)
DS4_VLLM_BLOCK_SIZE = 128
DS4_IE_LOGICAL_BLOCK_SIZE = 1024
DS4_CHUNK_SIZE = 512
DS4_COMPRESS_RATIOS = (8, 8, 8, 1, 32, 32, 8, 8, 8, 8, 16, 16)
DS4_SLOT_LENS_512 = (64, 64, 64, 512, 16, 16, 64, 64, 64, 64, 32, 32)
DS4_NUM_MODEL_LAYERS = 5
DS4_PRODUCTION_CHUNK_TOKENS = 256

# Backward-compatible aliases used by msprof and older references
DSV4_BLOCK_SIZES_BY_GROUP = DS4_BLOCK_SIZES_BY_GROUP
DSV4_NUM_SCHEDULER_GROUPS = DS4_NUM_SCHEDULER_GROUPS
DSV4_VLLM_BLOCK_SIZE = DS4_VLLM_BLOCK_SIZE
DSV4_IE_LOGICAL_BLOCK_SIZE = DS4_IE_LOGICAL_BLOCK_SIZE
DSV4_CHUNK_SIZE = DS4_CHUNK_SIZE
DSV4_COMPRESS_RATIOS = DS4_COMPRESS_RATIOS
DSV4_SLOT_LENS_512 = DS4_SLOT_LENS_512
DSV4_NUM_MODEL_LAYERS = DS4_NUM_MODEL_LAYERS
DSV4_PRODUCTION_CHUNK_TOKENS = DS4_PRODUCTION_CHUNK_TOKENS


def compress_ratios_from_block_sizes() -> tuple[int, ...]:
    return tuple(
        DS4_IE_LOGICAL_BLOCK_SIZE // int(bs) for bs in DS4_BLOCK_SIZES_BY_GROUP
    )


def set_bundle_multi_spec_env(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool
) -> None:
    monkeypatch.setenv(
        "LMCACHE_ASCEND_BUNDLE_MULTI_SPEC", "1" if enabled else "0"
    )
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.multi_spec_flatten.bundle_multi_spec_enabled",
        lambda: enabled,
    )


def set_skip_state_groups_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
    allowlist: str | None = ...,
) -> None:
    """Configure skip-state env vars for tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        enabled: When False, disables skip-state filtering entirely.
        allowlist: Explicit allowlist string, ``None`` to unset the env var
            (use built-in defaults), or omit to leave the env var unchanged.
    """
    monkeypatch.setenv("LMCACHE_ASCEND_SKIP_STATE_GROUPS", "1" if enabled else "0")
    if allowlist is ...:
        return
    env_key = "LMCACHE_ASCEND_SKIP_STATE_SPEC_ALLOWLIST"
    if allowlist is None:
        monkeypatch.delenv(env_key, raising=False)
        return
    monkeypatch.setenv(env_key, allowlist)


def _make_4d_page(
    num_blocks: int,
    block_size: int,
    head_size: int,
    dev: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.zeros(
        num_blocks, block_size, 1, head_size, dtype=dtype, device=dev
    )


def _make_shared_blob_layer(
    num_blocks: int,
    block_size: int,
    hidden: int,
    dev: torch.device,
) -> List[torch.Tensor]:
    return [_make_4d_page(num_blocks, block_size, hidden, dev, torch.bfloat16)]


def make_ds4_kv_caches_dict(
    dev: torch.device,
    num_blocks: int = 32,
) -> dict[str, list | tuple]:
    """Approximate DS4RandomQuarterLayers KV cache layout (5 model layers)."""
    layer2_subs = [
        _make_4d_page(num_blocks, 128, 512, dev),
        _make_4d_page(num_blocks, 128, 512, dev),
        _make_4d_page(num_blocks, 1024, 128, dev, torch.int8),
        _make_4d_page(num_blocks, 1024, 1, dev, torch.float16),
        _make_4d_page(num_blocks, 32, 1024, dev, torch.float32),
        _make_4d_page(num_blocks, 32, 1024, dev, torch.float32),
        _make_4d_page(num_blocks, 128, 256, dev, torch.float32),
        _make_4d_page(num_blocks, 128, 256, dev, torch.float32),
    ]
    return {
        L0: _make_shared_blob_layer(num_blocks, 128, 3713, dev),
        L1: _make_shared_blob_layer(num_blocks, 128, 512, dev),
        L2: layer2_subs,
        L3: (
            torch.zeros(num_blocks, 128, 1, 512, dtype=torch.bfloat16, device=dev),
            torch.zeros(num_blocks, 128, 1, 64, dtype=torch.bfloat16, device=dev),
            torch.zeros(num_blocks, 64, 1, 128, dtype=torch.float32, device=dev),
            torch.zeros(num_blocks, 64, 1, 1, dtype=torch.float32, device=dev),
        ),
        L4: _make_shared_blob_layer(num_blocks, 128, 512, dev),
    }


make_dsv4_kv_caches_dict = make_ds4_kv_caches_dict


def slot_mappings_for_ds4_groups(
    num_tokens: int,
    dev: torch.device,
    *,
    num_blocks: int = 16,
) -> tuple[torch.Tensor, ...]:
    block_ids = list(range(1, num_blocks + 1))
    mappings: list[torch.Tensor] = []
    for ratio, block_size in zip(
        compress_ratios_from_block_sizes(), DS4_BLOCK_SIZES_BY_GROUP
    ):
        sm = _build_slot_mapping_for_group(
            block_ids,
            int(block_size),
            num_tokens,
            is_store=False,
            compress_ratio=ratio,
        )
        mappings.append(sm.to(dev))
    return tuple(mappings)


def make_slot_mappings(
    num_tokens: int,
    dev: torch.device,
) -> tuple[torch.Tensor, ...]:
    return slot_mappings_for_ds4_groups(
        num_tokens, dev, num_blocks=32
    )


def make_slot_transfer_kwargs(
    slot_mappings_npu: tuple[torch.Tensor, ...],
    *,
    compress_ratios: tuple[int, ...] | None = None,
    connector: VLLMPagedMemNPUConnectorV2 | None = None,
    chunk_ranges: list[tuple[int, int]] | None = None,
) -> dict:
    """Build filtered NPU mappings + CPU prefix kwargs for multi-group transfer."""
    ratios = compress_ratios or DS4_COMPRESS_RATIOS[: len(slot_mappings_npu)]
    cpu_mappings = tuple(sm.cpu() for sm in slot_mappings_npu)
    filtered_cpu, prefixes = build_filtered_slot_mappings(
        cpu_mappings,
        compress_ratios=ratios,
    )
    dev = slot_mappings_npu[0].device
    filtered_npu = tuple(f.to(dev) for f in filtered_cpu)
    kwargs: dict = {
        "filtered_slot_mappings_npu": filtered_npu,
        "slot_valid_prefix_by_group": prefixes,
    }
    if connector is not None and chunk_ranges:
        kwargs["mp_launch_meta"] = build_mp_launch_meta(
            connector,
            chunk_ranges=chunk_ranges,
            slot_mappings_by_group=cpu_mappings,
            prefixes_by_group=prefixes,
            filtered_slot_mappings_npu=filtered_npu,
            compress_ratios=ratios,
        )
    return kwargs


def make_production_slot_mappings(
    num_tokens: int,
    dev: torch.device,
) -> tuple[torch.Tensor, ...]:
    return slot_mappings_for_ds4_groups(
        num_tokens, dev, num_blocks=32
    )


def build_connector_via_from_metadata(
    num_model_layers: int,
    *,
    head_size: int = 512,
    chunk_size: int = DS4_CHUNK_SIZE,
    layout_hints: dict | None = None,
) -> VLLMPagedMemNPUConnectorV2:
    hints = layout_hints or {}
    metadata = LMCacheMetadata(
        model_name="DS4RandomQuarterLayers",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_model_layers, 1, chunk_size, 1, head_size),
        use_mla=True,
        chunk_size=chunk_size,
        kv_layer_groups_manager=None,
    )
    connector = VLLMPagedMemNPUConnectorV2.from_metadata(
        metadata,
        layout_hints=hints,
    )
    connector.metadata = metadata
    return connector


def build_connector_and_metadata(
    kv_caches: list,
    *,
    sched_by_layer: tuple[int, ...],
    chunk_size: int = DS4_CHUNK_SIZE,
    num_model_layers: int = DS4_NUM_MODEL_LAYERS,
) -> tuple[VLLMPagedMemNPUConnectorV2, LMCacheMetadata]:
    bundled = any(isinstance(entry, (tuple, list)) for entry in kv_caches)
    layout_hints = {
        "inference_engine_logical_block_size": DS4_IE_LOGICAL_BLOCK_SIZE,
        "scheduler_group_by_flat_layer": sched_by_layer,
        "primary_kv_group_idx": 1,
        "bundle_multi_spec": bundled,
    }
    connector = build_connector_via_from_metadata(
        num_model_layers,
        chunk_size=chunk_size,
        layout_hints=layout_hints,
    )
    connector.kvcaches = kv_caches
    return connector, connector.metadata


def build_bundled_ds4_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[VLLMPagedMemNPUConnectorV2, LMCacheMetadata, list, torch.device]:
    monkeypatch.setenv("LMCACHE_ASCEND_BUNDLE_MULTI_SPEC", "1")
    dev = device()
    kv_dict = make_ds4_kv_caches_dict(dev, num_blocks=32)
    flat, sched, layer_to_groups, _ = build_flat_kv_caches(
        kv_dict,
        _make_ds4_kv_cache_config(),
        ie_logical_block_size=DS4_IE_LOGICAL_BLOCK_SIZE,
    )
    kv_list = list(flat.values())
    layout_hints = {
        "vllm_block_size": DS4_VLLM_BLOCK_SIZE,
        "inference_engine_logical_block_size": DS4_IE_LOGICAL_BLOCK_SIZE,
        "block_sizes_by_group": DS4_BLOCK_SIZES_BY_GROUP,
        "scheduler_group_by_flat_layer": sched,
        "layer_to_scheduler_groups": layer_to_groups,
        "flat_layer_names": list(flat.keys()),
        "bundle_multi_spec": True,
        "primary_kv_group_idx": 1,
    }
    connector = build_connector_via_from_metadata(
        DS4_NUM_MODEL_LAYERS, layout_hints=layout_hints
    )
    connector.kvcaches = kv_list
    connector.layout_hints = layout_hints
    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        connector._initialize_pointers(kv_list)
    assert connector.metadata is not None
    connector.kvcaches_device = dev
    return connector, connector.metadata, kv_list, dev


build_bundled_dsv4_connector = build_bundled_ds4_connector


def build_exploded_flat_connector(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_blocks: int = 32,
) -> tuple[VLLMPagedMemNPUConnectorV2, list, tuple[int, ...]]:
    set_bundle_multi_spec_env(monkeypatch, enabled=False)
    dev = device()
    kv_dict = make_ds4_kv_caches_dict(dev, num_blocks=num_blocks)
    flat, sched_by_layer, _, _ = build_flat_kv_caches(
        kv_dict,
        _make_ds4_kv_cache_config(),
        ie_logical_block_size=DS4_IE_LOGICAL_BLOCK_SIZE,
    )
    kv_list = list(flat.values())
    layout_hints = {
        "inference_engine_logical_block_size": DS4_IE_LOGICAL_BLOCK_SIZE,
        "scheduler_group_by_flat_layer": sched_by_layer,
        "kv_layout": "NHD",
    }
    connector = build_connector_via_from_metadata(
        DS4_NUM_MODEL_LAYERS,
        layout_hints=layout_hints,
    )
    connector.kvcaches = kv_list
    with patch(
        "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
        return_value=False,
    ):
        connector._initialize_pointers(kv_list)
    return connector, kv_list, sched_by_layer


_MULTI_GROUP_PINNED_ALLOCATOR: PinMemoryAllocator | None = None


def _get_multi_group_pinned_allocator() -> PinMemoryAllocator:
    global _MULTI_GROUP_PINNED_ALLOCATOR
    if _MULTI_GROUP_PINNED_ALLOCATOR is None:
        _MULTI_GROUP_PINNED_ALLOCATOR = PinMemoryAllocator(4 * 1024 * 1024 * 1024)
    return _MULTI_GROUP_PINNED_ALLOCATOR


def allocate_multi_group_memory_obj(
    metadata: LMCacheMetadata,
    num_tokens: int,
) -> TensorMemoryObj:
    shapes = metadata.get_shapes(num_tokens)
    dtypes = metadata.get_dtypes()
    mem_obj = _get_multi_group_pinned_allocator().allocate(
        shapes, dtypes, fmt=MemoryFormat.KV_2LTD
    )
    assert mem_obj is not None
    return mem_obj


def ds4_roundtrip_chunk_sizes_full() -> tuple[int, ...]:
    prod = DS4_PRODUCTION_CHUNK_TOKENS
    meta = DS4_CHUNK_SIZE
    ie = DS4_IE_LOGICAL_BLOCK_SIZE
    sizes = {
        0,
        1,
        prod - 1,
        prod,
        prod + 1,
        meta - 1,
        meta,
        meta + 1,
        ie - 1,
        ie,
        ie + 1,
        ie * 2 - 1,
        ie * 2,
        ie * 2 + 1,
        768,
        *power_of_two_boundary_triplet(15),
        LARGE_TOKEN_COPY_SIZE - 1,
        LARGE_TOKEN_COPY_SIZE,
        LARGE_TOKEN_COPY_SIZE + 1,
    }
    return tuple(sorted(sizes))


dsv4_roundtrip_chunk_sizes_full = ds4_roundtrip_chunk_sizes_full


DS4_CHUNK_SMOKE: tuple[int, ...] = ds4_smoke_chunk_sizes()


def ds4_roundtrip_chunk_sizes() -> tuple[int, ...]:
    if os.environ.get("DS4_CHUNK_TIER", "smoke") == "full":
        return ds4_roundtrip_chunk_sizes_full()
    return DS4_CHUNK_SMOKE


dsv4_roundtrip_chunk_sizes = ds4_roundtrip_chunk_sizes


def prefill_tokens_for_chunk(chunk: int) -> int:
    return max(int(chunk), 768, DS4_IE_LOGICAL_BLOCK_SIZE * 2)


def ds4_multi_plane_round_trip(
    connector: VLLMPagedMemNPUConnectorV2,
    gi: int,
    chunk: int,
    slot_mappings: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> None:
    multi_plane_round_trip_via_connector(
        connector,
        gi,
        chunk,
        slot_mappings,
        compress_ratios_from_block_sizes(),
        label=label,
    )


def ds4_separate_kv_round_trip(
    connector: VLLMPagedMemNPUConnectorV2,
    gi: int,
    chunk: int,
    slot_mappings: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> None:
    separate_kv_round_trip_via_connector(
        connector,
        gi,
        chunk,
        slot_mappings,
        compress_ratios_from_block_sizes(),
        label=label,
    )


def ds4_slot_concat_and_offsets(
    sched_groups: list[int],
    slot_mappings: tuple[torch.Tensor, ...],
    g_start: int,
    g_end: int,
) -> tuple[torch.Tensor, list[int]]:
    return slot_concat_and_offsets(
        sched_groups,
        slot_mappings,
        g_start,
        g_end,
        compress_ratios_from_block_sizes(),
    )


def make_ds4_slot_mappings_cpu(
    num_tokens: int = DS4_CHUNK_SIZE,
) -> tuple[torch.Tensor, ...]:
    mappings: list[torch.Tensor] = []
    block_ids = list(range(1, 17))
    for ratio, block_size in zip(DS4_COMPRESS_RATIOS, DS4_BLOCK_SIZES_BY_GROUP):
        mappings.append(
            _build_slot_mapping_for_group(
                block_ids,
                block_size,
                num_tokens,
                is_store=False,
                compress_ratio=ratio,
            )
        )
    assert tuple(len(m) for m in mappings) == DS4_SLOT_LENS_512
    return tuple(mappings)


def make_ascend_adapter_for_load(
    *, num_kv_groups: int = DS4_NUM_SCHEDULER_GROUPS
):
    pytest.importorskip("vllm")
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl,
    )

    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    adapter._num_kv_groups = num_kv_groups
    adapter._compress_ratios_by_group = DS4_COMPRESS_RATIOS[:num_kv_groups]
    adapter._block_sizes_by_group = DS4_BLOCK_SIZES_BY_GROUP[:num_kv_groups]
    adapter._block_size = DS4_VLLM_BLOCK_SIZE
    adapter._lmcache_chunk_size = DS4_CHUNK_SIZE
    adapter.use_layerwise = False
    adapter.enable_blending = False
    adapter.kv_caches = {"layer0": torch.zeros(1)}
    adapter.current_layer = 0
    adapter._wait_for_save_done = True
    adapter.layerwise_retrievers = []
    adapter._invalid_block_ids = set()
    adapter._stats_monitor = MagicMock()

    mock_engine = MagicMock()
    mock_gpu_connector = MagicMock()
    if npu_available():
        mock_gpu_connector.load_stream = torch.npu.Stream()
    else:
        mock_gpu_connector.load_stream = MagicMock()
    mock_engine.gpu_connector = mock_gpu_connector
    mock_engine.retrieve.return_value = torch.ones(DS4_CHUNK_SIZE, dtype=torch.bool)
    adapter._manager = SimpleNamespace(lmcache_engine=mock_engine)
    adapter._parent = MagicMock()
    return adapter


def make_load_req_meta(
    *,
    num_tokens: int = DS4_CHUNK_SIZE,
    can_load: bool = True,
    omit_load_spec: bool = False,
    num_kv_groups: int = DS4_NUM_SCHEDULER_GROUPS,
    primary_kv_group_idx: int = 1,
) -> ReqMeta:
    if omit_load_spec:
        load_spec = None
    elif can_load:
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=num_tokens,
            can_load=True,
        )
    else:
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=num_tokens,
            can_load=False,
        )
    slot_mappings = make_ds4_slot_mappings_cpu(num_tokens)[:num_kv_groups]
    return ReqMeta(
        req_id="ds4-load-test",
        token_ids=list(range(num_tokens)),
        slot_mapping=slot_mappings[primary_kv_group_idx],
        slot_mappings_by_group=slot_mappings,
        allocated_block_ids_by_group=tuple([list(range(16))] * num_kv_groups),
        primary_kv_group_idx=primary_kv_group_idx,
        load_spec=load_spec,
    )


def make_partial_fail_load_req_meta() -> ReqMeta:
    """Minimal 2-group request for partial-load failure regression tests."""
    load_spec = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=2,
        can_load=True,
    )
    return ReqMeta(
        req_id="partial-fail-test",
        token_ids=[0, 1],
        slot_mapping=torch.tensor([0, 1024], dtype=torch.long),
        slot_mappings_by_group=(
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 1024], dtype=torch.long),
        ),
        allocated_block_ids_by_group=([0], [0, 1]),
        primary_kv_group_idx=1,
        load_spec=load_spec,
    )


def make_forward_context():
    return SimpleNamespace(attn_metadata=SimpleNamespace())


def make_ds4_setup():
    """Build DS4 connector, metadata, KV caches, and device for integration tests."""
    if not npu_available():
        pytest.skip("NPU not available")
    dev = device()
    num_blocks = 32
    kv_dict = make_ds4_kv_caches_dict(dev, num_blocks=num_blocks)
    kv_caches, sched_by_layer, _, _ = build_flat_kv_caches(
        kv_dict,
        _make_ds4_kv_cache_config(),
        ie_logical_block_size=DS4_IE_LOGICAL_BLOCK_SIZE,
    )
    kv_caches = list(kv_caches.values())
    connector, metadata = build_connector_and_metadata(
        kv_caches, sched_by_layer=sched_by_layer
    )
    connector.ensure_kv_layer_groups(kv_caches)
    assert metadata.kv_layer_groups_manager is not None
    return connector, metadata, kv_caches, dev


dsv4_setup = make_ds4_setup
