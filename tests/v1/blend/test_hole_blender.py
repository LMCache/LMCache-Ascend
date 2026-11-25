# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import torch

# First Party
from lmcache_ascend.v1.blend.hole_blender import LMCBlenderHole
from lmcache_ascend.v1.npu_hole_connector import _slice_source_tokens


class _CacheKey:
    def split_layers(self, num_layers):
        return [self] * num_layers


def test_sparse_retrieve_keeps_full_key_when_vllm_prefix_splits_hit_range():
    key = _CacheKey()
    segment_helper = Mock()
    segment_helper.make_cache_key.return_value = key

    memory_obj = Mock()
    task = Mock()
    task.result.return_value = [memory_obj]
    storage_manager = Mock()
    storage_manager.layerwise_batched_get.return_value = iter([task])
    cache_engine = SimpleNamespace(num_layers=1, storage_manager=storage_manager)

    connector_call = {}

    def batched_to_gpu(starts, ends, **kwargs):
        connector_call.update(starts=starts, ends=ends, kwargs=kwargs)
        yield
        yield
        yield

    blender = LMCBlenderHole.__new__(LMCBlenderHole)
    blender.cache_engine = cache_engine
    blender.gpu_connector = SimpleNamespace(batched_to_gpu=batched_to_gpu)
    blender.segment_helper = segment_helper
    blender.num_layers = 1
    blender.emit_timer = Mock()

    tokens = torch.arange(600)
    load_spec = SimpleNamespace(
        hit_ranges=[(0, 500)],
        covered_tokens=500,
        location="LocalCPUBackend",
    )

    outputs = list(blender._sparse_retrieve_layer(tokens, load_spec, prefix_start=128))

    segment_helper.make_cache_key.assert_called_once_with(tokens, (0, 500), None)
    assert connector_call["starts"] == [128]
    assert connector_call["ends"] == [500]
    assert connector_call["kwargs"]["source_offsets"] == [128]
    assert connector_call["kwargs"]["prefix_end"] == 500
    expected_mask = torch.zeros(600, dtype=torch.bool)
    expected_mask[:500] = True
    assert torch.equal(outputs[-1], expected_mask)


def test_slice_source_tokens_selects_suffix_from_full_cached_segment():
    cached_segment = torch.arange(500)

    source = _slice_source_tokens(cached_segment, source_offset=128, num_tokens=372)

    assert torch.equal(source, torch.arange(128, 500))
