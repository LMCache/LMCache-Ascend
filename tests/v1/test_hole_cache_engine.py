# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import torch

# First Party
from lmcache_ascend.v1.hole_cache_engine import HoleLegacyCacheEngine


def test_hole_legacy_cache_engine_delegates_attributes():
    engine = SimpleNamespace(metadata="metadata")
    adapter = HoleLegacyCacheEngine(engine, Mock())

    assert adapter.metadata == "metadata"


def test_hole_legacy_cache_engine_empty_lookup_does_not_finalize_consumer():
    engine = Mock()
    engine.is_healthy.return_value = True
    engine.storage_manager = Mock()
    engine.token_database.process_tokens.return_value = iter(())
    engine.num_layers = 2
    engine.stats_monitor.on_retrieve_request.return_value = "monitor-id"
    engine._get_req_id.return_value = "req-id"
    engine._is_passive.return_value = True
    connector = Mock()
    adapter = HoleLegacyCacheEngine(engine, connector)

    outputs = list(adapter.retrieve_layer(torch.tensor([1, 2])))

    assert outputs[:-1] == [None, None, None]
    assert torch.equal(outputs[-1], torch.tensor([False, False]))
    connector.batched_to_gpu.assert_not_called()
    engine.stats_monitor.on_retrieve_finished.assert_called_once()
