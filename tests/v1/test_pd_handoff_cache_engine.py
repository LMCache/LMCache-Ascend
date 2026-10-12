# SPDX-License-Identifier: Apache-2.0
"""Regression tests for PD handoff integration in the cache engine."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import threading

# First Party
# This module retains the original base before the upstream export is patched.
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine, LMCacheEngine

HANDOFF_CONFIG = {
    "lmcache.pd_handoff_id": "handoff-cache-engine",
    "lmcache.tag.tenant": "tenant-a",
}
SANITIZED_CONFIG = {"lmcache.tag.tenant": "tenant-a"}


def test_lookup_promotes_handoff_and_sanitizes_upstream_request_config():
    engine = object.__new__(AscendLMCacheEngine)
    engine._engine_state_lock = threading.RLock()
    engine._promote_pd_handoff_lease = MagicMock()
    engine._is_pd_receiver = lambda: False

    with patch.object(LMCacheEngine, "lookup", return_value=2) as upstream_lookup:
        result = AscendLMCacheEngine.lookup(
            engine,
            tokens=[1, 2],
            lookup_id="decoder-request",
            pin=True,
            request_configs=HANDOFF_CONFIG,
        )

    assert result == 2
    engine._promote_pd_handoff_lease.assert_called_once_with(
        "handoff-cache-engine", "decoder-request"
    )
    assert upstream_lookup.call_args.kwargs["request_configs"] == SANITIZED_CONFIG
    assert HANDOFF_CONFIG["lmcache.pd_handoff_id"] == "handoff-cache-engine"


def test_retrieve_promotes_handoff_and_releases_real_request_lease():
    engine = object.__new__(AscendLMCacheEngine)
    engine._get_req_id = lambda kwargs: kwargs["request_id"]
    engine._promote_pd_handoff_lease = MagicMock()
    engine._is_pd_receiver = lambda: False
    engine._retrieve_impl = MagicMock(return_value="retrieved")
    engine._release_pd_request_lease = MagicMock()

    result = AscendLMCacheEngine.retrieve(
        engine,
        [1, 2],
        request_id="decoder-request",
        request_configs=HANDOFF_CONFIG,
    )

    assert result == "retrieved"
    engine._promote_pd_handoff_lease.assert_called_once_with(
        "handoff-cache-engine", "decoder-request"
    )
    retrieve_kwargs = engine._retrieve_impl.call_args.kwargs
    assert retrieve_kwargs["request_configs"] == SANITIZED_CONFIG
    engine._release_pd_request_lease.assert_called_once_with("decoder-request")


def test_store_sanitizes_handoff_before_capturing_pipeline_kwargs():
    engine = object.__new__(AscendLMCacheEngine)
    engine.is_healthy = lambda: True
    engine.gpu_connector = MagicMock()
    engine._is_passive = lambda: False
    engine.storage_manager = MagicMock()
    engine._get_req_id = lambda kwargs: kwargs["request_id"]
    engine.is_store_async = False
    engine._run_store_pipeline = MagicMock()

    AscendLMCacheEngine.store(
        engine,
        tokens=[1, 2],
        request_id="prefill-request",
        request_configs=HANDOFF_CONFIG,
    )

    pipeline_kwargs = engine._run_store_pipeline.call_args.args[6]
    assert pipeline_kwargs["request_configs"] == SANITIZED_CONFIG
    assert HANDOFF_CONFIG["lmcache.pd_handoff_id"] == "handoff-cache-engine"


def test_promote_handoff_dispatches_to_receiver_pd_backend():
    pd_backend = SimpleNamespace(promote_handoff_lease=MagicMock(return_value=2))
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(enable_pd=True, pd_role="receiver")
    engine.storage_manager = SimpleNamespace(storage_backends={"PDBackend": pd_backend})
    engine.is_healthy = lambda: True

    AscendLMCacheEngine._promote_pd_handoff_lease(
        engine,
        "handoff-dispatch",
        "decoder-dispatch",
    )

    pd_backend.promote_handoff_lease.assert_called_once_with(
        "handoff-dispatch", "decoder-dispatch"
    )
