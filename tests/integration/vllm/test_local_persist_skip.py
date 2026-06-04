# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``LMCacheAscendConnectorV1Impl._local_persist_skip``.

The helper decides whether a matched prefix was pulled from a *remote* peer and
therefore still has to be persisted into the local CPU backend (otherwise every
subsequent request re-pulls the same KV over the one-sided channel). These tests
exercise the local-vs-remote branch logic in isolation, mocking the cache engine
``lookup`` so no NPU / real backend is required.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

CHUNK = 16


def _adapter_mod():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    return pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")


def _make_adapter(*, kv_role="kv_both", local_cpu=True, local_present=0):
    adapter_mod = _adapter_mod()

    engine = MagicMock()
    engine.lookup.return_value = local_present

    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.kv_role = kv_role
    adapter.config = SimpleNamespace(local_cpu=local_cpu)
    adapter._lmcache_chunk_size = CHUNK
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    return adapter, engine


def _make_request(
    *,
    can_load=True,
    lmcache_cached_tokens=4 * CHUNK,
    has_save_spec=True,
    num_tokens=5 * CHUNK,
):
    load_spec = SimpleNamespace(
        can_load=can_load,
        lmcache_cached_tokens=lmcache_cached_tokens,
    )
    save_spec = (
        SimpleNamespace(can_save=False, skip_leading_tokens=lmcache_cached_tokens)
        if has_save_spec
        else None
    )
    return SimpleNamespace(
        req_id="req-1",
        token_ids=list(range(num_tokens)),
        request_configs=None,
        save_spec=save_spec,
        load_spec=load_spec,
    )


def test_remote_only_hit_persists_from_zero():
    """Cold local cache + remote hit -> back-fill the whole pulled prefix."""
    adapter, engine = _make_adapter(local_present=0)
    request = _make_request(lmcache_cached_tokens=4 * CHUNK)

    assert adapter._local_persist_skip(request, request.token_ids) == 0

    engine.lookup.assert_called_once()
    args, kwargs = engine.lookup.call_args
    assert args[0] is request.token_ids
    assert kwargs["search_range"] == ["LocalCPUBackend"]
    assert kwargs["pin"] is False
    assert kwargs["request_configs"] is request.request_configs


def test_full_local_hit_returns_none():
    """Whole matched prefix already local -> nothing to back-fill."""
    adapter, engine = _make_adapter(local_present=4 * CHUNK)
    request = _make_request(lmcache_cached_tokens=4 * CHUNK)

    assert adapter._local_persist_skip(request, request.token_ids) is None
    engine.lookup.assert_called_once()


def test_partial_local_hit_returns_chunk_aligned_local_prefix():
    """A partial (and unaligned) local prefix is reported chunk-aligned."""
    adapter, engine = _make_adapter(local_present=2 * CHUNK + 3)
    request = _make_request(lmcache_cached_tokens=4 * CHUNK)

    assert adapter._local_persist_skip(request, request.token_ids) == 2 * CHUNK


def test_consumer_role_skips_without_lookup():
    adapter, engine = _make_adapter(kv_role="kv_consumer", local_present=0)
    request = _make_request()

    assert adapter._local_persist_skip(request, request.token_ids) is None
    engine.lookup.assert_not_called()


def test_local_cpu_disabled_skips_without_lookup():
    adapter, engine = _make_adapter(local_cpu=False, local_present=0)
    request = _make_request()

    assert adapter._local_persist_skip(request, request.token_ids) is None
    engine.lookup.assert_not_called()


def test_no_load_skips_without_lookup():
    adapter, engine = _make_adapter(local_present=0)
    request = _make_request(can_load=False)

    assert adapter._local_persist_skip(request, request.token_ids) is None
    engine.lookup.assert_not_called()


def test_zero_hit_skips_without_lookup():
    adapter, engine = _make_adapter(local_present=0)
    request = _make_request(lmcache_cached_tokens=0)

    assert adapter._local_persist_skip(request, request.token_ids) is None
    engine.lookup.assert_not_called()


def test_missing_save_spec_skips_without_lookup():
    adapter, engine = _make_adapter(local_present=0)
    request = _make_request(has_save_spec=False)

    assert adapter._local_persist_skip(request, request.token_ids) is None
    engine.lookup.assert_not_called()
