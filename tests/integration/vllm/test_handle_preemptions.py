# SPDX-License-Identifier: Apache-2.0
# Standard
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

# Third Party
import pytest


def _import_and_patch_vllm_connector():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")

    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1,
    )

    lmcache_ascend = pytest.importorskip("lmcache_ascend")
    lmcache_ascend._patch_vllm_v1_adapter()
    return LMCacheConnectorV1


def _make_adapter(adapter_mod, *, store_async, kv_role, lmcache_engine):
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = store_async
    adapter.kv_role = kv_role
    adapter._manager = SimpleNamespace(lmcache_engine=lmcache_engine)
    return adapter


@pytest.mark.parametrize("preempted_req_ids", [{"req-1", "req-2"}, None])
def test_build_connector_meta_carries_preempted_request_ids(preempted_req_ids):
    """Scheduler metadata preserves LMCache requests and preemption hints."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    request_metadata = object()
    base_metadata = adapter_mod.LMCacheConnectorMetadata(requests=[request_metadata])
    scheduler_output = SimpleNamespace(preempted_req_ids=preempted_req_ids)

    with patch.object(
        adapter_mod.LMCacheConnectorV1Impl,
        "build_connector_meta",
        return_value=base_metadata,
    ):
        metadata = adapter.build_connector_meta(scheduler_output)

    assert isinstance(metadata, adapter_mod.LMCacheAscendConnectorMetadata)
    assert metadata.requests == [request_metadata]
    assert metadata.preempted_req_ids == set(preempted_req_ids or ())

    if preempted_req_ids is not None:
        preempted_req_ids.add("req-added-after-build")
        assert "req-added-after-build" not in metadata.preempted_req_ids


def test_ascend_connector_metadata_is_pickleable():
    """Custom metadata survives scheduler-to-worker style serialization."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    metadata = adapter_mod.LMCacheAscendConnectorMetadata(
        preempted_req_ids={"req-1", "req-2"}
    )
    restored = pickle.loads(pickle.dumps(metadata))

    assert isinstance(restored, adapter_mod.LMCacheAscendConnectorMetadata)
    assert restored.preempted_req_ids == {"req-1", "req-2"}


def test_lmcache_connector_extracts_preemptions_from_v023_metadata():
    """The vLLM 0.23 metadata payload is normalized to request IDs."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    metadata = adapter_mod.LMCacheAscendConnectorMetadata(
        preempted_req_ids={"req-1", "req-2"}
    )
    connector.handle_preemptions(metadata)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(
        {"req-1", "req-2"}
    )


def test_lmcache_connector_accepts_legacy_preemption_set():
    """The patched connector remains compatible with the vLLM 0.18 API."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    preempted_req_ids = {"req-1", "req-2"}
    connector.handle_preemptions(preempted_req_ids)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(
        preempted_req_ids
    )


def test_lmcache_connector_skips_empty_v023_metadata():
    """vLLM 0.23 calls the hook every step; empty metadata must be a no-op."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    connector.handle_preemptions(adapter_mod.LMCacheAscendConnectorMetadata())

    connector._lmcache_engine.handle_preemptions.assert_not_called()


def test_lmcache_connector_skips_metadata_without_preemption_field():
    """A mixed-version base LMCache metadata payload must not be iterated."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    connector.handle_preemptions(adapter_mod.LMCacheConnectorMetadata())

    connector._lmcache_engine.handle_preemptions.assert_not_called()


def test_lmcache_connector_preemption_patch_handles_no_inner_impl():
    """The Ascend patch should tolerate inner implementations without a hook."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = object()

    connector.handle_preemptions({"req-1"})


def test_ascend_adapter_drains_pending_stores_for_async_producer():
    """Async non-consumer workers must drain pending stores before reuse."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    lmcache_engine.wait_for_pending_stores.return_value = {"req-1"}
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    preempted_req_ids = {"req-1", "req-2"}
    adapter.handle_preemptions(preempted_req_ids)

    lmcache_engine.lookup_unpin.assert_has_calls(
        [call("req-1"), call("req-2")], any_order=True
    )
    lmcache_engine.wait_for_pending_stores.assert_called_once_with(preempted_req_ids)


@pytest.mark.parametrize(
    ("store_async", "kv_role", "has_engine"),
    [
        (False, "kv_both", True),
        (True, "kv_consumer", True),
        (True, "kv_both", False),
    ],
)
def test_ascend_adapter_skips_preemption_drain_when_not_required(
    store_async, kv_role, has_engine
):
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock() if has_engine else None
    adapter = _make_adapter(
        adapter_mod,
        store_async=store_async,
        kv_role=kv_role,
        lmcache_engine=lmcache_engine,
    )

    adapter.handle_preemptions({"req-1"})

    if has_engine:
        lmcache_engine.lookup_unpin.assert_called_once_with("req-1")
        lmcache_engine.wait_for_pending_stores.assert_not_called()


def test_ascend_adapter_skips_empty_preemption_ids():
    """An ordinary vLLM 0.23 step must not touch the cache engine."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    adapter.handle_preemptions(set())

    lmcache_engine.lookup_unpin.assert_not_called()
    lmcache_engine.wait_for_pending_stores.assert_not_called()
