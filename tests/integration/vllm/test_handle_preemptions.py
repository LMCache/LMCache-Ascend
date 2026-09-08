# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
import pickle

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


def test_build_connector_meta_carries_preempted_request_ids():
    """Scheduler metadata preserves LMCache requests and preemption hints."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    request_metadata = object()
    base_metadata = adapter_mod.LMCacheConnectorMetadata(requests=[request_metadata])
    preempted_req_ids = {"req-1", "req-2"}
    scheduler_output = SimpleNamespace(preempted_req_ids=preempted_req_ids)

    with patch.object(
        adapter_mod.LMCacheConnectorV1Impl,
        "build_connector_meta",
        return_value=base_metadata,
    ):
        metadata = adapter.build_connector_meta(scheduler_output)

    assert isinstance(metadata, adapter_mod.LMCacheAscendConnectorMetadata)
    assert metadata.requests == [request_metadata]
    assert metadata.preempted_req_ids == preempted_req_ids

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


@pytest.mark.parametrize("payload_kind", ["v023_metadata", "legacy_set"])
def test_lmcache_connector_normalizes_preemption_payload(payload_kind):
    """The patched connector accepts both vLLM 0.23 and legacy payloads."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    expected_req_ids = {"req-1", "req-2"}
    if payload_kind == "v023_metadata":
        payload = adapter_mod.LMCacheAscendConnectorMetadata(
            preempted_req_ids=expected_req_ids
        )
    else:
        payload = set(expected_req_ids)

    connector.handle_preemptions(payload)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(
        expected_req_ids
    )


@pytest.mark.parametrize("payload_kind", ["empty_metadata", "base_metadata"])
def test_lmcache_connector_skips_payloads_without_preemptions(payload_kind):
    """Payloads with no preempted request ids must be no-ops."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    if payload_kind == "empty_metadata":
        payload = adapter_mod.LMCacheAscendConnectorMetadata()
    else:
        payload = adapter_mod.LMCacheConnectorMetadata()

    connector.handle_preemptions(payload)

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
