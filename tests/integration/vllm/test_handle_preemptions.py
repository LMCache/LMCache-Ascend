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
    """The inherited builder preserves state executions and preemption hints."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    request_metadata = SimpleNamespace(
        slot_mappings_by_group=(object(), object()),
        allocated_block_ids_by_group=([1, 2], [3]),
        primary_kv_group_idx=1,
        filtered_slot_by_group=(object(), object()),
        slot_valid_prefix_by_group=(object(), object()),
    )
    mg = pytest.importorskip("lmcache_ascend.integration.vllm.multi_group_vllm_adapter")
    state_execution = SimpleNamespace(request_id="req-1", boundary=256)
    adapter.kv_role = "kv_consumer"
    preempted_req_ids = {"req-1", "req-2"}
    scheduler_output = SimpleNamespace(
        preempted_req_ids=preempted_req_ids,
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
    )

    def attach_state(metadata, output):
        assert output is scheduler_output
        metadata.requests.append(request_metadata)
        metadata.state_executions.append(state_execution)
        return metadata

    with patch.object(adapter, "_attach_state_executions", side_effect=attach_state):
        metadata = adapter.build_connector_meta(scheduler_output)

    assert isinstance(metadata, mg.AscendConnectorMetadata)
    assert metadata.requests[0] is request_metadata
    assert metadata.state_executions == [state_execution]
    assert metadata.preempted_req_ids == preempted_req_ids

    preempted_req_ids.add("req-added-after-build")
    assert "req-added-after-build" not in metadata.preempted_req_ids


def test_ascend_connector_metadata_is_pickleable():
    """Upstream metadata retains state executions across serialization."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    mg = pytest.importorskip("lmcache_ascend.integration.vllm.multi_group_vllm_adapter")

    metadata = mg.AscendConnectorMetadata(
        requests=[SimpleNamespace(req_id="req-1")],
        preempted_req_ids={"req-1", "req-2"},
        state_executions=[SimpleNamespace(request_id="req-1", boundary=256)],
    )
    restored = pickle.loads(pickle.dumps(metadata))

    assert isinstance(restored, mg.AscendConnectorMetadata)
    assert restored.requests == metadata.requests
    assert restored.state_executions == metadata.state_executions
    assert restored.preempted_req_ids == {"req-1", "req-2"}


@pytest.mark.parametrize("payload_kind", ["v023_metadata", "legacy_set"])
def test_lmcache_connector_normalizes_preemption_payload(payload_kind):
    """The patched connector accepts both vLLM 0.23 and legacy payloads."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    mg = pytest.importorskip("lmcache_ascend.integration.vllm.multi_group_vllm_adapter")

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    expected_req_ids = {"req-1", "req-2"}
    if payload_kind == "v023_metadata":
        payload = mg.AscendConnectorMetadata(preempted_req_ids=expected_req_ids)
    else:
        payload = set(expected_req_ids)

    connector.handle_preemptions(payload)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(
        expected_req_ids
    )


@pytest.mark.parametrize("payload_kind", ["empty_metadata", "empty_set"])
def test_lmcache_connector_skips_payloads_without_preemptions(payload_kind):
    """Payloads with no preempted request ids must be no-ops."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    mg = pytest.importorskip("lmcache_ascend.integration.vllm.multi_group_vllm_adapter")

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    if payload_kind == "empty_metadata":
        payload = mg.AscendConnectorMetadata()
    else:
        payload = set()

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


def test_lmcache_connector_metadata_path_releases_pins_and_drains_stores():
    """The outer connector passes recovered IDs to the real worker adapter."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    mg = pytest.importorskip("lmcache_ascend.integration.vllm.multi_group_vllm_adapter")

    lmcache_engine = MagicMock()
    lmcache_engine.wait_for_pending_stores.return_value = set()
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = adapter
    metadata = mg.AscendConnectorMetadata(preempted_req_ids={"req-1", "req-2"})
    connector.handle_preemptions(pickle.loads(pickle.dumps(metadata)))

    # lookup_unpin is called once per preempted id (request-scoped pins).
    assert lmcache_engine.lookup_unpin.call_count == 2
    unpinned = {call.args[0] for call in lmcache_engine.lookup_unpin.call_args_list}
    assert unpinned == {"req-1", "req-2"}
    # The recovered set reaches wait_for_pending_stores, not the metadata object.
    lmcache_engine.wait_for_pending_stores.assert_called_once_with({"req-1", "req-2"})


@pytest.mark.parametrize(
    "payload_kind", ["object", "none", "string", "list", "tuple", "base_metadata"]
)
def test_lmcache_connector_preemptions_fail_fast_on_unknown_arg(payload_kind):
    """Reject unknown payloads before delegation, including falsey ones."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    payload = {
        "object": object(),
        "none": None,
        "string": "",
        "list": [],
        "tuple": (),
        "base_metadata": adapter_mod.LMCacheConnectorMetadata(),
    }[payload_kind]
    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    with pytest.raises(TypeError, match="handle_preemptions expects") as exc:
        connector.handle_preemptions(payload)
    assert f"got {type(payload).__name__}" in str(exc.value)
    connector._lmcache_engine.handle_preemptions.assert_not_called()
