# SPDX-License-Identifier: Apache-2.0
# Standard
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_lmcache_connector_delegates_preemptions_after_ascend_patch():
    """Ascend patches the outer vLLM connector to delegate preemptions."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    preempted_req_ids = {"req-1", "req-2"}
    connector.handle_preemptions(preempted_req_ids)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(
        preempted_req_ids
    )


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
        lmcache_engine.wait_for_pending_stores.assert_not_called()


def _import_metadata():
    """Import LMCacheConnectorMetadata for constructing v0.25.1rc-style args."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata

    return LMCacheConnectorMetadata


def test_ascend_adapter_handles_v0251rc_metadata_path():
    """v0.25.1rc passes a KVConnectorMetadata with stashed preempted_req_ids.

    This is the primary fix path: ``build_connector_meta`` stashes the id set
    on the metadata, and ``handle_preemptions`` recovers it via ``hasattr``.
    """
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    LMCacheConnectorMetadata = _import_metadata()

    lmcache_engine = MagicMock()
    lmcache_engine.wait_for_pending_stores.return_value = set()
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    meta = LMCacheConnectorMetadata()
    meta.preempted_req_ids = {"req-1", "req-2"}

    adapter.handle_preemptions(meta)

    # lookup_unpin is called once per preempted id (request-scoped pins).
    assert lmcache_engine.lookup_unpin.call_count == 2
    unpinned = {call.args[0] for call in lmcache_engine.lookup_unpin.call_args_list}
    assert unpinned == {"req-1", "req-2"}
    # The recovered set reaches wait_for_pending_stores, not the metadata object.
    lmcache_engine.wait_for_pending_stores.assert_called_once_with({"req-1", "req-2"})


def test_ascend_adapter_preemption_ids_survive_pickle_round_trip():
    """Stashed preempted_req_ids must survive scheduler->worker pickle IPC.

    ``build_connector_meta`` runs scheduler-side; ``handle_preemptions``
    runs worker-side and receives the unpickled metadata. A plain @dataclass
    without ``slots=True`` carries dynamically attached attributes through the
    default ``__dict__`` pickle path — this test locks that contract so a
    future ``slots=True`` addition does not silently drop the attribute.
    """
    LMCacheConnectorMetadata = _import_metadata()

    meta = LMCacheConnectorMetadata()
    meta.preempted_req_ids = {"req-1", "req-2", "req-3"}

    restored = pickle.loads(pickle.dumps(meta))
    assert hasattr(restored, "preempted_req_ids")
    assert restored.preempted_req_ids == {"req-1", "req-2", "req-3"}


def test_ascend_adapter_handle_preemptions_fails_fast_on_unknown_arg():
    """An unknown argument type must raise, not silently no-op.

    Silent ``set()`` would skip ``lookup_unpin`` and async store drain with no
    signal — the exact failure mode this fix targets. Fail fast so future vLLM
    API drift surfaces immediately.
    """
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    adapter = _make_adapter(
        adapter_mod,
        store_async=False,
        kv_role="kv_both",
        lmcache_engine=MagicMock(),
    )

    with pytest.raises(TypeError, match="handle_preemptions expects"):
        adapter.handle_preemptions(object())
