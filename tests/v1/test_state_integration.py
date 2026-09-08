# SPDX-License-Identifier: Apache-2.0
"""State scheduling foundations; execute with the normal Ascend test bootstrap."""

# Standard
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec
import pytest
import torch

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
)
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
    AscendConnectorMetadata,
    LMCacheConnectorV1ImplMultiGroup,
    ReqMeta,
    RequestTracker,
    StateExecution,
)


def _connector(tracker):
    connector = LMCacheConnectorV1ImplMultiGroup.__new__(
        LMCacheConnectorV1ImplMultiGroup
    )
    state_spec = MambaSpec(
        block_size=512,
        shapes=((1, 3), (2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    connector._state_primary_kv_group_idx = 0
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[]),
            KVCacheGroupSpec(["gdn"], state_spec),
        ]
    )
    connector._block_sizes_by_group = (512, 512)
    connector._lmcache_chunk_size = 1024
    connector._request_trackers = {"r": tracker}
    connector._allocated_blocks = {}
    connector.config = SimpleNamespace(save_decode_cache=False)
    return connector


@pytest.mark.parametrize("kind", ["new", "cached", "resumed"])
@pytest.mark.parametrize("end, expected", [(1536, ()), (2048, ((1, 94),))])
def test_raw_execution_survives_attention_clipping_and_no_attention_work(
    kind, end, expected
):
    tracker = RequestTracker(
        req_id="r",
        prompt_len=4096,
        token_ids=list(range(end)),
        allocated_block_ids=[1, 2, 3, 4],
        allocated_block_ids_by_group=([1, 2, 3, 4], [71, 72, 93, 94]),
        num_saved_tokens=2048,
        request_configs={"lmcache.tag.tenant": "tenant-a"},
    )
    # Upstream Attention policy can omit the request altogether.
    assert (
        ReqMeta.from_request_tracker(tracker, (512, 512), 1024, primary_kv_group_idx=0)
        is None
    )
    connector = _connector(tracker)
    output = SimpleNamespace(
        scheduled_new_reqs=(
            [SimpleNamespace(req_id="r", num_computed_tokens=1024)]
            if kind == "new"
            else []
        ),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[] if kind == "new" else ["r"],
            num_computed_tokens=[1024],
            resumed_req_ids={"r"} if kind == "resumed" else set(),
        ),
        num_scheduled_tokens={"r": end - 1024},
        scheduled_spec_decode_tokens={},
    )
    meta = connector._attach_state_executions(AscendConnectorMetadata(), output)
    (execution,) = meta.state_executions
    assert execution.request_configs == {"lmcache.tag.tenant": "tenant-a"}
    tracker.request_configs["lmcache.tag.tenant"] = "changed"
    assert execution.request_configs["lmcache.tag.tenant"] == "tenant-a"
    assert execution.end == end
    assert execution.attention_end == end // 1024 * 1024
    assert len(execution.token_ids) == end
    assert execution.save_blocks(1024) == expected
    assert execution.load_blocks(1024) == (((1, 72),) if kind != "cached" else ())
    assert execution.block_ids_by_group[1][0] == 0


def test_complete_allocation_replaces_old_attempt_and_is_snapshotted():
    tracker = RequestTracker(
        req_id="r",
        prompt_len=2048,
        token_ids=list(range(2048)),
        allocated_block_ids=[1, 2, 3, 4],
        allocated_block_ids_by_group=([1, 2, 3, 4], [71, 72, 73, 74]),
    )
    connector = _connector(tracker)
    # The allocation callback's full table wins over the old tracker and delta.
    connector._allocated_blocks["r"] = ([5, 6, 7, 8], [0, 82, 0, 84])
    connector._apply_allocated_blocks(tracker)
    execution = StateExecution(
        "r",
        tuple(tracker.token_ids),
        1024,
        2048,
        2048,
        tuple(tuple(ids) for ids in tracker.allocated_block_ids_by_group),
        ((1, 512),),
        True,
        True,
    )
    assert execution.load_blocks(1024) == ((1, 82),)
    assert execution.save_blocks(1024) == ((1, 84),)
    # Ascend copies the restored initial block 82 into running block 84, then
    # forward updates 84; saving 82 would read the old prefix instead of S(E).
    tracker.allocated_block_ids_by_group[1][3] = 99
    assert execution.save_blocks(1024) == ((1, 84),)
    assert connector._allocated_blocks == {}


def test_state_mapping_rejects_unaligned_missing_and_wrong_restore_boundary():
    execution = StateExecution(
        "r",
        tuple(range(2048)),
        1024,
        2048,
        2048,
        ((1, 2, 3, 4), (0, 82, 0, 0)),
        ((1, 512),),
        True,
        True,
    )
    assert execution.save_blocks(1024) == ()
    assert execution.load_blocks(512) == ()
    assert execution.load_blocks(1024) == ((1, 82),)


def test_allocation_callback_copies_full_grouped_table(monkeypatch):
    # The Ascend outer connector must not lose the full table as upstream does.
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector import (
        LMCacheAscendConnector,
    )
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1 import (
        LMCacheAscendConnectorV1Dynamic,
    )

    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "update_state_after_alloc",
        lambda self, request, external: None,
    )
    impl = LMCacheConnectorV1ImplMultiGroup.__new__(LMCacheConnectorV1ImplMultiGroup)
    impl._allocated_blocks = {}
    impl._num_kv_groups = 2
    ids = ([5, 6], [0, 82])
    blocks = SimpleNamespace(get_block_ids=lambda: ids)
    request = SimpleNamespace(request_id="r")
    for cls in (LMCacheAscendConnector, LMCacheAscendConnectorV1Dynamic):
        outer = cls.__new__(cls)
        outer._lmcache_engine = impl
        outer.update_state_after_alloc(request, blocks, 1024)
        assert impl._allocated_blocks["r"] == ids
        assert impl._allocated_blocks["r"][1] is not ids[1]


@pytest.mark.parametrize(
    "role, passive", [("kv_both", False), ("kv_consumer", False), ("kv_both", True)]
)
@pytest.mark.parametrize("copy_error", [False, True])
def test_state_save_runs_without_attention_requests(
    monkeypatch, role, passive, copy_error
):
    worker = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    execution = StateExecution(
        "r", tuple(range(32)), 16, 32, 32, ((), (1, 2)), ((1, 16),), False, True
    )
    meta = AscendConnectorMetadata(state_executions=[execution])
    assert not meta.requests
    worker._parent = SimpleNamespace(_get_connector_metadata=lambda: meta)
    worker.kv_role = role
    worker.use_layerwise = False
    worker.kv_caches = {"attention": object()}
    worker.state_layouts = (object(),)
    worker.state_kv_caches = {"gdn": object()}
    worker.lmcache_engine = SimpleNamespace(
        _is_passive=lambda: passive,
        store_state=Mock(
            side_effect=RuntimeError("copy failed") if copy_error else None
        ),
        lookup_unpin=Mock(),
        metadata=SimpleNamespace(worker_id=0),
    )
    worker._replay_finished_stores_after_save = Mock()
    event = Mock()
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    if copy_error and role != "kv_consumer" and not passive:
        with pytest.raises(RuntimeError, match="copy failed"):
            worker.wait_for_save()
        worker.lmcache_engine.lookup_unpin.assert_called_once_with("r")
        return
    worker.wait_for_save()
    if role == "kv_consumer" or passive:
        worker.lmcache_engine.store_state.assert_not_called()
        event.record.assert_not_called()
        return
    event.record.assert_called_once()
    worker.lmcache_engine.lookup_unpin.assert_called_once_with("r")
    worker.lmcache_engine.store_state.assert_called_once_with(
        execution, worker.state_layouts, worker.state_kv_caches, event
    )


def test_generator_finally_forward_failure_skips_hybrid_publication():
    worker = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    worker._parent = SimpleNamespace(_get_connector_metadata=AscendConnectorMetadata)
    worker.state_layouts = (object(),)
    worker.lmcache_engine = SimpleNamespace(store_state=Mock())

    @contextmanager
    def forward_context():
        try:
            yield
        finally:
            worker.wait_for_save()

    with pytest.raises(RuntimeError, match="forward"):
        with forward_context():
            raise RuntimeError("forward")
    worker.lmcache_engine.store_state.assert_not_called()
    assert worker._wait_for_save_done


def test_worker_state_copy_error_propagates():
    worker = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    worker.state_layouts = (object(),)
    worker.state_kv_caches = {}
    worker.lmcache_engine = SimpleNamespace(
        metadata=SimpleNamespace(worker_id=2),
        store_state=Mock(side_effect=RuntimeError("device copy failed")),
    )
    meta = SimpleNamespace(state_executions=[SimpleNamespace(req_id="r", end=32)])
    with pytest.raises(RuntimeError, match="device copy failed"):
        worker._save_state_executions(meta, object())
