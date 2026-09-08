# SPDX-License-Identifier: Apache-2.0
"""State scheduling foundations; execute with the normal Ascend test bootstrap."""

# Standard
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec
import pytest
import torch

# First Party
from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
    AscendConnectorMetadata,
    LMCacheConnectorV1ImplMultiGroup,
    ReqMeta,
    RequestTracker,
    StateExecution,
)
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
)
from lmcache_ascend.v1.state_checkpoint import StateBlockBinding
from lmcache_ascend.v1.state_layout import build_state_group_layout
from lmcache_ascend.v1.state_lookup import StateLookupSelection
from lmcache_ascend.v1.state_memory import (
    StateCheckpointBuffer,
    state_checkpoint_metadata,
)


@pytest.fixture
def caplog(caplog):
    # First Party
    from lmcache_ascend.integration.vllm import vllm_v1_adapter
    from lmcache_ascend.v1 import state_cache

    # LMCache loggers do not propagate to pytest's root capture handler.
    loggers = (vllm_v1_adapter.logger, state_cache.logger)
    for logger in loggers:
        caplog.set_level("INFO", logger=logger.name)
        logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        for logger in loggers:
            logger.removeHandler(caplog.handler)


def _scheduler(monkeypatch, candidates, local=1024, skip=0, minimum=0):
    # First Party
    from lmcache_ascend.integration.vllm import multi_group_vllm_adapter as module

    connector = LMCacheConnectorV1ImplMultiGroup.__new__(
        LMCacheConnectorV1ImplMultiGroup
    )
    connector._state_primary_kv_group_idx = 1
    connector.kv_role = "kv_both"
    connector.worker_count = 2
    connector._manager = SimpleNamespace(lookup_client=Mock())
    connector._requests_priority = {}
    connector.load_specs = {}
    connector.skip_last_n_tokens = skip
    connector.config = SimpleNamespace(min_retrieve_tokens=minimum)
    request = SimpleNamespace(
        request_id="r",
        num_tokens=4096,
        all_token_ids=list(range(4096)),
        sampling_params=SimpleNamespace(
            extra_args={"kv_transfer_params": {"lmcache.tag.tenant": "a"}}
        ),
    )
    lookup = Mock(
        side_effect=lambda *a, **kw: max(
            (r for r in candidates if local < r <= kw["upper"]), default=0
        )
    )
    cancel = Mock()
    monkeypatch.setattr(module, "lookup_state", lookup)
    monkeypatch.setattr(module, "cancel_state_lookup", cancel)
    return connector, request, lookup, cancel


@pytest.mark.parametrize(
    "candidates,local,expected",
    [
        ([3072], 1024, 2048),
        ([1024], 1024, 0),
        ([1024], 1536, 0),
        ([4096], 0, 0),
        ([3072, 4096], 0, 3072),
    ],
)
def test_hybrid_scheduler_selects_real_boundary_before_full_hit_adjustment(
    monkeypatch, candidates, local, expected
):
    connector, request, lookup, _ = _scheduler(monkeypatch, candidates, local)
    assert connector.get_num_new_matched_tokens(request, local) == expected
    spec = connector.load_specs["r"]
    assert spec.vllm_cached_tokens == local
    assert spec.lmcache_cached_tokens == (local + expected if expected else 0)
    assert not spec.can_load
    assert lookup.call_args.kwargs["upper"] == 4095
    assert lookup.call_args.kwargs["required_world_size"] == 2
    assert lookup.call_args.kwargs["request_configs"] == {"lmcache.tag.tenant": "a"}
    assert lookup.call_args.args[1] == request.all_token_ids


def test_hybrid_minimum_rejects_entire_selection_and_skip_limits_query(monkeypatch):
    connector, request, lookup, cancel = _scheduler(
        monkeypatch, [2048, 3072], skip=1200, minimum=1500
    )
    assert connector.get_num_new_matched_tokens(request, 1024) == 0
    assert lookup.call_args.kwargs["upper"] == 2896
    assert connector.load_specs["r"].lmcache_cached_tokens == 0
    cancel.assert_called_once_with(connector.lookup_client, "r")


def test_hybrid_multimodal_request_is_rejected_before_lookup(monkeypatch):
    connector, request, lookup, _ = _scheduler(monkeypatch, [3072])
    request.mm_features = [object()]
    with pytest.raises(ValueError, match="text"):
        connector.get_num_new_matched_tokens(request, 0)
    lookup.assert_not_called()


def _load_worker(monkeypatch, ret_mask=None):
    # Third Party
    from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec

    # First Party
    from lmcache_ascend.integration.vllm import vllm_v1_adapter as module
    from lmcache_ascend.v1 import state_cache

    # These core tests inspect planning/ownership with host tensors. Native NPU
    # validation and transfer are separately exercised by the device suite.
    monkeypatch.setattr(state_cache, "_validate_load_device", lambda runtime: None)

    runtime = (torch.zeros(8, 3), torch.zeros(8, 2))
    layout = build_state_group_layout(1, ["gdn"], [runtime])
    obj = SimpleNamespace(
        is_valid=lambda: True,
        ref_count_down=Mock(),
        meta=SimpleNamespace(state_checkpoint=state_checkpoint_metadata(layout)),
    )
    buffer = StateCheckpointBuffer(
        layout,
        obj,
        tuple(torch.zeros(plane.shape, dtype=plane.dtype) for plane in layout.planes),
    )
    execution = StateExecution(
        "r",
        tuple(range(48)),
        32,
        48,
        48,
        ((1, 2, 3), (0, 5, 7)),
        ((1, 16),),
        True,
        True,
    )
    request = SimpleNamespace(
        req_id="r",
        token_ids=list(range(48)),
        load_spec=LoadSpec(16, 32, True),
        request_configs=None,
        num_kv_groups=2,
        primary_kv_group_idx=0,
        get_slot_mapping=lambda group: (
            torch.arange(48) if group == 0 else torch.empty(0)
        ),
        filtered_slot_by_group=None,
        slot_valid_prefix_by_group=None,
    )
    selection = StateLookupSelection(32, {1: buffer})
    lock = RLock()

    def selected(*args):
        assert lock._is_owned()
        return selection

    def retrieve(*args, **kwargs):
        assert lock._is_owned()
        return torch.ones(32, dtype=torch.bool) if ret_mask is None else ret_mask

    # Third Party
    from lmcache.utils import CacheEngineKey

    key = CacheEngineKey("model", 2, 1, 123, torch.float32)
    engine = SimpleNamespace(
        _engine_state_lock=lock,
        get_state_lookup=Mock(side_effect=selected),
        lookup_unpin=Mock(),
        token_database=SimpleNamespace(process_tokens=lambda *a, **k: [(16, 32, key)]),
        storage_manager=None,
        retrieve=Mock(side_effect=retrieve),
        gpu_connector=SimpleNamespace(load_stream=Mock()),
        metadata=SimpleNamespace(worker_id=1),
    )
    worker = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    worker._manager = SimpleNamespace(lmcache_engine=engine)
    worker._failed_state_loads = set()
    worker._finished_state_loads = set()
    worker.state_layouts = (layout,)
    worker.state_kv_caches = {"gdn": runtime}
    worker._lmcache_chunk_size = 16
    worker._invalid_block_ids = set()
    worker.kv_caches = {"attn": object()}
    worker._state_primary_kv_group_idx = 0
    worker._prepare_hybrid_attention = Mock(return_value={})
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    copy = Mock()
    monkeypatch.setattr(module, "transfer_state", copy)
    return worker, request, execution, selection, copy


def test_hybrid_restore_uses_selection_and_pre_movement_target(monkeypatch, caplog):
    caplog.set_level("INFO")
    worker, request, execution, selection, copy = _load_worker(monkeypatch)
    assert worker._load_hybrid_request(request, execution)
    operation = copy.call_args.args[0]
    assert operation.buffer is selection.buffers[1]
    assert operation.checkpoint.boundary == execution.start == 32
    assert operation.runtime.block_id == 5  # runner later moves 5 -> 7
    assert operation.runtime.tensors[0] == worker.state_kv_caches["gdn"]
    assert operation.direction == "load"
    worker.lmcache_engine.lookup_unpin.assert_called_once_with("r")
    assert not worker._failed_state_loads
    assert "Hybrid load complete" in caplog.text


@pytest.mark.parametrize(
    "failure", ["selection", "buffer", "metadata", "mapping", "execution"]
)
def test_hybrid_detectable_preflight_failure_does_not_copy(
    monkeypatch, caplog, failure
):
    worker, request, execution, selection, copy = _load_worker(monkeypatch)
    if failure == "selection":
        worker.lmcache_engine.get_state_lookup.side_effect = lambda *a: None
    elif failure == "buffer":
        selection.buffers.clear()
    elif failure == "metadata":
        selection.buffers[1].memory_obj.meta.state_checkpoint = {}
    elif failure == "mapping":
        execution = replace(execution, block_ids_by_group=((1, 2, 3), (0, 0, 7)))
    else:
        execution = None
    assert not worker._load_hybrid_request(request, execution)
    worker.lmcache_engine.retrieve.assert_not_called()
    copy.assert_not_called()
    assert worker._failed_state_loads == {"r"}
    assert not worker._invalid_block_ids
    worker.lmcache_engine.lookup_unpin.assert_called_once_with("r")
    for context in ("request=r", "rank=1", "group=", "R=32", "reason="):
        assert context in caplog.text
    assert "Hybrid load complete" not in caplog.text


def test_hybrid_partial_attention_checks_needed_interval_not_sum(monkeypatch, caplog):
    mask = torch.ones(32, dtype=torch.bool)
    mask[31] = False  # sum=31 >= needed=16; [C,R) is still incomplete.
    worker, request, execution, _, copy = _load_worker(monkeypatch, mask)
    assert not worker._load_hybrid_request(request, execution)
    copy.assert_not_called()
    assert worker._failed_state_loads == {"r"}
    assert not worker._invalid_block_ids
    assert "Attention coverage" in caplog.text
    assert "Hybrid load complete" not in caplog.text


@pytest.mark.parametrize("phase", ["validation", "attention", "copy", "sync"])
def test_hybrid_underlying_errors_log_and_propagate(monkeypatch, caplog, phase):
    worker, request, execution, _, copy = _load_worker(monkeypatch)
    error = RuntimeError("device failed")
    if phase == "validation":
        monkeypatch.setattr(StateBlockBinding, "validate", Mock(side_effect=error))
    elif phase == "attention":
        worker.lmcache_engine.retrieve.side_effect = error
    elif phase == "copy":
        copy.side_effect = error
    else:
        worker.lmcache_engine.gpu_connector.load_stream.synchronize.side_effect = error
    with pytest.raises(RuntimeError, match="device failed"):
        worker._load_hybrid_request(request, execution)
    assert worker._failed_state_loads == {"r"}
    worker.lmcache_engine.lookup_unpin.assert_called_once_with("r")
    assert "Hybrid load complete" not in caplog.text
    assert "request=r" in caplog.text and "R=32" in caplog.text


def test_failure_suppresses_later_local_state_save_until_finished(monkeypatch):
    worker, request, execution, _, _ = _load_worker(monkeypatch)
    worker._failed_state_loads.add("r")
    worker.lmcache_engine.store_state = Mock()
    worker._save_state_executions(
        SimpleNamespace(state_executions=[execution]), object()
    )
    worker.lmcache_engine.store_state.assert_not_called()
    worker._wait_for_save_done = True
    worker._late_finished_sending = set()
    worker.lmcache_engine.get_finished_stores = Mock(return_value=set())
    worker.get_finished({request.req_id})
    assert not worker._failed_state_loads
    worker.lmcache_engine.lookup_unpin.assert_called_once_with("r")


def test_no_load_spec_does_not_restore_new_request_state(monkeypatch):
    worker, request, execution, _, copy = _load_worker(monkeypatch)
    request.load_spec.can_load = False
    assert not worker._load_hybrid_request(request, execution)
    worker.lmcache_engine.get_state_lookup.assert_not_called()
    copy.assert_not_called()


@pytest.mark.parametrize("external", [0, 2048])
def test_allocation_preserves_selected_boundary_or_releases_it(monkeypatch, external):
    connector, request, _, cancel = _scheduler(monkeypatch, [3072])
    connector._unfinished_requests = {}
    connector._allocated_blocks = {}
    assert connector.get_num_new_matched_tokens(request, 1024) == 2048
    connector.update_state_after_alloc(request, external)
    spec = connector.load_specs["r"]
    assert spec.vllm_cached_tokens == 1024
    assert spec.can_load == bool(external)
    assert spec.lmcache_cached_tokens == (3072 if external else 0)
    connector.lookup_client.clear_lookup_status.assert_called_once_with("r")
    if external:
        cancel.assert_not_called()  # allocation-local clear retains worker selection
    else:
        cancel.assert_called_once_with(connector.lookup_client, "r")


def test_ordinary_scheduler_still_delegates(monkeypatch):
    # The normal bootstrap replaces the upstream module's exported class.
    # Patch the original base retained by the multi-group adapter instead.
    # First Party
    from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
        LMCacheConnectorV1Impl,
    )

    connector, request, lookup, _ = _scheduler(monkeypatch, [])
    connector._state_primary_kv_group_idx = None
    ordinary = Mock(return_value=123)
    monkeypatch.setattr(LMCacheConnectorV1Impl, "get_num_new_matched_tokens", ordinary)
    assert connector.get_num_new_matched_tokens(request, 17) == 123
    ordinary.assert_called_once_with(request, 17)
    lookup.assert_not_called()


def test_scheduler_cancel_before_allocation_releases_selection(monkeypatch):
    # Third Party
    from vllm.v1.request import RequestStatus

    # First Party
    from lmcache_ascend.integration.vllm import vllm_v1_adapter as module

    connector = LMCacheAscendConnectorV1Impl.__new__(LMCacheAscendConnectorV1Impl)
    connector._state_primary_kv_group_idx = 0
    connector._manager = SimpleNamespace(lookup_client=Mock(), lmcache_engine=None)
    connector.load_specs = {"r": object()}
    connector._allocated_blocks = {"r": ((1,), (2,))}
    connector.use_layerwise = connector.async_loading = connector.store_async = False
    connector.kv_role = "kv_both"
    connector.config = SimpleNamespace(get_extra_config_value=lambda *a: False)
    request = SimpleNamespace(request_id="r", status=RequestStatus.FINISHED_ABORTED)
    cancel = Mock()
    monkeypatch.setattr(module, "cancel_state_lookup", cancel)
    assert connector.request_finished_all_groups(request, ()) == (False, None)
    cancel.assert_called_once_with(connector.lookup_client, "r")
    assert not connector.load_specs and not connector._allocated_blocks


def test_preemption_releases_selection_and_restore_uses_new_target(monkeypatch):
    worker, request, execution, _, copy = _load_worker(monkeypatch)
    worker.store_async = False
    worker.handle_preemptions({"r"})
    worker.lmcache_engine.lookup_unpin.assert_called_once_with("r")
    remapped = replace(execution, block_ids_by_group=((8, 9, 10), (0, 6, 7)))
    assert worker._load_hybrid_request(request, remapped)
    assert copy.call_args.args[0].runtime.block_id == 6


@pytest.mark.parametrize("attn_metadata", [None, object()])
def test_start_load_cleans_all_batch_selections_on_no_metadata_or_error(
    monkeypatch, attn_metadata
):
    worker, request, execution, _, _ = _load_worker(monkeypatch)
    worker._num_kv_groups = 2
    meta = AscendConnectorMetadata(
        requests=[request],
        state_executions=[execution, replace(execution, req_id="other")],
    )
    worker._parent = SimpleNamespace(_get_connector_metadata=lambda: meta)
    worker._load_hybrid_request = Mock(side_effect=RuntimeError("load failed"))
    context = SimpleNamespace(attn_metadata=attn_metadata)
    if attn_metadata is None:
        worker.start_load_kv(context)
        worker._load_hybrid_request.assert_not_called()
        assert "r" in worker._failed_state_loads
    else:
        with pytest.raises(RuntimeError, match="load failed"):
            worker.start_load_kv(context)
    assert {
        call.args[0] for call in worker.lmcache_engine.lookup_unpin.call_args_list
    } == {"r", "other"}


def test_failed_request_cannot_publish_attention_or_state_even_if_finished_early(
    monkeypatch,
):
    worker, request, execution, _, _ = _load_worker(monkeypatch)
    worker._failed_state_loads.add("r")
    worker._wait_for_save_done = False
    worker._late_finished_sending = set()
    worker._finished_req_ids_waiting_for_save = set()
    worker.kv_role = "kv_both"
    worker.use_layerwise = False
    worker.lmcache_engine._is_passive = lambda: False
    worker.lmcache_engine.get_finished_stores = Mock(return_value=set())
    worker.lmcache_engine.store_state = Mock()
    worker.lmcache_engine.store = Mock()
    worker._local_persist_skip = Mock()
    meta = AscendConnectorMetadata(requests=[request], state_executions=[execution])
    worker._parent = SimpleNamespace(_get_connector_metadata=lambda: meta)
    monkeypatch.setattr(torch.npu, "Event", Mock)
    worker.get_finished({"r"})
    assert worker._failed_state_loads == {"r"}
    worker.wait_for_save()
    worker.lmcache_engine.store_state.assert_not_called()
    worker.lmcache_engine.store.assert_not_called()
    worker._local_persist_skip.assert_not_called()
    assert not worker._failed_state_loads and not worker._finished_state_loads


@pytest.mark.parametrize("failure", ["missing", "copy"])
def test_all_required_state_groups_preflight_and_partial_copy_failure(
    monkeypatch, caplog, failure
):
    worker, request, execution, selection, copy = _load_worker(monkeypatch)
    second = replace(worker.state_layouts[0], group_index=2, layer_names=("gdn2",))
    worker.state_layouts += (second,)
    worker.state_kv_caches["gdn2"] = worker.state_kv_caches["gdn"]
    execution = replace(
        execution,
        block_ids_by_group=execution.block_ids_by_group + ((0, 4, 6),),
        state_block_sizes=((1, 16), (2, 16)),
    )
    if failure == "copy":
        obj = SimpleNamespace(
            is_valid=lambda: True,
            meta=SimpleNamespace(state_checkpoint=state_checkpoint_metadata(second)),
        )
        selection.buffers[2] = StateCheckpointBuffer(
            second, obj, selection.buffers[1].planes
        )
        copy.side_effect = [None, RuntimeError("second group copy failed")]
        with pytest.raises(RuntimeError, match="second group"):
            worker._load_hybrid_request(request, execution)
        assert copy.call_count == 2
    else:
        assert not worker._load_hybrid_request(request, execution)
        worker.lmcache_engine.retrieve.assert_not_called()
        copy.assert_not_called()
    assert "group=2" in caplog.text
    assert "Hybrid load complete" not in caplog.text
    assert worker._failed_state_loads == {"r"}
    worker.lmcache_engine.lookup_unpin.assert_called_once_with("r")


def test_retained_buffer_reference_is_released_on_failure(monkeypatch):
    worker, request, execution, selection, copy = _load_worker(monkeypatch)
    buffer = selection.buffers[1]
    copy.side_effect = RuntimeError("copy failed")
    worker.lmcache_engine.lookup_unpin.side_effect = lambda req_id: selection.close(
        Mock()
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        worker._load_hybrid_request(request, execution)
    assert buffer._released
    buffer.memory_obj.ref_count_down.assert_called_once()


def test_state_load_device_preflight_rejects_host_runtime():
    # First Party
    from lmcache_ascend.v1.state_cache import _validate_load_device

    with pytest.raises(ValueError, match="one NPU"):
        _validate_load_device(StateBlockBinding(((torch.empty(2, 3),),), 0))


@pytest.mark.parametrize("drain_error", [False, True])
def test_hybrid_attention_copy_error_drains_and_releases_get_reference(drain_error):
    # First Party
    from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine.is_healthy = lambda: True
    engine._is_passive = lambda: False
    engine._log_kvcache_for_check = Mock()
    engine._get_req_id = lambda kwargs: "r"
    engine.async_loading = engine.save_only_first_rank = False
    engine.state_layouts = (object(),)
    profile = SimpleNamespace(
        profile_process_tokens=nullcontext, profile_to_gpu=nullcontext
    )
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda count: profile, on_retrieve_finished=Mock()
    )
    memory = Mock()
    engine._process_hybrid_tokens = Mock(return_value=([(object(), memory, 0, 16)], 16))
    copy_error = RuntimeError("Attention copy failed")
    stream = Mock()
    if drain_error:
        stream.synchronize.side_effect = RuntimeError("drain failed")
    engine.gpu_connector = SimpleNamespace(
        load_stream=stream, batched_to_gpu=Mock(side_effect=copy_error)
    )
    with pytest.raises(RuntimeError, match="Attention copy failed") as raised:
        engine.retrieve(list(range(16)))
    stream.synchronize.assert_called_once()
    memory.ref_count_down.assert_called_once()
    engine.stats_monitor.on_retrieve_finished.assert_not_called()
    if drain_error:
        assert str(raised.value.__cause__) == "drain failed"


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
    # First Party
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector import (
        LMCacheAscendConnector,
    )
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1 import (
        LMCacheAscendConnectorV1Dynamic,
    )
    from lmcache_ascend.integration.vllm.multi_group_vllm_adapter import (
        LMCacheConnectorV1Impl,
    )

    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "update_state_after_alloc",
        lambda self, request, external: None,
    )
    impl = LMCacheConnectorV1ImplMultiGroup.__new__(LMCacheConnectorV1ImplMultiGroup)
    impl._allocated_blocks = {}
    impl._state_primary_kv_group_idx = None
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
    worker._manager = SimpleNamespace(
        lmcache_engine=SimpleNamespace(
            _is_passive=lambda: passive,
            store_state=Mock(
                side_effect=RuntimeError("copy failed") if copy_error else None
            ),
            lookup_unpin=Mock(),
            metadata=SimpleNamespace(worker_id=0),
        )
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
    worker._manager = SimpleNamespace(
        lmcache_engine=SimpleNamespace(store_state=Mock())
    )

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
    worker._manager = SimpleNamespace(
        lmcache_engine=SimpleNamespace(
            metadata=SimpleNamespace(worker_id=2),
            store_state=Mock(side_effect=RuntimeError("device copy failed")),
        )
    )
    meta = SimpleNamespace(state_executions=[SimpleNamespace(req_id="r", end=32)])
    with pytest.raises(RuntimeError, match="device copy failed"):
        worker._save_state_executions(meta, object())


@pytest.mark.parametrize(
    "locations",
    [("LocalCPUBackend", "LocalDiskBackend"), ("LocalDiskBackend", "LocalDiskBackend")],
)
@pytest.mark.parametrize("failure", [True, False])
def test_hybrid_attention_acquisition_releases_prior_reads(locations, failure):
    # First Party
    from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

    first = Mock()
    first.get_size.return_value = 32
    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    pins = {}
    for key, location in zip(("a", "b"), locations, strict=True):
        pins.setdefault(location, []).append(key)
    engine.lookup_pins = {"r": pins}
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **kwargs: [(0, 16, "a"), (16, 32, "b")]
    )
    engine.storage_manager = SimpleNamespace(
        get=Mock(side_effect=[first, OSError("read failed") if failure else None])
    )
    mask = torch.zeros(32, dtype=torch.bool)
    if failure:
        with pytest.raises(OSError, match="read failed"):
            engine._process_hybrid_tokens(range(32), None, mask, req_id="r")
        first.ref_count_down.assert_called_once()
    else:
        chunks, size = engine._process_hybrid_tokens(range(32), None, mask, req_id="r")
        assert chunks == [("a", first, 0, 16)] and size == 32
        assert mask[:16].all() and not mask[16:].any()
        first.ref_count_down.assert_not_called()  # Returned owner belongs to retrieve.
        first.ref_count_down()
    assert [
        call.kwargs["location"] for call in engine.storage_manager.get.call_args_list
    ] == list(locations)


def test_hybrid_attention_does_not_fetch_unselected_key():
    # First Party
    from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine.lookup_pins = {}
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **kwargs: [(0, 16, "not-selected")]
    )
    engine.storage_manager = SimpleNamespace(get=Mock())
    mask = torch.zeros(16, dtype=torch.bool)
    assert engine._process_hybrid_tokens(range(16), None, mask, req_id="r") == ([], 0)
    engine.storage_manager.get.assert_not_called()
    assert not mask.any()
