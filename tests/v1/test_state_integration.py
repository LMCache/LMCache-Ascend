# SPDX-License-Identifier: Apache-2.0
"""State scheduling foundations; execute with the normal Ascend test bootstrap."""

# Standard
from types import SimpleNamespace

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
