# SPDX-License-Identifier: Apache-2.0
"""Image/native-bypass regression; run with stdlib unittest, no NPU needed."""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace as NS
from unittest.mock import Mock
import unittest

# Third Party
from _gdn_source import selected_class

MG = "lmcache_ascend/integration/vllm/multi_group_vllm_adapter.py"
ADAPTER = "lmcache_ascend/integration/vllm/vllm_v1_adapter.py"


@dataclass
class Metadata:
    preempted_req_ids: set = field(default_factory=set)
    requests: list = field(default_factory=list)
    state_executions: list = field(default_factory=list)
    native_mm_finished_req_ids: set = field(default_factory=set)

    def add_request(self, request):
        self.requests.append(request)


class Upstream:
    def update_state_after_alloc(self, *args, **kwargs):
        raise AssertionError("Native image must not allocate external cache state")

    def start_load_kv(self, *args, **kwargs):
        pass


def scheduler():
    lookup = Mock(return_value=3072)
    cls = selected_class(
        MG,
        "LMCacheConnectorV1ImplMultiGroup",
        {
            "_bypass_multimodal_state_request",
            "get_num_new_matched_tokens",
            "update_state_after_alloc",
            "build_connector_meta",
            "_attach_state_executions",
        },
        {
            "logger": Mock(),
            "lookup_state": lookup,
            "cancel_state_lookup": Mock(),
            "extract_request_configs": lambda _: None,
            "LoadSpec": lambda *a, **kw: NS(**kw),
            "AscendConnectorMetadata": Metadata,
        },
        Upstream,
    )
    obj = cls.__new__(cls)
    obj._state_primary_kv_group_idx = 0
    obj.kv_role = "kv_both"
    obj.worker_count = 2
    obj.skip_last_n_tokens = 0
    obj.config = NS(min_retrieve_tokens=0)
    obj.lookup_client = Mock()
    obj._requests_priority = {}
    obj.load_specs = {}
    obj._allocated_blocks = {}
    obj._request_trackers = {}
    obj._unfinished_requests = {}
    obj._kv_cache_config = NS(kv_cache_groups=[])
    obj.force_skip_save = False
    obj._num_kv_groups = 2
    return obj, lookup


def request(name="image", multimodal=True):
    return NS(
        request_id=name,
        req_id=name,
        mm_features=[object()] if multimodal else [],
        all_token_ids=list(range(4096)),
        num_tokens=4096,
        num_computed_tokens=0,
        sampling_params=NS(),
    )


def output(new=(), cached=None, finished=(), preempted=()):
    return NS(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=NS(
            req_ids=[], resumed_req_ids=set(), num_computed_tokens=[]
        )
        if cached is None
        else cached,
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
    )


def worker(rank):
    cls = selected_class(
        ADAPTER,
        "LMCacheAscendConnectorV1Impl",
        {
            "get_finished",
            "start_load_kv",
            "request_finished",
            "request_finished_all_groups",
        },
        {"logger": Mock(), "LMCacheConnectorMetadata": Metadata},
        Upstream,
        optional={"_get_finished_sync"},
    )
    obj = cls.__new__(cls)
    obj.lmcache_engine = NS(
        lookup_unpin=Mock(), get_finished_stores=Mock(side_effect=lambda ids: set(ids))
    )
    obj._wait_for_save_done = True
    obj._failed_state_loads = set()
    obj._finished_state_loads = set()
    obj._late_finished_sending = set()
    obj.state_layouts = (object(),)
    obj._num_kv_groups = 1
    obj._mark_failed_p2p_loads_for_recompute = Mock()
    obj.rank = rank
    return obj


class MultimodalBypassTest(unittest.TestCase):
    def test_image_never_calls_external_lookup_or_allocation(self):
        obj, lookup = scheduler()
        req = request()
        self.assertEqual(obj.get_num_new_matched_tokens(req, 0), 0)
        obj.update_state_after_alloc(req, 0, blocks=Mock())
        lookup.assert_not_called()
        self.assertEqual(obj.load_specs, {})
        self.assertEqual(obj._allocated_blocks, {})

    def test_bypass_survives_resumed_record_without_mm_features(self):
        obj, lookup = scheduler()
        obj.get_num_new_matched_tokens(request(), 0)
        resumed = NS(req_id="image")
        self.assertTrue(obj._bypass_multimodal_state_request(resumed))
        obj.update_state_after_alloc(resumed, 0)
        lookup.assert_not_called()

    def test_new_and_cached_image_steps_have_no_transfer_metadata(self):
        obj, lookup = scheduler()
        req = request()
        first = obj.build_connector_meta(output(new=[req]))
        for cached in (
            [NS(req_id="image", num_computed_tokens=128, resumed_from_preemption=True)],
            NS(req_ids=["image"], resumed_req_ids={"image"}, num_computed_tokens=[128]),
        ):
            result = obj.build_connector_meta(output(cached=cached))
            self.assertEqual(result.requests, [])
            self.assertEqual(result.state_executions, [])
        self.assertEqual(first.requests, [])
        self.assertEqual(first.state_executions, [])
        lookup.assert_not_called()

    def test_finished_ids_are_captured_before_scheduler_cleanup(self):
        obj, _ = scheduler()
        obj.get_num_new_matched_tokens(request(), 0)
        for mapping in (
            obj.load_specs,
            obj._requests_priority,
            obj._allocated_blocks,
            obj._request_trackers,
            obj._unfinished_requests,
        ):
            mapping["image"] = object()
        meta = obj.build_connector_meta(output(finished={"image"}, preempted={"image"}))
        self.assertEqual(meta.native_mm_finished_req_ids, {"image"})
        self.assertEqual(meta.preempted_req_ids, set())
        self.assertNotIn("image", obj._state_mm_bypass_req_ids)
        self.assertEqual(obj.load_specs, {})
        self.assertEqual(obj._request_trackers, {})
        next_meta = obj.build_connector_meta(output())
        self.assertEqual(next_meta.native_mm_finished_req_ids, set())

    def test_no_forward_finished_frame_never_synthesizes_send_on_either_rank(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                obj = worker(rank)
                obj._parent = NS(
                    _get_connector_metadata=lambda: Metadata(
                        native_mm_finished_req_ids={"image"}
                    )
                )
                obj.start_load_kv(NS())
                # A naive async store engine considers every supplied finished
                # ID complete. The native ID must be removed BEFORE this query.
                for _ in range(2):
                    sends, receives = obj.get_finished({"image", "text"})
                    self.assertEqual(sends, {"text"})
                    self.assertIsNone(receives)
                    self.assertTrue(sends <= {"text"})  # scheduler ownership
                for call in obj.lmcache_engine.get_finished_stores.call_args_list:
                    self.assertNotIn("image", call.args[0])
                for call in obj.lmcache_engine.lookup_unpin.call_args_list:
                    self.assertNotEqual(call.args[0], "image")

    def test_image_only_completion_has_no_transfer_ack(self):
        obj = worker(0)
        obj._state_mm_native_finished_req_ids = {"image"}
        self.assertEqual(obj.get_finished({"image"}), (None, None))

    def test_native_request_completion_claims_no_external_ownership(self):
        obj = worker(0)
        obj._bypass_multimodal_state_request = lambda _: True
        self.assertEqual(obj.request_finished(request(), [1, 2]), (False, None))
        self.assertEqual(
            obj.request_finished_all_groups(request(), ([1], [2])), (False, None)
        )

    def test_text_lookup_remains_on_hybrid_path(self):
        obj, lookup = scheduler()
        self.assertEqual(
            obj.get_num_new_matched_tokens(request("text", False), 1024), 2048
        )
        lookup.assert_called_once()
        self.assertEqual(obj.load_specs["text"].lmcache_cached_tokens, 3072)

    def test_non_gdn_request_is_not_classified_by_this_patch(self):
        obj, _ = scheduler()
        obj._state_primary_kv_group_idx = None
        self.assertFalse(obj._bypass_multimodal_state_request(request()))


if __name__ == "__main__":
    unittest.main()
