# SPDX-License-Identifier: Apache-2.0
"""Source-bound async host tests. All framework/device objects are test doubles."""

# Standard
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import queue
import threading
import unittest

# Third Party
from _gdn_source import package, source_module

CONFIG = "lmcache_ascend/v1/gdn_async_config.py"
CONTROL = "lmcache_ascend/v1/async_control.py"
LOOKUP = "lmcache_ascend/v1/gdn_async_lookup.py"
RECEIVE = "lmcache_ascend/integration/vllm/gdn_async_receive.py"
STORE = "lmcache_ascend/v1/gdn_async_store.py"


class Config:
    def __init__(self, extra=None, **kwargs):
        self.extra = extra or {}
        self.enable_async_loading = True
        self.store_async = True
        self.__dict__.update(kwargs)

    def get_extra_config_value(self, name, default):
        return self.extra.get(name, default)

    def get_lookup_server_worker_ids(self, *args):
        return []


class HostHarness(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict("os.environ", {}, clear=True))
        self.config_module = self.stack.enter_context(
            source_module(CONFIG, "lmcache_ascend.v1.gdn_async_config")
        )
        self.control = self.stack.enter_context(
            source_module(CONTROL, "lmcache_ascend.v1.async_control")
        )
        self.log = Mock()
        self.deps = {
            "lmcache": package("lmcache"),
            "lmcache.v1": package("lmcache.v1"),
            "lmcache.logging": package(
                "lmcache.logging", init_logger=lambda _: self.log
            ),
            "lmcache_ascend": package("lmcache_ascend"),
            "lmcache_ascend.v1": package("lmcache_ascend.v1"),
            "lmcache_ascend.v1.gdn_async_config": self.config_module,
            "lmcache_ascend.v1.async_control": self.control,
        }

    def load(self, path, name, deps=None):
        return self.stack.enter_context(
            source_module(path, name, {**self.deps, **(deps or {})})
        )


class GDNHostTest(HostHarness):
    def test_legacy_launch_flags_and_budget_are_accepted(self):
        with patch.dict(
            "os.environ",
            {
                "LMCACHE_GDN_ASYNC_EXPERIMENTAL": "1",
                "LMCACHE_GDN_FULL_ASYNC_EXPERIMENTAL": "1",
                "LMCACHE_GDN_ASYNC_STAGING_MIB": "128",
            },
        ):
            self.assertTrue(self.config_module.gdn_async_store_enabled(Config()))
            self.assertTrue(self.config_module.gdn_async_restore_enabled(Config()))
            self.assertEqual(self.config_module.gdn_async_staging_mib(Config()), 128)

    def test_explicit_false_overrides_legacy_opt_in(self):
        with patch.dict(
            "os.environ",
            {
                "LMCACHE_GDN_ASYNC_EXPERIMENTAL": "1",
                "LMCACHE_GDN_FULL_ASYNC_EXPERIMENTAL": "1",
            },
        ):
            cfg = Config({"gdn_async_store": False, "gdn_async_restore": False})
            self.assertFalse(self.config_module.gdn_async_store_enabled(cfg))
            self.assertFalse(self.config_module.gdn_async_restore_enabled(cfg))

    def test_default_is_synchronous_and_bad_extra_flags_are_rejected(self):
        self.assertFalse(self.config_module.gdn_async_store_enabled(Config()))
        self.assertFalse(self.config_module.gdn_async_restore_enabled(Config()))
        for value in ("false", "true", 1, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.config_module.gdn_async_store_enabled(
                    Config({"gdn_async_store": value})
                )

    def test_explicit_budget_wins_and_invalid_budgets_fail(self):
        with patch.dict("os.environ", {"LMCACHE_GDN_ASYNC_STAGING_MIB": "128"}):
            self.assertEqual(
                self.config_module.gdn_async_staging_mib(
                    Config({"gdn_async_staging_mib": 512})
                ),
                512,
            )
        for value in (0, 63, 1025, True, 128.5, "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.config_module.gdn_async_staging_mib(
                    Config({"gdn_async_staging_mib": value})
                )

    def test_owner_creates_uses_and_closes_resource_on_one_thread(self):
        events = []

        class Resource:
            def __init__(self):
                events.append(("create", threading.get_ident()))

            def close(self):
                events.append(("close", threading.get_ident()))

        owner = self.control.SerialOwner(Resource, "test-owner")
        try:
            owner.submit(
                lambda _: events.append(("use", threading.get_ident()))
            ).result(2)
        finally:
            owner.close(2)
        self.assertEqual([x[0] for x in events], ["create", "use", "close"])
        self.assertEqual(len({x[1] for x in events}), 1)
        self.assertNotEqual(events[0][1], threading.get_ident())

    def test_owner_close_can_retry_after_full_queue_timeout(self):
        entered, release = threading.Event(), threading.Event()
        closed = Mock()
        owner = self.control.SerialOwner(
            lambda: NS(close=closed), "test-owner", max_pending=1
        )
        first = owner.submit(lambda _: (entered.set(), release.wait(2)))
        self.assertTrue(entered.wait(2))
        second = owner.submit(lambda _: "second")
        try:
            with self.assertRaises(queue.Full):
                owner.close(0)
            self.assertFalse(owner._stop_sent)
        finally:
            release.set()
            first.result(2)
            second.result(2)
            owner.close(2)
        closed.assert_called_once()
        self.assertFalse(owner._thread.is_alive())

    def test_owner_factory_failure_reaches_future(self):
        def fail():
            raise RuntimeError("factory failure")

        owner = self.control.SerialOwner(fail, "test-owner")
        try:
            with self.assertRaisesRegex(RuntimeError, "factory failure"):
                owner.submit(lambda _: None).result(2)
        finally:
            owner.close(2)

    def test_receive_ack_retains_ownership_until_drain(self):
        ledger = self.control.CompletionLedger()
        ledger.begin("r")
        self.assertEqual(ledger.drain(), set())
        ledger.complete("r")
        self.assertEqual(ledger.owned(), {"r"})
        self.assertEqual(ledger.drain(), {"r"})
        self.assertEqual(ledger.drain(), set())
        self.assertEqual(ledger.owned(), set())

    def test_failed_receive_never_becomes_successful_ack(self):
        ledger = self.control.CompletionLedger()
        ledger.begin("r")
        ledger.fail("r", RuntimeError("copy failed"))
        self.assertEqual(ledger.owned(), {"r"})
        for operation in (
            lambda: ledger.drain(),
            lambda: ledger.complete("r"),
            lambda: ledger.begin("next"),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                operation()
        self.assertEqual(ledger.owned(), {"r"})

    def test_receive_ledger_bounds_unreported_completions_and_rejects_duplicates(self):
        ledger = self.control.CompletionLedger()
        ledger.begin("r")
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            ledger.begin("r")
        ledger.complete("r")
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            ledger.begin("r")
        for i in range(127):
            ledger.begin(str(i))
            ledger.complete(str(i))
        with self.assertRaisesRegex(RuntimeError, "limit"):
            ledger.begin("overflow")

    def lookup_module(self):
        events = []

        class CoreFactory:
            @staticmethod
            def create_lookup_client(config, metadata, engine=None):
                events.append(
                    ("create", threading.get_ident(), config.enable_async_loading)
                )
                return NS(
                    close=lambda: events.append(("close", threading.get_ident(), False))
                )

            @staticmethod
            def create_lookup_server(*args):
                return "ordinary-server"

            @staticmethod
            def _create_zmq_server_transport(metadata):
                return NS(world_size=metadata.world_size)

        class CoreManager:
            def post_init(self):
                self.ordinary_called = True

            def _init_health_monitor(self):
                self.health_called = True

            def _handle_post_init_failure(self, error):
                self.failure = error

        deps = {
            "lmcache.v1.lookup_client": package("lmcache.v1.lookup_client"),
            "lmcache.v1.lookup_client.factory": package(
                "factory", LookupClientFactory=CoreFactory
            ),
            "lmcache.v1.manager": package("manager", LMCacheManager=CoreManager),
            "lmcache.v1.token_database": package(
                "token_database", ChunkedTokenDatabase=lambda *a: object()
            ),
            "lmcache_ascend.v1.state_lookup": package(
                "state_lookup",
                cancel_state_lookup=lambda c, r: events.append(
                    ("cancel:" + r, threading.get_ident(), False)
                ),
            ),
            "lmcache.v1.lookup_client.lmcache_lookup_client": package(
                "lookup_client", LMCacheLookupServer=lambda *a: ("gdn-server", a)
            ),
        }
        self.stack.enter_context(patch.dict("sys.modules", {**self.deps, **deps}))
        return self.load(LOOKUP, "gdn_host_lookup", deps), events

    def test_non_gdn_factory_and_manager_delegate_unchanged(self):
        module, events = self.lookup_module()
        cfg, meta = Config(), NS(world_size=2)
        client = module.AscendGDNLookupFactory.create_lookup_client(cfg, meta)
        client.close()
        self.assertEqual(events[0][1], threading.get_ident())
        self.assertEqual(
            module.AscendGDNLookupFactory.create_lookup_server(NS(config=cfg), meta),
            "ordinary-server",
        )
        manager = module.AscendGDNManager()
        manager.config, manager.lmcache_engine, manager._init_failed = (
            cfg,
            object(),
            False,
        )
        manager.post_init()
        self.assertTrue(manager.ordinary_called)

    def test_async_lookup_is_pending_and_cancel_is_owner_ordered(self):
        module, events = self.lookup_module()
        cfg = Config({"gdn_async_restore": True})
        client = module.AscendGDNLookupFactory.create_lookup_client(
            cfg, NS(world_size=2)
        )
        entered, release = threading.Event(), threading.Event()

        def operation(resource):
            events.append(("probe", threading.get_ident(), False))
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            return 61440

        try:
            self.assertIsNone(client.poll("r", ("signature",), operation))
            self.assertTrue(entered.wait(2))
            self.assertIsNone(client.poll("r", ("signature",), operation))
            future = client._jobs["r"][1]
            release.set()
            self.assertEqual(future.result(2), 61440)
            self.assertEqual(client.poll("r", ("signature",), operation), 61440)
            client.cancel_lookup("r")
            client._owner.submit(lambda _: None).result(2)
        finally:
            release.set()
            client.close()
        self.assertEqual(
            [e[0] for e in events], ["create", "probe", "cancel:r", "close"]
        )
        self.assertEqual(len({e[1] for e in events}), 1)
        self.assertTrue(cfg.enable_async_loading)  # never mutate shared config
        self.assertFalse(events[0][2])  # synchronous GDN wire protocol

    def test_admitted_lookup_cancel_does_not_release_worker_lease(self):
        module, events = self.lookup_module()
        client = module.AsyncStateClient(
            Config(), NS(world_size=2), module.LookupClientFactory.create_lookup_client
        )
        try:
            client.mark_admitted("r")
            client.cancel_lookup("r")
            client._owner.submit(lambda _: None).result(2)
            self.assertNotIn("cancel:r", [e[0] for e in events])
            client.received({"r"})
            self.assertNotIn("r", client._admitted)
        finally:
            client.close()

    def pressured_lookup(self):
        module, events = self.lookup_module()
        with patch.object(
            module,
            "SerialOwner",
            lambda factory, name: self.control.SerialOwner(
                factory, name, max_pending=1
            ),
        ):
            client = module.AsyncStateClient(
                Config(),
                NS(world_size=2),
                module.LookupClientFactory.create_lookup_client,
            )
        entered, release = threading.Event(), threading.Event()

        def slow_lookup(resource):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            return 256

        client.poll("running", ("running",), slow_lookup)
        self.assertTrue(entered.wait(2))
        client.poll("queued", ("queued",), lambda _: 512)
        return client, events, release

    def test_lookup_queue_pressure_returns_pending_and_retries_once(self):
        client, _, release = self.pressured_lookup()
        operation = Mock(return_value=768)
        try:
            self.assertIsNone(client.poll("retry", ("retry",), operation))
            self.assertIsNone(client.poll("retry", ("retry",), operation))
            self.assertNotIn("retry", client._jobs)
            operation.assert_not_called()
            release.set()
            client._jobs["running"][1].result(2)
            client._jobs["queued"][1].result(2)
            self.assertIsNone(client.poll("retry", ("retry",), operation))
            client._jobs["retry"][1].result(2)
            self.assertEqual(client.poll("retry", ("retry",), operation), 768)
            operation.assert_called_once()
        finally:
            release.set()
            client.close()

    def test_lookup_cancellation_survives_full_queue_and_precedes_new_lookup(self):
        client, events, release = self.pressured_lookup()
        try:
            queued = client._jobs["queued"][1]
            client.cancel_lookup("queued")
            client.cancel_lookup("queued")
            self.assertIsNone(client.poll("new", ("new",), lambda _: 768))
            release.set()
            client._jobs["running"][1].result(2)
            queued.result(2)
            client._owner._queue.join()
            client.poll("new", ("new",), lambda _: events.append(("new", 0, False)))
            client._owner._queue.join()
            if "new" not in client._jobs:
                client.poll("new", ("new",), lambda _: events.append(("new", 0, False)))
            client._jobs["new"][1].result(2)
            names = [entry[0] for entry in events]
            self.assertEqual(names.count("cancel:queued"), 1)
            self.assertLess(names.index("cancel:queued"), names.index("new"))
        finally:
            release.set()
            client.close()

    def test_lookup_close_retries_pending_cancellation_after_queue_timeout(self):
        client, events, release = self.pressured_lookup()
        try:
            queued = client._jobs["queued"][1]
            running = client._jobs["running"][1]
            with self.assertRaises(queue.Full):
                client.close(0)
            release.set()
            running.result(2)
            queued.result(2)
            client.close(2)
            names = [entry[0] for entry in events]
            self.assertEqual(names.count("cancel:queued"), 1)
            self.assertEqual(names.count("cancel:running"), 1)
            self.assertEqual(names[-1], "close")
        finally:
            release.set()
            client.close(2)

    def test_gdn_manager_initializes_once_without_ordinary_prefetch_server(self):
        module, _ = self.lookup_module()
        manager = module.AscendGDNManager()
        cfg = Config({"gdn_async_restore": True})
        engine = NS(post_init=Mock())
        manager.config, manager.lmcache_engine, manager._init_failed = (
            cfg,
            engine,
            False,
        )
        manager.post_init()
        engine.post_init.assert_called_once_with(async_lookup_server=None)
        self.assertTrue(manager.health_called)
        self.assertTrue(cfg.enable_async_loading)
        self.assertFalse(getattr(manager, "ordinary_called", False))

    def receive_module(self):
        torch = NS(
            inference_mode=nullcontext,
            npu=NS(
                set_device=lambda _: None,
                Event=lambda: NS(record=Mock()),
                current_stream=lambda: "producer",
            ),
        )
        return self.load(RECEIVE, "gdn_host_receive", {"torch": torch}), torch

    def completed_receive(self):
        module, _ = self.receive_module()
        obj = module.GDNAsyncReceiveMixin()
        obj._state_primary_kv_group_idx = 0
        obj.config = Config({"gdn_async_restore": True})
        obj._vllm_config = NS(scheduler_config=NS(async_scheduling=True))
        obj._init_gdn_receive()
        obj._get_finished_sync = Mock(side_effect=lambda ids: (set(ids) or None, None))
        meta = NS(
            gdn_async_receives=[],
            native_mm_finished_req_ids=set(),
            gdn_async_forward_req_ids=set(),
            gdn_async_finished_req_ids=set(),
        )
        obj._parent = NS(_get_connector_metadata=lambda: meta)
        obj._full_ledger.begin("r")
        obj._full_ledger.complete("r")
        return obj, meta

    def test_late_abort_after_receive_ack_is_not_a_send_completion(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                obj, meta = self.completed_receive()
                self.assertEqual(obj._gdn_finished(set()), (None, {"r"}))
                meta.gdn_async_finished_req_ids = {"r"}
                obj._gdn_start_load(NS())
                self.assertEqual(obj._gdn_finished({"r"}), (None, None))
                self.assertEqual(obj._gdn_finished({"r"}), (None, None))
                self.assertEqual(obj._full_receive_acks, set())
                meta.gdn_async_finished_req_ids = set()
                obj._gdn_start_load(NS())
                self.assertEqual(obj._full_abort_receives, set())

    def test_resumed_receive_can_report_its_later_send_completion(self):
        obj, meta = self.completed_receive()
        self.assertEqual(obj._gdn_finished(set()), (None, {"r"}))
        meta.gdn_async_forward_req_ids = {"r"}
        obj._gdn_start_load(NS())
        self.assertEqual(obj._gdn_finished({"r"}), ({"r"}, None))
        self.assertEqual(obj._full_receive_acks, set())

    def test_abort_before_receive_ack_stays_suppressed_during_repeated_polling(self):
        obj, meta = self.completed_receive()
        meta.gdn_async_finished_req_ids = {"r"}
        obj._gdn_start_load(NS())
        self.assertEqual(obj._gdn_finished({"r"}), (None, {"r"}))
        self.assertEqual(obj._gdn_finished({"r"}), (None, None))
        meta.gdn_async_finished_req_ids = set()
        obj._gdn_start_load(NS())
        self.assertEqual(obj._full_abort_receives, set())

    def test_receive_metadata_identifies_forward_and_finished_requests(self):
        obj, _ = self.completed_receive()
        obj._full_receives = {"pending": object()}
        obj._full_resume_boundaries = {"finished": 256}
        obj._build_connector_meta_sync = Mock(return_value=NS())
        meta = obj._gdn_metadata(
            NS(
                num_scheduled_tokens={"resumed": 128, "pending": 0},
                finished_req_ids={"finished"},
            )
        )
        self.assertEqual(meta.gdn_async_forward_req_ids, {"resumed"})
        self.assertEqual(meta.gdn_async_finished_req_ids, {"finished"})
        self.assertEqual(len(meta.gdn_async_receives), 1)
        self.assertEqual(obj._full_receives, {})
        self.assertEqual(obj._full_resume_boundaries, {})

    def test_receive_worker_does_not_ack_before_transfer_finishes(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                module, _ = self.receive_module()
                entered, release = threading.Event(), threading.Event()

                def transfer(impl, receive, entered=entered, release=release):
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("test gate")

                module.perform_load = transfer
                obj = module.GDNAsyncReceiveMixin()
                obj._state_primary_kv_group_idx = 0
                obj.config = Config({"gdn_async_restore": True})
                obj._vllm_config = NS(scheduler_config=NS(async_scheduling=True))
                obj._init_gdn_receive()
                obj.lmcache_engine = NS(_device_id=rank, metadata=NS(worker_id=rank))
                received = module.Receive(NS(req_id="r"), NS(start=61440))
                meta = NS(
                    gdn_async_receives=[received], native_mm_finished_req_ids={"image"}
                )
                obj._parent = NS(_get_connector_metadata=lambda meta=meta: meta)
                obj._get_finished_sync = lambda ids: (
                    set(ids) - {"image"} or None,
                    None,
                )
                try:
                    obj._gdn_start_load(NS())
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(obj._gdn_finished({"image", "r"}), (None, None))
                    release.set()
                    obj._full_queue.join()
                    self.assertEqual(obj._gdn_finished({"image", "r"}), (None, {"r"}))
                    self.assertEqual(obj._state_mm_native_finished_req_ids, {"image"})
                finally:
                    release.set()
                    obj._gdn_shutdown()

    def test_receive_failure_retains_ownership_and_refuses_shutdown(self):
        module, _ = self.receive_module()
        module.perform_load = Mock(side_effect=RuntimeError("transfer failed"))
        obj = module.GDNAsyncReceiveMixin()
        obj._state_primary_kv_group_idx = 0
        obj.config = Config({"gdn_async_restore": True})
        obj._vllm_config = NS(scheduler_config=NS(async_scheduling=True))
        obj._init_gdn_receive()
        obj.lmcache_engine = NS(_device_id=0, metadata=NS(worker_id=0))
        received = module.Receive(NS(req_id="r"), NS(start=61440))
        obj._parent = NS(
            _get_connector_metadata=lambda: NS(
                gdn_async_receives=[received], native_mm_finished_req_ids=set()
            )
        )
        obj._get_finished_sync = Mock()
        obj._gdn_start_load(NS())
        obj._full_thread.join(2)
        self.assertFalse(obj._full_thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            obj._gdn_finished({"r"})
        self.assertEqual(obj._full_ledger.owned(), {"r"})
        obj._get_finished_sync.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "outstanding"):
            obj._gdn_shutdown()

    def test_actual_restore_fences_all_groups_before_releasing_sources(self):
        module, torch = self.receive_module()
        trace = []
        torch.npu.stream = lambda _: nullcontext()
        deps = {
            "lmcache_ascend.v1.state_transfer": package(
                "state_transfer",
                transfer_state=lambda op: trace.append(
                    "state:" + str(op.checkpoint.group_index)
                ),
            )
        }
        self.stack.enter_context(patch.dict("sys.modules", deps))
        chunk = NS(ref_count_down=lambda: trace.append("release-attention"))
        selection = NS(
            boundary=61440, close=lambda _: trace.append("release-selection")
        )
        operations = [NS(checkpoint=NS(group_index=i)) for i in (1, 2, 3)]
        module.acquire_load = lambda *a: (selection, [(chunk, 0, 61440)], operations)
        stream = NS(
            wait_event=lambda _: trace.append("wait-init"),
            synchronize=lambda: trace.append("fence"),
        )
        engine = NS(
            _gdn_transfer_lock=threading.RLock(),
            metadata=NS(worker_id=0),
            storage_manager=object(),
            wait_for_pending_stores=lambda _: trace.append("drain-stores"),
            gpu_connector=NS(
                load_stream=stream,
                batched_to_gpu=lambda *a, **k: trace.append("attention"),
            ),
        )
        obj = NS(
            lmcache_engine=engine,
            _prepare_hybrid_attention=lambda *a: {},
            _full_quarantine=[],
        )
        receive = module.Receive(NS(req_id="r"), object(), object())
        module.perform_load(obj, receive)
        self.assertEqual(
            trace,
            [
                "wait-init",
                "attention",
                "state:1",
                "state:2",
                "state:3",
                "fence",
                "drain-stores",
                "release-attention",
                "release-selection",
            ],
        )

    def test_failed_device_fence_quarantines_sources_without_release(self):
        module, torch = self.receive_module()
        torch.npu.stream = lambda _: nullcontext()
        self.stack.enter_context(
            patch.dict(
                "sys.modules",
                {
                    "lmcache_ascend.v1.state_transfer": package(
                        "state_transfer", transfer_state=Mock()
                    )
                },
            )
        )
        chunk = NS(ref_count_down=Mock())
        selection = NS(boundary=61440, close=Mock())
        module.acquire_load = lambda *a: (selection, [(chunk, 0, 61440)], [])
        stream = NS(
            wait_event=Mock(),
            synchronize=Mock(side_effect=RuntimeError("device fence failed")),
        )
        engine = NS(
            _gdn_transfer_lock=threading.RLock(),
            storage_manager=object(),
            gpu_connector=NS(load_stream=stream, batched_to_gpu=Mock()),
        )
        obj = NS(
            lmcache_engine=engine,
            _prepare_hybrid_attention=lambda *a: {},
            _full_quarantine=[],
        )
        receive = module.Receive(NS(req_id="r"), object(), object())
        with self.assertRaisesRegex(RuntimeError, "device fence failed"):
            module.perform_load(obj, receive)
        chunk.ref_count_down.assert_not_called()
        selection.close.assert_not_called()
        self.assertEqual(len(obj._full_quarantine), 1)


@dataclass(frozen=True)
class Device:
    type: str = "npu"


class Plane:
    def __init__(self, values):
        self.values = list(values)
        self.device = Device()

    def __getitem__(self, key):
        return Plane(self.values[key])

    def clone(self, **kwargs):
        return Plane(self.values)


class SnapshotHostTest(HostHarness):
    def store_module(self):
        trace = []

        class Stream:
            def wait_event(self, event):
                trace.append("wait")

            def synchronize(self):
                trace.append("host-fence")

        class Event:
            def record(self, stream):
                trace.append("record")

            def synchronize(self):
                trace.append("snapshot-fence")

        class Binding:
            def __init__(self, tensors, block):
                self.tensors = tensors

            def validate(self, layout):
                pass

        producer = Stream()
        torch = NS(
            inference_mode=nullcontext,
            contiguous_format="contiguous",
            npu=NS(
                device=lambda _: nullcontext(),
                stream=lambda _: nullcontext(),
                Stream=Stream,
                Event=Event,
                current_stream=lambda: producer,
                set_device=lambda _: None,
            ),
        )
        module = self.load(
            STORE,
            "gdn_host_store",
            {
                "torch": torch,
                "lmcache_ascend.v1.state_checkpoint": package(
                    "checkpoint", StateBlockBinding=Binding
                ),
            },
        )
        engine = NS(
            config=Config({"gdn_async_store": True}),
            is_store_async=True,
            _store_queue_maxsize=8,
        )
        module.GDNAsyncStoreMixin._init_gdn_async_store(engine)
        engine.config.chunk_size = 256
        layout = NS(group_index=1, layer_names=("gdn",), planes=(NS(nbytes=8),))
        execution = NS(
            req_id="r",
            token_ids=tuple(range(256)),
            end=256,
            request_configs={"tag": [1]},
            save_blocks=lambda _: ((1, 2),),
        )
        caches = {"gdn": (Plane([10, 20, 30, 40]),)}
        return module, engine, layout, execution, caches, trace

    def test_capture_is_private_and_success_has_no_host_fence(self):
        module, engine, layout, execution, caches, trace = self.store_module()
        job = module.capture(engine, execution, (layout,), caches, object())
        caches["gdn"][0].values[2] = 999
        execution.request_configs["tag"].append(2)
        self.assertEqual(job.caches["gdn"][0].values, [30])
        self.assertEqual(job.execution.request_configs, {"tag": [1]})
        self.assertEqual(job.execution.save_blocks(256), ((1, 0),))
        self.assertEqual(trace, ["wait", "record", "wait"])
        self.assertEqual(engine._gdn_reserved, 8)

    def test_snapshot_budget_exhaustion_publishes_nothing(self):
        module, engine, layout, execution, caches, trace = self.store_module()
        engine._gdn_budget = 7
        self.assertIsNone(
            module.capture(engine, execution, (layout,), caches, object())
        )
        self.assertEqual(engine._gdn_reserved, 0)
        self.assertEqual(engine._gdn_skipped, 1)
        self.assertEqual(trace, [])

    def test_attention_and_state_share_pending_store_accounting(self):
        module, engine, layout, execution, caches, trace = self.store_module()
        job = module.capture(engine, execution, (layout,), caches, object())
        engine._device_id = 0
        engine._store_cv = threading.Condition()
        engine._pending_store_reqs = {"r": 2}
        engine._store_queue = queue.Queue()
        engine._store_queue.put(job)
        engine._store_queue.put(("r", None, None, None, None, None, {}))
        engine._store_queue.put(None)
        counts = []
        engine._store_state_sync = lambda *args: counts.append(
            engine._pending_store_reqs["r"]
        )
        engine._run_store_pipeline = lambda *args: counts.append(
            engine._pending_store_reqs["r"]
        )
        module.GDNAsyncStoreMixin._gdn_store_worker_loop(engine)
        self.assertEqual(counts, [2, 1])
        self.assertEqual(engine._pending_store_reqs, {})
        self.assertEqual(engine._gdn_reserved, 0)
        self.assertEqual(job.caches, {})
        self.assertEqual(engine._gdn_processed, 1)
        self.assertEqual(engine._gdn_kv_processed, 1)


if __name__ == "__main__":
    unittest.main()
