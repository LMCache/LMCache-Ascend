# SPDX-License-Identifier: Apache-2.0
"""Opt-in snapshot-backed GDN stores.

Snapshots own private NPU tensors. Producer/capture ordering is device-side;
the existing store worker owns CPU persistence and request completion. Unknown
device failures fail closed rather than recycling potentially live buffers.
"""

# Standard
from copy import deepcopy
from dataclasses import dataclass
import queue
import threading
import time

# Third Party
from lmcache.logging import init_logger
import torch

# First Party
from lmcache_ascend.v1.gdn_async_config import (
    gdn_async_staging_mib,
    gdn_async_store_enabled,
)
from lmcache_ascend.v1.state_checkpoint import StateBlockBinding

logger = init_logger(__name__)


class CheckedQueue(queue.Queue):
    """Bound producer waits; a dead/failed worker must not hang the forward."""

    def __init__(self, engine, maxsize):
        super().__init__(maxsize=maxsize)
        self.engine = engine

    def put(self, item, block=True, timeout=None):
        if not block:
            if item is not None:
                check_failure(self.engine)
            return super().put(item, block=False)
        deadline = time.monotonic() + (30 if timeout is None else timeout)
        while True:
            if item is not None:
                check_failure(self.engine)
                worker = self.engine._store_worker_thread
                if worker is None or not worker.is_alive():
                    raise RuntimeError("Async store worker is not alive")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Full("Bounded asynchronous queue wait exceeded")
            try:
                return super().put(item, timeout=min(remaining, 0.1))
            except queue.Full:
                continue


@dataclass(frozen=True)
class FrozenExecution:
    req_id: str
    token_ids: tuple
    end: int
    request_configs: dict | None
    groups: tuple

    def save_blocks(self, chunk_size):
        if self.end <= 0 or self.end % chunk_size or len(self.token_ids) != self.end:
            raise ValueError("Invalid frozen GDN endpoint")
        # Each private tensor has exactly one block. Zero here is its real
        # physical index, NOT the original runtime's sentinel block zero.
        return tuple((group, 0) for group in self.groups)


@dataclass
class StateJob:
    execution: FrozenExecution
    layouts: tuple
    caches: dict
    ready: object
    nbytes: int
    enqueued_at: float

    @property
    def req_id(self):
        return self.execution.req_id


def check_failure(engine):
    message = getattr(engine, "_gdn_async_error", None)
    if message:
        raise RuntimeError("GDN_ASYNC_FATAL: " + message)


def latch_error(engine, error, req_id):
    # Keep a string, not a traceback retaining tensors/allocator references.
    with engine._gdn_error_lock:
        if not engine._gdn_async_error:
            engine._gdn_async_error = (
                f"request={req_id}: {type(error).__name__}: {error}"
            )
    logger.exception("GDN_ASYNC_BACKGROUND_ERROR req_id=%s; fail closed", req_id)
    with engine._store_cv:
        engine._store_cv.notify_all()


def release_budget(engine, nbytes):
    with engine._gdn_budget_lock:
        engine._gdn_reserved -= nbytes
        if engine._gdn_reserved < 0:
            raise RuntimeError("GDN staging accounting underflow")


def decrement_pending(engine, req_id):
    with engine._store_cv:
        count = engine._pending_store_reqs.get(req_id, 0)
        if count <= 0:
            raise RuntimeError("GDN pending-store accounting underflow")
        if count == 1:
            del engine._pending_store_reqs[req_id]
        else:
            engine._pending_store_reqs[req_id] = count - 1
        engine._store_cv.notify_all()


def capture(engine, execution, layouts, kv_caches, ordering_event):
    """Reserve -> freeze -> producer fence. No CPU synchronize on success."""
    blocks = tuple(execution.save_blocks(engine.config.chunk_size))
    if not blocks:
        return None
    by_group = {layout.group_index: layout for layout in layouts}
    if len(by_group) != len(layouts) or {g for g, _ in blocks} != set(by_group):
        raise ValueError("GDN snapshot requires every state group exactly once")
    if len({g for g, _ in blocks}) != len(blocks):
        raise ValueError("Duplicate GDN state group")
    devices, names = set(), set()
    nbytes = 0
    for group, block in blocks:
        layout = by_group[group]
        binding = StateBlockBinding(
            tuple(tuple(kv_caches[n]) for n in layout.layer_names), block
        )
        binding.validate(layout)
        for entry in binding.tensors:
            devices.update(t.device for t in entry)
        if names.intersection(layout.layer_names):
            raise ValueError("Duplicate GDN layer in snapshot")
        names.update(layout.layer_names)
        nbytes += sum(plane.nbytes for plane in layout.planes)
    if len(devices) != 1 or next(iter(devices)).type != "npu":
        raise ValueError("GDN snapshot requires one NPU device")
    if nbytes <= 0:
        raise ValueError("Empty GDN snapshot")
    frozen = FrozenExecution(
        execution.req_id,
        tuple(execution.token_ids),
        execution.end,
        deepcopy(execution.request_configs),
        tuple(g for g, _ in blocks),
    )
    frozen.save_blocks(engine.config.chunk_size)
    with engine._gdn_budget_lock:
        if engine._gdn_reserved + nbytes > engine._gdn_budget:
            engine._gdn_skipped += 1
            logger.warning(
                "GDN_ASYNC_SKIP req_id=%s boundary=%d reason=staging_budget "
                "bytes=%d reserved=%d limit=%d",
                execution.req_id,
                execution.end,
                nbytes,
                engine._gdn_reserved,
                engine._gdn_budget,
            )
            return None
        engine._gdn_reserved += nbytes
    device = next(iter(devices))
    snapshots = {}
    stream = None
    try:
        with torch.inference_mode(), torch.npu.device(device):
            if engine._gdn_capture_stream is None:
                engine._gdn_capture_stream = torch.npu.Stream()
            stream = engine._gdn_capture_stream
            producer = torch.npu.current_stream()
            with torch.npu.stream(stream):
                stream.wait_event(ordering_event)
                for group, block in blocks:
                    for name in by_group[group].layer_names:
                        snapshots[name] = tuple(
                            t[block : block + 1].clone(
                                memory_format=torch.contiguous_format
                            )
                            for t in kv_caches[name]
                        )
                ready = torch.npu.Event()
                ready.record(stream)
            # Device-side dependency, not a host wait. Next forward/movement
            # on this producer stream cannot overwrite source before freezing.
            producer.wait_event(ready)
        return StateJob(
            frozen, tuple(layouts), snapshots, ready, nbytes, time.monotonic()
        )
    except BaseException:
        # A partly submitted clone may still own source/snapshot memory.
        if stream is not None:
            try:
                stream.synchronize()
            except BaseException:
                engine._gdn_quarantine.append(snapshots)
                raise
        snapshots.clear()
        release_budget(engine, nbytes)
        raise


def wait_pending(engine, req_ids, timeout=120):
    requested = set(req_ids)
    with engine._store_cv:
        initially = requested.intersection(engine._pending_store_reqs)
        deadline = time.monotonic() + timeout
        while requested.intersection(engine._pending_store_reqs):
            check_failure(engine)
            worker = engine._store_worker_thread
            if worker is None or not worker.is_alive():
                raise RuntimeError("Async store worker stopped with pending jobs")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Async stores did not drain; blocks must not be reused"
                )
            engine._store_cv.wait(timeout=min(remaining, 0.1))
        check_failure(engine)
    return initially


class GDNAsyncStoreMixin:
    """Private helpers used explicitly by AscendLMCacheEngine."""

    def _init_gdn_async_store(self):
        self._gdn_async_enabled = gdn_async_store_enabled(self.config)
        if not self._gdn_async_enabled:
            return
        if not self.is_store_async:
            raise ValueError("gdn_async_store requires store_async=true")
        if not 1 <= self._store_queue_maxsize <= 32:
            raise ValueError("GDN asynchronous stores require queue size 1..32")
        self._gdn_budget_lock = threading.Lock()
        self._gdn_error_lock = threading.Lock()
        self._gdn_reserved = 0
        self._gdn_enqueued = self._gdn_processed = self._gdn_skipped = (
            self._gdn_kv_processed
        ) = 0
        self._gdn_quarantine = []
        self._gdn_capture_stream = None
        self._gdn_async_error = None
        self._gdn_stopping = False
        self._gdn_stop_sent = False
        mib = gdn_async_staging_mib(self.config)
        self._gdn_budget = mib * 1024**2

    def _gdn_ensure_store_worker(self):
        check_failure(self)
        if self._store_queue is not None:
            if (
                self._store_worker_thread is None
                or not self._store_worker_thread.is_alive()
            ):
                raise RuntimeError("Async worker stopped unexpectedly")
            return
        if self._gdn_stopping:
            raise RuntimeError("Cannot start a store worker during shutdown")
        self._store_queue = CheckedQueue(self, self._store_queue_maxsize)
        self._store_worker_thread = threading.Thread(
            target=self._store_worker_loop, daemon=True, name="lmcache-gdn-async-store"
        )
        self._store_worker_thread.start()

    def _gdn_store_state(self, execution, layouts, kv_caches, ordering_event):
        check_failure(self)
        if self._gdn_stopping:
            raise RuntimeError("GDN store attempted during shutdown")
        if self._is_passive() or not self.is_healthy() or self.is_frozen():
            return
        if self.store_location not in (None, "LocalCPUBackend"):
            raise ValueError("Asynchronous GDN snapshots support CPU storage only")
        self._ensure_store_worker()
        if self._store_queue.full():
            self._gdn_skipped += 1
            logger.warning(
                "GDN_ASYNC_SKIP req_id=%s boundary=%d reason=queue_full",
                execution.req_id,
                execution.end,
            )
            return
        started = time.monotonic()
        try:
            work = capture(self, execution, tuple(layouts), kv_caches, ordering_event)
        except BaseException as error:
            latch_error(self, error, execution.req_id)
            raise
        if work is None:
            return
        with self._store_cv:
            self._pending_store_reqs[work.req_id] = (
                self._pending_store_reqs.get(work.req_id, 0) + 1
            )
        try:
            self._store_queue.put_nowait(work)
        except BaseException as enqueue_error:
            # Rare concurrent producer race. Drain ONLY this snapshot before
            # releasing budget; never publish it as stored.
            try:
                work.ready.synchronize()
            except BaseException as error:
                self._gdn_quarantine.append(work)
                latch_error(self, error, work.req_id)
                raise
            work.caches.clear()
            release_budget(self, work.nbytes)
            decrement_pending(self, work.req_id)
            self._gdn_skipped += 1
            logger.warning(
                "GDN_ASYNC_SKIP req_id=%s reason=queue_full_after_capture", work.req_id
            )
            if not isinstance(enqueue_error, queue.Full):
                raise
        else:
            self._gdn_enqueued += 1
            logger.info(
                "GDN_ASYNC_ENQUEUED req_id=%s boundary=%d bytes=%d "
                "capture_enqueue_ms=%.3f",
                work.req_id,
                work.execution.end,
                work.nbytes,
                (time.monotonic() - started) * 1000,
            )
        check_failure(self)

    def _gdn_store_worker_loop(self):
        try:
            if self._device_id is None:
                raise RuntimeError(
                    "NPU device was not bound before starting async worker"
                )
            torch.npu.set_device(self._device_id)
        except BaseException as error:
            latch_error(self, error, "worker-init")
            return
        while True:
            work = self._store_queue.get()
            if work is None:
                self._store_queue.task_done()
                break
            is_state = isinstance(work, StateJob)
            req_id = work.req_id if is_state else work[0]
            safe_to_release = True
            try:
                if is_state:
                    # Only the background thread waits for capture/CPU copies.
                    work.ready.synchronize()
                check_failure(self)
                with torch.inference_mode():
                    if is_state:
                        delay = (time.monotonic() - work.enqueued_at) * 1000
                        self._store_state_sync(
                            work.execution, work.layouts, work.caches, work.ready
                        )
                        self._gdn_processed += 1
                        logger.info(
                            "GDN_ASYNC_PROCESSED req_id=%s boundary=%d "
                            "queue_delay_ms=%.3f; "
                            "verify Stored state checkpoint for actual publication",
                            req_id,
                            work.execution.end,
                            delay,
                        )
                    else:
                        self._run_store_pipeline(*work)
                        self._gdn_kv_processed += 1
            except BaseException as error:
                latch_error(self, error, req_id)
                # If a native operation partially submitted, drain before freeing
                # tensors/blocks. If draining fails, retain ownership and fail closed.
                try:
                    if is_state:
                        work.ready.synchronize()
                    self.gpu_connector.store_stream.synchronize()
                except BaseException:
                    safe_to_release = False
                    self._gdn_quarantine.append(work)
                    logger.exception(
                        "GDN_ASYNC_QUARANTINED req_id=%s; storage must not close",
                        req_id,
                    )
            finally:
                if safe_to_release:
                    if is_state:
                        work.caches.clear()
                        release_budget(self, work.nbytes)
                    decrement_pending(self, req_id)
                del work  # Do not retain the final job's GPU tensors while idle.
                self._store_queue.task_done()

    def _gdn_close_store(self):
        if self.is_store_async:
            self._gdn_stopping = True
            worker, work_queue = self._store_worker_thread, self._store_queue
            if work_queue is not None:
                deadline = time.monotonic() + 60
                if not self._gdn_stop_sent:
                    while True:
                        if worker is None or not worker.is_alive():
                            raise RuntimeError(
                                "Cannot close allocator: async worker is dead"
                            )
                        try:
                            work_queue.put(None, timeout=0.1)
                            self._gdn_stop_sent = True
                            break
                        except queue.Full:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "Cannot close allocator: store queue did not drain"
                                ) from None
                worker.join(timeout=max(0, deadline - time.monotonic()))
                if (
                    worker.is_alive()
                    or self._gdn_quarantine
                    or self._pending_store_reqs
                ):
                    raise RuntimeError(
                        "Cannot close allocator while asynchronous operations remain"
                    )
                self._store_queue = self._store_worker_thread = None
