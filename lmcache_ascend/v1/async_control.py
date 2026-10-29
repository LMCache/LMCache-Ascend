# SPDX-License-Identifier: Apache-2.0
"""Host-only async ownership primitives; no torch/NPU imports."""

# Standard
from concurrent.futures import Future
import queue
import threading
import time


class SerialOwner:
    """Create/use/close a thread-affine resource on one owner thread."""

    def __init__(self, factory, name, max_pending=128):
        self._factory = factory
        self._queue = queue.Queue(maxsize=max_pending)
        self._ready = Future()
        self._closed = False
        self._stop_sent = False
        self._guard = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, operation, *, timeout=0):
        """Enqueue work; only shutdown callers may wait for queue capacity."""
        future = Future()
        with self._guard:
            if self._closed:
                raise RuntimeError("Owner already closed")
            self._queue.put((operation, future), timeout=timeout)
        return future

    def _run(self):
        resource = None
        try:
            resource = self._factory()
            self._ready.set_result(True)
        except BaseException as error:
            self._ready.set_exception(error)
        try:
            while True:
                job = self._queue.get()
                try:
                    if job is None:
                        break
                    operation, future = job
                    if not future.set_running_or_notify_cancel():
                        continue
                    try:
                        self._ready.result()
                        future.set_result(operation(resource))
                    except BaseException as error:
                        future.set_exception(error)
                finally:
                    self._queue.task_done()
        finally:
            if resource is not None:
                resource.close()

    def close(self, timeout=120):
        if threading.current_thread() is self._thread:
            raise RuntimeError("Owner cannot join itself")
        deadline = time.monotonic() + timeout
        with self._guard:
            self._closed = True
            if not self._stop_sent:
                # If a full queue times out, a later close must retry the
                # sentinel; marking the owner closed is not proof it was sent.
                self._queue.put(None, timeout=max(0, deadline - time.monotonic()))
                self._stop_sent = True
        self._thread.join(max(0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError("Owner did not drain; resource must not be freed")


class CompletionLedger:
    """Receive ACK exists only after copies AND reference cleanup complete."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = set()
        self.done = set()
        self.error = None

    def check(self):
        with self.lock:
            self._check_locked()

    def _check_locked(self):
        if self.error:
            raise RuntimeError("GDN_FULL_ASYNC_FATAL: " + self.error)

    def begin(self, req_id):
        with self.lock:
            self._check_locked()
            if req_id in self.active or req_id in self.done:
                raise RuntimeError("Duplicate in-flight receive: " + req_id)
            if len(self.active) + len(self.done) >= 128:
                raise RuntimeError("Receive ownership limit exceeded")
            self.active.add(req_id)

    def complete(self, req_id):
        with self.lock:
            self._check_locked()
            if req_id not in self.active:
                raise RuntimeError("Receive accounting underflow")
            self.active.remove(req_id)
            self.done.add(req_id)

    def fail(self, req_id, error):
        with self.lock:
            if self.error is None:
                self.error = f"{req_id}: {type(error).__name__}: {error}"
        # Keep active ownership. Never manufacture a successful ACK.

    def owned(self):
        with self.lock:
            return self.active | self.done

    def drain(self):
        with self.lock:
            self._check_locked()
            result, self.done = self.done, set()
            return result
