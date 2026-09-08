# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Unit tests for host-staging H2H copy cancellation safety.

These tests exercise ``HcclChannel._async_h2h_copy`` on a minimally
constructed instance (no live HCCL/NPU traffic). Import still requires
the extension module to be present.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import asyncio
import threading
import time

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache_ascend.v1.transfer_channel.hccl_channel import HcclChannel

    _hccl_channel_available = True
except ImportError:
    _hccl_channel_available = False
    HcclChannel = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(
    not _hccl_channel_available,
    reason="hccl channel extension not built (set HCOMM_SRC_PATH at build time)",
)

# How long a parked fake copy waits to be released. Only a safety net so a
# broken test cannot hang the suite; the test always releases explicitly.
_HOLD_TIMEOUT_S = 10.0
# How long we keep asserting the caller stays blocked while workers are live.
_OBSERVE_S = 0.5
# Duration of the slow sibling slice in the failing-slice test.
_SIBLING_COPY_S = 0.5


class _ParkedCopy:
    """Fake ``torch._foreach_copy_`` that parks *inside* the copy body.

    Makes worker liveness directly observable instead of assumed:
    ``live_workers`` counts threads currently inside the copy, and no worker
    can leave until :meth:`release` is called. A test can therefore assert
    "N threads are still writing" at the same moment it asserts the caller
    has not been resumed.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self._all_inside = threading.Barrier(num_workers + 1)
        self._gate = threading.Event()
        self._lock = threading.Lock()
        self._live = 0
        self.finished: list[str] = []

    def __call__(self, dst_slice, src_slice) -> None:
        with self._lock:
            self._live += 1
        try:
            self._all_inside.wait(timeout=_HOLD_TIMEOUT_S)
            if not self._gate.wait(timeout=_HOLD_TIMEOUT_S):
                raise AssertionError("copy gate was never released by the test")
            with self._lock:
                self.finished.append(threading.current_thread().name)
        finally:
            with self._lock:
                self._live -= 1

    @property
    def live_workers(self) -> int:
        with self._lock:
            return self._live

    def wait_until_all_inside(self) -> None:
        """Block until every worker has entered the copy body."""
        self._all_inside.wait(timeout=_HOLD_TIMEOUT_S)

    def release(self) -> None:
        self._gate.set()


def _make_channel(*, copy_threads: int = 2) -> HcclChannel:
    channel = object.__new__(HcclChannel)
    channel._os_staging_copy_threads = copy_threads
    channel._staging_copy_pool = ThreadPoolExecutor(
        max_workers=copy_threads,
        thread_name_prefix="test-staging-copy",
    )
    return channel


def _run(coro):
    return asyncio.run(coro)


async def _assert_blocked_while_workers_live(task, copy: _ParkedCopy, seconds: float):
    """Assert the caller stays blocked for *seconds* while workers are live.

    Both halves matter: ``live_workers`` proves the copy threads really are
    still inside the copy, and ``task.done()`` proves the caller has not been
    resumed (which is what would let it hand the staging page back).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        live = copy.live_workers
        assert live == copy.num_workers, (
            f"expected {copy.num_workers} copy threads still inside the copy, "
            f"saw {live} (gate has not been released yet)"
        )
        assert not task.done(), (
            f"caller resumed while {live} copy thread(s) were still writing"
        )
        await asyncio.sleep(0.02)


class TestAsyncH2HCopyDrain:
    def test_cancel_waits_for_executor_workers(self, monkeypatch):
        """Cancelled await must not return while H2H workers still write.

        Sync-get timeout cancels ``_handle_pull_mode_transfer`` during
        ``copy_receiver_staging_to``. Callers then ``release_staged()``.
        If ThreadPoolExecutor workers are orphaned, arena freelist reuse
        UAFs the pages under in-flight ``torch._foreach_copy_``.
        """
        channel = _make_channel(copy_threads=2)
        copy = _ParkedCopy(num_workers=2)
        monkeypatch.setattr(torch, "_foreach_copy_", copy)

        src = [torch.ones(8), torch.ones(8)]
        dst = [torch.zeros(8), torch.zeros(8)]

        async def _cancel_mid_copy():
            loop = asyncio.get_running_loop()
            task = asyncio.create_task(channel._async_h2h_copy(src, dst))
            await loop.run_in_executor(None, copy.wait_until_all_inside)

            # Release inside the coroutine even on failure: parked workers
            # would otherwise stall asyncio.run()'s executor shutdown until
            # the _HOLD_TIMEOUT_S safety net fires.
            try:
                task.cancel()
                await _assert_blocked_while_workers_live(task, copy, _OBSERVE_S)
            finally:
                copy.release()

            with pytest.raises(asyncio.CancelledError):
                await task
            # Every worker ran to completion before the await returned.
            assert len(copy.finished) == 2
            assert copy.live_workers == 0

        try:
            _run(_cancel_mid_copy())
        finally:
            copy.release()
            channel._staging_copy_pool.shutdown(wait=True)

    def test_repeated_cancel_still_drains(self, monkeypatch):
        """Repeated cancels must not resume the caller before workers finish.

        A sync-get timeout can be followed by a second cancel (e.g. loop
        teardown cancelling pending tasks). A single shielded await would
        resume on the second cancel, freeing pages under live writers.
        """
        channel = _make_channel(copy_threads=2)
        copy = _ParkedCopy(num_workers=2)
        monkeypatch.setattr(torch, "_foreach_copy_", copy)

        src = [torch.ones(8), torch.ones(8)]
        dst = [torch.zeros(8), torch.zeros(8)]

        async def _cancel_repeatedly():
            loop = asyncio.get_running_loop()
            task = asyncio.create_task(channel._async_h2h_copy(src, dst))
            await loop.run_in_executor(None, copy.wait_until_all_inside)

            # Cancel repeatedly, and keep checking across the whole window.
            try:
                for _ in range(3):
                    task.cancel()
                    await _assert_blocked_while_workers_live(task, copy, _OBSERVE_S / 3)
            finally:
                copy.release()

            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(copy.finished) == 2
            assert copy.live_workers == 0

        try:
            _run(_cancel_repeatedly())
        finally:
            copy.release()
            channel._staging_copy_pool.shutdown(wait=True)

    def test_exception_in_one_slice_drains_siblings(self, monkeypatch):
        """A failing slice must not freelist while sibling workers still run."""
        channel = _make_channel(copy_threads=2)
        sibling_started = threading.Event()
        sibling_done = threading.Event()
        call_count = {"n": 0}
        lock = threading.Lock()

        def flaky_foreach(dst_slice, src_slice):
            with lock:
                call_count["n"] += 1
                idx = call_count["n"]
            if idx == 1:
                sibling_started.set()
                # Slow sibling — must finish before _async_h2h_copy raises.
                time.sleep(_SIBLING_COPY_S)
                sibling_done.set()
                return
            # Fast failing slice: wait until sibling has started, then raise.
            assert sibling_started.wait(timeout=_HOLD_TIMEOUT_S)
            raise RuntimeError("boom-slice")

        monkeypatch.setattr(torch, "_foreach_copy_", flaky_foreach)

        src = [torch.ones(4), torch.ones(4)]
        dst = [torch.zeros(4), torch.zeros(4)]

        async def _run_copy():
            with pytest.raises(RuntimeError, match="boom-slice"):
                await channel._async_h2h_copy(src, dst)
            assert sibling_done.is_set(), (
                "error surfaced while the sibling slice was still copying"
            )

        try:
            _run(_run_copy())
        finally:
            channel._staging_copy_pool.shutdown(wait=True)


class TestStageCancelReleasesArena:
    def test_stage_cancelled_error_releases_staged(self, monkeypatch):
        """Cancelled stage() must return arena pages (after H2H drain)."""
        channel = _make_channel(copy_threads=1)
        channel._use_host_staging = True
        channel._staging_arena = MagicMock()
        channel._staging_lock = threading.Lock()

        slot = MagicMock()
        slot.tensor = torch.zeros(4)
        slot.meta = MagicMock(shape=(4,), dtype=torch.float32, fmt=MagicMock())
        channel._staging_arena.allocate.return_value = slot

        released = []

        def _release(objs):
            released.extend(objs)

        channel.release_staged = _release  # type: ignore[method-assign]
        channel.get_local_buffer_refs = MagicMock(return_value=([], []))

        async def cancel_copy(*_args, **_kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(channel, "_async_h2h_copy", cancel_copy)

        src_obj = MagicMock()
        src_obj.tensor = torch.ones(4)
        src_obj.meta = slot.meta

        async def _run_stage():
            with pytest.raises(asyncio.CancelledError):
                await channel.stage([src_obj])

        try:
            _run(_run_stage())
            assert released == [slot]
        finally:
            channel._staging_copy_pool.shutdown(wait=True)
