# SPDX-License-Identifier: Apache-2.0
"""Ascend patches for LMCache controller worker RPCs."""

# Standard
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any
import asyncio

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (
    HeartbeatMsg,
    Msg,
    WorkerReqMsg,
    WorkerReqRetMsg,
)
import msgspec
import zmq

logger = init_logger(__name__)


def _request_timeout_s(worker: Any) -> float:
    timeout_ms = getattr(worker, "socket_recv_timeout_ms", 30000)
    return max(float(timeout_ms) / 1000.0, 0.001)


def _outer_timeout_s(worker: Any) -> float:
    # Give the worker-loop coroutine a little room to run its own timeout
    # handling and recreate the socket before the caller gives up.
    return _request_timeout_s(worker) + 1.0


def _get_req_socket_lock(worker: Any) -> asyncio.Lock:
    lock = getattr(worker, "_ascend_req_socket_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        worker._ascend_req_socket_lock = lock
    return lock


async def _async_put_and_wait_msg_on_worker_loop(
    worker: Any,
    msg: WorkerReqMsg,
) -> WorkerReqRetMsg:
    """Run controller request/reply I/O on ``worker.loop`` only.

    ``zmq.asyncio.Socket`` instances are event-loop-bound in practice.  The
    upstream worker creates/uses ``req_socket`` on ``LMCacheWorker.loop`` during
    registration, but P2P async lookup can call ``async_put_and_wait_msg`` from
    the storage-manager loop.  Keeping the DEALER socket on one loop avoids late
    wakeups where a reply is not observed until another request pokes the loop.
    """
    if isinstance(msg, HeartbeatMsg):
        return await worker._send_heartbeat_msg(msg)

    msg_type = type(msg).__name__
    async with _get_req_socket_lock(worker):
        try:
            await worker.req_socket.send_multipart([b"", msgspec.msgpack.encode(msg)])
            frames = await asyncio.wait_for(
                worker.req_socket.recv_multipart(),
                timeout=_request_timeout_s(worker),
            )
            serialized_ret_msg = frames[-1]
            ret_msg = msgspec.msgpack.decode(serialized_ret_msg, type=Msg)
            return ret_msg
        except asyncio.CancelledError:
            # Outer ``asyncio.wait_for`` / task cancellation can interrupt between
            # send and recv, leaving the DEALER socket in an inconsistent state.
            # Recreate immediately so the next request does not fail first.
            logger.warning(
                "LMCacheWorker controller request cancelled, recreating socket: "
                "worker_id=%s msg_type=%s",
                getattr(worker, "worker_id", None),
                msg_type,
            )
            worker._recreate_req_socket()
            raise
        except (asyncio.TimeoutError, zmq.Again) as e:
            logger.error(
                "LMCacheWorker controller request timed out, recreating socket: "
                "worker_id=%s msg_type=%s error=%s",
                getattr(worker, "worker_id", None),
                msg_type,
                e,
            )
            worker._recreate_req_socket()
            return worker._on_request_failure(msg)
        except zmq.ZMQError as e:
            logger.error(
                "LMCacheWorker controller request hit ZMQ error, recreating socket: "
                "worker_id=%s msg_type=%s error=%s",
                getattr(worker, "worker_id", None),
                msg_type,
                e,
            )
            worker._recreate_req_socket()
            return worker._on_request_failure(msg)
        except Exception as e:
            logger.error(
                "LMCacheWorker controller request failed: worker_id=%s "
                "msg_type=%s error=%s",
                getattr(worker, "worker_id", None),
                msg_type,
                e,
                exc_info=True,
            )
            return worker._on_request_failure(msg)


async def async_put_and_wait_msg(
    self: Any,
    msg: WorkerReqMsg,
) -> WorkerReqRetMsg:
    """Patched ``LMCacheWorker.async_put_and_wait_msg``.

    If called from another event loop, marshal the actual socket operation to
    ``self.loop`` and await the thread-safe future without blocking the caller's
    loop.  If already on ``self.loop``, execute directly.
    """
    worker_loop = getattr(self, "loop", None)
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if worker_loop is None or running_loop is worker_loop:
        return await _async_put_and_wait_msg_on_worker_loop(self, msg)

    future = asyncio.run_coroutine_threadsafe(
        _async_put_and_wait_msg_on_worker_loop(self, msg),
        worker_loop,
    )
    try:
        return await asyncio.wait_for(
            asyncio.wrap_future(future),
            timeout=_outer_timeout_s(self),
        )
    except (asyncio.TimeoutError, FutureTimeoutError) as e:
        future.cancel()
        logger.error(
            "LMCacheWorker controller request did not finish on worker loop: "
            "worker_id=%s msg_type=%s timeout_s=%.2f error=%s",
            getattr(self, "worker_id", None),
            type(msg).__name__,
            _outer_timeout_s(self),
            e,
        )
        return self._on_request_failure(msg)
