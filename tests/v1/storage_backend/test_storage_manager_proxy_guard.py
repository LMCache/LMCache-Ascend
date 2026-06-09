# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the StorageManager delay-pull proxy write-back guard.

``lmcache_ascend.v1.storage_backend.storage_manager.get`` / ``batched_get``
mirror every non-local backend hit into ``LocalCPUBackend`` for faster reuse --
EXCEPT data-less delay-pull placeholders (``is_proxy``), which would poison the
hot cache with a stale, never-resolving entry. These tests verify that skip (and
that ordinary hits are still written back) with mocks; no NPU required.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# First Party
from lmcache_ascend.v1.storage_backend.storage_manager import batched_get, get


def _obj(is_proxy: bool) -> MagicMock:
    obj = MagicMock()
    obj.is_proxy = is_proxy
    return obj


def _manager(backend_name, *, blocking=None, batched=None, has_local=True):
    """Build a fake ``StorageManager`` whose single active backend is named
    ``backend_name`` and returns the supplied object(s)."""
    remote = MagicMock()
    remote.get_blocking.return_value = blocking
    remote.batched_get_blocking.return_value = batched
    local = MagicMock(spec=LocalCPUBackend)
    storage_backends = {"LocalCPUBackend": local} if has_local else {}
    manager = SimpleNamespace(
        get_active_storage_backends=lambda location=None: [(backend_name, remote)],
        storage_backends=storage_backends,
    )
    return manager, local


def test_get_non_proxy_is_written_back_to_local():
    obj = _obj(is_proxy=False)
    manager, local = _manager("P2PBackend", blocking=obj)

    assert get(manager, "key") is obj
    local.submit_put_task.assert_called_once()


def test_get_proxy_is_not_written_back():
    proxy = _obj(is_proxy=True)
    manager, local = _manager("P2PBackend", blocking=proxy)

    assert get(manager, "key") is proxy
    local.submit_put_task.assert_not_called()


def test_get_local_hit_is_not_rewritten():
    obj = _obj(is_proxy=False)
    manager, local = _manager("LocalCPUBackend", blocking=obj)

    assert get(manager, "key") is obj
    local.submit_put_task.assert_not_called()


def test_get_without_local_backend_is_not_written_back():
    obj = _obj(is_proxy=False)
    manager, local = _manager("P2PBackend", blocking=obj, has_local=False)

    assert get(manager, "key") is obj
    local.submit_put_task.assert_not_called()


def test_batched_get_non_proxy_is_written_back():
    objs = [_obj(is_proxy=False), _obj(is_proxy=False)]
    manager, local = _manager("P2PBackend", batched=objs)

    assert batched_get(manager, ["k1", "k2"]) == objs
    local.batched_submit_put_task.assert_called_once()


def test_batched_get_with_any_proxy_is_not_written_back():
    objs = [_obj(is_proxy=False), _obj(is_proxy=True)]
    manager, local = _manager("P2PBackend", batched=objs)

    assert batched_get(manager, ["k1", "k2"]) == objs
    local.batched_submit_put_task.assert_not_called()


def test_batched_get_with_none_is_not_written_back():
    objs = [_obj(is_proxy=False), None]
    manager, local = _manager("P2PBackend", batched=objs)

    assert batched_get(manager, ["k1", "k2"]) == objs
    local.batched_submit_put_task.assert_not_called()
