# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the best-effort ``touch_cache`` overrides.

Upstream ``LocalCPUBackend.touch_cache`` / ``LocalDiskBackend.touch_cache`` call
``cache_policy.update_on_hit(key, ...)`` for every key pinned during a lookup
and only clear ``keys_in_request`` *after* the loop. When a pinned key is
removed (concurrent overwrite/eviction) between ``contains(pin=True)`` and
``touch_cache``, the LRU/MRU ``move_to_end(key)`` (or LFU ``key_to_freq[key]``)
raises ``KeyError(key)``. Because ``CacheEngine.lookup`` calls ``touch_cache``
in a ``finally`` block, that exception:

* discards the already-computed local hit count and aborts the lookup, so the
  lookup RPC handler sends no reply and the scheduler times out / recreates
  sockets, and
* leaves ``keys_in_request`` uncleared, so the stale key poisons every later
  ``touch_cache`` -- turning a transient race into a permanent failure.

The Ascend overrides make the eviction-policy bookkeeping strictly best-effort:
each key is updated independently and ``keys_in_request`` is always cleared.
"""

# Standard
from collections import OrderedDict
from types import SimpleNamespace
import threading

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.storage_backend.storage_manager import (
    local_cpu_touch_cache,
    local_disk_touch_cache,
)


class _LRULikePolicy:
    """Mimics LRU/MRU ``update_on_hit``: ``move_to_end`` KeyErrors on a missing key."""

    def __init__(self):
        self.attempted = []

    def update_on_hit(self, key, cache_dict):
        self.attempted.append(key)
        # Raises KeyError(key) when the key was evicted -- the exact upstream
        # failure this override must tolerate.
        cache_dict.move_to_end(key)


def _cpu_backend(cache, keys_in_request, policy):
    return SimpleNamespace(
        cpu_lock=threading.Lock(),
        hot_cache=cache,
        keys_in_request=list(keys_in_request),
        cache_policy=policy,
    )


def _disk_backend(cache, keys_in_request, policy):
    return SimpleNamespace(
        disk_lock=threading.Lock(),
        dict=cache,
        keys_in_request=list(keys_in_request),
        cache_policy=policy,
    )


@pytest.mark.parametrize(
    "touch, make_backend",
    [
        (local_cpu_touch_cache, _cpu_backend),
        (local_disk_touch_cache, _disk_backend),
    ],
)
class TestTouchCacheResilience:
    def test_evicted_key_does_not_raise_and_is_cleared(self, touch, make_backend):
        cache = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        policy = _LRULikePolicy()
        # "x" was pinned during lookup but evicted before touch_cache runs.
        backend = make_backend(cache, ["a", "x", "c"], policy)

        touch(backend)  # must not raise

        # keys_in_request is always cleared so the stale key cannot poison
        # subsequent lookups.
        assert backend.keys_in_request == []
        # Every key is attempted (reversed order), including the evicted one.
        assert policy.attempted == ["c", "x", "a"]
        # Present keys still get their LRU refresh; "x" is silently skipped.
        assert list(cache.keys()) == ["b", "c", "a"]

    def test_all_keys_missing_still_clears(self, touch, make_backend):
        cache = OrderedDict()
        policy = _LRULikePolicy()
        backend = make_backend(cache, ["gone1", "gone2"], policy)

        touch(backend)

        assert backend.keys_in_request == []

    def test_happy_path_refreshes_order(self, touch, make_backend):
        cache = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        policy = _LRULikePolicy()
        backend = make_backend(cache, ["a", "b"], policy)

        touch(backend)

        assert backend.keys_in_request == []
        # reversed iteration => move_to_end("b") then move_to_end("a")
        assert list(cache.keys()) == ["c", "b", "a"]

    def test_not_permanently_poisoned_after_eviction(self, touch, make_backend):
        """A transient eviction must not break subsequent lookups."""
        cache = OrderedDict([("a", 1)])
        policy = _LRULikePolicy()
        backend = make_backend(cache, ["missing"], policy)

        touch(backend)  # clears despite the KeyError
        assert backend.keys_in_request == []

        # Next lookup pins a live key; touch must behave normally.
        backend.keys_in_request = ["a"]
        touch(backend)
        assert backend.keys_in_request == []
        assert list(cache.keys()) == ["a"]
