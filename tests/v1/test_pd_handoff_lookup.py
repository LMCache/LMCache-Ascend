# SPDX-License-Identifier: Apache-2.0
"""Keep hybrid control operations separate from PD handoff promotion."""

# Standard
from threading import Lock
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache_ascend.v1 import state_lookup

# This module retains the original base before the upstream export is patched.
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine, LMCacheEngine
from lmcache_ascend.v1.storage_backend import storage_manager as sm


@pytest.mark.parametrize(
    "operation", [{"op": "probe", "C": 0, "upper": 256}, {"op": "release"}]
)
def test_state_lookup_does_not_promote_pd_handoff(operation):
    engine = object.__new__(AscendLMCacheEngine)
    engine._engine_state_lock = Lock()
    engine._promote_pd_handoff_lease = MagicMock()
    configs = {
        "lmcache.pd_handoff_id": "handoff-1",
        "lmcache.tag.tenant": "tenant-a",
        state_lookup.OP_KEY: operation,
    }
    original = dict(configs)

    def dispatch(actual_engine, lookup_id, op, sanitized, **inputs):
        assert actual_engine is engine
        assert engine._engine_state_lock.locked()
        assert lookup_id == "req-1"
        assert op == operation
        assert sanitized == {"lmcache.tag.tenant": "tenant-a"}
        assert inputs == {"tokens": [1, 2], "hashes": None, "offsets": None}
        return 256 if op["op"] == "probe" else 0

    with (
        patch.object(
            state_lookup, "dispatch_state_lookup", side_effect=dispatch
        ) as routed,
        patch.object(LMCacheEngine, "lookup") as ordinary,
    ):
        result = engine.lookup(
            [1, 2], lookup_id="req-1", pin=True, request_configs=configs
        )

    assert result == (256 if operation["op"] == "probe" else 0)
    routed.assert_called_once()
    ordinary.assert_not_called()
    engine._promote_pd_handoff_lease.assert_not_called()
    assert configs == original


@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_pd_lookup_promotes_before_lookup_and_restores_context(pin, fail):
    engine = object.__new__(AscendLMCacheEngine)
    engine._engine_state_lock = Lock()
    engine._is_pd_receiver = lambda: True
    engine._promote_pd_handoff_lease = MagicMock()
    configs = {"lmcache.pd_handoff_id": "handoff-1", "lmcache.tag.tenant": "tenant-a"}
    original = dict(configs)

    def lookup(**kwargs):
        assert engine._engine_state_lock.locked()
        if pin:
            engine._promote_pd_handoff_lease.assert_called_once_with(
                "handoff-1", "req-1"
            )
        else:
            engine._promote_pd_handoff_lease.assert_not_called()
        assert sm._current_pd_lookup_id.get() == ("req-1" if pin else "outer")
        assert kwargs["request_configs"] == {"lmcache.tag.tenant": "tenant-a"}
        if fail:
            raise RuntimeError("lookup failed")
        return 256

    token = sm.set_current_pd_lookup_id("outer")
    try:
        with patch.object(LMCacheEngine, "lookup", side_effect=lookup):
            if fail:
                with pytest.raises(RuntimeError, match="lookup failed"):
                    engine.lookup(
                        [1, 2], lookup_id="req-1", pin=pin, request_configs=configs
                    )
            else:
                assert (
                    engine.lookup(
                        [1, 2], lookup_id="req-1", pin=pin, request_configs=configs
                    )
                    == 256
                )
        assert sm._current_pd_lookup_id.get() == "outer"
        assert not engine._engine_state_lock.locked()
        assert configs == original
    finally:
        sm.reset_current_pd_lookup_id(token)
