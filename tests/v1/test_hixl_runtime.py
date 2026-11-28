# SPDX-License-Identifier: Apache-2.0
"""Exercise real ACL context helpers with CPU-only native boundary substitutes."""

# Standard
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call
import importlib.util
import logging
import sys

# Third Party
import pytest
import torch

_TARGET_CONTEXT = 0x2000
_OTHER_CONTEXT = 0x1000


@pytest.fixture
def acl(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Load the real helper without activating unrelated Ascend package patches."""
    root = Path(__file__).resolve().parents[2] / "lmcache_ascend"
    package = ModuleType("lmcache_ascend")
    package.__path__ = [str(root)]
    native = SimpleNamespace(
        get_current_device=Mock(return_value=2),
        get_current_context=Mock(),
        set_current_context=Mock(),
    )
    npu = SimpleNamespace(set_device=Mock())
    boundary = SimpleNamespace(native=native, npu=npu, current_context=_OTHER_CONTEXT)

    def set_context(context: int) -> None:
        boundary.current_context = context

    native.get_current_context.side_effect = lambda: boundary.current_context
    native.set_current_context.side_effect = set_context
    monkeypatch.setitem(sys.modules, "lmcache_ascend", package)
    monkeypatch.setitem(sys.modules, "lmcache_ascend.hixl_npu_comms", native)
    monkeypatch.setattr(package, "hixl_npu_comms", native, raising=False)
    monkeypatch.setattr(torch, "npu", npu, raising=False)

    spec = importlib.util.spec_from_file_location(
        "acl_runtime_under_test", root / "v1/hixl_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    boundary.runtime = runtime
    return boundary


@pytest.mark.parametrize(
    "actual_device,context,error_match",
    [
        (2, _OTHER_CONTEXT, None),
        (3, _OTHER_CONTEXT, "device mismatch"),
        (2, None, "No current ACL context"),
    ],
    ids=["borrow-context", "device-mismatch", "empty-context"],
)
def test_prepare_acl_context(
    acl: SimpleNamespace,
    actual_device: int,
    context: int | None,
    error_match: str | None,
) -> None:
    """Borrow only a nonempty context on the explicitly selected device."""
    acl.native.get_current_device.return_value = actual_device
    acl.current_context = context
    if error_match is None:
        assert acl.runtime.prepare_acl_context(2) == context
    else:
        with pytest.raises(RuntimeError, match=error_match):
            acl.runtime.prepare_acl_context(2)

    acl.npu.set_device.assert_called_once_with(2)
    acl.native.get_current_device.assert_called_once_with()
    if actual_device != 2:
        acl.native.get_current_context.assert_not_called()
    else:
        acl.native.get_current_context.assert_called_once_with()


@pytest.mark.parametrize(
    "previous,body_fails",
    [(None, False), (_TARGET_CONTEXT, False), (_OTHER_CONTEXT, True)],
    ids=["no-context", "same-context-changed-by-call", "other-context-call-fails"],
)
def test_use_acl_context_restores_thread_context(
    acl: SimpleNamespace, previous: int | None, body_fails: bool
) -> None:
    """Restore a borrowed context even when entry did not change its identity."""
    acl.current_context = previous
    error = ValueError("native call failed")
    expectation = pytest.raises(ValueError) if body_fails else nullcontext()
    with expectation as caught:
        with acl.runtime.use_acl_context(_TARGET_CONTEXT):
            assert acl.current_context == _TARGET_CONTEXT
            if previous == _TARGET_CONTEXT:
                # Simulate a native call switching the current thread's context.
                acl.current_context = _OTHER_CONTEXT
            if body_fails:
                raise error
    if body_fails:
        assert caught is not None and caught.value is error

    expected_calls = [call(_TARGET_CONTEXT)]
    if previous is not None:
        expected_calls.append(call(previous))
    assert acl.native.set_current_context.call_args_list == expected_calls
    assert acl.current_context == (previous or _TARGET_CONTEXT)


@pytest.mark.parametrize("body_fails", [False, True], ids=["success", "failure"])
def test_restore_failure_preserves_call_outcome(
    acl: SimpleNamespace, caplog: pytest.LogCaptureFixture, body_fails: bool
) -> None:
    """A restoration RuntimeError must not replace a result or the original error."""
    restore_error = RuntimeError("restore failed")
    body_error = ValueError("native call failed")
    result = object()
    acl.native.set_current_context.side_effect = [None, restore_error]

    def invoke() -> object:
        with acl.runtime.use_acl_context(_TARGET_CONTEXT):
            if body_fails:
                raise body_error
            return result

    with caplog.at_level(logging.WARNING, logger=acl.runtime.logger.name):
        if body_fails:
            with pytest.raises(ValueError) as caught:
                invoke()
            assert caught.value is body_error
        else:
            assert invoke() is result

    assert acl.native.set_current_context.call_args_list == [
        call(_TARGET_CONTEXT),
        call(_OTHER_CONTEXT),
    ]
    assert any(
        record.name == acl.runtime.logger.name
        and record.levelno >= logging.WARNING
        and record.exc_info is not None
        and record.exc_info[1] is restore_error
        for record in caplog.records
    )


def test_scope_entry_failure_does_not_run_body_or_restore(acl: SimpleNamespace) -> None:
    """Failure to install the target propagates before acquiring the scope."""
    error = RuntimeError("set target failed")
    acl.native.set_current_context.side_effect = error
    body = Mock()
    with pytest.raises(RuntimeError) as caught:
        with acl.runtime.use_acl_context(_TARGET_CONTEXT):
            body()
    assert caught.value is error
    body.assert_not_called()
    acl.native.set_current_context.assert_called_once_with(_TARGET_CONTEXT)
    assert acl.current_context == _OTHER_CONTEXT
