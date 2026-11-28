# SPDX-License-Identifier: Apache-2.0
"""Borrow ACL contexts for HIXL calls without owning their lifetime."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


def prepare_acl_context(device_id: int) -> int:
    """Select a logical NPU device and return its nonempty ACL context.

    The initialization thread keeps the selected device. The caller must keep
    the borrowed context and runtime alive through HIXL cleanup.

    Raises:
        RuntimeError: Device selection fails, the device differs, or no context
            is available.
    """
    # Third Party
    import torch

    # First Party
    import lmcache_ascend.hixl_npu_comms as hixl_comms

    # torch_npu adds this namespace at runtime.
    torch.npu.set_device(device_id)  # type: ignore[attr-defined]
    actual_device = hixl_comms.get_current_device()
    if actual_device != device_id:
        raise RuntimeError(
            f"ACL device mismatch: requested {device_id}, current {actual_device}"
        )
    context = hixl_comms.get_current_context()
    if not context:
        raise RuntimeError(f"No current ACL context for device {device_id}")
    return context


@contextmanager
def use_acl_context(target_context: int) -> Iterator[None]:
    """Set the calling thread's context for one HIXL call, then restore it.

    Restore a nonempty previous context even if it equals the target: HIXL may
    change it during the call. If none existed, leave the target installed.
    Entry failures propagate; restoration failures are logged without replacing
    the call's result or exception. The caller manages Engine lifetime and locks.
    """
    # First Party
    import lmcache_ascend.hixl_npu_comms as hixl_comms

    previous = hixl_comms.get_current_context()
    hixl_comms.set_current_context(target_context)
    try:
        yield
    finally:
        if previous:
            try:
                hixl_comms.set_current_context(previous)
            except RuntimeError:
                logger.warning("Failed to restore previous ACL context", exc_info=True)
