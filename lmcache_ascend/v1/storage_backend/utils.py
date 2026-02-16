# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for Ascend storage backends (PD and P2P).

Extracts common patterns used by both ``pd_backend.py`` and
``p2p_backend.py`` to reduce code duplication.
"""

# Standard
from typing import Any, Callable, List, Optional
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
import torch

logger = init_logger(__name__)


def resolve_memory_format(use_mla: bool) -> MemoryFormat:
    """Return the appropriate :class:`MemoryFormat` based on MLA usage."""
    return MemoryFormat.KV_MLA_FMT if use_mla else MemoryFormat.KV_2LTD


def build_channel_transfer_spec(
    receiver_id: str,
    remote_buffer_uuids: list[str],
    remote_mem_indexes: list[int],
) -> dict[str, Any]:
    """Build a transfer-spec dict consumed by the HCCL transfer channel."""
    return {
        "receiver_id": receiver_id,
        "remote_buffer_uuids": remote_buffer_uuids,
        "remote_mem_indexes": remote_mem_indexes,
    }


def release_memory_objects(
    mem_objs: List[MemoryObj],
    unpin: bool = False,
) -> None:
    """Call ``ref_count_down()`` (and optionally ``unpin()``) on each object."""
    for mem_obj in mem_objs:
        mem_obj.ref_count_down()
        if unpin:
            mem_obj.unpin()


def allocate_with_retry(
    allocate_fn: Callable[..., Optional[MemoryObj]],
    shape: torch.Size,
    dtype: torch.dtype,
    fmt: MemoryFormat,
    poll_interval: float = 0.01,
    timeout: float = 5.0,
) -> Optional[MemoryObj]:
    """Retry ``allocate_fn`` until it succeeds or *timeout* elapses.

    Parameters
    ----------
    allocate_fn:
        Callable with signature ``(shape, dtype, fmt) -> Optional[MemoryObj]``.
    shape, dtype, fmt:
        Arguments forwarded to *allocate_fn*.
    poll_interval:
        Seconds to sleep between retries.
    timeout:
        Maximum seconds to keep retrying.  Returns ``None`` on timeout.

    Returns
    -------
    Optional[MemoryObj]
        A successfully allocated memory object, or ``None`` if the
        allocation could not be fulfilled within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    mem_obj = allocate_fn(shape, dtype, fmt)
    while mem_obj is None:
        if time.monotonic() >= deadline:
            logger.error(
                "Memory allocation timed out after %.1fs", timeout
            )
            return None
        logger.warning("Failed to allocate memory object, retrying...")
        time.sleep(poll_interval)
        mem_obj = allocate_fn(shape, dtype, fmt)
    return mem_obj

def adjust_last_chunk_shape(
    shape: list[int],
    idx: int,
    total_allocs: int,
    fmt: MemoryFormat,
    last_chunk_toks: int,
) -> list[int]:
    """Return *shape* with the token dimension adjusted for the last chunk.

    If ``idx`` is not the last allocation, the shape is returned unchanged.
    """
    alloc_shape = list(shape)
    if idx == total_allocs - 1:
        token_dim = fmt.token_dim()
        alloc_shape[token_dim] = last_chunk_toks
    return alloc_shape
