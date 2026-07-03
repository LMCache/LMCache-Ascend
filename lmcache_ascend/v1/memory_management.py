# SPDX-License-Identifier: Apache-2.0
"""Ascend helpers for multi-group ``MemoryObj`` legacy metadata."""

# Third Party
import torch
from lmcache.v1.memory_management import MemoryObj


def is_multi_group_memory_obj(memory_obj: MemoryObj) -> bool:
    """Return True when ``memory_obj`` spans more than one KV layer group."""
    return len(getattr(memory_obj, "group_prefix_sum", (0,))) > 2


def maybe_normalize_multi_group_metadata(memory_obj: MemoryObj) -> None:
    """Rewrite legacy ``meta.shape``/``meta.dtype`` to a flat uint8 byte view.

    Per-group structure remains in ``meta.shapes`` / ``meta.dtypes`` for
    ``get_tensor(i)``. Idempotent for already-normalized objects.
    """
    if not is_multi_group_memory_obj(memory_obj):
        return
    meta = memory_obj.meta
    assert meta.shapes is not None and meta.dtypes is not None
    num_bytes = memory_obj.get_size()
    meta.shape = torch.Size([num_bytes])
    meta.dtype = torch.uint8
