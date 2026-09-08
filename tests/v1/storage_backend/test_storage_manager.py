# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
# Standard
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

# Third Party
from lmcache_tests.v1.storage_backend.test_storage_manager import (
    TestStorageManagerPrefetchCallback,
    event_manager,
    storage_manager,
    storage_manager_config,
    storage_manager_metadata,
)


def _make_copy_source():
    src = MagicMock()
    src.tensor = MagicMock()
    src.meta.fmt = "fmt"
    src.get_shape.return_value = "shape"
    src.get_dtype.return_value = "dtype"
    return src


def _make_copy_target():
    dst = MagicMock()
    dst.tensor = MagicMock()
    return dst


def test_allocate_and_copy_objects_preserves_key_alignment_after_existing_key():
    """Skipped existing keys must not shift copied objects onto wrong keys."""
    # First Party
    from lmcache_ascend.v1.storage_backend import storage_manager as sm

    keys = ["k0", "k1", "k2"]
    src_objs = [_make_copy_source(), _make_copy_source(), _make_copy_source()]
    dst_objs = [_make_copy_target(), _make_copy_target()]

    allocator = MagicMock()
    allocator.contains.side_effect = lambda key: key == "k0"
    allocator.allocate.side_effect = dst_objs

    with patch.object(sm.torch.cuda, "stream", return_value=nullcontext()):
        allocated_keys, allocated_objs = sm.allocate_and_copy_objects(
            allocator,
            keys,
            src_objs,
            stream=None,
        )

    assert allocated_keys == ["k1", "k2"]
    assert allocated_objs == dst_objs
    assert allocator.allocate.call_count == 2
    dst_objs[0].tensor.copy_.assert_called_once_with(
        src_objs[1].tensor,
        non_blocking=True,
    )
    dst_objs[1].tensor.copy_.assert_called_once_with(
        src_objs[2].tensor,
        non_blocking=True,
    )


def test_batched_contains_uses_pd_request_lease_when_context_set():
    """Pinned PD receiver lookup should create a request-scoped lease."""
    # First Party
    from lmcache_ascend.v1.storage_backend import storage_manager as sm

    keys = ["k0", "k1"]
    pd_backend = MagicMock()
    pd_backend.batched_contains_and_lease.return_value = len(keys)
    manager = MagicMock()
    manager.get_active_storage_backends.return_value = [("PDBackend", pd_backend)]

    token = sm.set_current_pd_lookup_id("req-1")
    try:
        hit_chunks, block_mapping = sm.batched_contains(manager, keys, pin=True)
    finally:
        sm.reset_current_pd_lookup_id(token)

    assert hit_chunks == len(keys)
    assert block_mapping == {"PDBackend": keys}
    pd_backend.batched_contains_and_lease.assert_called_once_with(keys, "req-1")
    pd_backend.batched_contains.assert_not_called()
