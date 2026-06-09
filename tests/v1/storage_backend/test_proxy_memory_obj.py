# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ProxyMemoryObj`` delay-pull lifecycle bookkeeping.

A delay-pull lookup returns several data-less ``ProxyMemoryObj`` placeholders
that share one ``AscendBaseTransferContext``. The context must fire its Done
signal to the sender exactly once -- after every proxy in the batch has either
been consumed by the connector or discarded. These tests exercise that
ref-counting (``ref_count_down`` -> ``decref``) with mocks; no NPU required.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
from lmcache.v1.memory_management import MemoryFormat
import torch

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj

DEFAULT_SHAPE = torch.Size([2, 2, 256, 512])
DEFAULT_DTYPE = torch.bfloat16


def _make_proxy(context: MagicMock, chunk_index: int = 0) -> ProxyMemoryObj:
    return ProxyMemoryObj(
        backing_obj=None,
        transfer_channel=MagicMock(),
        target_peer_url="target_peer_url",
        remote_buffer_uuid=f"remote-buffer-{chunk_index}",
        remote_mem_index=chunk_index,
        transfer_context=context,
        chunk_index=chunk_index,
        shapes=[DEFAULT_SHAPE],
        dtypes=[DEFAULT_DTYPE],
        fmt=MemoryFormat.KV_2LTD,
    )


def test_proxy_ref_count_down_decrefs_context_once():
    """Discarded delay-pull proxies decrement their shared context once."""
    context = MagicMock()
    proxy = _make_proxy(context)

    proxy.ref_count_down()
    proxy.ref_count_down()

    context.decref.assert_called_once()


def test_proxy_mark_consumed_suppresses_later_ref_count_down():
    """Connector-consumed proxies should not later decref during cleanup."""
    context = MagicMock()
    proxy = _make_proxy(context)

    proxy.mark_consumed()
    proxy.ref_count_down()

    context.decref.assert_not_called()


def test_proxy_shared_context_sends_done_after_all_discards():
    """A shared transfer context sends Done once all proxies are discarded."""
    # First Party
    from lmcache_ascend.v1.transfer_context import AscendBaseTransferContext

    class _StubTransferContext(AscendBaseTransferContext):
        def __init__(self):
            super().__init__(num_proxies=2)
            self.send_done = MagicMock()

        def _send_done(self):
            self.send_done()

    context = _StubTransferContext()

    proxy_0 = _make_proxy(context, chunk_index=0)
    proxy_1 = _make_proxy(context, chunk_index=1)

    proxy_0.ref_count_down()
    context.send_done.assert_not_called()

    proxy_1.ref_count_down()
    context.send_done.assert_called_once()
