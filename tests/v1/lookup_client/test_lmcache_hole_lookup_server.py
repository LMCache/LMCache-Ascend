# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import Mock

# First Party
from lmcache_ascend.v1.lookup_client.lmcache_hole_lookup_server import (
    LMCacheHoleLookupServer,
)


class _SegmentHelper:
    def __init__(self, ranges):
        self.ranges = ranges

    def split_ranges(self, tokens):
        return self.ranges


def _make_server(ranges, segment_results):
    server = object.__new__(LMCacheHoleLookupServer)
    server.segment_helper = _SegmentHelper(ranges)
    server._segment_hit = Mock(side_effect=segment_results)
    return server


def test_first_segment_miss_stops_lookup_and_uses_legacy_mode():
    ranges = [(0, 2), (2, 4), (4, 6)]
    server = _make_server(ranges, [(False, None)])

    result = server._lookup_tokens(list(range(6)), "request-1", None)

    assert result.mode == "legacy"
    assert result.covered_tokens == 0
    assert result.hit_flags == [False, False, False]
    server._segment_hit.assert_called_once()


def test_later_miss_checks_tail_and_can_produce_hole_mode():
    ranges = [(0, 2), (2, 4), (4, 6), (6, 8)]
    server = _make_server(
        ranges,
        [
            (True, "LocalCPUBackend"),
            (False, None),
            (True, "LocalCPUBackend"),
            (False, None),
        ],
    )

    result = server._lookup_tokens(list(range(8)), "request-2", None)

    assert result.mode == "hole"
    assert result.covered_tokens == 6
    assert result.hit_ranges == [(0, 2), (4, 6)]
    assert result.prefix_miss_ranges == [(2, 4)]
    assert server._segment_hit.call_count == 4
