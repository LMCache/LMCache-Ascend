# SPDX-License-Identifier: Apache-2.0
"""Unified request ledger for the single-connector MP+PD design.

This module is the core increment of the monolithic
``LMCacheAscendMPPDConnector``: every request owns one :class:`RequestLedger`
that resolves three cumulative prefix coverages into a disjoint segment map
with a fixed source priority APC > L1 > PD_IN:

    APC (engine-internal, passed in as num_computed_tokens)
      -> L1  (dedup: only the part beyond APC counts)
      -> PD_IN (arrived transfer chunks beyond the L1 reach)

Live consumers (Step 3/5 wiring):

* ``get_num_new_matched_tokens`` reports ``external_tokens`` to the engine
* ``pd_receive_segments`` drives the worker's D2D load instructions
* ``describe`` feeds the connector's debug logging

The ledger is scheduler-side state.  It never moves KV itself; it only
decides *which source* serves *which token range*.  Earlier draft faces
(block bookkeeping, chunk-aligned store cadence, per-segment failure
state, and the separate SourceCoordinator plan builder) were superseded
by the live wiring — see git history (Step 1 commit ``1f33fe2``) if a
unified load plan returns with the Step-6 deferred work.
"""

# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class KVSource(Enum):
    """Where a token segment's KV comes from."""

    APC = "apc"      # engine GPU prefix cache (soft-released blocks)
    L1 = "l1"        # local lmcache server pool (ZMQ + IPC data plane)
    PD_IN = "pd_in"  # incoming PD transfer (hixl/hccl from producer)


@dataclass
class Segment:
    """A half-open token interval [start, end) with a resolved source.

    ``end`` is exclusive, matching vLLM's token-range conventions.
    """

    start: int
    end: int
    source: KVSource

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"Invalid segment: end ({self.end}) < start ({self.start})"
            )


@dataclass
class RequestLedger:
    """Per-request coverage map and scheduling decisions.

    The ledger keeps:

    * ``apc_tokens``    engine APC coverage (from num_computed_tokens)
    * ``l1_tokens``     local L1 prefix hit length (from ZMQ lookup)
    * ``pd_tokens``     union reach of (L1 + arrived PD chunks)
    * ``segments``      resolved coverage map, always non-overlapping
                        and sorted by ``start``

    The three prefix lengths are *cumulative from position 0*; the ledger
    resolves them into disjoint segments with a fixed source priority
    APC > L1 > PD_IN (a cheaper source never loses its range to a more
    expensive one, and no token range is served twice).
    """

    request_id: str

    # Cumulative prefix coverages (re-measured on each accounting pass).
    apc_tokens: int = 0
    l1_tokens: int = 0
    pd_tokens: int = 0

    # Resolved coverage map.
    segments: list[Segment] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._resolve_segments()

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------

    def update_coverage(
        self,
        apc_tokens: int,
        l1_tokens: int,
        pd_tokens: int,
    ) -> None:
        """Record a fresh measurement of the three cumulative coverages.

        Called whenever the scheduler-side connector re-runs the lookup
        pipeline for this request (LMCache lookup futures completing,
        PD arrival reports landing, ...).  The whole segment map is
        re-resolved from scratch, so no incremental state can rot.
        """
        self.apc_tokens = max(0, apc_tokens)
        self.l1_tokens = max(0, l1_tokens)
        self.pd_tokens = max(0, pd_tokens)
        self._resolve_segments()

    def _resolve_segments(self) -> None:
        """Collapse the three cumulative coverages into disjoint segments.

        All three reaches are prefixes from token 0.  Every token
        position is owned by the *cheapest* source that covers it,
        with priority APC > L1 > PD_IN (a cheaper source never loses
        its range to a more expensive one, and no range is served
        twice).  Walk the sorted breakpoints and assign each slice to
        the best covering source.
        """
        reaches = [
            (self.apc_tokens, KVSource.APC),
            (self.l1_tokens, KVSource.L1),
            (self.pd_tokens, KVSource.PD_IN),
        ]
        # Cheapest source first: for each position the first entry in
        # this order whose reach covers it wins.
        priorities = [KVSource.APC, KVSource.L1, KVSource.PD_IN]
        reach_by_source = {src: reach for reach, src in reaches}

        def owner(pos: int) -> Optional[KVSource]:
            for src in priorities:
                if pos < reach_by_source[src]:
                    return src
            return None

        breakpoints = sorted({r for r, _ in reaches if r > 0})
        segments: list[Segment] = []
        prev = 0
        for bp in breakpoints:
            src = owner(prev)
            if src is not None and bp > prev:
                # Merge with the previous segment when the owner is the
                # same source (adjacent same-source slices collapse).
                if segments and segments[-1].source is src and segments[-1].end == prev:
                    segments[-1].end = bp
                else:
                    segments.append(Segment(prev, bp, src))
            prev = bp
        self.segments = segments

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def external_tokens(self) -> int:
        """Tokens loadable through *this connector*: L1 + PD beyond APC.

        This is what ``get_num_new_matched_tokens`` reports to the
        engine.  APC coverage is already accounted for by the engine as
        ``num_computed_tokens`` and must not be double counted.
        """
        total_reach = max(self.l1_tokens, self.pd_tokens)
        return max(0, total_reach - self.apc_tokens)

    def segments_for_source(self, source: KVSource) -> list[Segment]:
        """All segments currently assigned to ``source``."""
        return [seg for seg in self.segments if seg.source is source]

    def pd_receive_segments(self) -> list[Segment]:
        """Segments the worker must land from the PD receive buffer.

        These are the arrived PD chunks beyond the L1 reach; the worker
        copies them from the receiver's NPU buffer into the paged pool.
        """
        return self.segments_for_source(KVSource.PD_IN)

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def describe(self) -> str:
        segs = ", ".join(
            f"[{s.start}:{s.end})={s.source.value}" for s in self.segments
        )
        return (
            f"Ledger({self.request_id}: apc={self.apc_tokens} "
            f"l1={self.l1_tokens} pd={self.pd_tokens} "
            f"external={self.external_tokens} | {segs})"
        )
