# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import Literal, Optional

Range = tuple[int, int]
HoleMode = Literal["legacy", "pure_hit", "hole"]


@dataclass
class HoleLookupResult:
    mode: HoleMode
    covered_tokens: int
    tail_start: int
    location: Optional[str] = None
    segment_ranges: list[Range] = field(default_factory=list)
    hit_flags: list[bool] = field(default_factory=list)
    hit_ranges: list[Range] = field(default_factory=list)
    prefix_miss_ranges: list[Range] = field(default_factory=list)

    def to_wire(self) -> dict:
        return {
            "mode": self.mode,
            "covered_tokens": self.covered_tokens,
            "tail_start": self.tail_start,
            "location": self.location,
            "segment_ranges": self.segment_ranges,
            "hit_flags": self.hit_flags,
            "hit_ranges": self.hit_ranges,
            "prefix_miss_ranges": self.prefix_miss_ranges,
        }

    @classmethod
    def from_wire(cls, payload: dict) -> "HoleLookupResult":
        return cls(
            mode=payload["mode"],
            covered_tokens=payload["covered_tokens"],
            tail_start=payload["tail_start"],
            location=payload.get("location"),
            segment_ranges=[tuple(item) for item in payload["segment_ranges"]],
            hit_flags=list(payload["hit_flags"]),
            hit_ranges=[tuple(item) for item in payload["hit_ranges"]],
            prefix_miss_ranges=[tuple(item) for item in payload["prefix_miss_ranges"]],
        )


@dataclass
class HoleLoadSpec:
    mode: HoleMode
    covered_tokens: int
    tail_start: int
    hit_ranges: list[Range] = field(default_factory=list)
    prefix_miss_ranges: list[Range] = field(default_factory=list)
    vllm_cached_tokens: int = 0
    can_load: bool = False
    location: Optional[str] = None


@dataclass
class HoleSaveSpec:
    num_saved_tokens: int
    prefix_misses_saved: bool
    prefix_miss_ranges: list[Range] = field(default_factory=list)
    covered_tokens: int = 0
    num_tokens_to_save: int = 0
    can_save: bool = False
