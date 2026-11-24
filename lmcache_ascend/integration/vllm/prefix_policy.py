# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side load/recompute policy for experimental LMCache lookups.

This module deliberately has no vLLM, torch, or LMCache imports so that policy
semantics can be tested without an NPU. It does not observe worker execution:
``h_lookup`` is a lookup result, never proof that KV was actually loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from typing import Any, Mapping, Optional


ENV_MODE = "LMCACHE_PREFIX_POLICY_MODE"
ENV_STATIC_H = "LMCACHE_PREFIX_POLICY_STATIC_H"
ENV_STATIC_H2 = "LMCACHE_PREFIX_POLICY_STATIC_H2"
ENV_LOG_DECISION = "LMCACHE_PREFIX_POLICY_LOG_DECISION"
STATIC_H_THRESHOLD = 1024


def log_decision_enabled(env_value=None):
    """Per-request decision-log switch for the vLLM v1 adapter.

    ``LMCACHE_PREFIX_POLICY_LOG_DECISION`` defaults to ``"0"`` (disabled): the
    per-request ``LMCACHE_PREFIX_POLICY_DECISION`` JSON line is a measurement
    aid, not a production hot-path log. Set it to ``"1"`` to enable the line;
    the decide/apply path and all return values are identical either way.
    """
    if env_value is None:
        env_value = os.environ.get(ENV_LOG_DECISION, "0")
    return env_value == "1"


class PrefixPolicyError(ValueError):
    """Raised when policy configuration or scheduling semantics are invalid."""


class PrefixPolicyMode(str, Enum):
    OFF = "OFF"
    FORCED_LOAD = "FORCED_LOAD"
    FORCED_RECOMPUTE = "FORCED_RECOMPUTE"
    STATIC_H = "STATIC_H"


class PrefixPath(str, Enum):
    NATIVE = "NATIVE"
    LOAD = "LOAD"
    RECOMPUTE = "RECOMPUTE"


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrefixPolicyError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise PrefixPolicyError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class PrefixDecision:
    path: PrefixPath
    reason: str

    def log_fields(self) -> dict[str, Any]:
        return {
            "chosen_path": self.path.value,
            "decision_reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyApplication:
    returned_external_tokens: int
    policy_applied: bool


class PrefixPolicy:
    """Pure decision logic plus the native return-0 recompute primitive."""

    def __init__(
        self,
        mode: PrefixPolicyMode = PrefixPolicyMode.OFF,
        *,
        static_h_threshold: int = STATIC_H_THRESHOLD,
        static_h_threshold2: Optional[int] = None,
    ) -> None:
        self.mode = mode
        self.static_h_threshold = _integer(
            "static_h_threshold", static_h_threshold, minimum=1
        )
        if static_h_threshold2 is not None:
            static_h_threshold2 = _integer(
                "static_h_threshold2", static_h_threshold2, minimum=1
            )
            if static_h_threshold2 <= self.static_h_threshold:
                raise PrefixPolicyError(
                    f"{ENV_STATIC_H2}={static_h_threshold2} must be > "
                    f"{ENV_STATIC_H}={self.static_h_threshold}"
                )
        self.static_h_threshold2 = static_h_threshold2

    @property
    def enabled(self) -> bool:
        return self.mode != PrefixPolicyMode.OFF

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "PrefixPolicy":
        env = os.environ if environ is None else environ
        raw_mode = env.get(ENV_MODE, PrefixPolicyMode.OFF.value).strip().upper()
        try:
            mode = PrefixPolicyMode(raw_mode)
        except ValueError as exc:
            choices = ", ".join(item.value for item in PrefixPolicyMode)
            raise PrefixPolicyError(
                f"{ENV_MODE}={raw_mode!r} is invalid; choose one of: {choices}"
            ) from exc
        if mode == PrefixPolicyMode.STATIC_H:
            raw_threshold = env.get(ENV_STATIC_H)
            t1 = STATIC_H_THRESHOLD
            if raw_threshold is not None and raw_threshold.strip():
                try:
                    t1 = int(raw_threshold.strip(), 10)
                except ValueError as exc:
                    raise PrefixPolicyError(
                        f"{ENV_STATIC_H}={raw_threshold!r} must be a positive integer"
                    ) from exc
                t1 = _integer(ENV_STATIC_H, t1, minimum=1)
            raw_threshold2 = env.get(ENV_STATIC_H2)
            if raw_threshold2 is None or not raw_threshold2.strip():
                # H2 unset -> legacy single-threshold behavior (backward compatible)
                return cls(mode, static_h_threshold=t1)
            try:
                t2 = int(raw_threshold2.strip(), 10)
            except ValueError as exc:
                raise PrefixPolicyError(
                    f"{ENV_STATIC_H2}={raw_threshold2!r} must be a positive integer"
                ) from exc
            t2 = _integer(ENV_STATIC_H2, t2, minimum=1)
            return cls(mode, static_h_threshold=t1, static_h_threshold2=t2)
        return cls(mode)

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mode": self.mode.value,
        }
        if self.mode == PrefixPolicyMode.STATIC_H:
            result["static_h_threshold"] = self.static_h_threshold
            if self.static_h_threshold2 is not None:
                result["static_h_threshold2"] = self.static_h_threshold2
        return result

    def decide(
        self, *, h_lookup: int, full_prompt: int, num_computed_tokens: int
    ) -> PrefixDecision:
        h_lookup = _integer("h_lookup", h_lookup)
        full_prompt = _integer("full_prompt", full_prompt, minimum=1)
        num_computed_tokens = _integer("num_computed_tokens", num_computed_tokens)
        if h_lookup > full_prompt:
            raise PrefixPolicyError(
                f"h_lookup={h_lookup} exceeds full_prompt={full_prompt}"
            )
        if num_computed_tokens > full_prompt:
            raise PrefixPolicyError(
                f"num_computed_tokens={num_computed_tokens} exceeds "
                f"full_prompt={full_prompt}"
            )

        if self.mode == PrefixPolicyMode.OFF:
            return PrefixDecision(PrefixPath.NATIVE, "policy_off")
        if self.mode == PrefixPolicyMode.FORCED_LOAD:
            return PrefixDecision(PrefixPath.LOAD, "forced_load")
        if self.mode == PrefixPolicyMode.FORCED_RECOMPUTE:
            return PrefixDecision(PrefixPath.RECOMPUTE, "forced_recompute")
        return self._static_decision(
            h_lookup,
            self.static_h_threshold,
            "static_h",
            threshold2=self.static_h_threshold2,
        )

    @staticmethod
    def _static_decision(
        h_lookup: int,
        threshold: int,
        reason: str,
        threshold2: Optional[int] = None,
    ) -> PrefixDecision:
        """Single threshold (legacy) or dual-threshold window.

        Dual-threshold window semantics (fixed):
          H == tau1 -> LOAD;  H == tau2 -> LOAD;
          tau1 < H < tau2 -> RECOMPUTE;  H <= tau1 -> LOAD;  H >= tau2 -> LOAD.
        With threshold2=None this degrades to the legacy single-threshold rule
        (H <= tau1 -> LOAD else RECOMPUTE).
        """
        if threshold2 is None:
            if h_lookup <= threshold:
                return PrefixDecision(PrefixPath.LOAD, f"{reason}:H<={threshold}")
            return PrefixDecision(PrefixPath.RECOMPUTE, f"{reason}:H>{threshold}")
        if h_lookup <= threshold:
            return PrefixDecision(PrefixPath.LOAD, f"{reason}_window:H<={threshold}")
        if h_lookup >= threshold2:
            return PrefixDecision(PrefixPath.LOAD, f"{reason}_window:H>={threshold2}")
        return PrefixDecision(
            PrefixPath.RECOMPUTE,
            f"{reason}_window:{threshold}<H<{threshold2}",
        )

    @staticmethod
    def apply(
        *,
        native_matched_tokens: Optional[int],
        load_spec: Any,
        decision: PrefixDecision,
    ) -> PolicyApplication:
        """Apply a decision using upstream LMCache's native scheduler contract.

        For a real external hit, RECOMPUTE returns zero newly matched external
        tokens and leaves ``lmcache_cached_tokens`` unchanged. The latter remains
        available to RequestTracker/store skip logic. ``can_load`` stays false;
        update_state_after_alloc() will keep it false when zero is allocated.
        """
        if native_matched_tokens is None:
            raise PrefixPolicyError(
                "async lookup (native_matched_tokens=None) is unsupported by "
                "Build E active modes; use synchronous lookup or OFF"
            )
        native = _integer("native_matched_tokens", native_matched_tokens)
        if decision.path != PrefixPath.RECOMPUTE or native == 0:
            return PolicyApplication(native, False)
        if load_spec is None:
            raise PrefixPolicyError(
                "native lookup returned external tokens without a LoadSpec; "
                "refusing to claim a safe full-recompute transition"
            )
        h_before = _integer(
            "load_spec.lmcache_cached_tokens",
            getattr(load_spec, "lmcache_cached_tokens", None),
        )
        if h_before == 0:
            raise PrefixPolicyError(
                "native lookup returned external tokens with lmcache_cached_tokens=0"
            )
        load_spec.can_load = False
        h_after = load_spec.lmcache_cached_tokens
        if h_after != h_before:
            raise PrefixPolicyError("lmcache_cached_tokens changed unexpectedly")
        return PolicyApplication(0, True)
