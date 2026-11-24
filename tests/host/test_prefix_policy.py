# SPDX-License-Identifier: Apache-2.0
"""Host-only Build E tests; no vLLM, LMCache engine, torch, or NPU required.

These tests follow the repository host-test convention (see tests/host/README.md):
the policy source is loaded by file path instead of importing the package, so
``python -m unittest discover -s tests/host`` works on a bare checkout without
the setuptools-scm generated ``lmcache_ascend/_version`` module.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import importlib.util
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "prefix_policy_under_test",
    ROOT / "lmcache_ascend" / "integration" / "vllm" / "prefix_policy.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD  # must be registered before exec: dataclass
# decoration resolves names through sys.modules
_SPEC.loader.exec_module(_MOD)

ENV_LOG_DECISION = _MOD.ENV_LOG_DECISION
ENV_MODE = _MOD.ENV_MODE
ENV_STATIC_H = _MOD.ENV_STATIC_H
ENV_STATIC_H2 = _MOD.ENV_STATIC_H2
PrefixPath = _MOD.PrefixPath
PrefixPolicy = _MOD.PrefixPolicy
PrefixPolicyError = _MOD.PrefixPolicyError
PrefixPolicyMode = _MOD.PrefixPolicyMode
log_decision_enabled = _MOD.log_decision_enabled


@dataclass
class FakeLoadSpec:
    lmcache_cached_tokens: int
    can_load: bool = False
    vllm_cached_tokens: int = 0


class PrefixPolicyTest(unittest.TestCase):
    def test_unknown_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(PrefixPolicyError, ENV_MODE):
            PrefixPolicy.from_env({ENV_MODE: "P3"})

    def test_forced_modes(self) -> None:
        load = PrefixPolicy(PrefixPolicyMode.FORCED_LOAD)
        recompute = PrefixPolicy(PrefixPolicyMode.FORCED_RECOMPUTE)
        kwargs = dict(h_lookup=512, full_prompt=522, num_computed_tokens=0)
        self.assertEqual(load.decide(**kwargs).path, PrefixPath.LOAD)
        self.assertEqual(recompute.decide(**kwargs).path, PrefixPath.RECOMPUTE)

    def test_static_h_boundary_is_inclusive(self) -> None:
        policy = PrefixPolicy(PrefixPolicyMode.STATIC_H)
        at_threshold = policy.decide(
            h_lookup=1024, full_prompt=1034, num_computed_tokens=0
        )
        above_threshold = policy.decide(
            h_lookup=1280, full_prompt=1290, num_computed_tokens=0
        )
        self.assertEqual(at_threshold.path, PrefixPath.LOAD)
        self.assertEqual(above_threshold.path, PrefixPath.RECOMPUTE)

    def test_static_h_default_threshold_is_1024(self) -> None:
        policy = PrefixPolicy(PrefixPolicyMode.STATIC_H)
        self.assertEqual(policy.describe()["static_h_threshold"], 1024)
        self.assertEqual(
            policy.decide(h_lookup=1024, full_prompt=1034, num_computed_tokens=0).path,
            PrefixPath.LOAD,
        )
        self.assertEqual(
            policy.decide(h_lookup=1025, full_prompt=1035, num_computed_tokens=0).path,
            PrefixPath.RECOMPUTE,
        )

    def test_static_h_env_override_validated_and_frozen(self) -> None:
        policy = PrefixPolicy.from_env({ENV_MODE: "STATIC_H", ENV_STATIC_H: "512"})
        self.assertEqual(policy.mode, PrefixPolicyMode.STATIC_H)
        self.assertEqual(policy.describe()["static_h_threshold"], 512)
        self.assertEqual(
            policy.decide(h_lookup=512, full_prompt=522, num_computed_tokens=0).path,
            PrefixPath.LOAD,
        )
        self.assertEqual(
            policy.decide(h_lookup=513, full_prompt=523, num_computed_tokens=0).path,
            PrefixPath.RECOMPUTE,
        )

    def test_static_h_env_invalid_values_rejected(self) -> None:
        for bad in ("0", "-5", "abc", "1.5", "12.0", "2.5"):
            with self.subTest(value=bad), self.assertRaises(PrefixPolicyError):
                PrefixPolicy.from_env({ENV_MODE: "STATIC_H", ENV_STATIC_H: bad})

    def test_static_h_env_blank_uses_default(self) -> None:
        policy = PrefixPolicy.from_env({ENV_MODE: "STATIC_H", ENV_STATIC_H: "   "})
        self.assertEqual(policy.describe()["static_h_threshold"], 1024)

    def test_static_h_env_ignored_in_other_modes(self) -> None:
        off = PrefixPolicy.from_env({ENV_MODE: "OFF", ENV_STATIC_H: "1"})
        self.assertFalse(off.enabled)
        rec = PrefixPolicy.from_env({ENV_MODE: "FORCED_RECOMPUTE", ENV_STATIC_H: "1"})
        self.assertEqual(rec.mode, PrefixPolicyMode.FORCED_RECOMPUTE)

    def test_recompute_returns_zero_and_preserves_lookup_h(self) -> None:
        spec = FakeLoadSpec(lmcache_cached_tokens=512, can_load=True)
        policy = PrefixPolicy(PrefixPolicyMode.FORCED_RECOMPUTE)
        decision = policy.decide(h_lookup=512, full_prompt=522, num_computed_tokens=0)
        result = policy.apply(
            native_matched_tokens=512, load_spec=spec, decision=decision
        )
        self.assertEqual(result.returned_external_tokens, 0)
        self.assertTrue(result.policy_applied)
        self.assertEqual(spec.lmcache_cached_tokens, 512)
        self.assertFalse(spec.can_load)

    def test_load_preserves_native_match(self) -> None:
        spec = FakeLoadSpec(lmcache_cached_tokens=512)
        policy = PrefixPolicy(PrefixPolicyMode.FORCED_LOAD)
        decision = policy.decide(h_lookup=512, full_prompt=522, num_computed_tokens=0)
        result = policy.apply(
            native_matched_tokens=512, load_spec=spec, decision=decision
        )
        self.assertEqual(result.returned_external_tokens, 512)
        self.assertFalse(result.policy_applied)
        self.assertEqual(spec.lmcache_cached_tokens, 512)

    def test_scheduler_length_is_suffix_for_load_but_full_for_recompute(self) -> None:
        full_prompt = 522
        h_lookup = 512
        local_computed = 0
        spec = FakeLoadSpec(lmcache_cached_tokens=h_lookup)

        load_policy = PrefixPolicy(PrefixPolicyMode.FORCED_LOAD)
        load_decision = load_policy.decide(
            h_lookup=h_lookup,
            full_prompt=full_prompt,
            num_computed_tokens=local_computed,
        )
        loaded = load_policy.apply(
            native_matched_tokens=h_lookup,
            load_spec=spec,
            decision=load_decision,
        )
        load_prefill = full_prompt - local_computed - loaded.returned_external_tokens

        recompute_policy = PrefixPolicy(PrefixPolicyMode.FORCED_RECOMPUTE)
        recompute_decision = recompute_policy.decide(
            h_lookup=h_lookup,
            full_prompt=full_prompt,
            num_computed_tokens=local_computed,
        )
        recomputed = recompute_policy.apply(
            native_matched_tokens=h_lookup,
            load_spec=spec,
            decision=recompute_decision,
        )
        recompute_prefill = (
            full_prompt - local_computed - recomputed.returned_external_tokens
        )

        self.assertEqual(load_prefill, full_prompt - h_lookup)
        self.assertEqual(recompute_prefill, full_prompt)
        self.assertEqual(spec.lmcache_cached_tokens, h_lookup)

    def test_no_hit_is_not_relabelled_as_applied_recompute(self) -> None:
        spec = FakeLoadSpec(lmcache_cached_tokens=0)
        policy = PrefixPolicy(PrefixPolicyMode.FORCED_RECOMPUTE)
        decision = policy.decide(h_lookup=0, full_prompt=522, num_computed_tokens=0)
        result = policy.apply(
            native_matched_tokens=0, load_spec=spec, decision=decision
        )
        self.assertEqual(result.returned_external_tokens, 0)
        self.assertFalse(result.policy_applied)

    def test_async_lookup_is_rejected_not_silently_mislabelled(self) -> None:
        policy = PrefixPolicy(PrefixPolicyMode.FORCED_RECOMPUTE)
        decision = policy.decide(h_lookup=512, full_prompt=522, num_computed_tokens=0)
        with self.assertRaisesRegex(PrefixPolicyError, "async lookup"):
            policy.apply(
                native_matched_tokens=None,
                load_spec=FakeLoadSpec(512),
                decision=decision,
            )

    def test_positive_match_without_load_spec_fails(self) -> None:
        policy = PrefixPolicy(PrefixPolicyMode.FORCED_RECOMPUTE)
        decision = policy.decide(h_lookup=512, full_prompt=522, num_computed_tokens=0)
        with self.assertRaisesRegex(PrefixPolicyError, "without a LoadSpec"):
            policy.apply(native_matched_tokens=512, load_spec=None, decision=decision)

    def test_invalid_lengths_are_rejected(self) -> None:
        policy = PrefixPolicy(PrefixPolicyMode.STATIC_H)
        with self.assertRaisesRegex(PrefixPolicyError, "exceeds full_prompt"):
            policy.decide(h_lookup=523, full_prompt=522, num_computed_tokens=0)
        with self.assertRaisesRegex(PrefixPolicyError, "exceeds"):
            policy.decide(h_lookup=512, full_prompt=522, num_computed_tokens=523)


class WindowPolicyTest(unittest.TestCase):
    """Build E v4: dual-threshold STATIC_H window (tau1=1024, tau2=23296)."""

    T1 = 1024
    T2 = 23296

    def _policy(self, t1: int = T1, t2: Optional[int] = T2) -> PrefixPolicy:
        return PrefixPolicy(
            PrefixPolicyMode.STATIC_H,
            static_h_threshold=t1,
            static_h_threshold2=t2,
        )

    def test_window_boundary_branches(self) -> None:
        p = self._policy()
        cases = [
            # (H, expected path, expected reason fragment)
            (self.T1, PrefixPath.LOAD, "H<=1024"),  # H == tau1 -> LOAD
            (self.T1 + 256, PrefixPath.RECOMPUTE, "1024<H<23296"),
            (self.T2 - 256, PrefixPath.RECOMPUTE, "1024<H<23296"),
            (self.T2, PrefixPath.LOAD, "H>=23296"),  # H == tau2 -> LOAD
            (self.T2 + 256, PrefixPath.LOAD, "H>=23296"),
            (256, PrefixPath.LOAD, "H<=1024"),  # H < tau1
            (30000, PrefixPath.LOAD, "H>=23296"),  # H > tau2
        ]
        for h, path, tag in cases:
            d = p.decide(h_lookup=h, full_prompt=h + 10, num_computed_tokens=0)
            self.assertEqual(d.path, path, f"H={h}")
            self.assertIn(tag, d.reason, f"H={h}")

    def test_reject_h2_le_h1(self) -> None:
        with self.assertRaisesRegex(PrefixPolicyError, "must be >"):
            PrefixPolicy(
                PrefixPolicyMode.STATIC_H,
                static_h_threshold=1024,
                static_h_threshold2=1024,
            )
        with self.assertRaisesRegex(PrefixPolicyError, "must be >"):
            PrefixPolicy(
                PrefixPolicyMode.STATIC_H,
                static_h_threshold=1024,
                static_h_threshold2=512,
            )

    def test_reject_bad_h2_values(self) -> None:
        with self.assertRaises(PrefixPolicyError):
            PrefixPolicy.from_env({ENV_MODE: "STATIC_H", ENV_STATIC_H2: "abc"})
        with self.assertRaises(PrefixPolicyError):
            PrefixPolicy.from_env({ENV_MODE: "STATIC_H", ENV_STATIC_H2: "0"})
        with self.assertRaises(PrefixPolicyError):
            PrefixPolicy.from_env({ENV_MODE: "STATIC_H", ENV_STATIC_H2: "-5"})
        with self.assertRaises(PrefixPolicyError):
            PrefixPolicy.from_env(
                {ENV_MODE: "STATIC_H", ENV_STATIC_H: "2048", ENV_STATIC_H2: "1024"}
            )

    def test_legacy_single_threshold_when_h2_unset(self) -> None:
        # direct construction with threshold2=None -> legacy rule
        p = PrefixPolicy(PrefixPolicyMode.STATIC_H, static_h_threshold=1024)
        self.assertIsNone(p.static_h_threshold2)
        self.assertEqual(
            p.decide(h_lookup=1024, full_prompt=1034, num_computed_tokens=0).path,
            PrefixPath.LOAD,
        )
        self.assertEqual(
            p.decide(h_lookup=1280, full_prompt=1290, num_computed_tokens=0).path,
            PrefixPath.RECOMPUTE,
        )
        # from_env without STATIC_H2 -> legacy single threshold
        pe = PrefixPolicy.from_env({ENV_MODE: "STATIC_H"})
        self.assertIsNone(pe.static_h_threshold2)

    def test_from_env_dual_threshold(self) -> None:
        pe = PrefixPolicy.from_env(
            {ENV_MODE: "STATIC_H", ENV_STATIC_H: "1024", ENV_STATIC_H2: "23296"}
        )
        self.assertEqual(pe.static_h_threshold, 1024)
        self.assertEqual(pe.static_h_threshold2, 23296)
        self.assertEqual(
            pe.decide(h_lookup=23552, full_prompt=23562, num_computed_tokens=0).path,
            PrefixPath.LOAD,
        )

    def test_describe_dual_threshold(self) -> None:
        desc = self._policy().describe()
        self.assertEqual(desc["static_h_threshold"], 1024)
        self.assertEqual(desc["static_h_threshold2"], 23296)
        # legacy describe has no threshold2 key
        self.assertNotIn(
            "static_h_threshold2",
            PrefixPolicy(PrefixPolicyMode.STATIC_H, static_h_threshold=1024).describe(),
        )


class DecisionLogSwitchTest(unittest.TestCase):
    """LMCACHE_PREFIX_POLICY_LOG_DECISION switch semantics (default off)."""

    def test_default_disabled(self):
        # unset env -> disabled (measurement aid, not a hot-path log)
        try:
            del os.environ[ENV_LOG_DECISION]
        except KeyError:
            pass
        self.assertFalse(log_decision_enabled())

    def test_zero_disables(self):
        self.assertFalse(log_decision_enabled("0"))
        self.assertFalse(log_decision_enabled(env_value="0"))

    def test_only_explicit_one_enables(self):
        self.assertTrue(log_decision_enabled("1"))
        self.assertFalse(log_decision_enabled("true"))
        self.assertFalse(log_decision_enabled(""))
        self.assertFalse(log_decision_enabled("2"))


if __name__ == "__main__":
    unittest.main()
