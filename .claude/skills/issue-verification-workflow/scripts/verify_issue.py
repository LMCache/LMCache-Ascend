#!/usr/bin/env python3
"""
LMCache-Ascend Issue Verification Runner

Utility that automates source-level verification of issue fixes.
Run on the target environment (local or container).

Usage:
    python3 verify_issue.py --issue-type pin_memory --file lmcache_ascend/v1/npu_connector/npu_connectors.py
    python3 verify_issue.py --issue-type offload_time  --file lmcache_ascend/v1/cache_engine.py
    python3 verify_issue.py --issue-type dead_code     --file lmcache_ascend/v1/storage_backend/p2p_backend.py --pattern local_lookup_cache
    python3 verify_issue.py --issue-type index_fix     --file lmcache_ascend/v1/blend/blender.py --bad-pattern "recomp_ratios[0]"
"""

import argparse
import sys
import time
from pathlib import Path

# Try to import torch for performance tests (optional)
try:
    import torch
    import torch_npu
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Base path for LMCache-Ascend source
DEFAULT_BASE = "/mnt/sdb/csy/p2p/LMCache-Ascend"


def read_file(filepath: str, base: str = DEFAULT_BASE) -> str:
    """Read source file, returning contents."""
    path = Path(base) / filepath
    if not path.exists():
        print(f"FAIL: File not found: {path}")
        sys.exit(1)
    return path.read_text()


# ---------------------------------------------------------------------------
# Issue-specific verification functions
# ---------------------------------------------------------------------------

def verify_pin_memory(filepath: str, base: str = DEFAULT_BASE) -> dict:
    """Verify #238: pin_memory() + non_blocking=True in npu_connectors."""
    src = read_file(filepath, base)
    results = {}

    # Source audit
    results["pin_memory_found"] = ".pin_memory()" in src
    results["non_blocking_found"] = "non_blocking=True" in src

    # Count occurrences in _initialize_pointers via line analysis
    pin_count = src.count(".pin_memory()")
    nb_count = src.count("non_blocking=True")
    results["pin_memory_count"] = pin_count
    results["non_blocking_count"] = nb_count
    results["pin_memory_branches"] = "all 4" if pin_count >= 4 else f"{pin_count}/4"

    # Micro-benchmark (if torch available)
    if HAS_TORCH:
        try:
            n_iters = 2000
            n_elem = 96
            pinned = torch.empty(n_elem, dtype=torch.int64, device="cpu").pin_memory()
            unpinned = torch.empty(n_elem, dtype=torch.int64, device="cpu")
            npu_p = torch.empty(n_elem, dtype=torch.int64, device="npu")
            npu_u = torch.empty(n_elem, dtype=torch.int64, device="npu")

            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                npu_u.copy_(unpinned)
            t_old = (time.perf_counter() - t0) / n_iters * 1e6

            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                npu_p.copy_(pinned, non_blocking=True)
            t_new = (time.perf_counter() - t0) / n_iters * 1e6

            results["benchmark_old_us"] = round(t_old, 1)
            results["benchmark_new_us"] = round(t_new, 1)
            results["benchmark_speedup"] = round(t_old / max(t_new, 0.001), 0)
            results["is_pinned"] = pinned.is_pinned()
        except Exception as e:
            results["benchmark_error"] = str(e)

    results["passed"] = (
        results.get("pin_memory_found") and
        results.get("non_blocking_found") and
        results.get("pin_memory_count", 0) >= 4 and
        results.get("non_blocking_count", 0) >= 1
    )
    return results


def verify_offload_time(filepath: str, base: str = DEFAULT_BASE) -> dict:
    """Verify #233: offload_time log format correction."""
    src = read_file(filepath, base)
    results = {}

    results["old_label_removed"] = "offload_time: %.4f ms, put_time" not in src
    results["new_label_present"] = "offload_total_time" in src
    results["breakdown_present"] = "process_tokens: %.4f ms, from_gpu" in src
    results["comment_present"] = "NOTE(#233)" in src and "wait_for_forward" in src

    results["passed"] = (
        results["old_label_removed"] and
        results["new_label_present"] and
        results["breakdown_present"] and
        results["comment_present"]
    )
    return results


def verify_dead_code(filepath: str, bad_pattern: str, base: str = DEFAULT_BASE) -> dict:
    """Verify dead code removal."""
    src = read_file(filepath, base)
    results = {}

    results["pattern_absent"] = bad_pattern not in src

    results["passed"] = results["pattern_absent"]
    return results


def verify_index_fix(filepath: str, bad_pattern: str, base: str = DEFAULT_BASE) -> dict:
    """Verify hardcoded index fix."""
    src = read_file(filepath, base)
    results = {}

    results["bad_pattern_removed"] = bad_pattern not in src

    results["passed"] = results["bad_pattern_removed"]
    return results


def verify_stream_sync(filepath: str, expected_pattern: str, base: str = DEFAULT_BASE) -> dict:
    """Verify stream synchronization fix (e.g., store_stream wrapping)."""
    src = read_file(filepath, base)
    results = {}

    results["pattern_found"] = expected_pattern in src

    results["passed"] = results["pattern_found"]
    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_results(issue_type: str, results: dict):
    """Pretty-print verification results."""
    print(f"\n{'='*55}")
    print(f"Verification: {issue_type}")
    print(f"{'='*55}")

    for key, value in results.items():
        if key == "passed":
            continue
        if key.startswith("benchmark_"):
            if key == "benchmark_error":
                print(f"  [WARN] Benchmark failed: {value}")
            else:
                print(f"  {key}: {value}")
        else:
            status = "✓" if value else "✗"
            print(f"  [{status}] {key}: {value}")

    passed = results.get("passed", False)
    print(f"\n  Overall: {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*55}")
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LMCache-Ascend Issue Verification Runner"
    )
    parser.add_argument(
        "--issue-type", required=True,
        choices=["pin_memory", "offload_time", "dead_code", "index_fix", "stream_sync"],
        help="Type of issue to verify"
    )
    parser.add_argument(
        "--file", required=True,
        help="Relative path of target file (from lmcache-ascend root)"
    )
    parser.add_argument(
        "--pattern", default=None,
        help="Pattern to check for dead_code or stream_sync"
    )
    parser.add_argument(
        "--bad-pattern", default=None,
        help="Bad pattern to check against for index_fix"
    )
    parser.add_argument(
        "--base", default=DEFAULT_BASE,
        help="Base directory of lmcache-ascend source"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all checks for PR #253"
    )

    args = parser.parse_args()

    if args.all:
        # Run all PR #253 checks
        checks = {
            "pin_memory": lambda: verify_pin_memory("lmcache_ascend/v1/npu_connector/npu_connectors.py", args.base),
            "offload_time": lambda: verify_offload_time("lmcache_ascend/v1/cache_engine.py", args.base),
            "dead_code_p2p": lambda: verify_dead_code(
                "lmcache_ascend/v1/storage_backend/p2p_backend.py", "local_lookup_cache", args.base),
            "index_fix_blend": lambda: verify_index_fix(
                "lmcache_ascend/v1/blend/blender.py", "recomp_ratios[0]", args.base),
            "stream_sync_ms": lambda: verify_stream_sync(
                "lmcache_ascend/mindspore/v1/npu_connector.py", "store_stream", args.base),
        }

        all_passed = True
        for name, check_fn in checks.items():
            results = check_fn()
            if not print_results(name, results):
                all_passed = False

        if all_passed:
            print("\n✓ All checks PASSED")
        else:
            print("\n✗ Some checks FAILED")
            sys.exit(1)
        return

    # Single check
    if args.issue_type == "pin_memory":
        results = verify_pin_memory(args.file, args.base)
    elif args.issue_type == "offload_time":
        results = verify_offload_time(args.file, args.base)
    elif args.issue_type == "dead_code":
        if not args.pattern:
            print("ERROR: --pattern required for dead_code check")
            sys.exit(1)
        results = verify_dead_code(args.file, args.pattern, args.base)
    elif args.issue_type == "index_fix":
        if not args.bad_pattern:
            print("ERROR: --bad-pattern required for index_fix check")
            sys.exit(1)
        results = verify_index_fix(args.file, args.bad_pattern, args.base)
    elif args.issue_type == "stream_sync":
        if not args.pattern:
            print("ERROR: --pattern required for stream_sync check")
            sys.exit(1)
        results = verify_stream_sync(args.file, args.pattern, args.base)

    if not print_results(args.issue_type, results):
        sys.exit(1)


if __name__ == "__main__":
    main()
