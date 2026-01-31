# SPDX-License-Identifier: Apache-2.0
# Standard
from importlib.metadata import PackageNotFoundError, version
import warnings

# Third Party
import pytest

# First Party
import lmcache_ascend


def test_dependency_compatibility():
    """
    Verifies that the installed 'lmcache' package matches the version
    that 'lmcache-ascend' was designed to patch.

    Logic:
      1. PASS if versions match exactly (e.g., Target v0.3.7 vs Installed 0.3.7).
      2. WARN (but PASS) if installed version is a Dev/Dirty build (e.g., 0.3.8.dev0).
      3. FAIL if it is a clean release mismatch
         (e.g., Target v0.3.7 vs Installed 0.4.0).
    """

    # 1. Get the installed version safely
    try:
        installed_ver = version("lmcache")
    except PackageNotFoundError:
        pytest.fail("❌ 'lmcache' is not installed in the current environment.")

    # 2. Get the target version from your package
    target_tag = lmcache_ascend.LMCACHE_UPSTREAM_TAG

    # 3. Normalize (remove 'v' prefix for comparison)
    clean_target = target_tag.lstrip("v")

    print(f"\n   [Version Check] Installed: {installed_ver} vs Targeted: {target_tag}")

    # --- CHECK LOGIC ---

    # A. Exact/Compatible Match
    # If installed starts with target (e.g. "0.3.7" starts with "0.3.7"), we are good.
    if installed_ver.startswith(clean_target):
        return

    # B. Development Build Bypass
    # If the installed version contains dev/dirty markers,
    # we assume you are fixing things.
    dev_markers = [".dev", "+", "dirty", "a", "b", "rc"]
    is_dev_build = any(marker in installed_ver for marker in dev_markers)

    if is_dev_build:
        warnings.warn(
            f"\n⚠️  ALLOWING MISMATCH FOR DEV BUILD\n"
            f"   Targeted Tag:      {target_tag}\n"
            f"   Installed Version: {installed_ver}\n"
            f"   -> Assuming local development compatibility.",
            stacklevel=2,
        )
        return

    # C. Critical Mismatch (Production Safety)
    # If we get here, it's a clean release that doesn't match. This is dangerous.
    pytest.fail(
        f"CRITICAL: Upstream Version Mismatch!\n"
        f"   Your code expects LMCache version: {target_tag}\n"
        f"   But the installed LMCache version is: {installed_ver}\n"
        f"   -> ACTION: Update 'LMCACHE_UPSTREAM_TAG' in lmcache_ascend/__init__.py "
        f"OR install the correct upstream version."
    )
