# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib.util
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
import lmcache_ascend

"""
LMCache Test Bootstrap & Fixture Inheritance
============================================

This configuration file bootstraps the upstream `LMCache` repository
to allow this project to reuse its test suite and fixtures.

It executes the following critical setup steps immediately upon import, 
ensuring the environment is ready before Pytest begins test collection:

1.  **Dependency Synchronization**:
    - Checks for `LMCache` in the workspace.
    - Clones or checks out the specific `VERSION_TAG` to ensure we are testing 
      against the correct API contract.

2.  **Dynamic Module Aliasing**:
    - Adds the `LMCache` source to `sys.path`.
    - Registers the upstream `tests/` directory as a new python module named 
      `lmcache_tests`. This prevents naming collisions with the local `tests/` folder.

3.  **Global Pre-Import Patching**:
    - Monkey-patches utility functions in the upstream modules *before* they are 
      imported by the test suite. This guarantees that all reused tests use our 
      custom logic (e.g., custom GPU connectors) instead of the defaults.

4.  **Fixture Inheritance**:
    - Uses `pytest_plugins` to load the upstream `conftest.py`. This automatically 
      exposes fixtures like `mock_redis` and `autorelease_v1` to the local session.

NOTE: The setup functions in this file run at the module level (not inside a hook) 
to ensure `sys.modules` is populated before Pytest attempts to resolve plugins.
"""

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
LMCACHEPATH = os.environ.get("LMCACHEPATH", "/workspace/LMCache")
LMCACHEGITREPO = "https://github.com/LMCache/LMCache.git"
VERSION_TAG = lmcache_ascend.LMCACHE_UPSTREAM_TAG
TEST_ALIAS = "lmcache_tests"


# ==============================================================================
# 2. BOOTSTRAP DEPENDENCY (Must run before plugins load)
# ==============================================================================
def run_git_cmd(cmd_list, cwd=None):
    """Helper to run git commands with error handling."""
    try:
        # Use subprocess.check_call for all git operations so failures
        #  stop test setup early
        subprocess.check_call(["git"] + cmd_list, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {' '.join(cmd_list)}")
        raise e


def get_current_git_tag(path):
    """Returns the current tag name if HEAD is exactly on a tag, else None."""
    try:
        # describe --tags --exact-match fails if we are not exactly on a tag
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match"],
                cwd=path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return tag
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# TODO (gingfung): consider moving the git clone setup into submodule
# with version pinning.
def setup_lmcache_dependency():
    print(f"\n🔍 Checking dependency: {LMCACHEGITREPO} @ {VERSION_TAG}...")

    # 1. Check if Repo Exists
    if os.path.exists(LMCACHEPATH):
        current_tag = get_current_git_tag(LMCACHEPATH)

        if current_tag == VERSION_TAG:
            print(
                f"   ✅ Already on correct version ({VERSION_TAG}). "
                "Skipping git operations."
            )
        else:
            print(
                f"   ⚠️  Version mismatch (Found: {current_tag}). "
                "Syncing to {VERSION_TAG}..."
            )
            # Only fetch/checkout if tags don't match
            run_git_cmd(["fetch", "--tags"], cwd=LMCACHEPATH)
            run_git_cmd(["checkout", f"tags/{VERSION_TAG}"], cwd=LMCACHEPATH)

    # 2. Clone if Repo Missing
    else:
        print("   📦 LMCache missing. Cloning...")
        run_git_cmd(
            [
                "clone",
                "--branch",
                VERSION_TAG,
                "--depth",
                "1",
                LMCACHEGITREPO,
                LMCACHEPATH,
            ]
        )

    # 3. Register Module (Must always run to update sys.path for this session)
    if LMCACHEPATH not in sys.path:
        sys.path.append(LMCACHEPATH)

    tests_init_path = os.path.join(LMCACHEPATH, "tests", "__init__.py")

    if not os.path.exists(tests_init_path):
        pytest.exit(f"❌ Critical: {tests_init_path} does not exist. Clone failed?")

    spec = importlib.util.spec_from_file_location(TEST_ALIAS, tests_init_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[TEST_ALIAS] = module
        spec.loader.exec_module(module)
        print(f"   ✅ Registered '{tests_init_path}' as module alias '{TEST_ALIAS}'")
    else:
        pytest.exit(f"❌ Failed to register {TEST_ALIAS} as a module.", returncode=1)


def setup_npu_backend():
    try:
        # First Party
        from lmcache_ascend import _build_info

        print(f"\n⚡ [NPU Setup] Detected framework: {_build_info.__framework_name__}")

        if _build_info.__framework_name__ == "pytorch":
            # This applies the monkeypatch to torch.cuda -> torch.npu
            # Third Party
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
            import torch

            print("   ✅ Applied 'transfer_to_npu' patch.")
            # initialize context
            _ = torch.randn((100, 100), device="npu")

    except ImportError as e:
        pytest.exit(f"❌ lmcache_ascend or torch_npu not found: {e}", returncode=1)


def patch_lmcache_test_utils():
    try:
        # Third Party
        import lmcache_tests.v1.utils as original_utils

        # 1. Construct path to your local file
        local_utils_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v1", "utils.py"
        )

        # 2. Load it safely as a standalone module
        #    We give it a unique name "local_npu_utils" to avoid conflicts
        spec = importlib.util.spec_from_file_location(
            "local_npu_utils", local_utils_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load spec for local utils module at {local_utils_path}"
            )
        npu_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(npu_utils)

        # 3. Patch
        original_utils.create_gpu_connector = npu_utils.create_npu_connector
        print("✅ Successfully patched create_gpu_connector with NPU implementation.")

    except (ImportError, FileNotFoundError, AttributeError) as e:
        pytest.exit(f"❌ Failed to patch lmcache_tests: {e}", returncode=1)


def _run_module_setup():
    """
    Run all LMCache test bootstrap steps with logging and robust error handling.

    This function is invoked at import time so that the environment is prepared
    before pytest begins test collection. Any unexpected error will cause
    pytest to exit with a clear, contextual message.

    The timing must be the following:
    1. setup lmcache git tags
    2. setup npu backend
    3. patch lmcache tests utils
    """
    phases = [
        ("LMCache dependency setup", setup_lmcache_dependency),
        ("NPU backend setup", setup_npu_backend),
        ("LMCache test utils patching", patch_lmcache_test_utils),
    ]

    for description, func in phases:
        print(f"\n[LMCache Test Bootstrap] Starting: {description} ...")
        try:
            func()
        except Exception as exc:
            pytest.exit(
                f"❌ Error during {description}: {exc}\n"
                "This error occurred while importing tests/conftest.py,\n"
                "before pytest started test collection. Please verify your\n"
                "LMCache checkout, environment, and dependency configuration.",
                returncode=1,
            )


_run_module_setup()
# ==============================================================================
# 3. PLUGIN REGISTRATION
# ==============================================================================
# This tells Pytest to load the remote conftest file as if it were local.
# It inherits all fixtures (autorelease_v1, mock_redis, etc.) automatically.
pytest_plugins = [f"{TEST_ALIAS}.conftest"]
