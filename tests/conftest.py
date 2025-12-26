# SPDX-License-Identifier: Apache-2.0
import os
import sys
import subprocess
import importlib.util
import pytest

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
# TODO (gingfung): obtain tag from setup.py
VERSION_TAG = "v0.3.7"
TEST_ALIAS = "lmcache_tests"

# ==============================================================================
# 2. BOOTSTRAP DEPENDENCY (Must run before plugins load)
# ==============================================================================
def run_git_cmd(cmd_list, cwd=None):
    """Helper to run git commands with error handling."""
    try:
        # Using subprocess.call for config/trust commands to avoid crashing on errors
        # Using check_call for critical operations like clone/checkout
        subprocess.check_call(["git"] + cmd_list, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {' '.join(cmd_list)}")
        raise e

# TODO (gingfung): consider moving the git clone setup into submodule with version pinning.
def setup_lmcache_dependency():
    print(f"\n🔍 Checking dependency: {LMCACHEGITREPO} @ {VERSION_TAG}...")

    if not os.path.exists(LMCACHEPATH):
        print("   📦 LMCache missing. Cloning...")
        run_git_cmd([
            "clone", 
            "--branch", VERSION_TAG,
            "--depth", "1",
            LMCACHEGITREPO, 
            LMCACHEPATH
        ])
    else:
        print("   🔄 LMCache exists. Syncing version...")
        run_git_cmd(["fetch", "--tags"], cwd=LMCACHEPATH)
        run_git_cmd(["checkout", f"tags/{VERSION_TAG}"], cwd=LMCACHEPATH)
    
    print(f"   ✅ LMCache is ready at tag: {VERSION_TAG}")

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
        import lmcache_ascend 
        from lmcache_ascend import _build_info

        print(f"\n⚡ [NPU Setup] Detected framework: {_build_info.__framework_name__}")

        if _build_info.__framework_name__ == "pytorch":
            # This applies the monkeypatch to torch.cuda -> torch.npu
            import torch
            from torch_npu.contrib import transfer_to_npu
            print("   ✅ Applied 'transfer_to_npu' patch.")
            # initialize context
            _ = torch.randn((100, 100), device="npu")
            
    except ImportError as e:
        pytest.exit(f"❌ lmcache_ascend or torch_npu not found: {e}", returncode=1)


def patch_lmcache_test_utils():
    imported = False
    local_tests_dir = os.path.dirname(os.path.abspath(__file__))
    if local_tests_dir not in sys.path:
        sys.path.append(local_tests_dir)
        imported = True

    try:
        import lmcache_tests.v1.utils as original_utils
        import v1.utils as npu_utils
        original_utils.create_gpu_connector = npu_utils.create_npu_connector
    except ImportError as e:
        print(f"❌ Import error when patching lmcache_tests: {e}")
        pytest.exit(f"❌ Failed to patch lmcache_tests for NPU usage.", returncode=1)
    finally:
        if imported:
            # avoid polluting the namespace in later tests
            sys.path.remove(local_tests_dir)


setup_lmcache_dependency()
setup_npu_backend()
patch_lmcache_test_utils()

# ==============================================================================
# 3. PLUGIN REGISTRATION
# ==============================================================================
# This tells Pytest to load the remote conftest file as if it were local.
# It inherits all fixtures (autorelease_v1, mock_redis, etc.) automatically.
pytest_plugins = [f"{TEST_ALIAS}.conftest"]