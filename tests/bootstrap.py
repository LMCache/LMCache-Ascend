# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib.util
import os
import subprocess
import sys

# First Party
import lmcache_ascend

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LMCACHEPATH = os.environ.get("LMCACHEPATH", "/workspace/LMCache")
LMCACHEGITREPO = "https://github.com/LMCache/LMCache.git"
VERSION_TAG = lmcache_ascend.LMCACHE_UPSTREAM_TAG
TEST_ALIAS = "lmcache_tests"


def run_git_cmd(cmd_list, cwd=None):
    """Helper to run git commands with error handling."""
    try:
        subprocess.check_call(["git"] + cmd_list, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {' '.join(cmd_list)}")
        raise e


def get_current_git_tag(path):
    """Returns the current tag name if HEAD is exactly on a tag, else None."""
    try:
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


def setup_lmcache_dependency():
    """Clones or updates the upstream LMCache repo."""
    # 1. Check if Repo Exists
    if os.path.exists(LMCACHEPATH):
        current_tag = get_current_git_tag(LMCACHEPATH)
        if current_tag == VERSION_TAG:
            return  # Already on correct version

        print(f"⚠️ Version mismatch (Found: {current_tag}). Syncing to {VERSION_TAG}...")
        run_git_cmd(["fetch", "--tags"], cwd=LMCACHEPATH)
        run_git_cmd(["checkout", f"tags/{VERSION_TAG}"], cwd=LMCACHEPATH)

    # 2. Clone if Repo Missing
    else:
        print(f"📦 LMCache missing. Cloning {VERSION_TAG}...")
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


def register_alias():
    """Injects the upstream tests into sys.modules as 'lmcache_tests'."""
    if LMCACHEPATH not in sys.path:
        sys.path.append(LMCACHEPATH)

    # Check if already registered to avoid double-loading
    if TEST_ALIAS in sys.modules:
        return

    tests_init_path = os.path.join(LMCACHEPATH, "tests", "__init__.py")
    if not os.path.exists(tests_init_path):
        raise FileNotFoundError(f"Critical: {tests_init_path} does not exist.")

    spec = importlib.util.spec_from_file_location(TEST_ALIAS, tests_init_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[TEST_ALIAS] = module
        spec.loader.exec_module(module)
        print(f"✅ Registered module alias '{TEST_ALIAS}'")


def prepare_environment():
    """Main entry point to prepare the test environment."""
    setup_lmcache_dependency()
    register_alias()
