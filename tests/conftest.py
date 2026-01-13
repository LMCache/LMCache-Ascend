# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib
import os

# Third Party
import pytest

# Local
# Local Bootstrap
# We import the function we just created to handle the git/alias logic
from .bootstrap import TEST_ALIAS, prepare_environment

# ==============================================================================
# 1. RUN BOOTSTRAP
# ==============================================================================
try:
    prepare_environment()
except Exception as e:
    pytest.exit(f"❌ Bootstrap failed: {e}", returncode=1)


# ==============================================================================
# 2. NPU ENVIRONMENT SETUP
# ==============================================================================
def setup_npu_backend():
    try:
        # First Party
        from lmcache_ascend import _build_info

        print(f"\n⚡ [NPU Setup] Detected framework: {_build_info.__framework_name__}")

        if _build_info.__framework_name__ == "pytorch":
            # Third Party
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
            import torch

            # Sanity check
            _ = torch.randn((10), device="npu")
            print("   ✅ NPU Backend initialized successfully.")

    except ImportError as e:
        pytest.exit(f"❌ lmcache_ascend or torch_npu not found: {e}", returncode=1)


def patch_lmcache_test_utils():
    """
    NOTE (gingfung): in some of the tests like test_cache_engine directly uses
    fixtures for gpu_connector, and we want to patch this prior the tests loaded.
    """
    try:
        # Third Party
        import lmcache_tests.v1.utils as original_utils

        # 1. Construct path to the utils file
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


# Run NPU setup
setup_npu_backend()
patch_lmcache_test_utils()

# ==============================================================================
# 3. PLUGIN REGISTRATION
# ==============================================================================
# Inherit fixtures from the upstream repo
pytest_plugins = [f"{TEST_ALIAS}.conftest"]
