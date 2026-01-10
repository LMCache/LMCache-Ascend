# SPDX-License-Identifier: Apache-2.0
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


# Run NPU setup
setup_npu_backend()


# ==============================================================================
# 3. PLUGIN REGISTRATION
# ==============================================================================
# Inherit fixtures from the upstream repo
pytest_plugins = [f"{TEST_ALIAS}.conftest"]
