# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import sys

# Third Party
import pytest

# Local
# Local Bootstrap
# We import the function we just created to handle the git/alias logic
from .bootstrap import TEST_ALIAS, prepare_environment

# ==============================================================================
# 1. RUN BOOTSTRAP
# ==============================================================================
_lmcache_path = os.environ.get("LMCACHEPATH")
if _lmcache_path and _lmcache_path not in sys.path:
    sys.path.insert(0, _lmcache_path)

try:
    prepare_environment()
except Exception as e:
    pytest.exit(f"❌ Bootstrap failed: {e}", returncode=1)


# ==============================================================================
# 2. NPU ENVIRONMENT SETUP
# ==============================================================================
def setup_npu_backend():
    try:
        # Third Party
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            _ = torch.randn((10), device="npu")
            print("   ✅ NPU Backend initialized successfully.")
        else:
            print("   ℹ️ CPU-only pytest: skipped NPU sanity check.")

    except ImportError as e:
        pytest.exit(f"❌ torch_npu not found: {e}", returncode=1)


setup_npu_backend()

# ==============================================================================
# 3. PLUGIN REGISTRATION
# ==============================================================================
# Inherit fixtures from the upstream repo
pytest_plugins = [f"{TEST_ALIAS}.conftest"]
