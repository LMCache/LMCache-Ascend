# SPDX-License-Identifier: Apache-2.0
"""The plugin must activate from inside ``import lmcache`` on current dev."""

# Standard
import subprocess
import sys

# Third Party
from lmcache.v1.platform import torch_device_type
import pytest


@pytest.mark.skipif(
    torch_device_type != "npu", reason="only meaningful on the NPU platform"
)
def test_import_lmcache_first_binds_npu_device_ops() -> None:
    """Regression for #369: an import failing during activation must not leave
    ``lmcache.device_ops`` on the lazy proxy."""
    probe = (
        "import lmcache\n"
        "from lmcache.v1.platform.devices.npu.device_ops import NpuDeviceOps\n"
        "bound = isinstance(lmcache.device_ops, NpuDeviceOps)\n"
        "print(type(lmcache.device_ops).__qualname__)\n"
        "raise SystemExit(0 if bound else 1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"{proc.stdout}{proc.stderr}"
