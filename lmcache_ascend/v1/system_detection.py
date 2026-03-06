# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.system_detection import NUMAMapping
from vllm.platforms import current_platform
import torch

logger = init_logger(__name__)

if torch.npu.is_available():
    try:
        # First Party
        from lmcache_ascend.c_ops import get_gpu_pci_bus_id
    except ImportError:
        # Fallback if c_ops is not available
        get_gpu_pci_bus_id = None


def _read_from_sys() -> Optional[NUMAMapping]:
    """
    Read NUMA mapping from system configuration.
    """

    try:
        device_index = torch.npu.current_device()
        phy_device_id = current_platform.device_id_to_physical_device_id(device_index)
        pci_bus_id = get_gpu_pci_bus_id(phy_device_id).lower()

        numa_node_file = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
        with open(numa_node_file) as f:
            numa_node = int(f.read())

        # Sanitizing the output as on some hardware setups the numa_node variable
        # appeared to return with -1 value, causing failure.
        if numa_node >= 0:
            return NUMAMapping(gpu_to_numa_mapping={device_index: numa_node})
        else:
            logger.warning("No valid NUMA mapping for current device, returning None")
            return None
    except Exception as e:
        logger.warning(f"Failed to auto read NUMA mapping from system: {e}")
        return None
