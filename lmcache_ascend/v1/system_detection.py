from dataclasses import dataclass
from typing import Optional
import torch
from lmcache.v1.system_detection import NUMAMapping
from lmcache.logging import init_logger

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
        pci_bus_id = get_gpu_pci_bus_id(device_index).lower()

        numa_node_file = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
        with open(numa_node_file) as f:
            numa_node = int(f.read())

        # Sanitizing the output as on some hardware setups the numa_node variable
        # appeared to return with -1 value, causing failure.
        if numa_node >= 0:
            return NUMAMapping(gpu_to_numa_mapping={device_index: numa_node})
        else:
            logger.warning(f"No valid NUMA mapping for current device, returning None")
            return None
    except Exception as e:
        logger.warning(f"Failed to auto read NUMA mapping from system: {e}")
        return None
