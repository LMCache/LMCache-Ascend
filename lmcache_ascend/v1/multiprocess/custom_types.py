# SPDX-License-Identifier: Apache-2.0
# Standard
import re
import subprocess

# Third Party
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper
import torch


class AscendIPCWrapper(CudaIPCWrapper):
    """
    We patch the CudaIPCWrapper because of the following reasons:
    1. acl runtime currently does not support getting uuid
    2. the torch_npu transfer from cuda cannot directly convert
        _new_share_cuda / _share_cuda -> _new_share_npu / _share_npu
    Potentially, we should let torch_npu to update the patch.
        we should also beware that the uuid we created might not be *unique*.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        assert tensor.storage_offset() == 0
        assert tensor.is_contiguous()
        storage = tensor.untyped_storage()
        handle = storage._share_npu_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        device_index = tensor.device.index
        self.device_uuid = AscendIPCWrapper._get_device_uuid(device_index)

    @staticmethod
    def _get_device_uuid(device_index: int) -> str:
        """
        Ascend does not support uuid from the get_device_properties.
        Retrieves the VDie ID (Silicon ID) for Ascend device.
        Falls back to PCIe Bus ID if VDie ID is unavailable.
        """
        device_name = torch.npu.get_device_name()

        try:
            # Run the npu-smi command
            cmd = ["npu-smi", "info", "-t", "board", "-i", str(device_index), "-c", "0"]
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode(
                "utf-8"
            )

            # 1. Try to find VDie ID
            # Matches: "VDie ID : XXXXX XXXX..."
            vdie_match = re.search(r"VDie ID\s*:\s*([0-9A-F ]+)", result)
            if vdie_match:
                raw_id = vdie_match.group(1).replace(" ", "")
                if raw_id and not all(c == "0" for c in raw_id):
                    return f"{device_name}-{raw_id}"

            # 2. Fallback to PCIe Bus Info (Best Local ID)
            # Matches: "PCIe Bus Info : 0000:C1:00.0"
            pci_match = re.search(r"PCIe Bus Info\s*:\s*([0-9A-Fa-f:.]+)", result)
            if pci_match:
                return f"{device_name}-{pci_match.group(1)}"

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("Failed to retrieve device UUID from npu-smi.") from e

        # 3. Final Fallback (Unlikely to be unique globally)
        return f"{device_name}-{device_index}"

    @staticmethod
    def _discover_gpu_devices():
        """Discover all available GPU devices and map their UUIDs to
        the physical device ordinals.
        """
        if not torch.npu.is_available():
            return

        num_devices = torch.npu.device_count()
        with AscendIPCWrapper._device_mapping_lock:
            if AscendIPCWrapper._discovered_device_mapping:
                return  # Already discovered

            for i in range(num_devices):
                device_uuid = AscendIPCWrapper._get_device_uuid(i)
                AscendIPCWrapper._discovered_device_mapping[device_uuid] = i

    @staticmethod
    def _get_device_index_from_uuid(device_uuid: str) -> int:
        """Get the physical device ordinal from its UUID."""
        AscendIPCWrapper._discover_gpu_devices()

        with AscendIPCWrapper._device_mapping_lock:
            device_index = AscendIPCWrapper._discovered_device_mapping.get(
                device_uuid, None
            )

        if device_index is None:
            raise RuntimeError(
                f"Device UUID {device_uuid} not found in the discovered devices."
                "Please make sure the process can see all the GPU devices"
            )
        return device_index

    def to_tensor(self):
        """
        Note:
            This function may break if torch cuda is not initialized.
            We should call `torch.cuda.init()` before using this function.
        """
        device = AscendIPCWrapper._get_device_index_from_uuid(self.device_uuid)
        storage = torch.UntypedStorage._new_shared_npu(device, *self.handle[1:])
        t = torch.tensor(0, device=device, dtype=self.dtype)
        t.set_(storage)
        return t.view(self.shape)
