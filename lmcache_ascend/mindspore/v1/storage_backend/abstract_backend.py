# SPDX-License-Identifier: Apache-2.0
# Standard

# Third Party
import torch


def StorageBackendInterface___init__(
    self,
    dst_device: str = "Ascend",
):
    dst_device = "Ascend"
    try:
        torch.device(dst_device)
    except RuntimeError:
        raise

    self.dst_device = dst_device
