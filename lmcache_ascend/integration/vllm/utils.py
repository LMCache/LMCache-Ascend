# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Tuple

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)


def _is_npu_available() -> bool:
    """Check if NPU is available via torch_npu."""
    return hasattr(torch, "npu") and torch.npu.is_available()


def get_vllm_torch_dev() -> Tuple:
    """
    Returns the torch device and device name for the vLLM engine.
    e.g. (torch.cuda, "cuda") or (torch.xpu, "xpu") or (torch.npu, "npu")

    This is the Ascend-specific version that patches the upstream
    lmcache.integration.vllm.utils.get_vllm_torch_dev function.
    """
    if _is_npu_available():
        logger.info("NPU device is available. Using NPU for LMCache engine.")
        torch_dev = torch.npu
        dev_name = "npu"
    else:
        raise RuntimeError("Unsupported device platform for LMCache engine.")
    return torch_dev, dev_name
