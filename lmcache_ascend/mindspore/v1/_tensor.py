# SPDX-License-Identifier: Apache-2.0
"""
This file contain methods for  torch.Tensor or np.ndarray.
"""

# Standard
from typing import Union
import ctypes

# Third Party
import numpy as np
import torch

# First Party
from lmcache_ascend import _build_info

USE_MS = False


if _build_info.__framework_name__ == "mindspore":
    USE_MS = True
    # Third Party
    from mindspore.common import np_dtype
    import mindspore as ms

MS_DTYPE_SIZE = {}


def get_dtype_compat(dtype: torch.dtype):
    global USE_MS
    if USE_MS and isinstance(dtype, ms.dtype.Type):
        return ms.dtype_to_nptype(dtype)
    return dtype


def get_itemsize(dtype: torch.dtype):
    """FIXME: is there a better way to do this ? :/"""
    m = getattr(dtype, "itemsize", None)
    global USE_MS
    global MS_DTYPE_SIZE
    if USE_MS:
        if m is None:
            # we are probably at mindspore
            if dtype in MS_DTYPE_SIZE:
                return MS_DTYPE_SIZE[dtype]
            tmp = ms.Tensor([1.0], dtype=dtype)
            MS_DTYPE_SIZE[dtype] = tmp.itemsize
            m = MS_DTYPE_SIZE[dtype]
        elif dtype == np.float16 or dtype == np_dtype.bfloat16:
            # np does not have bfloat16
            return 2
    return m


def get_data_ptr(tensor: Union[torch.Tensor, np.ndarray]):
    """Get the data pointer of a torch.Tensor or np.ndarray."""
    if isinstance(tensor, torch.Tensor):
        return tensor.data_ptr()
    elif isinstance(tensor, np.ndarray):
        return tensor.ctypes.data_as(ctypes.c_void_p).value
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


def get_numel(tensor: Union[torch.Tensor, np.ndarray, torch.Size]):
    """Get the number of elements in a torch.Tensor or np.ndarray."""
    if isinstance(tensor, torch.Tensor):
        return tensor.numel()
    elif isinstance(tensor, np.ndarray):
        return tensor.size
    elif isinstance(tensor, torch.Size):
        return torch.numel(tensor)
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


def get_element_size(tensor: Union[torch.Tensor, np.ndarray]):
    """Get the size of each element in a torch.Tensor or np.ndarray."""
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size()
    elif isinstance(tensor, np.ndarray):
        return tensor.itemsize
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


def view_and_shape(
    tensor: Union[torch.Tensor, np.ndarray], dtype: torch.dtype, shape: torch.Size
):
    """Get the view and shape of a torch.Tensor or np.ndarray."""
    if isinstance(tensor, torch.Tensor):
        return tensor.view(dtype).view(shape)
    elif isinstance(tensor, np.ndarray):
        return tensor.reshape(-1).view(dtype).reshape(shape)
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")
