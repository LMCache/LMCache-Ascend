# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Any
import json

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryFormat

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

HAS_MS_TYPE = False
try:
    from mindspore.common import np_dtype
    HAS_MS_TYPE = True
except ImportError as ie:
    pass

def DiskCacheMetadata_to_dict(self) -> Dict[str, Any]:
    """
    Converts the metadata to a JSON-serializable dictionary.
    """
    global TORCH_DTYPE_TO_STR_DTYPE
    return {
        "path": self.path,
        "size": self.size,
        # Convert torch.Size to a list, which is JSON-safe
        "shape": list(self.shape) if self.shape is not None else None,
        # Convert torch.dtype to a string, which is JSON-safe
        "dtype": TORCH_DTYPE_TO_STR_DTYPE[self.dtype] if self.dtype is not None else None,
    }

@classmethod
def DiskCacheMetadata_from_dict(cls, data_dict: Dict[str, Any]) -> 'DiskCacheMetadata':
    """
    Creates a DiskCacheMetadata instance from a dictionary.
    """
    path = data_dict.get("path")
    size = data_dict.get("size")
    global TORCH_STR_TO_DTYPE
    # Convert shape from a list back to torch.Size
    shape_list = data_dict.get("shape")
    shape = torch.Size(shape_list) if shape_list is not None else None
    
    # Convert dtype from a string back to torch.dtype using the safe map
    dtype_str = data_dict.get("dtype")
    dtype = TORCH_STR_TO_DTYPE[dtype_str] if dtype_str is not None else None

    return cls(path=path, size=size, shape=shape, dtype=dtype)

TORCH_DTYPE_TO_STR_DTYPE = {
    torch.half: "half",
    torch.float16: "half",
    torch.bfloat16: "bfloat16",
    torch.float: "float",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.double: "double",
    torch.uint8: "fp8",
}

TORCH_STR_TO_DTYPE = {
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "float32": torch.float32,
    "float64": torch.float64,
    "double": torch.double,
    "fp8": torch.uint8,
}

def update_dtypes():
    global TORCH_DTYPE_TO_STR_DTYPE
    global HAS_MS_TYPE
    try:
        TORCH_DTYPE_TO_STR_DTYPE.update({torch.float8_e4m3fn: "fp8_e4m3"})
        TORCH_DTYPE_TO_STR_DTYPE.update({torch.float8_e5m2: "float8_e5m2"})
        TORCH_STR_TO_DTYPE.update({"fp8_e4m3": torch.float8_e4m3fn})
        TORCH_STR_TO_DTYPE.update({"float8_e5m2": torch.float8_e5m2})
    except AttributeError as ae:
        if not HAS_MS_TYPE:
            logger.error("Unable to update dtype: ", ae)
            raise ae
        else:
            logger.warn("not using dtypes: torch.float8_e4m3fn and torch.float8_e5m2")
            pass
    
    if HAS_MS_TYPE:    
        try:
            TORCH_DTYPE_TO_STR_DTYPE.update({np_dtype.bfloat16: "np_bfloat16"})
            TORCH_STR_TO_DTYPE.update({"np_bfloat16": np_dtype.bfloat16})
        except AttributeError as ae:
            logger.error("Unable to update dtype: ", ae)
            pass

update_dtypes()


def CacheEngineKey_to_string(self):
    model_name = self.model_name.replace("-", "#")
    return (
        f"{self.fmt}@{model_name}@{self.world_size}"
        f"@{self.worker_id}@{self.chunk_hash}"
    )

@staticmethod
def CacheEngineKey_from_string(s):
    parts = s.split("@")
    if len(parts) != 5:
        raise ValueError(f"Invalid key string: {s}")
    model_name = parts[1].replace("#", "-")
    return CacheEngineKey(
        parts[0], model_name, int(parts[2]), int(parts[3]), parts[4]
    )

def _lmcache_nvtx_annotate(func, domain="lmcache"):
    """Decorator for applying nvtx annotations to methods in lmcache."""
    return (func)
