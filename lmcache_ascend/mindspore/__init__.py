
import lmcache

import sys

from lmcache_ascend.mindspore import c_ops
sys.modules['lmcache.c_ops'] = c_ops

from lmcache_ascend.mindspore.v1.memory_management import _allocate_cpu_memory
lmcache.v1.memory_management._allocate_cpu_memory = _allocate_cpu_memory

from lmcache_ascend.mindspore.v1.memory_management import NumpyAndTensorMemoryObj
lmcache.v1.memory_management.TensorMemoryObj = NumpyAndTensorMemoryObj

from lmcache_ascend.mindspore.v1.memory_management import NumpyAndTensorMemoryAllocator
lmcache.v1.memory_management.TensorMemoryAllocator = NumpyAndTensorMemoryAllocator

import lmcache.v1.storage_backend
from lmcache_ascend.mindspore.v1.storage_backend.abstract_backend import StorageBackendInterface___init__
lmcache.v1.storage_backend.StorageBackendInterface.__init__ = StorageBackendInterface___init__

from lmcache_ascend.mindspore.v1.npu_connector import VLLMPagedMemNPUConnectorV2
import lmcache.v1.gpu_connector
lmcache.v1.gpu_connector.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemNPUConnectorV2

import lmcache.v1.system_detection
from lmcache_ascend.mindspore.v1.system_detection import _read_from_sys
lmcache.v1.system_detection.NUMADetector._read_from_sys = _read_from_sys