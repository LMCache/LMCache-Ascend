# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, E501

# Standard
import sys

# Third Party
import lmcache

# First Party
from lmcache_ascend import c_ops

sys.modules["lmcache.c_ops"] = c_ops

# First Party
from lmcache_ascend.mindspore.v1.memory_management import _allocate_cpu_memory

lmcache.v1.memory_management._allocate_cpu_memory = _allocate_cpu_memory

# First Party
from lmcache_ascend.mindspore.v1.memory_management import NumpyAndTensorMemoryObj

lmcache.v1.memory_management.TensorMemoryObj = NumpyAndTensorMemoryObj

# First Party
from lmcache_ascend.mindspore.v1.memory_management import NumpyAndTensorMemoryAllocator

lmcache.v1.memory_management.TensorMemoryAllocator = NumpyAndTensorMemoryAllocator

# Third Party
import lmcache.v1.storage_backend

# First Party
from lmcache_ascend.mindspore.v1.storage_backend.abstract_backend import (
    StorageBackendInterface___init__,
)

lmcache.v1.storage_backend.StorageBackendInterface.__init__ = (
    StorageBackendInterface___init__
)

# Third Party
import lmcache.v1.storage_backend.connector.mooncakestore_connector as mooncakestore_connector

# First Party
from lmcache_ascend.mindspore.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConnector__register_cpu_buffer,
)

mooncakestore_connector.MooncakestoreConnector._register_cpu_buffer = (
    MooncakeStoreConnector__register_cpu_buffer
)

# First Party
from lmcache_ascend.mindspore.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConnector__batch_get_into,
)

mooncakestore_connector.MooncakestoreConnector._batch_get_into = (
    MooncakeStoreConnector__batch_get_into
)

# First Party
from lmcache_ascend.mindspore.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConnector__put_without_metadata,
)

mooncakestore_connector.MooncakestoreConnector._put_without_metadata = (
    MooncakeStoreConnector__put_without_metadata
)

# Third Party
import lmcache.v1.gpu_connector

# First Party
from lmcache_ascend.mindspore.v1.npu_connector import VLLMPagedMemNPUConnectorV2

lmcache.v1.gpu_connector.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemNPUConnectorV2

# Third Party
import lmcache.v1.system_detection

# First Party
from lmcache_ascend.mindspore.v1.system_detection import _read_from_sys

lmcache.v1.system_detection.NUMADetector._read_from_sys = _read_from_sys
