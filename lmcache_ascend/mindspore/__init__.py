# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, E501

# Standard
import sys

# First Party
from lmcache_ascend import c_ops

# Third Party
import lmcache

sys.modules["lmcache.c_ops"] = c_ops

# Third Party
import lmcache.v1.storage_backend.storage_manager as sm_module

# First Party
from lmcache_ascend.mindspore.v1.storage_backend.storage_manager import (
    StorageManager__init__,
    allocate_and_copy_objects_310p,
)

sm_module.StorageManager.__init__ = StorageManager__init__
sm_module.StorageManager.allocate_and_copy_objects = allocate_and_copy_objects_310p

# First Party
from lmcache_ascend.mindspore.v1.memory_management import _allocate_cpu_memory

lmcache.v1.memory_management._allocate_cpu_memory = _allocate_cpu_memory

# First Party
from lmcache_ascend.mindspore.v1.memory_management import NumpyAndTensorMemoryObj

lmcache.v1.memory_management.TensorMemoryObj = NumpyAndTensorMemoryObj

# First Party
from lmcache_ascend.mindspore.v1.memory_management import NumpyAndTensorMemoryAllocator

lmcache.v1.memory_management.TensorMemoryAllocator = NumpyAndTensorMemoryAllocator

# First Party
from lmcache_ascend.mindspore.v1.storage_backend.abstract_backend import (
    StorageBackendInterface___init__,
)

# Third Party
import lmcache.v1.storage_backend

lmcache.v1.storage_backend.StorageBackendInterface.__init__ = (
    StorageBackendInterface___init__
)

# First Party
from lmcache_ascend.mindspore.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConnector__register_cpu_buffer,
)

# Third Party
import lmcache.v1.storage_backend.connector.mooncakestore_connector as mooncakestore_connector

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

# First Party
from lmcache_ascend.mindspore.v1.npu_connector import VLLMPagedMemNPUConnectorV2

# Third Party
import lmcache.v1.gpu_connector

lmcache.v1.gpu_connector.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemNPUConnectorV2

# First Party
from lmcache_ascend.mindspore.v1.system_detection import _read_from_sys

# Third Party
import lmcache.v1.system_detection

lmcache.v1.system_detection.NUMADetector._read_from_sys = _read_from_sys

# Third Party
import lmcache.integration.vllm.vllm_v1_adapter

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    init_lmcache_engine as ascend_init_lmcache_engine,
)

lmcache.integration.vllm.vllm_v1_adapter._init_lmcache_engine = (
    ascend_init_lmcache_engine
)
