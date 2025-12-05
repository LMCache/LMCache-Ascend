
import lmcache

import sys

from lmcache_ascend.mindspore import c_ops
sys.modules['lmcache.c_ops'] = c_ops

import lmcache.v1.storage_backend
from lmcache_ascend.mindspore.v1.storage_backend.abstract_backend import StorageBackendInterface___init__
lmcache.v1.storage_backend.StorageBackendInterface.__init__ = StorageBackendInterface___init__


import lmcache.utils
from lmcache_ascend.mindspore.utils import DiskCacheMetadata_to_dict
lmcache.utils.DiskCacheMetadata.to_dict = DiskCacheMetadata_to_dict

from lmcache_ascend.mindspore.utils import DiskCacheMetadata_from_dict
lmcache.utils.DiskCacheMetadata.from_dict = DiskCacheMetadata_from_dict

from lmcache_ascend.mindspore.utils import TORCH_DTYPE_TO_STR_DTYPE
lmcache.utils.TORCH_DTYPE_TO_STR_DTYPE = TORCH_DTYPE_TO_STR_DTYPE

from lmcache_ascend.mindspore.utils import TORCH_STR_TO_DTYPE
lmcache.utils.TORCH_STR_TO_DTYPE = TORCH_STR_TO_DTYPE

from lmcache_ascend.mindspore.utils import CacheEngineKey_to_string
lmcache.utils.CacheEngineKey.to_string = CacheEngineKey_to_string

from lmcache_ascend.mindspore.utils import CacheEngineKey_from_string
lmcache.utils.CacheEngineKey.from_string = CacheEngineKey_from_string

from lmcache_ascend.mindspore.utils import _lmcache_nvtx_annotate
lmcache.utils._lmcache_nvtx_annotate = _lmcache_nvtx_annotate

import lmcache.v1.memory_management
from lmcache_ascend.mindspore.v1.memory_management import MemoryObjMetadata_get_size
lmcache.v1.memory_management.MemoryObjMetadata.get_size = MemoryObjMetadata_get_size

from lmcache_ascend.mindspore.v1.memory_management import _allocate_cpu_memory
lmcache.v1.memory_management._allocate_cpu_memory = _allocate_cpu_memory

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryObj___init__
lmcache.v1.memory_management.TensorMemoryObj.__init__ = TensorMemoryObj___init__

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryObj_get_size
lmcache.v1.memory_management.TensorMemoryObj.get_size = TensorMemoryObj_get_size

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryObj_tensor
lmcache.v1.memory_management.TensorMemoryObj.tensor = TensorMemoryObj_tensor

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryObj_byte_array
lmcache.v1.memory_management.TensorMemoryObj.byte_array = TensorMemoryObj_byte_array

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryAllocator___init__
lmcache.v1.memory_management.TensorMemoryAllocator.__init__ = TensorMemoryAllocator___init__

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryAllocator__Compute_raw_size
lmcache.v1.memory_management.TensorMemoryAllocator._Compute_raw_size = TensorMemoryAllocator__Compute_raw_size

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryAllocator__is_uint8_type
lmcache.v1.memory_management.TensorMemoryAllocator._is_uint8_type = TensorMemoryAllocator__is_uint8_type

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryAllocator_allocate
lmcache.v1.memory_management.TensorMemoryAllocator.allocate = TensorMemoryAllocator_allocate

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryAllocator_batched_allocate
lmcache.v1.memory_management.TensorMemoryAllocator.batched_allocate = TensorMemoryAllocator_batched_allocate

from lmcache_ascend.mindspore.v1.memory_management import TensorMemoryAllocator_memcheck
lmcache.v1.memory_management.TensorMemoryAllocator.memcheck = TensorMemoryAllocator_memcheck

# from lmcache_ascend.mindspore.v1.memory_management import MixedMemoryAllocator___init__
# lmcache.v1.memory_management.MixedMemoryAllocator.__init__ = MixedMemoryAllocator___init__

# from lmcache_ascend.mindspore.v1.memory_management import AdHocMemoryAllocator_allocate
# lmcache.v1.memory_management.AdHocMemoryAllocator.allocate = AdHocMemoryAllocator_allocate

# from lmcache_ascend.mindspore.v1.memory_management import HostMemoryAllocator___init__
# lmcache.v1.memory_management.HostMemoryAllocator.__init__ = HostMemoryAllocator___init__

# from lmcache_ascend.mindspore.v1.memory_management import GPUMemoryAllocator___init__
# lmcache.v1.memory_management.GPUMemoryAllocator.__init__ = GPUMemoryAllocator___init__

from lmcache_ascend.mindspore.v1.npu_connector import VLLMPagedMemNPUConnectorV2, VLLMPagedMemLayerwiseNPUConnector
import lmcache.v1.gpu_connector
lmcache.v1.gpu_connector.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemNPUConnectorV2
lmcache.v1.gpu_connector.VLLMPagedMemLayerwiseGPUConnector = VLLMPagedMemLayerwiseNPUConnector

# from lmcache.v1.token_database import TokenDatabase
# from lmcache_ascend.mindspore.v1.token_database import TokenDatabase__hash_tokens
# TokenDatabase._hash_tokens = TokenDatabase__hash_tokens

# from lmcache.v1.token_database import ChunkedTokenDatabase
# from lmcache_ascend.mindspore.v1.token_database import ChunkedTokenDatabase_process_tokens
# ChunkedTokenDatabase.process_tokens = ChunkedTokenDatabase_process_tokens

# from lmcache_ascend.mindspore.usage_context import InitializeUsageContext
# import lmcache.usage_context
# lmcache.usage_context.InitializeUsageContext = InitializeUsageContext

# from lmcache.v1.cache_engine import LMCacheEngine
# from lmcache_ascend.mindspore.v1.cache_engine import LMCacheEngine_store
# LMCacheEngine.store = LMCacheEngine_store

# from lmcache_ascend.mindspore.v1.cache_engine import LMCacheEngine_retrieve
# LMCacheEngine.retrieve = LMCacheEngine_retrieve

from lmcache_ascend.mindspore.v1.protocol import DTYPE_TO_INT, INT_TO_DTYPE
import lmcache.v1.protocol
lmcache.v1.protocol.DTYPE_TO_INT = DTYPE_TO_INT
lmcache.v1.protocol.INT_TO_DTYPE = INT_TO_DTYPE

# from lmcache_ascend.mindspore.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient_lookup
# from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient
# LMCacheLookupClient.lookup = LMCacheLookupClient_lookup

# from lmcache_ascend.mindspore.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer___init__
# from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer
# LMCacheLookupServer.__init__ = LMCacheLookupServer___init__

# from lmcache_ascend.mindspore.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl_start_load_kv
# from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
# LMCacheConnectorV1Impl.start_load_kv = LMCacheConnectorV1Impl_start_load_kv

# from lmcache_ascend.mindspore.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl_wait_for_save
# from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
# LMCacheConnectorV1Impl.wait_for_save = LMCacheConnectorV1Impl_wait_for_save

# from lmcache_ascend.mindspore.v1.memory_management import PinMemoryAllocator___init__
# lmcache.v1.memory_management.PinMemoryAllocator.__init__ = PinMemoryAllocator___init__

import lmcache.v1.system_detection
from lmcache_ascend.mindspore.v1.system_detection import _read_from_sys
lmcache.v1.system_detection.NUMADetector._read_from_sys = _read_from_sys