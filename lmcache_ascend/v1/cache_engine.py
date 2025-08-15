# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryAllocatorInterface
from .memory_management import AscendMixedMemoryAllocator


def _ascend_create_memory_allocator(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
) -> MemoryAllocatorInterface:
    if config.enable_nixl:
        raise NotImplementedError("Ascend does not support nixl.")

    if config.weka_path is not None or config.gds_path is not None:
        raise NotImplementedError("Ascend does not support Direct Storage.")

    max_local_cpu_size = config.max_local_cpu_size
    return AscendMixedMemoryAllocator(int(max_local_cpu_size * 1024**3))
