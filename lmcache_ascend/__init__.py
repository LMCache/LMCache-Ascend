# SPDX-License-Identifier: Apache-2.0
import sys
import lmcache
import lmcache_ascend
import lmcache_ascend.c_ops as ascend_c_ops

sys.modules["lmcache.c_ops"] = ascend_c_ops

from lmcache_ascend.v1.cache_engine import _ascend_create_memory_allocator
import lmcache.v1.cache_engine

lmcache.v1.cache_engine.LMCacheEngineBuilder._Create_memory_allocator = (
    _ascend_create_memory_allocator
)

from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    init_lmcache_engine as ascend_init_lmcache_engine,
)
import lmcache.integration.vllm.vllm_adapter

lmcache.integration.vllm.vllm_adapter.init_lmcache_engine = ascend_init_lmcache_engine
