# SPDX-License-Identifier: Apache-2.0
from functools import partial
import sys
import lmcache
import lmcache_ascend
import lmcache_ascend.c_ops as ascend_c_ops

sys.modules["lmcache.c_ops"] = ascend_c_ops

from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    init_lmcache_engine as ascend_init_lmcache_engine,
)

from lmcache_ascend.v1.blend.utils import get_or_create_blender
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
LMCBlenderBuilder.get_or_create = partial(get_or_create_blender, LMCBlenderBuilder)

import lmcache.integration.vllm.vllm_v1_adapter
lmcache.integration.vllm.vllm_v1_adapter._init_lmcache_engine = ascend_init_lmcache_engine
