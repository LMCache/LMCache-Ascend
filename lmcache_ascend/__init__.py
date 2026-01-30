# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# First Party
from lmcache_ascend import _build_info

# NOTE: Must be manually edited per each version and
# is also used by the test infrastructure.
LMCACHE_UPSTREAM_TAG = "v0.3.7"

# Check if we've already patched to avoid redundant work
if os.environ.get("LMCACHE_ASCEND_PATCHED") != "1":
    if _build_info.__framework_name__ == "pytorch":
        # Standard
        import sys
        from functools import partial

        # First Party
        import lmcache_ascend.c_ops as ascend_c_ops

        # Third Party
        # TODO (gingfung): Currently we patch all the cuda calls
        # due to effort to port all torch.cuda will disabled torch.jit
        # NOTE: this must be done early in the patch prior to the cache engine
        # to avoid falling into non_cuda_equivalent
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

        import lmcache

        sys.modules["lmcache.c_ops"] = ascend_c_ops

        # The following patches are related for single-layer offload in sync mode
        # i.e. enable_async_loading = False
        # in pre LMCache v0.3.9, the sync mode was broken for layerwise
        # due to storage_manager post init as seen here:
        #  https://github.com/LMCache/LMCache/issues/1794
        #  https://github.com/LMCache/LMCache/pull/1852
        #  https://github.com/LMCache/LMCache/pull/1795
        #  TODO (gingfung): we should remove these once we release v0.3.9
        # Third Party
        # First Party
        from lmcache_ascend.v1.storage_backend.storage_manager import (
            post_init_fix as storage_post_init_fix,
        )

        from lmcache.v1.storage_backend.storage_manager import StorageManager

        StorageManager.post_init = storage_post_init_fix
        # Third Party
        # First Party
        from lmcache_ascend.v1.cache_engine import (
            post_init_fix as cache_engine_post_init_fix,
        )

        from lmcache.v1.cache_engine import LMCacheEngine

        LMCacheEngine.post_init = cache_engine_post_init_fix

        # Third Party
        # First Party
        from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
            init_lmcache_engine as ascend_init_lmcache_engine,
        )
        from lmcache_ascend.v1.blend.utils import get_or_create_blender

        from lmcache.v1.compute.blend.utils import LMCBlenderBuilder

        LMCBlenderBuilder.get_or_create = partial(
            get_or_create_blender, LMCBlenderBuilder
        )

        # Third Party
        import lmcache.integration.vllm.vllm_v1_adapter

        lmcache.integration.vllm.vllm_v1_adapter._init_lmcache_engine = (
            ascend_init_lmcache_engine
        )

        # On OpenEuler and python3.10,
        # the _hash_tokens func hash(None) seems to run into
        # ASLR lead to non-deterministic hashing for builtin hash
        # Third Party
        # First Party
        from lmcache_ascend.v1.tokens_hash import _hash_tokens

        import lmcache.v1.token_database

        lmcache.v1.token_database.TokenDatabase._hash_tokens = _hash_tokens

        # Patching this as on some Ascend machines
        # as the kernel can set the NUMA node to -1.
        # If propagated in the NUMA mapping, this can cause failures to the caller.
        # The patch sanitizes negative values with None,
        # and is up to the caller to handle it.
        # Third Party
        # First Party
        from lmcache_ascend.v1.system_detection import _read_from_sys

        import lmcache.v1.system_detection

        lmcache.v1.system_detection.NUMADetector._read_from_sys = _read_from_sys

        # Third Party
        # First Party
        from lmcache_ascend.v1.lookup_client.lmcache_lookup_client import lookup

        import lmcache.v1.lookup_client.lmcache_lookup_client as lmc_lookup_client

        lmc_lookup_client.LMCacheLookupClient.lookup = lookup

        # Third Party
        # First Party
        from lmcache_ascend.v1.token_database import process_tokens

        import lmcache.v1.token_database as lmc_token_database

        lmc_token_database.SegmentTokenDatabase.process_tokens = process_tokens

        # Third Party
        # First Party
        from lmcache_ascend.v1.config import _validate_config

        import lmcache.v1.config as lmc_config

        lmc_config._validate_config = _validate_config

        # Third Party
        # First Party
        from lmcache_ascend.v1.storage_backend.storage_manager import (
            __init__,
            async_lookup_and_prefetch,
            layerwise_batched_get,
        )

        import lmcache.v1.storage_backend.storage_manager as lmc_storage_manager

        sm = lmc_storage_manager.StorageManager
        sm.__init__ = __init__
        sm.layerwise_batched_get = layerwise_batched_get
        sm.async_lookup_and_prefetch = async_lookup_and_prefetch

    elif _build_info.__framework_name__ == "mindspore":
        # First Party
        import lmcache_ascend.mindspore  # noqa: F401
    else:
        raise ValueError("Unsupported framework!")

    os.environ["LMCACHE_ASCEND_PATCHED"] = "1"
