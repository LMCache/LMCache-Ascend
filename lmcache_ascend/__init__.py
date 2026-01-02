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
        from functools import partial
        import sys

        # Third Party
        # TODO (gingfung): Currently we patch all the cuda calls
        # due to effort to port all torch.cuda will disabled torch.jit
        # NOTE: this must be done early in the patch prior to the cache engine
        # to avoid falling into non_cuda_equivalent
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        import lmcache

        # First Party
        import lmcache_ascend.c_ops as ascend_c_ops

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
        from lmcache.v1.storage_backend.storage_manager import StorageManager

        # First Party
        from lmcache_ascend.v1.storage_backend.storage_manager import (
            post_init_fix as storage_post_init_fix,
        )

        StorageManager.post_init = storage_post_init_fix
        # Third Party
        from lmcache.v1.cache_engine import LMCacheEngine

        # First Party
        from lmcache_ascend.v1.cache_engine import (
            post_init_fix as cache_engine_post_init_fix,
        )

        LMCacheEngine.post_init = cache_engine_post_init_fix

        # Third Party
        from lmcache.v1.compute.blend.utils import LMCBlenderBuilder

        # First Party
        from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
            init_lmcache_engine as ascend_init_lmcache_engine,
        )
        from lmcache_ascend.v1.blend.utils import get_or_create_blender

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
        import lmcache.v1.token_database

        # First Party
        from lmcache_ascend.v1.tokens_hash import _hash_tokens

        lmcache.v1.token_database.TokenDatabase._hash_tokens = _hash_tokens

        # Patching this as on some Ascend machines
        # as the kernel can set the NUMA node to -1.
        # If propagated in the NUMA mapping, this can cause failures to the caller.
        # The patch sanitizes negative values with None,
        # and is up to the caller to handle it.
        # Third Party
        import lmcache.v1.system_detection

        # First Party
        from lmcache_ascend.v1.system_detection import _read_from_sys

        lmcache.v1.system_detection.NUMADetector._read_from_sys = _read_from_sys
    elif _build_info.__framework_name__ == "mindspore":
        # First Party
        import lmcache_ascend.mindspore  # noqa: F401
    else:
        raise ValueError("Unsupported framework!")

    os.environ["LMCACHE_ASCEND_PATCHED"] = "1"
