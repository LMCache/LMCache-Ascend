# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache_ascend import _build_info

# NOTE: Must be manually edited per each version and
# is also used by the test infrastructure.
LMCACHE_UPSTREAM_TAG = "v0.3.12"
LMCACHE_ASCEND_PATCHED = False

# Check if we've already patched to avoid redundant work
if not LMCACHE_ASCEND_PATCHED:
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

        # First Party
        from lmcache_ascend.v1.transfer_channel import (
            CreateTransferChannel as AscendCreateTransferChannel,
        )
        from lmcache_ascend.v1.transfer_channel import (
            get_correct_device as ascend_get_correct_device,
        )

        # Make sure to import before importing init_lmcache_engine, otherwise
        # CreateTransferChannel gets patched after the original version is
        # already imported.
        sys.modules[
            "lmcache.v1.transfer_channel"
        ].CreateTransferChannel = AscendCreateTransferChannel
        sys.modules[
            "lmcache.v1.transfer_channel.transfer_utils"
        ].get_correct_device = ascend_get_correct_device

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
        import lmcache.v1.multiprocess.custom_types as lm_mp_types

        # First Party
        from lmcache_ascend.v1.multiprocess.custom_types import AscendIPCWrapper

        lm_mp_types.CudaIPCWrapper = AscendIPCWrapper

        # Third Party
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager

        # First Party
        import lmcache_ascend.v1.kv_layer_groups as ascend_kv_layer_groups

        KVLayerGroupsManager.build_kv_layer_groups = (
            ascend_kv_layer_groups.build_kv_layer_groups
        )
        KVLayerGroupInfo.hidden_dim_size = property(
            ascend_kv_layer_groups.patched_hidden_dim_size
        )

        # Third Party
        import lmcache.v1.storage_backend.connector.mooncakestore_connector as lmc_mks_connector  # noqa: E501

        # First Party
        from lmcache_ascend.v1.storage_backend.connector.mooncakestore_connector import (  # noqa: E501
            _batched_put_with_metadata,
            _batched_put_zero_copy,
        )

        # NOTE (gingfung): these two function patches fixes the double free ref counts
        # we took the upstream merged post v0.3.12 into our current branch,
        # please remove after.
        # Ref - https://github.com/LMCache/LMCache/pull/2415
        lmc_mks_connector.MooncakestoreConnector._batched_put_zero_copy = (
            _batched_put_zero_copy
        )
        lmc_mks_connector.MooncakestoreConnector._batched_put_with_metadata = (
            _batched_put_with_metadata
        )

        # Third Party
        import lmcache.integration.vllm.vllm_v1_adapter

        # NOTE (gingfung): this is the main entry point of LMCache, and since we are
        # patching this, every time we upgrade, we should re-evaluate the function, as
        # the experience is that this function signatures or init process will change
        # every N versions.
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

        # Third Party
        import lmcache.v1.lookup_client.lmcache_lookup_client as lmc_lookup_client

        # First Party
        from lmcache_ascend.v1.lookup_client.lmcache_lookup_client import lookup

        lmc_lookup_client.LMCacheLookupClient.lookup = lookup

        # Third Party
        import lmcache.v1.token_database as lmc_token_database

        # First Party
        from lmcache_ascend.v1.token_database import process_tokens

        lmc_token_database.SegmentTokenDatabase.process_tokens = process_tokens

    elif _build_info.__framework_name__ == "mindspore":
        # First Party
        import lmcache_ascend.mindspore  # noqa: F401
    else:
        raise ValueError("Unsupported framework!")

    LMCACHE_ASCEND_PATCHED = True
