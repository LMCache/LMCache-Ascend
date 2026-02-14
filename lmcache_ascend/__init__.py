# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache_ascend import _build_info

# NOTE: Must be manually edited per each version and
# is also used by the test infrastructure.
LMCACHE_UPSTREAM_TAG = "v0.3.12"
LMCACHE_ASCEND_PATCHED = False


def _is_sglang_runtime():
    return "sglang" in sys.modules or any("sglang" in arg for arg in sys.argv)

<<<<<<< HEAD
def _is_vllm_runtime():
    return "vllm" in sys.modules or any("vllm" in arg for arg in sys.argv)


=======
>>>>>>> 11b4e37469b203fcb447226bce5237512854b672
def _patch_ops():
    # First Party
    import lmcache_ascend.c_ops as ascend_c_ops

    sys.modules["lmcache.c_ops"] = ascend_c_ops

def _patch_torch_capability():
    # First Party
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    import torch

    capability_mock = lambda *args: (0, 0)
    torch.npu.get_device_capability = capability_mock
    torch.cuda.get_device_capability = capability_mock


def _patch_transfer_channel():
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


def _patch_cacheblend():
    # Third Party
    from lmcache.v1.compute.blend.utils import LMCBlenderBuilder

    # First Party
    from lmcache_ascend.v1.blend.utils import get_or_create_blender

    LMCBlenderBuilder.get_or_create = partial(get_or_create_blender, LMCBlenderBuilder)


def _patch_multi_process():
    # Third Party
    import lmcache.v1.multiprocess.custom_types as lm_mp_types

    # First Party
    from lmcache_ascend.v1.multiprocess.custom_types import AscendIPCWrapper

    lm_mp_types.CudaIPCWrapper = AscendIPCWrapper


def _patch_kv_layer_group():
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


def _patch_mooncake_store_connector():
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


def _patch_init_engine():
    # Third Party
    import lmcache.integration.vllm.vllm_v1_adapter

    # First Party
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        init_lmcache_engine as ascend_init_lmcache_engine,
    )

    # NOTE (gingfung): this is the main entry point of LMCache, and since we are
    # patching this, every time we upgrade, we should re-evaluate the function, as
    # the experience is that this function signatures or init process will change
    # every N versions.
    lmcache.integration.vllm.vllm_v1_adapter._init_lmcache_engine = (
        ascend_init_lmcache_engine
    )


def _patch_hash_token():
    # On OpenEuler and python3.10,
    # the _hash_tokens func hash(None) seems to run into
    # ASLR lead to non-deterministic hashing for builtin hash
    # Third Party
    import lmcache.v1.token_database

    # First Party
    from lmcache_ascend.v1.tokens_hash import _hash_tokens

    lmcache.v1.token_database.TokenDatabase._hash_tokens = _hash_tokens

    # First Party
    from lmcache_ascend.v1.token_database import TokenDatabase_process_tokens

    lmcache.v1.token_database.SegmentTokenDatabase.process_tokens = (
        TokenDatabase_process_tokens
    )


def _patch_lookup_client():
    # Third Party
    import lmcache.v1.lookup_client.lmcache_lookup_client as lmc_lookup_client

    # First Party
    from lmcache_ascend.v1.lookup_client.lmcache_lookup_client import (
        LMCacheLookupClient_lookup,
    )

    lmc_lookup_client.LMCacheLookupClient.lookup = LMCacheLookupClient_lookup


def _patch_sys_detection():
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


def _patch_sgl():
    # Third Party
    import lmcache.integration.sglang.sglang_adapter as lmc_sglang_adapter

    # First Party
    from lmcache_ascend.integration.sglang.sglang_adapter import (
        sglang_init_lmcache_engine,
        LMCacheConnector__init__,
<<<<<<< HEAD
        LMCacheLayerwiseConnector_global_min_tokens,
=======
>>>>>>> 11b4e37469b203fcb447226bce5237512854b672
        LMCacheLayerwiseConnector_start_load_kv,
    )

    lmc_sglang_adapter.init_lmcache_engine = sglang_init_lmcache_engine

    lmc_sglang_adapter.LMCacheConnector.__init__ = LMCacheConnector__init__

<<<<<<< HEAD
    lmc_sglang_adapter.LMCacheLayerwiseConnector.global_min_tokens = (
        LMCacheLayerwiseConnector_global_min_tokens
    )

=======
>>>>>>> 11b4e37469b203fcb447226bce5237512854b672
    lmc_sglang_adapter.LMCacheLayerwiseConnector.start_load_kv = (
        LMCacheLayerwiseConnector_start_load_kv
    )

    # Third Party
    import lmcache.v1.memory_management as lmc_memory_management

    # First Party
    from lmcache_ascend.v1.memory_management import (
        GPUMemoryAllocator__init__,
    )

    lmc_memory_management.GPUMemoryAllocator.__init__ = GPUMemoryAllocator__init__


# Check if we've already patched to avoid redundant work
if not LMCACHE_ASCEND_PATCHED:
    # Standard
    from functools import partial
    import sys

    is_sgl = _is_sglang_runtime()
<<<<<<< HEAD
    is_vllm = _is_vllm_runtime()
=======
>>>>>>> 11b4e37469b203fcb447226bce5237512854b672

    if _build_info.__framework_name__ == "pytorch":
        # Third Party
        # TODO (gingfung): Currently we patch all the cuda calls
        # due to effort to port all torch.cuda will disabled torch.jit
        # NOTE: this must be done early in the patch prior to the cache engine
        # to avoid falling into non_cuda_equivalent
        _patch_torch_capability()

    _patch_ops()

    if _build_info.__framework_name__ == "pytorch":
        _patch_transfer_channel()
        _patch_cacheblend()
        _patch_multi_process()
        _patch_lookup_client()

    _patch_kv_layer_group()
    _patch_mooncake_store_connector()
    _patch_hash_token()

    if is_sgl:
        _patch_sgl()
<<<<<<< HEAD
    elif is_vllm:
        _patch_init_engine()

=======
    else:
        _patch_init_engine()
    
>>>>>>> 11b4e37469b203fcb447226bce5237512854b672
        if _build_info.__framework_name__ == "pytorch":
            _patch_sys_detection()

    if _build_info.__framework_name__ == "mindspore":
        # First Party
        import lmcache_ascend.mindspore  # noqa: F401

    LMCACHE_ASCEND_PATCHED = True
