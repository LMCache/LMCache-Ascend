# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache_ascend import _build_info

# NOTE: Must be manually edited per each version and
# is also used by the test infrastructure.
LMCACHE_UPSTREAM_TAG = "v0.3.12"
LMCACHE_ASCEND_PATCHED = False


def _patch_config():
    # Third Party
    from lmcache.v1.config_base import _to_bool, create_config_class
    import lmcache.v1.config

    # Add new config item for p2p npu usage
    lmcache.v1.config._CONFIG_DEFINITIONS["p2p_use_npu"] = {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Whether to use NPU memory for P2P transfers. "
        "If True, the P2P transfers will be performed on NPU. ",
    }

    # Add new p2p_npu_buffer_size config
    lmcache.v1.config._CONFIG_DEFINITIONS["p2p_npu_buffer_size"] = {
        "type": int,
        "default": 1 * 1024 * 1024 * 1024,
        "description": "The total buffer size in bytes for P2P transfers. "
        "This config is only used when p2p_use_npu is set to True.",
    }

    # Add new p2p_pull_mode config
    lmcache.v1.config._CONFIG_DEFINITIONS["p2p_pull_mode"] = {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Whether to use pull mode for P2P transfers "
        "when using NPU memory. If False, push mode will be used. "
        "This config is only used when p2p_use_npu is set to True.",
    }

    # Add new p2p_delay_pull config
    lmcache.v1.config._CONFIG_DEFINITIONS["p2p_delay_pull"] = {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Whether to delay the pull operation for P2P transfers "
        "when using NPU memory. If True, the pull operation will be delayed "
        "until the data is actually needed. This can help improve performance "
        "in some cases. This config is only used when p2p_use_npu is set to True "
        "and p2p_pull_mode is set to True.",
    }

    # Add new pd_pull_mode config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_pull_mode"] = {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Whether to use pull mode for PD disaggregated transfers. "
        "In pull mode the receiver (decoder) reads KV cache data from the "
        "sender (prefiller) on-demand during batched_to_gpu, using a pipelined "
        "ping-pong approach that overlaps RDMA reads with KV cache scatter. "
        "This avoids bulk NPU memory pre-allocation on the receiver side.",
    }

    # Add new pd_delay_pull config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_delay_pull"] = {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Whether to delay the pull operation for "
        "PD disaggregated transfers when using NPU memory. "
        "If True, the pull operation will be delayed "
        "until the data is actually needed. "
        "This can help improve performance in some cases. "
        "This config is only used when "
        "pd_pull_mode is set to True and p2p_use_npu is set to True."
        "Set at the receiver side.",
    }

    # Add new pd_pull_done_port config (list of ports, one per TP rank)
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_pull_done_port"] = {
        "type": list,
        "default": None,
        "description": "List of ports (one per TP rank) on which the sender "
        "binds a ZMQ PULL socket to receive Done signals from the receiver "
        "in PD pull mode.  If not set, the port is derived as "
        "peer_alloc_port + 100.  Example: [18100, 18101].",
    }

    # Add pd_use_cpu_offload config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_use_cpu_offload"] = {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Whether to use CPU offload for PD transfers. "
        "If True, the KV caches will be offloaded to CPU first "
        "and then transferred to remote npu later. "
        "This config is only used when the role is `sender` "
        "and pd_pull_mode is set to True.",
    }

    # Add pd_cpu_buffer_size config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_cpu_buffer_size"] = {
        "type": int,
        "default": None,
        "description": "The total buffer size in bytes for PD CPU offload. "
        "This config is used when the role is `sender`, "
        "because the kvcaches can be offloaded to cpu first, "
        "and then transferred to remote npu later. "
        "This config is only used when pd_pull_mode is set to True.",
    }

    # Add pd_alloc_fail_backoff_ttl config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_alloc_fail_backoff_ttl"] = {
        "type": float,
        "default": 2.0,
        "description": "The timeout in seconds for the allocation failure backoff"
        "This config is used to avoid infinite loop for memory allocation.",
    }

    # Add pd_pull_pending_ttl config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_pull_pending_ttl"] = {
        "type": float,
        "default": 360.0,
        "description": "TTL in seconds for pull-pending entries on the sender side. "
        "If a receiver crashes and never sends PullDoneSignal, "
        "pinned MemObjs are released after this timeout. "
        "This config is only used when pd_pull_mode is set to True.",
    }

    # Add pd_pull_backpressure_reserve_pct config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_pull_backpressure_reserve_pct"] = {
        "type": float,
        "default": 2.0,
        "description": "Percentage of the sender buffer pool to reserve as free "
        "headroom in pull mode. New put tasks block when pinned pages "
        "exceed (1 - reserve_pct/100) * total_pages. "
        "This config is only used when pd_pull_mode is set to True.",
    }

    namespace_extras = {
        "validate": lmcache.v1.config._validate_config,
        "log_config": lmcache.v1.config._log_config,
        "get_extra_config_value": lmcache.v1.config._get_extra_config_value,
        "get_lmcache_worker_ids": lmcache.v1.config._get_lmcache_worker_ids,
        "from_legacy": classmethod(lmcache.v1.config._from_legacy),
        "get_lookup_server_worker_ids": lmcache.v1.config._get_lookup_server_worker_ids,
    }

    # Re-create the configuration class with the updated definitions
    lmcache.v1.config.LMCacheEngineConfig = create_config_class(
        config_name="LMCacheEngineConfig",
        config_definitions=lmcache.v1.config._CONFIG_DEFINITIONS,
        config_aliases=lmcache.v1.config._CONFIG_ALIASES,
        deprecated_configs=lmcache.v1.config._DEPRECATED_CONFIGS,
        namespace_extras=namespace_extras,
    )

    # If lmcache.integration.vllm.utils was already imported before this
    # patch ran, its module-level ``LMCacheEngineConfig`` still points to
    # the OLD class whose ``_from_file`` closure now iterates the mutated
    # _CONFIG_DEFINITIONS dict (with keys like ``p2p_use_npu``), while the
    # OLD ``__init__`` doesn't accept them → TypeError.  Fix by updating
    # the stale reference.
    _utils_mod = sys.modules.get("lmcache.integration.vllm.utils")
    if _utils_mod is not None:
        _utils_mod.LMCacheEngineConfig = lmcache.v1.config.LMCacheEngineConfig


def _patch_ops():
    # First Party
    import lmcache_ascend.c_ops as ascend_c_ops

    sys.modules["lmcache.c_ops"] = ascend_c_ops


def _patch_storage_backend_init():
    # Third Party
    import lmcache.v1.storage_backend as lm_storage_backend

    # First Party
    from lmcache_ascend.v1.storage_backend import (
        CreateStorageBackends as ascend_create_storage_backends,
    )

    lm_storage_backend.CreateStorageBackends = ascend_create_storage_backends


def _patch_transfer_channel():
    # First Party
    from lmcache_ascend.v1.transfer_channel import (
        get_correct_device as ascend_get_correct_device,
    )

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


def _patch_wait_for_save():
    # Third Party
    import lmcache.integration.vllm.vllm_v1_adapter

    # First Party
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        wait_for_save as ascend_wait_for_save,
    )

    # Fixes a bug where disagg_spec.num_transferred_tokens (initialized to 0)
    # overrides save_spec.skip_leading_tokens via min(), causing redundant
    # full re-saves when there is an existing cache hit.
    lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl.wait_for_save = (
        ascend_wait_for_save
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


# Check if we've already patched to avoid redundant work
if not LMCACHE_ASCEND_PATCHED:
    # Standard
    from functools import partial
    import sys

    _patch_config()

    if _build_info.__framework_name__ == "pytorch":
        # Third Party
        # TODO (gingfung): Currently we patch all the cuda calls
        # due to effort to port all torch.cuda will disabled torch.jit
        # NOTE: this must be done early in the patch prior to the cache engine
        # to avoid falling into non_cuda_equivalent
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

    _patch_ops()
    _patch_hash_token()

    if _build_info.__framework_name__ == "pytorch":
        _patch_storage_backend_init()
        _patch_transfer_channel()
        _patch_cacheblend()
        _patch_multi_process()
        _patch_lookup_client()

    _patch_kv_layer_group()
    _patch_mooncake_store_connector()
    _patch_init_engine()
    _patch_wait_for_save()

    if _build_info.__framework_name__ == "pytorch":
        _patch_sys_detection()

    if _build_info.__framework_name__ == "mindspore":
        # First Party
        import lmcache_ascend.mindspore  # noqa: F401

    LMCACHE_ASCEND_PATCHED = True
