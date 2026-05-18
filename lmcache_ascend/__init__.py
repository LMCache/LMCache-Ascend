# SPDX-License-Identifier: Apache-2.0

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from ._version import __version__ as __version__  # noqa: F401  # isort:skip
from ._version import __version_tuple__ as __version_tuple__  # noqa: F401  # isort:skip

# Standard
import asyncio
import functools
import sys
import time

# First Party
from lmcache_ascend import _build_info

# NOTE: Must be manually edited per each version and
# is also used by the test infrastructure.
LMCACHE_UPSTREAM_TAG = "v0.4.4"
LMCACHE_ASCEND_PATCHED = False


def _is_sglang_runtime():
    return "sglang" in sys.modules or any("sglang" in arg for arg in sys.argv)


def _is_vllm_runtime():
    return "vllm" in sys.modules or any("vllm" in arg for arg in sys.argv)


def _patch_config():
    # Third Party
    from lmcache.v1.config_base import _to_bool, _to_int_list, create_config_class
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
        "env_converter": int,
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

    # Add new p2p_pull_pending_ttl config
    lmcache.v1.config._CONFIG_DEFINITIONS["p2p_pull_pending_ttl"] = {
        "type": float,
        "default": 360.0,
        "env_converter": float,
        "description": "TTL in seconds for pull-pending entries on the sender side. "
        "If a receiver crashes and never sends PullDoneSignal, "
        "pinned MemObjs are released after this timeout. "
        "This config is only used when p2p_pull_mode is set to True.",
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
        "pd_pull_mode is set to True and pd_use_npu is set to True."
        "Set at the receiver side.",
    }

    # Add new pd_pull_done_port config (list of ports, one per TP rank)
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_pull_done_port"] = {
        "type": list,
        "default": None,
        "env_converter": _to_int_list,
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
        "env_converter": int,
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
        "env_converter": float,
        "description": "The timeout in seconds for the allocation failure backoff. "
        "This config is used to avoid infinite loop for memory allocation.",
    }

    # Add pd_pull_pending_ttl config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_pull_pending_ttl"] = {
        "type": float,
        "default": 360.0,
        "env_converter": float,
        "description": "TTL in seconds for pull-pending entries on the sender side. "
        "If a receiver crashes and never sends PullDoneSignal, "
        "pinned MemObjs are released after this timeout. "
        "This config is only used when pd_pull_mode is set to True.",
    }

    # Add pd_pull_backpressure_reserve_pct config
    lmcache.v1.config._CONFIG_DEFINITIONS["pd_pull_backpressure_reserve_pct"] = {
        "type": float,
        "default": 2.0,
        "env_converter": float,
        "description": "Percentage of the sender buffer pool to reserve as free "
        "headroom in pull mode. New put tasks block when pinned pages "
        "exceed (1 - reserve_pct/100) * total_pages. "
        "This config is only used when pd_pull_mode is set to True.",
    }

    # Add store async
    lmcache.v1.config._CONFIG_DEFINITIONS["store_async"] = {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Whether to use store kvcache asynchronously. "
        "If True, the kvcache will be stored asynchronously. ",
    }

    # Add async store queue size. 0 keeps queue unbounded.
    lmcache.v1.config._CONFIG_DEFINITIONS["store_async_max_queue_size"] = {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": "Maximum number of pending async store tasks in queue. "
        "Set 0 for an unbounded queue; values > 0 enable bounded backpressure.",
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
    # Standard
    from enum import IntEnum

    # First Party
    import lmcache_ascend.c_ops as ascend_c_ops

    # LMCache v0.4.2 introduces GPUKVFormat enum in c_ops (CUDA pybind).
    # Ascend c_ops doesn't have it, so we provide a compatible mock
    # to avoid AttributeError when upstream code references it.
    if not hasattr(ascend_c_ops, "GPUKVFormat"):

        class GPUKVFormat(IntEnum):
            NB_NL_TWO_BS_NH_HS = 0
            NL_X_TWO_NB_BS_NH_HS = 1
            NL_X_NB_TWO_BS_NH_HS = 2
            NL_X_NB_BS_HS = 3
            TWO_X_NL_X_NBBS_NH_HS = 4
            NL_X_NBBS_ONE_HS = 5
            NL_X_TWO_NB_NH_BS_HS = 6
            NL_X_NB_TWO_NH_BS_HS = 7

        ascend_c_ops.GPUKVFormat = GPUKVFormat

    sys.modules["lmcache.c_ops"] = ascend_c_ops


def _patch_storage_backend_init():
    # Third Party
    import lmcache.v1.storage_backend as lm_storage_backend

    # First Party
    from lmcache_ascend.v1.storage_backend import (
        CreateStorageBackends as ascend_create_storage_backends,
    )

    lm_storage_backend.CreateStorageBackends = ascend_create_storage_backends


def _patch_torch_capability():
    # Third Party
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    import torch

    # Note: torch_npu do not support get_device_capability
    capability_mock = lambda *args: (0, 0)
    torch.npu.get_device_capability = capability_mock


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


def _patch_gpu_connector():
    """Patch CreateGPUConnector to return NPU connectors on Ascend.

    In LMCache 0.4.2, engine initialization uses CreateGPUConnector()
    as a factory function. We patch it to return Ascend NPU connectors
    instead of the default CUDA ones.

    ``permute_kv_caches_to_contiguous`` must be patched on
    ``lmcache.v1.gpu_connector.utils`` *before* importing
    ``lmcache.v1.gpu_connector``, so the import in ``gpu_connectors`` binds
    the Ascend implementation. If ``gpu_connectors`` was already loaded,
    also replace its cached reference (same pattern as ``CreateGPUConnector``
    on ``lmcache.v1.manager``).
    """
    # Standard

    # Third Party
    import lmcache.v1.gpu_connector.utils as gpu_utils

    # First Party
    from lmcache_ascend.v1.npu_connector.utils import permute_kv_caches_to_contiguous

    gpu_utils.permute_kv_caches_to_contiguous = permute_kv_caches_to_contiguous

    _gpu_connectors_mod = sys.modules.get("lmcache.v1.gpu_connector.gpu_connectors")
    if _gpu_connectors_mod is not None:
        _gpu_connectors_mod.permute_kv_caches_to_contiguous = (
            permute_kv_caches_to_contiguous
        )

    # Third Party
    import lmcache.v1.gpu_connector as lm_gpu_connector

    # First Party
    from lmcache_ascend.v1.npu_connector import CreateNPUConnector

    lm_gpu_connector.CreateGPUConnector = CreateNPUConnector

    # Also patch the reference in lmcache.v1.manager module, in case it
    # was imported before this patch ran
    _manager_mod = sys.modules.get("lmcache.v1.manager")
    if _manager_mod is not None:
        _manager_mod.CreateGPUConnector = CreateNPUConnector


def _patch_get_vllm_torch_dev():
    """Patch get_vllm_torch_dev to return NPU device on Ascend.

    The upstream function only supports CUDA and XPU. This patch adds
    NPU support by replacing the function with our Ascend-specific version.
    """
    # Third Party
    import lmcache.integration.vllm.utils as lm_utils

    # First Party
    from lmcache_ascend.integration.vllm.utils import (
        get_vllm_torch_dev as ascend_get_vllm_torch_dev,
    )

    lm_utils.get_vllm_torch_dev = ascend_get_vllm_torch_dev


def _patch_vllm_v1_adapter():
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        lmcache_connector as vllm_lmcache_connector,
    )
    import lmcache.integration.vllm.vllm_v1_adapter as lmc_vllm_v1_adapter

    # First Party
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl as ascend_LMCacheAscendConnectorV1Impl,
    )

    lmc_vllm_v1_adapter.LMCacheConnectorV1Impl = ascend_LMCacheAscendConnectorV1Impl

    def handle_preemptions(self, preempted_req_ids):
        method = getattr(self._lmcache_engine, "handle_preemptions", None)
        if callable(method):
            method(preempted_req_ids)

    vllm_lmcache_connector.LMCacheConnectorV1.handle_preemptions = handle_preemptions


def _patch_cache_engine():
    # Third Party
    import lmcache.v1.cache_engine as lmc_cache_engine

    # First Party
    from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

    lmc_cache_engine.LMCacheEngine = AscendLMCacheEngine

    for mod_name in (
        "lmcache.v1.manager",
        "lmcache.integration.vllm.vllm_service_factory",
        "lmcache.v1.standalone.standalone_service_factory",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "LMCacheEngine"):
            mod.LMCacheEngine = AscendLMCacheEngine


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
        normalize_token_ids,
    )

    lmc_lookup_client.LMCacheLookupClient.lookup = normalize_token_ids(
        lmc_lookup_client.LMCacheLookupClient.lookup
    )


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
        LMCacheConnector__init__,
        LMCacheLayerwiseConnector_global_min_tokens,
        LMCacheLayerwiseConnector_start_load_kv,
    )

    lmc_sglang_adapter.LMCacheConnector.__init__ = LMCacheConnector__init__

    lmc_sglang_adapter.LMCacheLayerwiseConnector.global_min_tokens = (
        LMCacheLayerwiseConnector_global_min_tokens
    )

    lmc_sglang_adapter.LMCacheLayerwiseConnector.start_load_kv = (
        LMCacheLayerwiseConnector_start_load_kv
    )

    # Third Party
    import lmcache.v1.memory_management as lmc_memory_management

    # First Party
    from lmcache_ascend.v1.memory_management import GPUMemoryAllocator__init__

    lmc_memory_management.GPUMemoryAllocator.__init__ = GPUMemoryAllocator__init__


def _patch_rpc_utils():
    # Patching this to fix socket path length issues on some systems.
    # The original socket path can exceed Unix domain socket's 107 character
    # limit, causing ZMQ errors. The patched version uses shorter, hash-based
    # identifiers to ensure paths are always under the limit.
    # Third Party
    from lmcache.v1.lookup_client import (
        lmcache_async_lookup_client as lmc_async_lookup_client,
    )
    from lmcache.v1.lookup_client import lmcache_lookup_client as lmc_lookup_client
    import lmcache.v1.offload_server.zmq_server as zmq_server
    import lmcache.v1.rpc_utils

    # First Party
    from lmcache_ascend.v1.rpc_utils import use_short_engine_id

    get_zmq_rpc_path_lmcache = use_short_engine_id(
        lmcache.v1.rpc_utils.get_zmq_rpc_path_lmcache
    )

    lmcache.v1.rpc_utils.get_zmq_rpc_path_lmcache = get_zmq_rpc_path_lmcache

    lmc_lookup_client.get_zmq_rpc_path_lmcache = get_zmq_rpc_path_lmcache
    lmc_async_lookup_client.get_zmq_rpc_path_lmcache = get_zmq_rpc_path_lmcache
    zmq_server.get_zmq_rpc_path_lmcache = get_zmq_rpc_path_lmcache

    # Also patch the factory module if already imported
    _factory_mod = sys.modules.get("lmcache.v1.lookup_client.factory")
    if _factory_mod is not None:
        _factory_mod.get_zmq_rpc_path_lmcache = get_zmq_rpc_path_lmcache


def _patch_cache_engine_agentos():
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.event_manager import EventStatus, EventType

    _original_cleanup = LMCacheEngine.cleanup_memory_objs
    _original_compress = LMCacheEngine.compress
    _original_decompress = LMCacheEngine.decompress
    _original_retrieve = LMCacheEngine.retrieve

    def _patched_cleanup(self, lookup_id):
        try:
            if (
                self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.DONE
            ):
                return
            future = self.event_manager.pop_event(EventType.LOADING, lookup_id)
            memory_objs = future.result()
            memory_objs_flat = []
            for m in memory_objs:
                memory_objs_flat.extend(m)
            for key, memory_obj in memory_objs_flat:
                try:
                    memory_obj.ref_count_down()
                except Exception:
                    pass
        except Exception:
            pass

    def _patched_compress(self, location="LocalCPU"):
        if self.lookup_pins is None:
            return 0
        event_id = list(self.lookup_pins.keys())[0]
        if event_id not in self.lookup_pins:
            return 0
        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]
        memory_objs = self.storage_manager.batched_get(
            keys=keys, location=location
        )
        serializer = self.serializer
        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)
        self.lookup_pins.pop(event_id, None)
        self.storage_manager.batched_remove(keys, locations=[location])
        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )
        return len(compressed_memory_objs)

    def _patched_decompress(self, location="LocalCPU"):
        if self.lookup_pins is None:
            return 0
        event_id = list(self.lookup_pins.keys())[0]
        if event_id not in self.lookup_pins:
            return 0
        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]
        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys, location=location
        )
        deserializer = self.deserializer
        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)
        self.lookup_pins.pop(event_id, None)
        self.storage_manager.batched_remove(keys, locations=[location])
        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )
        return len(memory_objs)

    def _patched_retrieve(
        self,
        tokens=None,
        token_mask=None,
        hashes=None,
        offsets=None,
        **kwargs,
    ):
        t0 = time.perf_counter()
        result = _original_retrieve(
            self,
            tokens=tokens,
            token_mask=token_mask,
            hashes=hashes,
            offsets=offsets,
            **kwargs,
        )
        return result

    LMCacheEngine.cleanup_memory_objs = _patched_cleanup
    LMCacheEngine.compress = _patched_compress
    LMCacheEngine.decompress = _patched_decompress
    LMCacheEngine.retrieve = _patched_retrieve


def _patch_storage_manager_agentos():
    from lmcache.v1.storage_backend.storage_manager import StorageManager
    from lmcache.v1.event_manager import EventStatus, EventType

    _original_callback = StorageManager.prefetch_all_done_callback

    def _patched_callback(
        self,
        task,
        lookup_id,
        cum_chunk_lengths_total,
        tier_expected_chunks,
        loading_task_backends=None,
    ):
        _original_callback(
            self, task, lookup_id,
            cum_chunk_lengths_total, tier_expected_chunks,
        )
        if not loading_task_backends:
            return

        res = task.result()

        total_retrieved_chunks = 0
        for tier_idx, tier_result in enumerate(res):
            actual_chunks = len(tier_result)
            total_retrieved_chunks += actual_chunks
            if actual_chunks < tier_expected_chunks[tier_idx]:
                break

        if (
            self.local_cpu_backend is not None
            and self.local_cpu_backend.use_hot
            and total_retrieved_chunks > 0
        ):
            chunk_count = 0
            for tier_idx, tier_result in enumerate(res):
                tier_keys = []
                tier_objs = []
                for key, mem_obj in tier_result:
                    if chunk_count >= total_retrieved_chunks:
                        break
                    tier_keys.append(key)
                    tier_objs.append(mem_obj)
                    chunk_count += 1
                if tier_keys:
                    self.local_cpu_backend.batched_submit_put_task(
                        tier_keys, tier_objs
                    )
                    if (
                        tier_idx < len(loading_task_backends)
                        and loading_task_backends[tier_idx]
                        not in ("LocalCPUBackend", "PDBackend", "MaruBackend")
                    ):
                        for mem_obj in tier_objs:
                            mem_obj.unpin()
                if chunk_count >= total_retrieved_chunks:
                    break

        for tier_idx, tier_result in enumerate(res):
            if tier_idx >= len(loading_task_backends):
                break
            backend_name = loading_task_backends[tier_idx]
            backend = self.storage_backends.get(backend_name)
            if backend is not None:
                for key, _ in tier_result:
                    backend.unpin(key)

    async def _patched_async(
        self,
        lookup_id,
        keys,
        cum_chunk_lengths,
        search_range=None,
        pin=False,
        log_timing=False,
    ):
        num_total_chunks = len(keys)
        num_total_hit_chunks = 0
        cum_chunk_lengths_total = list(cum_chunk_lengths)
        loading_tasks = []
        tier_expected_chunks = []
        loading_task_keys = []
        loading_task_backends = []
        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            num_hit_chunks = await backend.batched_async_contains(
                lookup_id, keys, pin
            )
            if num_hit_chunks == 0:
                continue
            num_total_hit_chunks += num_hit_chunks
            tier_expected_chunks.append(num_hit_chunks)
            backend_keys = keys[:num_hit_chunks]
            loading_task_keys.append(backend_keys)
            loading_task_backends.append(backend_name)
            assert self.async_serializer is not None
            get_coro = self.async_serializer.run(
                backend.batched_get_non_blocking(
                    lookup_id,
                    backend_keys,
                    {
                        "cum_chunk_lengths": cum_chunk_lengths[
                            : num_hit_chunks + 1
                        ]
                    },
                ),
                num_hit_chunks,
            )
            loading_task = asyncio.create_task(get_coro)
            loading_task.add_done_callback(
                functools.partial(
                    self.prefetch_single_done_callback,
                    keys=keys,
                    backend_name=backend_name,
                )
            )
            loading_tasks.append(loading_task)
            cum_chunk_lengths = cum_chunk_lengths[num_hit_chunks:]
            if num_total_hit_chunks == num_total_chunks:
                break
            keys = keys[num_hit_chunks:]

        if num_total_hit_chunks == 0:
            if self.async_lookup_server is not None:
                self.async_lookup_server.send_response_to_scheduler(
                    lookup_id, 0
                )
            return

        async def gather_with_keys():
            loading_results = await asyncio.gather(*loading_tasks)
            return [
                list(zip(keys, results))
                for keys, results in zip(
                    loading_task_keys, loading_results
                )
            ]

        all_done = asyncio.create_task(gather_with_keys())
        self.event_manager.add_event(
            EventType.LOADING,
            lookup_id,
            all_done,
        )
        all_done.add_done_callback(
            lambda future: self.prefetch_all_done_callback(
                future,
                lookup_id,
                cum_chunk_lengths_total,
                tier_expected_chunks,
                loading_task_backends,
            )
        )

    StorageManager.prefetch_all_done_callback = _patched_callback
    StorageManager.async_lookup_and_prefetch = _patched_async


def _patch_pin_monitor_agentos():
    from contextlib import nullcontext

    from lmcache.v1.pin_monitor import PinMonitor

    def _patched_force_unpin(self, memory_obj, elapsed_time):
        obj_lock = getattr(memory_obj, "lock", None) or nullcontext()
        with obj_lock:
            pin_count_to_release = memory_obj.meta.pin_count
            if pin_count_to_release <= 0:
                return
        for _ in range(pin_count_to_release):
            memory_obj.unpin()

    PinMonitor._force_unpin_timeout_object = _patched_force_unpin  # type: ignore[assignment]


def _patch_api_server_agentos():
    from lmcache.v1.internal_api_server.api_server import (
        InternalAPIServer,
        app,
    )

    _original_init = InternalAPIServer.__init__

    def _patched_init(self, lmcache_manager):
        _original_init(self, lmcache_manager)
        if hasattr(lmcache_manager, "config"):
            config = lmcache_manager.config
            lmcache_engine = lmcache_manager.lmcache_engine
            if lmcache_engine is None:
                port_offset = 0
            else:
                port_offset = 1 + lmcache_engine.metadata.worker_id
            app.state.internal_api_server_port_offset = port_offset
            app.state.internal_api_server_port_start = (
                config.internal_api_server_port_start
            )
        from lmcache_ascend.v1.internal_api_server.memory.memory_api import (
            router as memory_router,
        )
        app.include_router(memory_router)

    InternalAPIServer.__init__ = _patched_init


# Check if we've already patched to avoid redundant work
if not LMCACHE_ASCEND_PATCHED:
    # Standard
    from functools import partial
    import sys

    _patch_config()

    is_sgl = _is_sglang_runtime()
    is_vllm = _is_vllm_runtime()

    if _build_info.__framework_name__ == "pytorch":
        # Third Party
        # TODO (gingfung): Currently we patch all the cuda calls
        # due to effort to port all torch.cuda will disabled torch.jit
        # NOTE: this must be done early in the patch prior to the cache engine
        # to avoid falling into non_cuda_equivalent
        _patch_torch_capability()

    _patch_ops()
    if is_vllm:
        _patch_get_vllm_torch_dev()
        _patch_gpu_connector()

    _patch_hash_token()

    if _build_info.__framework_name__ == "pytorch":
        _patch_storage_backend_init()
        _patch_transfer_channel()
        _patch_cacheblend()
        _patch_multi_process()
        _patch_lookup_client()
        _patch_rpc_utils()

    _patch_kv_layer_group()

    if is_sgl:
        _patch_sgl()
    elif is_vllm:
        if _build_info.__framework_name__ == "pytorch":
            _patch_sys_detection()

        _patch_vllm_v1_adapter()

        _patch_cache_engine()

        _patch_cache_engine_agentos()
        _patch_storage_manager_agentos()
        _patch_pin_monitor_agentos()
        _patch_api_server_agentos()

    if _build_info.__framework_name__ == "mindspore":
        # First Party
        import lmcache_ascend.mindspore  # noqa: F401

    LMCACHE_ASCEND_PATCHED = True
