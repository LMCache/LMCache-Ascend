# SPDX-License-Identifier: Apache-2.0

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from ._version import __version__ as __version__  # noqa: F401  # isort:skip
from ._version import __version_tuple__ as __version_tuple__  # noqa: F401  # isort:skip

# Standard
from typing import Any
import sys

# First Party
from lmcache_ascend import _build_info

# NOTE: Must be manually edited per each version and
# is also used by the test infrastructure.
LMCACHE_UPSTREAM_TAG = "v0.4.4"
LMCACHE_ASCEND_PATCHED = False
# Set once _patch_lazy_memory_allocator successfully swaps the pool to
# alloc_pinned_ptr; the patch is exception-safe and retried after _patch_ops.
_lazy_alloc_patched = False


def _is_sglang_runtime():
    return "sglang" in sys.modules or any("sglang" in arg for arg in sys.argv)


def _is_vllm_runtime():
    return "vllm" in sys.modules or any("vllm" in arg for arg in sys.argv)


def _seed_partial_lmcache_device_ops() -> None:
    """Bridge ``lmcache.device_ops`` while ``import lmcache`` is still running.

    On the serving path the plugin is first pulled in by lmcache's own
    platform init (``NpuDeviceOps.ensure_native`` -> ``import
    lmcache_ascend.c_ops``), i.e. in the middle of ``import lmcache`` --
    before ``lmcache/__init__.py`` binds its ``device_ops`` singleton (which
    lives after the platform import that invoked us). Core modules imported
    during the patch activation (``lmcache.v1.kv_layer_groups``,
    ``lmcache.storage_backend.serde.cachegen_decoder``, ...) do
    ``from lmcache import device_ops`` at module level and would raise
    ``ImportError`` on the half-initialized package, aborting the whole
    activation -- lmcache then soft-fails to "plugin not found" and every MP
    transfer stays on the torch baseline (~8x slower).

    Seed a lazy proxy for exactly that window: ``lmcache/__init__.py``
    overwrites the attribute with the real singleton right after platform
    init returns, and any proxy holder bound during the window transparently
    delegates to it on every access. No-op whenever ``device_ops`` already
    exists (plugin imported after lmcache finished, or plugin-first).
    """
    # Third Party
    import lmcache

    if hasattr(lmcache, "device_ops"):
        return

    class _LazyDeviceOpsProxy:
        """Resolve the real ``lmcache.device_ops`` singleton per access.

        Accesses before lmcache binds its singleton (module-level grabs such
        as ``system_detection``'s ``get_gpu_pci_bus_id = device_ops.
        get_gpu_pci_bus_id``) are served from the ascend extension directly
        -- the same function objects ``bind_native`` later puts on the real
        instance -- and anything the extension does not export falls back to
        the torch-baseline method, so no caller degrades to a broken binding.
        """

        def __getattr__(self, name: str) -> Any:
            # Standard
            import sys

            real = getattr(sys.modules.get("lmcache"), "device_ops", None)
            if real is not None and not isinstance(real, _LazyDeviceOpsProxy):
                return getattr(real, name)
            # First Party
            import lmcache_ascend.c_ops as ascend_c_ops

            sym = getattr(ascend_c_ops, name, None)
            if sym is not None:
                return sym
            # Third Party
            from lmcache.v1.platform.base.device_ops import DeviceOps

            return getattr(DeviceOps(), name)

    # Overwritten with the real singleton by lmcache's own __init__ as soon
    # as the platform import currently in progress returns.
    lmcache.device_ops = _LazyDeviceOpsProxy()  # type: ignore[assignment]

    # Third Party
    from lmcache.logging import init_logger

    init_logger(__name__).debug("Seeded lazy lmcache.device_ops proxy (partial import)")


def _patch_lazy_memory_allocator():
    """Back the LazyMemoryAllocator pool with aclrtMallocHost-pinned memory.

    Upstream allocates the cache-server pool with ``torch.empty`` and pins each
    chunk post-hoc via ``torch_dev.ext.pin_memory`` (``aclrtHostRegister``).
    ``aclrtHostRegister`` is unreliable on ``torch.empty``/malloc memory at pool
    scale (intermittent ``507899``), so the per-chunk pins fail noisily and the
    pool stays unpinned. The compiled ascend helper ``alloc_pinned_ptr`` instead
    does ``aclrtMallocHost`` + an internal ``register_ptr`` in C++, yielding
    memory that is pinned and registered at allocation time (async D2H works with
    no post-hoc register).

    This replaces ``__init__`` to source the whole pool from ``alloc_pinned_ptr``
    (NUMA binding is intentionally not applied -- the server runs without a NUMA
    mapping), makes ``_pin_memory_chunk`` a no-op (the pool is already pinned), and
    frees via ``free_pinned_ptr`` at close. No-op when ``alloc_pinned_ptr`` is
    absent (non-ascend build).
    """
    # Applied once; idempotent and exception-safe so it can be retried after
    # ``_patch_ops`` sets ``lmcache.c_ops`` (the early call during ``import
    # lmcache`` hits a circular ``import lmcache.c_ops`` and is skipped).
    global _lazy_alloc_patched
    if _lazy_alloc_patched:
        return

    # Standard
    import ctypes
    import threading

    # Third Party
    from lmcache import torch_dev
    from lmcache.logging import init_logger
    import torch

    _logger = init_logger(__name__)
    try:
        # Third Party
        from lmcache.v1.memory_allocators.lazy_memory_allocator import (
            AddressManager,
            LazyMemoryAllocator,
            TensorMemoryAllocator,
            align_to,
        )
    except Exception as exc:  # circular during early activation; retry later
        _logger.debug(
            "LazyMemoryAllocator patch deferred (lmcache.v1 not ready yet): %r",
            exc,
        )
        return

    def _ascend_init(
        self: LazyMemoryAllocator,
        init_size: int,
        final_size: int,
        align_bytes: int = AddressManager.ALIGN_BYTES,
        numa_mapping: Any = None,
    ) -> None:
        """Mirror upstream ``__init__`` but back the buffer with aclrtMallocHost."""
        # Lazy import: at plugin-activation time ``lmcache.c_ops`` has not yet been
        # swapped to the ascend backend (``_patch_ops`` runs later), so resolve the
        # ascend extension directly here, at construction time.
        # First Party
        import lmcache_ascend.c_ops as ascend_c_ops

        self._use_numa = False
        self._curr_size = align_to(init_size, self.PIN_CHUNK_SIZE)
        self._final_size = align_to(final_size, self.PIN_CHUNK_SIZE)
        try:
            # Third Party
            from lmcache.v1.platform import current_device_spec

            pin_supported = current_device_spec.is_pin_supported
        except ImportError:  # upstream <= #4001 exposed torch_dev.ext
            pin_supported = torch_dev.ext.is_pin_supported
        if not pin_supported:
            raise RuntimeError(
                "Backend does not support memory pinning. "
                "LazyMemoryAllocator requires pinned memory."
            )
        self._pin_record: list[tuple[int, int]] = []
        # Ensure an ACL context on this thread: aclrtMallocHost (inside
        # alloc_pinned_ptr) fails with 107002 if no device op has run yet, e.g.
        # on the cache-server thread. Idempotent on workers that already have one.
        if torch.npu.is_available():
            torch.npu.set_device(torch.npu.current_device())
        # Whole pool from aclrtMallocHost: pinned + registered at allocation time,
        # so no per-chunk aclrtHostRegister is needed (or reliable).
        ptr = ascend_c_ops.alloc_pinned_ptr(self._final_size, 0)
        arr_type = ctypes.c_uint8 * self._final_size
        self._buffer = torch.frombuffer(arr_type.from_address(ptr), dtype=torch.uint8)
        self._ascend_pool_ptr: int = ptr
        self._allocator = TensorMemoryAllocator(
            tensor=self._buffer,
            align_bytes=align_bytes,
            init_address_space=self._curr_size,
        )
        self._address_manager = self._allocator.address_manager
        self._stop_expand = threading.Event()
        self._expand_thread = threading.Thread(
            target=self._expand_worker, daemon=True, name="lazy-mem-expand-thread"
        )
        self._expand_thread.start()

    def _ascend_pin_memory_chunk(self, offset: int, size: int) -> None:
        """No-op: the pool is already pinned by ``alloc_pinned_ptr``."""
        return

    def _ascend_close(self: LazyMemoryAllocator) -> None:
        """Stop the expand thread and release the aclrtMallocHost allocation."""
        # First Party
        import lmcache_ascend.c_ops as ascend_c_ops

        self._stop_expand.set()
        self._expand_thread.join()
        # Releases the internal register_ptr + the aclrtMallocHost allocation.
        ascend_c_ops.free_pinned_ptr(self._ascend_pool_ptr)

    LazyMemoryAllocator.__init__ = _ascend_init  # type: ignore[assignment]
    LazyMemoryAllocator._pin_memory_chunk = _ascend_pin_memory_chunk  # type: ignore[assignment]
    LazyMemoryAllocator.close = _ascend_close  # type: ignore[assignment]
    _lazy_alloc_patched = True
    _logger.info(
        "Routed LazyMemoryAllocator pool through alloc_pinned_ptr (aclrtMallocHost)"
    )


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

    # P2P sync control-plane lookup cache (scheduler lookup daemon thread)
    lmcache.v1.config._CONFIG_DEFINITIONS["p2p_sync_lookup_cache_ttl"] = {
        "type": float,
        "default": 5.0,
        "env_converter": float,
        "description": "TTL in seconds for entries in the P2P sync lookup cache. "
        "Used on the sync ZMQ control-plane path for batched peer lookups.",
    }
    lmcache.v1.config._CONFIG_DEFINITIONS["p2p_sync_lookup_cache_max_entries"] = {
        "type": int,
        "default": 1024,
        "env_converter": int,
        "description": "Maximum number of entries in the P2P sync lookup cache. "
        "Oldest entries are evicted when the limit is exceeded.",
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

    # Add enable_chunk_hashes_return config
    lmcache.v1.config._CONFIG_DEFINITIONS["enable_chunk_hashes_return"] = {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Whether to track chunk hashes during lookup and "
        "include them in request_finished return params. "
        "If True, chunk_hashes will be available in return_params. "
        "Default is False (disabled, no impact on original functionality).",
    }

    # Add lookup_hashes_cache_size config
    lmcache.v1.config._CONFIG_DEFINITIONS["lookup_hashes_cache_size"] = {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": "Maximum number of cached chunk hash entries. "
        "When exceeded, the oldest entry is evicted to prevent unbounded "
        "memory growth. Default is 0 (unlimited).",
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

    # Third Party
    # Merge fallback functions that ascend c_ops doesn't implement
    # (e.g., alloc_shm_pinned_ptr, free_shm_pinned_ptr, hugepage
    # functions).  This must happen BEFORE any downstream module
    # imports lmcache.c_ops, otherwise the bound reference will miss
    # these symbols.
    import lmcache.v1.platform.torch_ops as python_ops_fallback

    # First Party
    import lmcache_ascend.c_ops as ascend_c_ops

    for attr_name in dir(python_ops_fallback):
        if not attr_name.startswith("__") and not hasattr(ascend_c_ops, attr_name):
            setattr(ascend_c_ops, attr_name, getattr(python_ops_fallback, attr_name))

    # LMCache v0.4.2 introduces GPUKVFormat enum in c_ops (CUDA pybind).
    # Ascend c_ops doesn't have it, so we provide a compatible mock
    # to avoid AttributeError when upstream code references it.
    if not hasattr(ascend_c_ops, "GPUKVFormat"):

        class GPUKVFormat(IntEnum):
            # Keep numeric values in lockstep with ``csrc/mem_kernels.cuh``
            # (CUDA ``lmcache.c_ops.GPUKVFormat``) so IntEnum comparisons stay
            # consistent when upstream MP code passes raw ints.
            NB_NL_TWO_BS_NH_HS = 0
            NL_X_TWO_NB_BS_NH_HS = 1
            NL_X_NB_TWO_BS_NH_HS = 2
            NL_X_NB_BS_HS = 3
            TWO_X_NL_X_NBBS_NH_HS = 4
            NL_X_NBBS_ONE_HS = 5
            NL_X_TWO_NB_NH_BS_HS = 6
            NL_X_NB_TWO_NH_BS_HS = 7
            NB_NL_TWO_NH_BS_HS = 8

        ascend_c_ops.GPUKVFormat = GPUKVFormat

    # Block kernel: 16 separate K/V, 17 packed multi-plane (MLA/DSA/DSv4),
    # 13 fused NH_CS. Annotation has no Tensor so MP stays in ptr mode.
    # The host prepare_group/validate_launch pair is the SINGLE authority on
    # which geometries launch natively: this wrapper only does
    # tensor/descriptor conversion and lifetime management — no narrow-tail
    # admission checks here.
    _native_block = ascend_c_ops.multi_layer_block_kv_transfer
    _fmt_13 = int(ascend_c_ops.EngineKVFormat.NL_X_NB_BS_NH_CS)
    _fmt_16 = int(ascend_c_ops.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS)
    _fmt_17 = int(ascend_c_ops.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS)

    def _paged_arg_to_ptr_tensor(paged, device):
        """NPU cache context returns per-layer tensors; the C++ op wants int64 ptrs."""
        import torch

        if isinstance(paged, torch.Tensor):
            return paged
        ptrs: list[int] = []
        for layer in paged:
            if isinstance(layer, (tuple, list)):
                ptrs.extend(int(plane.data_ptr()) for plane in layer)
            else:
                ptrs.append(int(layer.data_ptr()))
        return torch.tensor(ptrs, dtype=torch.int64, device=device)

    def multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor,
        lmcache_objects_ptrs: list[int],
        block_ids,
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    ):
        if int(engine_kv_format) in (_fmt_13, _fmt_16, _fmt_17):
            import torch

            paged = _paged_arg_to_ptr_tensor(paged_buffer_ptrs_tensor, device)
            if isinstance(block_ids, torch.Tensor):
                block_ids = block_ids.contiguous()
            return _native_block(
                paged,
                lmcache_objects_ptrs,
                block_ids,
                device,
                direction,
                shape_desc,
                lmcache_chunk_size,
                engine_kv_format,
                skip_prefix_n_blocks,
            )
        return python_ops_fallback.multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )

    ascend_c_ops.multi_layer_block_kv_transfer = multi_layer_block_kv_transfer
    dop = getattr(sys.modules.get("lmcache"), "device_ops", None)
    if dop is not None:
        dop.multi_layer_block_kv_transfer = multi_layer_block_kv_transfer
        # bind_native may have run before this wrap; re-bind the CUDA plan
        # surface so hasattr(device_ops, "execute_object_group_transfer")
        # matches a CUDA build.
        for _plan_name in (
            "execute_object_group_transfer",
            "KernelGroupSpec",
            "StagingCopy",
            "LaunchVar",
            "BatchStep",
        ):
            if hasattr(ascend_c_ops, _plan_name):
                setattr(dop, _plan_name, getattr(ascend_c_ops, _plan_name))

    sys.modules["lmcache.c_ops"] = ascend_c_ops

    try:
        import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as _ldt

        _ldt._HAS_NATIVE_OBJECT_GROUP_TRANSFER = hasattr(
            ascend_c_ops, "execute_object_group_transfer"
        )
    except ImportError:
        pass


def _patch_storage_backend_init():
    # Third Party
    import lmcache.v1.storage_backend as lm_storage_backend

    # First Party
    from lmcache_ascend.v1.storage_backend import (
        CreateStorageBackends as ascend_create_storage_backends,
    )

    lm_storage_backend.CreateStorageBackends = ascend_create_storage_backends


def _patch_storage_manager():
    # Rebind StorageManager.get / batched_get so the delay-pull proxy
    # write-back guard lives in the Ascend overlay instead of upstream LMCache.
    # Also rebind LocalCPUBackend/LocalDiskBackend.touch_cache so a key evicted
    # between lookup-pin and touch_cache degrades the eviction-policy update to a
    # no-op instead of aborting the lookup (which dropped local hits and timed
    # out the lookup RPC). Prefetch all-done callback mirrors loaded tiers into
    # the local hot cache when enabled. See lmcache_ascend.v1.storage_backend
    # .storage_manager.
    # Third Party
    import lmcache.v1.storage_backend.local_cpu_backend as lm_local_cpu_backend
    import lmcache.v1.storage_backend.local_disk_backend as lm_local_disk_backend
    import lmcache.v1.storage_backend.storage_manager as lm_storage_manager

    # First Party
    from lmcache_ascend.v1.storage_backend.storage_manager import (
        batched_get as ascend_batched_get,
    )
    from lmcache_ascend.v1.storage_backend.storage_manager import get as ascend_get
    from lmcache_ascend.v1.storage_backend.storage_manager import (
        local_cpu_touch_cache,
        local_disk_touch_cache,
        patched_prefetch_all_done_callback,
    )

    lm_storage_manager.StorageManager.get = ascend_get
    lm_storage_manager.StorageManager.batched_get = ascend_batched_get
    lm_storage_manager.StorageManager.prefetch_all_done_callback = (
        patched_prefetch_all_done_callback
    )
    lm_local_cpu_backend.LocalCPUBackend.touch_cache = local_cpu_touch_cache
    lm_local_disk_backend.LocalDiskBackend.touch_cache = local_disk_touch_cache


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


def _patch_cachegen():
    # Third Party
    import lmcache.storage_backend.serde.cachegen_decoder as cachegen_decoder
    import lmcache.storage_backend.serde.cachegen_encoder as cachegen_encoder

    # First Party
    from lmcache_ascend.serde.pac import pac_decode_function, pac_encode_function

    cachegen_encoder.encode_function = pac_encode_function
    cachegen_decoder.decode_function_gpu = pac_decode_function


def _patch_remote_backend():
    # Standard
    from typing import List, Optional

    # Third Party
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.storage_backend.naive_serde import CacheGenDeserializer
    from lmcache.v1.storage_backend.remote_backend import RemoteBackend

    # The core remote backend implementation deserializes an NPU resident tensor that
    # isn't managed by a parent allocator. To mesh with the rest of LMCache it needs to
    # be a host registered CPU tensor.
    #
    # Patch the get function with that functionality - allocate managed CPU memory,
    # copy over the data
    old_batched_get_blocking = RemoteBackend.batched_get_blocking

    def new_batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        source_bufs = old_batched_get_blocking(self, keys)

        if isinstance(self.deserializer, CacheGenDeserializer):
            allocator = self.get_allocator_backend()

            target_bufs = []
            for source_buf in source_bufs:
                shape = source_buf.tensor.shape
                dtype = source_buf.tensor.dtype

                target_buf = allocator.allocate(shape, dtype)
                target_buf.tensor.copy_(source_buf.tensor, non_blocking=True)
                target_bufs.append(target_buf)
        else:
            target_bufs = source_bufs

        return target_bufs

    RemoteBackend.batched_get_blocking = new_batched_get_blocking


def _patch_gpu_connector():
    """Patch CreateGPUConnector to return NPU connectors on Ascend.

    In LMCache 0.4.2, engine initialization uses CreateGPUConnector()
    as a factory function. We patch it to return Ascend NPU connectors
    instead of the default CUDA ones.

    ``permute_kv_caches_to_contiguous`` was renamed upstream to
    ``attempt_permute_to_contiguous_view`` (now defined in
    ``lmcache.v1.gpu_connector.kv_format.contiguity``). Only the kvcaches-list
    call site in ``gpu_connectors.initialize_kvcaches_ptr`` takes the Ascend
    override (K/V-tuple and shared-pool-view handling), so the by-value import
    inside ``gpu_connectors`` is rebound directly -- importing the module here
    first keeps the rebind deterministic regardless of import order. The
    single-tensor callers of the contiguity module (cuda/cpu IPC wrappers,
    shm) must keep upstream semantics and are left untouched.
    """
    # Standard

    # Third Party
    import lmcache.v1.gpu_connector.gpu_connectors as gpu_connectors_mod

    # First Party
    from lmcache_ascend.v1.npu_connector.utils import permute_kv_caches_to_contiguous

    # LMC-A: rebind the renamed symbol at its (list) call site instead of the
    # pre-import module patch the old upstream layout required.
    gpu_connectors_mod.attempt_permute_to_contiguous_view = (
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


def _patch_logical_block_size():
    """Wrap LMCache ``get_tokens_per_block`` when Ascend ``block_size`` is physical.

    Pre-#13242 pool math still uses ``block_size * compress_ratio``; the wrap
    reports that span for Ascend leaves. Post-#13242 ``block_size`` is already
    logical, so LMCache core is left unpatched. Never mutate ``spec.block_size``.
    """
    from lmcache_ascend.integration.vllm.logical_block_size import install_overrides

    install_overrides()


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
    # Third Party
    import lmcache.v1.token_database

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


def _patch_cache_controller_worker():
    # Third Party
    import lmcache.v1.cache_controller.worker as lmc_worker

    # First Party
    from lmcache_ascend.v1.cache_controller.worker import async_put_and_wait_msg

    lmc_worker.LMCacheWorker.async_put_and_wait_msg = async_put_and_wait_msg


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
    from lmcache.v1.memory_allocators import gpu_memory_allocator as lmc_gpu_allocator

    # First Party
    from lmcache_ascend.v1.memory_management import GPUMemoryAllocator__init__

    # LMC-A: GPUMemoryAllocator moved out of lmcache.v1.memory_management in
    # the upstream #4077 refactor; patch the class where it now lives.
    lmc_gpu_allocator.GPUMemoryAllocator.__init__ = GPUMemoryAllocator__init__


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


def _patch_lookup_client_factory():
    # Replace LMCacheAsyncLookupClient with Ascend subclass that caches
    # chunk_hashes during lookup and exposes them via get_cached_hashes().
    # Third Party
    import lmcache.v1.lookup_client.lmcache_async_lookup_client as lmc_async

    # First Party
    from lmcache_ascend.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupClient,
    )

    lmc_async.LMCacheAsyncLookupClient = LMCacheAsyncLookupClient


def _patch_api_server():
    # Register /memory/prefetch and /memory/evict REST endpoints.
    # Third Party
    from lmcache.v1.internal_api_server.api_server import InternalAPIServer

    # First Party
    from lmcache_ascend.v1.internal_api_server import (
        InternalAPIServer__init__,
        _capture_original_init,
    )

    _capture_original_init(InternalAPIServer.__init__)
    InternalAPIServer.__init__ = InternalAPIServer__init__


# Check if we've already patched to avoid redundant work
if not LMCACHE_ASCEND_PATCHED:
    # Standard
    from functools import partial
    import sys

    # Must run before any patch: the activation imports core modules that
    # bind ``lmcache.device_ops`` at module level, which only exists once
    # ``import lmcache`` finished -- and we are typically invoked from inside
    # it (see the docstring).
    _seed_partial_lmcache_device_ops()

    if _build_info.__framework_name__ == "pytorch":
        _patch_lazy_memory_allocator()

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
    # Retry now that _patch_ops has set lmcache.c_ops (the early call during
    # import lmcache hits a circular import lmcache.c_ops and is skipped).
    _patch_lazy_memory_allocator()
    if is_vllm:
        _patch_gpu_connector()

    _patch_hash_token()

    _patch_cachegen()
    _patch_remote_backend()

    if _build_info.__framework_name__ == "pytorch":
        _patch_storage_backend_init()
        _patch_storage_manager()
        _patch_transfer_channel()
        _patch_cacheblend()
        _patch_lookup_client()
        _patch_cache_controller_worker()
        _patch_rpc_utils()

    if is_sgl:
        _patch_sgl()
    elif is_vllm:
        if _build_info.__framework_name__ == "pytorch":
            _patch_sys_detection()

        _patch_lookup_client_factory()
        _patch_vllm_v1_adapter()
        _patch_logical_block_size()

        _patch_cache_engine()

        _patch_api_server()

    if _build_info.__framework_name__ == "mindspore":
        # First Party
        import lmcache_ascend.mindspore  # noqa: F401

    LMCACHE_ASCEND_PATCHED = True
