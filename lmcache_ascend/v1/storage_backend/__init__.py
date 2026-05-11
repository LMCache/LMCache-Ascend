# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, AbstractSet, Optional
import asyncio

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import storage_plugin_launcher
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
import torch
import torch_npu  # noqa: F401

# First Party
from lmcache_ascend.v1.storage_backend.p2p_backend import AscendP2PBackend

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def is_npu_worker(metadata: LMCacheMetadata) -> bool:
    """
    Check if the current role is worker and NPU is available.

    Args:
        metadata: The LMCache engine metadata.

    Returns:
        True if the worker is not a scheduler and NPU is available.
    """
    return metadata.role != "scheduler" and torch.npu.is_available()


"""
NOTE (gingfung): Patching the CreateStorageBackends function
to replace with AscendP2PBackend when p2p is enabled on Ascend.
Also remove NIXL as it is not supported.
"""


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    loop: asyncio.AbstractEventLoop,
    dst_device: str = "cuda",
    lmcache_worker: Optional["LMCacheWorker"] = None,  # noqa: F821
    skip_backends: Optional[AbstractSet[str]] = None,
    existing_backends: Optional[OrderedDict[str, StorageBackendInterface]] = None,
) -> OrderedDict[str, StorageBackendInterface]:
    if is_npu_worker(metadata):
        dst_device = f"npu:{torch.npu.current_device()}"
    else:
        dst_device = "cpu"
    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
    _skip = skip_backends or set()

    if config.enable_pd:
        # First Party
        from lmcache_ascend.v1.storage_backend.pd import AscendPDBackend

        if config.use_layerwise:
            raise ValueError(
                "Invalid LMCache-Ascend config: `enable_pd=true` is not compatible "
                "with `use_layerwise=true`. PD backend does not support layerwise "
                "mode (including pull/delay-pull paths). Disable one of them."
            )
        storage_backends["PDBackend"] = AscendPDBackend(config, metadata)

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    # Reuse existing LocalCPUBackend when available so that
    # dependent backends (disk, remote, p2p, …) keep working.
    local_cpu_backend: Optional[LocalCPUBackend] = None
    if existing_backends and "LocalCPUBackend" in existing_backends:
        _existing_cpu = existing_backends["LocalCPUBackend"]
        if isinstance(_existing_cpu, LocalCPUBackend):
            local_cpu_backend = _existing_cpu

    if metadata.role == "scheduler":
        # For scheduler role, local_cpu_backend is None
        pass
    elif not config.enable_pd or config.local_cpu:
        if "LocalCPUBackend" in _skip:
            pass  # Skipped — already exists
        elif config.max_local_cpu_size > 0:
            local_cpu_backend = LocalCPUBackend(
                config,
                metadata,
                dst_device,
                lmcache_worker,
            )
            backend_name = str(local_cpu_backend)
            storage_backends[backend_name] = local_cpu_backend
        else:
            logger.info("No cpu memory is allocated as max_local_cpu_size <= 0")

    if config.enable_p2p and "P2PBackend" not in _skip:
        if config.use_layerwise:
            raise ValueError(
                "Invalid LMCache-Ascend config: `enable_p2p=true` is not compatible "
                "with `use_layerwise=true`. The Ascend P2P backend does not support "
                "layerwise mode in current implementation. Disable one of them."
            )
        assert local_cpu_backend is not None
        assert lmcache_worker is not None
        p2p_backend = AscendP2PBackend(
            config,
            metadata,
            loop,
            local_cpu_backend,
            lmcache_worker,
        )
        backend_name = str(p2p_backend)
        storage_backends[backend_name] = p2p_backend

    if (
        config.local_disk
        and config.max_local_disk_size > 0
        and "LocalDiskBackend" not in _skip
    ):
        assert local_cpu_backend is not None
        local_disk_backend = LocalDiskBackend(
            config,
            loop,
            local_cpu_backend,
            dst_device,
            lmcache_worker,
            metadata,
        )

        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    # Handle remote storage plugins (new way)
    if config.remote_storage_plugins and "RemoteBackend" not in _skip:
        for plugin_name in config.remote_storage_plugins:
            assert local_cpu_backend is not None, (
                "Remote backend requires local CPU backend as a buffer."
                "Please turn on local cpu backend with max_local_cpu_size > 0"
            )
            try:
                remote_backend = RemoteBackend(
                    config,
                    metadata,
                    loop,
                    local_cpu_backend,
                    dst_device,
                    plugin_name=plugin_name,
                )
                backend_name = "RemoteBackend-%s" % plugin_name
                storage_backends[backend_name] = remote_backend
                logger.info(
                    "Created remote backend for plugin: %s",
                    plugin_name,
                )
            except Exception as e:
                logger.error(
                    "Failed to create remote backend for plugin %s: %s",
                    plugin_name,
                    e,
                )

    # Handle legacy remote_url (deprecated but still supported)
    if config.remote_url is not None and "RemoteBackend" not in _skip:
        # Log deprecation warning
        logger.warning(
            "remote_url is deprecated and will be removed in a future release. "
            "Please use remote_storage_plugins instead."
        )
        remote_backend = RemoteBackend(
            config,
            metadata,
            loop,
            local_cpu_backend,
            dst_device,
        )
        backend_name = str(remote_backend)
        storage_backends[backend_name] = remote_backend

    if not config.enable_pd or config.local_cpu:
        # Load storage backends from configuration
        storage_plugin_launcher(
            config,
            metadata,
            loop,
            local_cpu_backend,
            dst_device,
            storage_backends,
        )

    # Only wrap if audit is enabled in config
    if config.extra_config is not None and config.extra_config.get(
        "audit_backend_enabled", False
    ):
        # Third Party
        from lmcache.v1.storage_backend.audit_backend import AuditBackend

        # Conditionally wrap backends with audit logging if enabled in config
        audited_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
        for name, backend in storage_backends.items():
            # Wrap each normal backend with AuditBackend
            if not isinstance(backend, LocalCPUBackend):
                audited_backend = AuditBackend(backend)
                audited_backends[name] = audited_backend
                logger.info(f"Wrapped {name} with AuditBackend")
            else:
                audited_backends[name] = backend
                logger.info(f"Do not wrap {name} as it is a LocalCPUBackend")
        return audited_backends
    else:
        # If audit is not enabled, use the original backends
        return storage_backends
