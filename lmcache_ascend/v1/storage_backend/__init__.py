# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional
import asyncio

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
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


def is_npu_worker(metadata: LMCacheEngineMetadata) -> bool:
    """
    Check if the current role is worker and CUDA is available.

    Args:
        metadata: The LMCache engine metadata.

    Returns:
        True if the worker is not a scheduler and CUDA is available.
    """
    return metadata.role != "scheduler" and torch.npu.is_available()


"""
NOTE (gingfung): Patching the CreateStorageBackends function 
to replace with AscendP2PBackend when p2p is enabled on Ascend. 
Also remove NIXL as it is not supported.
"""


def CreateStorageBackends(
    config: LMCacheEngineConfig,
    metadata: LMCacheEngineMetadata,
    loop: asyncio.AbstractEventLoop,
    dst_device: str = "cuda",
    lmcache_worker: Optional["LMCacheWorker"] = None,  # noqa: F821
) -> OrderedDict[str, StorageBackendInterface]:
    if is_npu_worker(metadata):
        dst_device = f"npu:{torch.npu.current_device()}"
    else:
        dst_device = "cpu"
    storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()

    if config.enable_pd:
        # First Party
        from lmcache_ascend.v1.storage_backend.pd_backend import AscendPDBackend

        storage_backends["PDBackend"] = AscendPDBackend(config, metadata)

    # TODO(Jiayi): The hierarchy is fixed for now
    # NOTE(Jiayi): The local_cpu backend is always created because
    # other backends might need it as a buffer.
    local_cpu_backend: Optional[LocalCPUBackend] = None
    if metadata.role == "scheduler":
        # For scheduler role, local_cpu_backend is None
        pass
    elif not config.enable_pd or config.local_cpu:
        if config.max_local_cpu_size > 0:
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

    if config.enable_p2p:
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

    if config.local_disk and config.max_local_disk_size > 0:
        assert local_cpu_backend is not None
        local_disk_backend = LocalDiskBackend(
            config, loop, local_cpu_backend, dst_device, lmcache_worker, metadata
        )

        backend_name = str(local_disk_backend)
        storage_backends[backend_name] = local_disk_backend

    if config.remote_url is not None:
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
