# SPDX-License-Identifier: Apache-2.0
"""Ascend-only MP L1 construction and cleanup over pure ACL Host memory."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar
import ctypes
import sys
import threading

# Third Party
from lmcache.logging import init_logger
import lmcache
import torch

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.distributed.config import (
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
    from lmcache.v1.memory_management import MemoryAllocatorInterface
    from lmcache.v1.multiprocess.config import MPServerConfig

_ALLOCATOR_ENV = "LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR"
_installed = False
logger = init_logger(__name__)
_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass
class AclHostMemoryAllocation:
    """Own an ACL address while exposing a non-owning CPU Tensor view.

    This record owns the successful-release state. It does not synchronize,
    register memory, or free it in a destructor. All views must stop being used
    before release. Release calls must be sequential.
    """

    raw_ptr: int
    size: int
    buffer: torch.Tensor
    _free: Callable[[int], None] = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def released(self) -> bool:
        """Whether the saved native release function has returned successfully."""
        return self._released

    def release(self) -> None:
        """Free once after success, retaining state if native release fails.

        Raises:
            RuntimeError: ACL release failed; memory usability is not guaranteed.
        """
        if self._released:
            return
        self._free(self.raw_ptr)
        self._released = True


def allocate_host_memory(size: int) -> AclHostMemoryAllocation:
    """Allocate pure ACL Host bytes and transfer ownership to the caller.

    The caller prepares the NPU runtime. No device selection, Host mapping,
    registration or base-address adjustment is performed here.

    Args:
        size: Positive byte count representable by ctypes and Tensor views.

    Returns:
        An owned allocation whose Tensor aliases the original ACL address.

    Raises:
        ValueError: Size is outside the supported range.
        RuntimeError: Native symbols are missing or ACL allocation fails.
        Exception: View construction failed; the acquired address is released
            without replacing that exception if cleanup also fails.
    """
    if not 0 < size <= sys.maxsize:
        raise ValueError(f"size must be between 1 and {sys.maxsize}, got {size}")
    array_type = ctypes.c_uint8 * size
    try:
        # First Party
        import lmcache_ascend.c_ops as c_ops

        alloc = c_ops.alloc_host_ptr
        free = c_ops.free_host_ptr
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(
            "ACL Host memory requires c_ops.alloc_host_ptr and c_ops.free_host_ptr"
        ) from exc

    raw_ptr = alloc(size)
    try:
        buffer = torch.frombuffer(array_type.from_address(raw_ptr), dtype=torch.uint8)
        return AclHostMemoryAllocation(raw_ptr, size, buffer, free)
    except BaseException:
        try:
            free(raw_ptr)
        except Exception:
            logger.exception("Failed to release ACL Host allocation at %d", raw_ptr)
        raise


def validate_acl_l1_config(
    config: L1MemoryManagerConfig, device_type: str, *, shm_name: str | None = None
) -> None:
    """Reject unsupported ACL backing before initializing or allocating.

    Args:
        config: DRAM L1 configuration, left unchanged.
        device_type: Resolved LMCache device type; must be npu.
        shm_name: Effective MP override, or None to use the L1 setting.

    Raises:
        ValueError: Platform, backing, size or alignment is unsupported.
    """
    if device_type != "npu":
        raise ValueError("aclrt_malloc_host requires the NPU platform")
    if config.use_lazy:
        raise ValueError("aclrt_malloc_host requires use_lazy=False")
    effective_shm_name = config.shm_name if shm_name is None else shm_name
    if effective_shm_name:
        raise ValueError("aclrt_malloc_host requires an empty shm_name")
    if config.use_hugepages or config.devdax_path:
        raise ValueError("aclrt_malloc_host does not support hugepages or Device-DAX")
    if config.size_in_bytes <= 0:
        raise ValueError("aclrt_malloc_host requires a positive L1 size")
    if config.align_bytes <= 0 or config.align_bytes & (config.align_bytes - 1):
        raise ValueError("align_bytes must be a positive power of two")


def validate_mp_host_memory_config(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    device_type: str,
) -> None:
    """Validate effective MP backing for the installed ACL path before SHM fallback.

    Args:
        mp_config: MP transfer mode and SHM override.
        storage_manager_config: L1 settings; shadow L1s own no memory.
        device_type: Resolved LMCache device type.

    Raises:
        ValueError: An ACL selection contains unsupported memory backing.
    """
    # Third Party
    from lmcache.v1.distributed.config import ShadowL1ManagerConfig

    configs = storage_manager_config.l1_manager_configs
    override_shm = (
        mp_config.supported_transfer_mode != "lmcache_driven"
        and len(configs) == 1
        and mp_config.shm_name is not None
    )
    for config in configs:
        if isinstance(config, ShadowL1ManagerConfig):
            continue
        if config.gds_l1_config is not None:
            raise ValueError("aclrt_malloc_host does not support a GDS L1")
        validate_acl_l1_config(
            config.memory_config,
            device_type,
            shm_name=mp_config.shm_name if override_shm else None,
        )


def make_mp_memory_allocator(
    original_create: Callable[[L1MemoryManagerConfig], MemoryAllocatorInterface],
) -> Callable[..., MemoryAllocatorInterface]:
    """Make the opt-in MP creation wrapper without changing upstream sources.

    Args:
        original_create: The upstream L1 factory whose metadata the wrapper retains.

    Returns:
        An ACL-only factory that validates configuration and initializes the NPU
        runtime before constructing Mixed. Selection happens at installation;
        this wrapper does not reread the environment or call the original factory.
        The close wrapper must be installed first. Errors propagate without
        fallback or independent runtime finalization.
    """

    @wraps(original_create)
    def create(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
        validate_acl_l1_config(config, lmcache.torch_device_type)
        initialize = getattr(lmcache.torch_dev, "init", None)
        if not callable(initialize):
            raise RuntimeError("aclrt_malloc_host requires callable torch_dev.init()")
        initialize()
        return create_acl_mixed_allocator(config, lmcache.torch_device_type)

    return create


def create_acl_mixed_allocator(
    config: L1MemoryManagerConfig, device_type: str
) -> MixedMemoryAllocator:
    """Construct existing Mixed over an Ascend-owned arena without default allocation.

    The caller prepares the runtime and installs make_mixed_close first. Field
    initialization follows the non-paging Mixed constructor at LMCache
    460e46a9; review it when updating the upstream baseline. The allocation's
    released state replaces Mixed's default-path _unregistered flag.

    Args:
        config: Non-lazy, non-SHM DRAM settings for the ACL path.
        device_type: Resolved LMCache device type, required to be npu.

    Returns:
        A Mixed instance whose KV objects are views of the ACL Host arena.

    Raises:
        ValueError: The configuration is unsupported.
        Exception: Allocation or pool construction failed. Acquired memory is
            released on construction failure without masking the original error.
    """
    validate_acl_l1_config(config, device_type)
    # Third Party
    from lmcache.v1.memory_allocators import mixed_memory_allocator as mixed

    allocator = object.__new__(mixed.MixedMemoryAllocator)
    allocation = allocate_host_memory(config.size_in_bytes)
    try:
        # This Ascend-owned attribute identifies the matching close path.
        allocator.ascend_host_allocation = allocation  # type: ignore[attr-defined]
        allocator.numa_mapping = None
        allocator.use_hugepages = False
        allocator.align_bytes = config.align_bytes
        allocator.shm_name = None
        allocator.size = allocation.size
        allocator.buffer = allocation.buffer
        allocator.pin_allocator = mixed.TensorMemoryAllocator(
            allocator.buffer, align_bytes=config.align_bytes
        )
        allocator.host_mem_lock = threading.Lock()
        allocator.buffer_allocator = mixed.BufferAllocator("cpu")
        return allocator
    except BaseException:
        try:
            allocation.release()
        except Exception:
            logger.exception(
                "Failed to release ACL Host allocation at %d", allocation.raw_ptr
            )
        raise


def make_mixed_close(
    original_close: Callable[[MixedMemoryAllocator], None],
) -> Callable[..., None]:
    """Make a close wrapper that dispatches by each instance's allocation owner.

    Args:
        original_close: Upstream close, used unchanged by ordinary instances.

    Returns:
        A sequential close operation. ACL instances synchronize local device
        work then release their saved allocation, once after success. Errors
        propagate without marking success. Callers must already have ended
        communication use; synchronization does not drain HIXL or remote access.
    """

    @wraps(original_close)
    def close(allocator: MixedMemoryAllocator) -> None:
        allocation = getattr(allocator, "ascend_host_allocation", None)
        if allocation is None:
            original_close(allocator)
            return
        if allocation.released:
            return
        # Third Party
        from lmcache.v1.memory_allocators import mixed_memory_allocator as mixed

        if mixed.torch_dev.is_available():
            mixed.torch_dev.synchronize()
        allocation.release()

    return close


def _validated_start(
    original: Callable[Concatenate[MPServerConfig, StorageManagerConfig, _P], _R],
) -> Callable[..., _R]:
    """Keep validation ahead of SHM mutation while preserving the server signature."""

    @wraps(original)
    def start(
        mp_config: MPServerConfig,
        storage_manager_config: StorageManagerConfig,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        validate_mp_host_memory_config(
            mp_config, storage_manager_config, lmcache.torch_device_type
        )
        return original(mp_config, storage_manager_config, *args, **kwargs)

    return start


def install_mp_host_memory(selection: str) -> None:
    """Install the three memory wrappers directly during Ascend activation.

    Args:
        selection: Environment value captured by the package activation hook.
            Default leaves upstream memory handling untouched.

    Calls are sequential and idempotent. No runtime initialization or allocation
    occurs here; later environment changes do not change the selected path.
    During LMCache-first activation, imported modules can use the existing
    device_ops proxy until native binding finishes. Installation errors propagate
    to the package activation hook, which terminates startup rather than letting
    LMCache's backend-import exception handler continue with default memory.
    """
    global _installed
    if _installed or selection == "default":
        return
    if selection != "aclrt_malloc_host":
        raise ValueError(
            f"{_ALLOCATOR_ENV} must be 'default' or 'aclrt_malloc_host', "
            f"got {selection!r}"
        )
    # Third Party
    from lmcache.v1.distributed.memory_manager import l1_memory_manager
    from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
    from lmcache.v1.multiprocess import server

    close = make_mixed_close(MixedMemoryAllocator.close)
    create = make_mp_memory_allocator(l1_memory_manager.create_memory_allocator)
    start = _validated_start(server._start_cache_server)
    # Complete all wrappers first; close must be ready before ACL instances exist.
    MixedMemoryAllocator.close = close  # type: ignore[method-assign]
    l1_memory_manager.create_memory_allocator = create
    server._start_cache_server = start
    _installed = True
