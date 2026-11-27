# SPDX-License-Identifier: Apache-2.0
"""Ops-only binding surface for ``NpuDeviceOps.ensure_native``.

Mirrors upstream's ``lmcache.cuda_ops`` contract: ops plus
``PageBufferShapeDesc`` only. The compiled ``c_ops`` additionally exports
the format enums and ``is_*`` predicates that upstream keeps on
``lmcache_native`` alone. Binding either through
``bind_native`` would violate the device-ops facade contract or shadow
the class-level NPU overrides, so both are filtered out here.

Host allocation prepares the NPU runtime on demand before calling ``c_ops``.
"""

# Third Party
from lmcache.v1.platform import torch_ops as _torch_ops
import torch as _torch

# First Party
import lmcache_ascend.c_ops as _c_ops

# Names upstream keeps on lmcache_native only; never on the ops facade.
_FACADE_EXCLUDES = frozenset(
    (
        "TransferDirection",
        "EngineKVFormat",
        "GPUKVFormat",
        "is_cross_layer",
        "is_kv_list",
        "is_layer_list",
        "is_mla",
    )
)

for _name in dir(_c_ops):
    if _name.startswith("_") or _name in _FACADE_EXCLUDES:
        continue
    _symbol = getattr(_c_ops, _name)
    if not (callable(_symbol) or isinstance(_symbol, type)):
        continue
    if _symbol is getattr(_torch_ops, _name, None):
        continue  # merged torch fallback; class-level NPU overrides win
    globals()[_name] = _symbol


def alloc_pinned_ptr(size: int, flags: int) -> int:
    """Initialize NPU runtime on demand and allocate unmapped ACL Host bytes.

    L1 allocation can precede transfer-channel and MP server initialization.
    Initialize through PyTorch without overriding an existing device selection.

    Args:
        size: Nonzero allocation size in bytes.
        flags: Forwarded to the native allocator, which currently ignores it.

    Returns:
        Host address owned by the caller. Keep runtime alive until all users and
        communication registrations are finished and the address is freed.

    Raises:
        ValueError: The allocation size is zero.
        RuntimeError: NPU initialization or ACL Host allocation fails.
    """
    _torch.npu.init()  # type: ignore[attr-defined]
    return _c_ops.alloc_pinned_ptr(size, flags)
