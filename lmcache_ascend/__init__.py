# SPDX-License-Identifier: Apache-2.0

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from ._version import __version__ as __version__  # noqa: F401  # isort:skip
from ._version import __version_tuple__ as __version_tuple__  # noqa: F401  # isort:skip

# Standard
from typing import Any
import sys

# NOTE: Must be manually edited per each version and
# is also used by the test infrastructure.
LMCACHE_UPSTREAM_TAG = "v0.5.6"
LMCACHE_ASCEND_PATCHED = False


def _is_vllm_runtime():
    return "vllm" in sys.modules or any("vllm" in arg for arg in sys.argv)


def _seed_partial_lmcache_device_ops() -> None:
    """Bridge ``lmcache.device_ops`` while ``import lmcache`` is still running.

    On the serving path the plugin is first pulled in by lmcache's own
    platform init (``NpuDeviceOps.ensure_native`` -> ``import
    lmcache_ascend.c_ops``), i.e. in the middle of ``import lmcache`` --
    before ``lmcache/__init__.py`` binds its ``device_ops`` singleton (which
    lives after the platform import that invoked us). Core modules imported
    during the patch activation (``lmcache.v1.kv_layer_groups``, ...) do
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


def _patch_logical_block_size():
    """Wrap LMCache ``get_tokens_per_block`` when Ascend ``block_size`` is physical.

    Pre-#13242 pool math still uses ``block_size * compress_ratio``; the wrap
    reports that span for Ascend leaves. Post-#13242 ``block_size`` is already
    logical, so LMCache core is left unpatched. Never mutate ``spec.block_size``.
    """
    # First Party
    from lmcache_ascend.integration.vllm.logical_block_size import install_overrides

    install_overrides()


def _patch_make_page_buffer_shape_desc():
    """Attach per-plane byte geometry to upstream's desc factory (fmt 16/17)."""
    # First Party
    from lmcache_ascend.v1.shape_desc import install_plane_geometry_fill

    install_plane_geometry_fill()


# Check if we've already patched to avoid redundant work
if not LMCACHE_ASCEND_PATCHED:
    # Must run before any patch: the activation imports core modules that
    # bind ``lmcache.device_ops`` at module level, which only exists once
    # ``import lmcache`` finished -- and we are typically invoked from inside
    # it (see the docstring).
    _seed_partial_lmcache_device_ops()

    if _is_vllm_runtime():
        _patch_logical_block_size()

    _patch_make_page_buffer_shape_desc()

    LMCACHE_ASCEND_PATCHED = True
