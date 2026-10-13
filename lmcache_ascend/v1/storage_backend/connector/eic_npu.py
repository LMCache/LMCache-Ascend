# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch that makes the upstream EIC remote connector importable on Ascend.

The EIC connector lives in LMCache core
(``lmcache.v1.storage_backend.connector.eic_connector``, introduced in
LMCache PR #1930, ``eic://`` url scheme). This module does not reimplement it.
The only Ascend-specific blocker is that ``EICConnector.__init__`` eagerly
loads ``libcudart.so`` through ctypes to support CUDA GDR. CANN images do not
ship libcudart, so merely importing the connector raises OSError before the
adapter can be selected. On NPU the transport is RDMA and GDR is not used.
"""

# Standard
import ctypes

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class _CudaNoOp:
    """A callable whose ctypes-style ``argtypes``/``restype`` are assignable."""

    argtypes = None
    restype = None

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "CUDA GDR path invoked without libcudart.so; this path is not "
            "supported on Ascend NPU"
        )


class _CudaLibShim:
    """Stand-in for ctypes.CDLL("libcudart.so") when CUDA runtime is absent.

    The connector binds a cudaMemcpy signature at init and calls it only on the
    CUDA GDR path that NPU never enters, so the binding must be assignable and
    persisted while calls remain no-ops.
    """

    def __init__(self):
        self.cudaMemcpy = _CudaNoOp()

    def __getattr__(self, _name):
        return _CudaNoOp()


class _GuardedCtypes:
    """Module-level proxy for the connector's view of ``ctypes``.

    Delegates every attribute to the real ``ctypes`` module except ``CDLL``:
    a libcudart lookup that fails returns a no-op shim. The proxy is installed
    once as an attribute of the connector module, so concurrent EICConnector
    constructions share it with no global mutation or restore window, and
    unrelated ``ctypes`` callers in the process are untouched.
    """

    def __init__(self, real_ctypes):
        self._real = real_ctypes
        self._real_cdll = real_ctypes.CDLL

    def CDLL(self, name, *args, **kwargs):
        if name == "libcudart.so":
            try:
                return self._real_cdll(name, *args, **kwargs)
            except OSError:
                logger.info(
                    "libcudart.so not found; EIC CUDA GDR is disabled "
                    "(expected on Ascend NPU, RDMA transport is unaffected)"
                )
                return _CudaLibShim()
        return self._real_cdll(name, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _make_cudart_optional(connector_module):
    """Make the connector's eager libcudart load tolerant of its absence.

    ``cuda_lib`` backs only the CUDA GDR receive path. When libcudart is
    absent (Ascend CANN images), the connector module's ``ctypes`` name is
    rebound once to a proxy that yields a no-op shim for the cudart lookup;
    the NPU RDMA path never calls it.
    """
    if getattr(connector_module, "_lmcache_ascend_eic_patched", False):
        return

    # Idempotent even if a proxy is already present.
    current = getattr(connector_module, "ctypes", None)
    if isinstance(current, _GuardedCtypes):
        connector_module._lmcache_ascend_eic_patched = True
        return

    connector_module.ctypes = _GuardedCtypes(ctypes)
    connector_module._lmcache_ascend_eic_patched = True


def _core_loads_cudart_optional(connector_module) -> bool:
    """Whether core already tolerates a missing libcudart.

    Core LMCache#5141 loads libcudart defensively: ``cuda_lib`` stays None
    without the library, RDMA still constructs, and only an explicit
    TRANSPORT_GDR is rejected. Against such core the ctypes shim is both
    unnecessary and wrong: a truthy CDLL stand-in makes core take its
    load-success branch, so ``cuda_lib`` is non-None and the ``cuda_lib is
    None`` GDR guard no longer fires.
    """
    return getattr(connector_module, "_LMCACHE_EIC_CUDART_OPTIONAL", False)


def patch_eic_connector():
    """Apply Ascend compatibility patch to the upstream EIC connector."""
    # The module imports the vendor-only `eic` package at module scope. It is
    # not available on public PyPI and is absent in CI, so import lazily and
    # skip silently, mirroring the optional-connector convention in core.
    try:
        import lmcache.v1.storage_backend.connector.eic_connector as eic_mod
    except ImportError:
        logger.debug(
            "upstream EIC connector or its `eic` dependency is not available; "
            "skipping EIC NPU compatibility patch"
        )
        return

    if _core_loads_cudart_optional(eic_mod):
        logger.info(
            "core already loads libcudart defensively; leaving the EIC "
            "connector unpatched so its GDR guard stays effective"
        )
        return

    _make_cudart_optional(eic_mod)
    logger.info("Applied Ascend compatibility patch to EIC connector")
