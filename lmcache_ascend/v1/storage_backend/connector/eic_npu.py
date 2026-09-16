# SPDX-License-Identifier: Apache-2.0
"""
Runtime patches that make the upstream EIC remote connector work on Ascend NPU.

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


def _make_cudart_optional(connector_module):
    """Replace the eager libcudart load with a guarded one.

    ``cuda_lib`` is only needed by the CUDA GDR receive path. When libcudart is
    absent (Ascend CANN images), bind a no-op shim instead of failing at import
    time; the NPU RDMA path never calls it.
    """
    if getattr(connector_module, "_lmcache_ascend_eic_patched", False):
        return

    original_init = connector_module.EICConnector.__init__
    original_cdll = ctypes.CDLL

    def guarded_cdll(name, *a, **kw):
        if name == "libcudart.so":
            try:
                return original_cdll(name, *a, **kw)
            except OSError:
                logger.info(
                    "libcudart.so not found; EIC CUDA GDR is disabled "
                    "(expected on Ascend NPU, RDMA transport is unaffected)"
                )
                return _CudaLibShim()
        return original_cdll(name, *a, **kw)

    def patched_init(self, *args, **kwargs):
        ctypes.CDLL = guarded_cdll
        try:
            original_init(self, *args, **kwargs)
        finally:
            ctypes.CDLL = original_cdll

    connector_module.EICConnector.__init__ = patched_init
    connector_module._lmcache_ascend_eic_patched = True


def patch_eic_connector():
    """Apply Ascend compatibility patches to the upstream EIC connector."""
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

    _make_cudart_optional(eic_mod)
    logger.info("Applied Ascend compatibility patch to EIC connector")
