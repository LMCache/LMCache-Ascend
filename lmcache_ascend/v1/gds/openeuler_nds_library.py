# SPDX-License-Identifier: Apache-2.0
"""Compatibility re-export. The implementation lives in LMCache."""

from lmcache.v1.gpu_connector.gds_backends._openeuler_nds_library import (
    NDSCapabilities,
    OpenEulerNDSLibrary,
    get_openeuler_nds_library,
)

__all__ = [
    "NDSCapabilities",
    "OpenEulerNDSLibrary",
    "get_openeuler_nds_library",
]
