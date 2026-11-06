# SPDX-License-Identifier: Apache-2.0
"""Huawei NDS (NPU Direct Storage) bindings for LMCache-Ascend."""

from lmcache_ascend.v1.gds.openeuler_nds_library import (
    NDSCapabilities,
    OpenEulerNDSLibrary,
    get_openeuler_nds_library,
)

__all__ = ["NDSCapabilities", "OpenEulerNDSLibrary", "get_openeuler_nds_library"]
