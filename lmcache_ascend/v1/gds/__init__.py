# SPDX-License-Identifier: Apache-2.0
"""Huawei NDS (NPU Direct Storage) bindings for LMCache-Ascend."""

from lmcache_ascend.v1.gds.nds_library import (
    NDSCapabilities,
    NDSLibrary,
    get_nds_library,
)

__all__ = ["NDSCapabilities", "NDSLibrary", "get_nds_library"]
