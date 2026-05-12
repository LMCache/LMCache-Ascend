# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache_ascend.v1.internal_api_server.memory.memory_api import (
    router as memory_router,
)

_original_init = None


def _capture_original_init(original):
    global _original_init
    _original_init = original


def InternalAPIServer__init__(self, lmcache_manager):
    # Third Party
    from lmcache.v1.internal_api_server.api_server import app

    _original_init(self, lmcache_manager)

    config = lmcache_manager.config
    if getattr(config, "enable_chunk_hashes_return", False):
        app.include_router(memory_router)
