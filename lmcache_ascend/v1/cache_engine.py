# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


# Fix from https://github.com/LMCache/LMCache/pull/1852
# TODO (gingfung): remove when in v0.3.9
def post_init_fix(self, **kwargs) -> None:
    if "async_lookup_server" in kwargs:
        self.async_lookup_server = kwargs["async_lookup_server"]

    if not self.post_inited:
        self.storage_manager.post_init(**kwargs)
        logger.info("Post-initializing LMCacheEngine")
        self.gpu_connector.initialize_kvcaches_ptr(**kwargs)
        self.post_inited = True
