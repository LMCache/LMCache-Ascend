# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache.v1.storage_backend.storage_manager import AsyncSerializer


# Fix from https://github.com/LMCache/LMCache/pull/1795
# TODO (gingfung): remove when in v0.3.9
def post_init_fix(self, **kwargs) -> None:
    if "async_lookup_server" in kwargs:
        assert not self.config.save_unfull_chunk, (
            "save_unfull_chunk should be automatically set to False "
            "when using async loading."
        )
        self.async_lookup_server = kwargs.pop("async_lookup_server")
    self.async_serializer = AsyncSerializer(self.allocator_backend, self.loop)
