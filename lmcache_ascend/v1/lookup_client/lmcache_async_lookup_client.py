# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import time

# Third Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.async_lookup_message import (
    LookupRequestMsg,
)
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient as _BaseLMCacheAsyncLookupClient,
)
from lmcache.v1.metadata import LMCacheMetadata
import msgspec
import torch


class LMCacheAsyncLookupClient(_BaseLMCacheAsyncLookupClient):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        super().__init__(config, metadata)
        # lookup_hashes_cache init start---------------------
        self._lookup_hashes_cache = {}
        self._max_cache_size = int(getattr(config, "lookup_hashes_cache_size", 0))
        # lookup_hashes_cache init end ---------------------

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        hashes: list[int] = []
        offsets = []
        for start, end, hash_val in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(hash_val)
            offsets.append(end - start)

        # lookup_hashes_cache fill start ---------------------
        if (
            self._max_cache_size > 0
            and len(self._lookup_hashes_cache) >= self._max_cache_size
        ):
            oldest_key = next(iter(self._lookup_hashes_cache))
            del self._lookup_hashes_cache[oldest_key]

        self._lookup_hashes_cache[lookup_id] = [f"{h:x}" for h in hashes]
        # lookup_hashes_cache fill end ---------------------

        msg = LookupRequestMsg(
            lookup_id=lookup_id,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        )
        msg_buf = msgspec.msgpack.encode(msg)
        for i in range(self.world_size):
            self.push_sockets[i].send(msg_buf, copy=False)
        time.sleep(self.lookup_backoff_time)
        return None

    # get_cached_hashes start ---------------------
    def get_cached_hashes(self, lookup_id: str) -> Optional[list[str]]:
        return self._lookup_hashes_cache.pop(lookup_id, None)

    # get_cached_hashes end ---------------------
