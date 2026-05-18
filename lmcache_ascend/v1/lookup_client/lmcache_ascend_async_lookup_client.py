# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)

from lmcache.v1.lookup_client.async_lookup_message import (
    LookupRequestMsg,
)

import msgspec
import time


class LMCacheAscendAsyncLookupClient(LMCacheAsyncLookupClient):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Use LMCacheAscendAsyncLookupClient.from_existing() "
            "to upgrade an existing LMCacheAsyncLookupClient instance."
        )

    @classmethod
    def from_existing(
        cls, original: LMCacheAsyncLookupClient
    ) -> "LMCacheAscendAsyncLookupClient":
        original.__class__ = cls
        original._lookup_hashes_cache = {}  # type: ignore[attr-defined]
        return original  # type: ignore[return-value]

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
            hashes.append(hash_val)  # type: ignore[arg-type]
            offsets.append(end - start)

        self._lookup_hashes_cache[lookup_id] = [f"{h:x}" for h in hashes]

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

    def get_cached_hashes(self, lookup_id: str) -> Optional[list[str]]:
        return self._lookup_hashes_cache.pop(lookup_id, None)  # type: ignore[attr-defined]

    def clear_lookup_status(self, lookup_id: str) -> None:
        super().clear_lookup_status(lookup_id)
