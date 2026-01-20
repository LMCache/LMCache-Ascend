# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List
import asyncio

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

"""
The below are fixes from upstream post v0.3.12 for the double free bug:
https://github.com/LMCache/LMCache-Ascend/issues/145
https://github.com/LMCache/LMCache/pull/2415
TODO (gingfung): remove these patches post v0.3.12
"""


async def _batched_put_zero_copy(
    self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
) -> None:
    key_strs = [k.to_string() for k in keys]
    buffer_ptrs: List[int] = []
    buffer_sizes: List[int] = []
    for obj in memory_objs:
        tensor = obj.tensor
        assert tensor is not None
        buffer_ptrs.append(tensor.data_ptr())
        buffer_sizes.append(tensor.numel() * tensor.element_size())

    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                self.store.batch_put_from,
                key_strs,
                buffer_ptrs,
                buffer_sizes,
                self.replica_config,
            ),
            timeout=self.config.transfer_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Timeout during batch_put_from; some decoders may redo prefill.")


async def _batched_put_with_metadata(
    self,
    keys: List[CacheEngineKey],
    memory_objs: List[MemoryObj],
) -> None:
    for key, obj in zip(keys, memory_objs, strict=False):
        await self._put_with_metadata(key.to_string(), obj)
