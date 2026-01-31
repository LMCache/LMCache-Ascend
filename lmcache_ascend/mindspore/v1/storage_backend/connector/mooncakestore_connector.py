# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import asyncio

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj

# First Party
from lmcache_ascend.mindspore.v1._tensor import (
    get_data_ptr,
    get_element_size,
    get_numel,
)

logger = init_logger(__name__)


def MooncakeStoreConnector__register_cpu_buffer(self):
    """Register CPU buffer for zero-copy operations."""
    try:
        allocator = self.local_cpu_backend.memory_allocator
        if hasattr(allocator, "pin_allocator") and hasattr(
            allocator.pin_allocator, "buffer"
        ):
            buffer = allocator.pin_allocator.buffer
            self.registered_buffer_ptr = get_data_ptr(buffer)
            result = self.store.register_buffer(get_data_ptr(buffer), get_numel(buffer))
            if result == 0:
                logger.info(
                    f"Registered: {hex(get_data_ptr(buffer))}, "
                    f"{get_numel(buffer)} bytes"
                )
            else:
                logger.warning(f"Buffer registration failed: error={result}")
                self.registered_buffer_ptr = None
        else:
            self.registered_buffer_ptr = None
    except Exception as e:
        logger.error(f"Buffer registration error: {e}")
        self.registered_buffer_ptr = None


async def MooncakeStoreConnector__batch_get_into(
    self, keys: List[CacheEngineKey]
) -> List[Optional[MemoryObj]]:
    """
    Zero-copy batch get using batch_get_into when metadata is available locally.
    This is used when save_chunk_meta=False (metadata not stored remotely).
    """
    if not self.meta_shape or not self.meta_dtype or not self.meta_fmt:
        logger.error(
            f"Metadata required for batch_get_into but not available: "
            f"meta_shape={self.meta_shape}, "
            f"meta_dtype={self.meta_dtype}, "
            f"meta_fmt={self.meta_fmt}"
        )
        return [None] * len(keys)

    logger.debug(f"Using batch_get_into for {len(keys)} keys (zero-copy mode)")

    # Reserve a buffer for every requested chunk
    memory_objs: list[Optional[MemoryObj]] = []
    valid_idx: list[int] = []

    key_strs: list[str] = []
    buffer_ptrs: list[int] = []
    buffer_sizes: list[int] = []

    for i, _ in enumerate(keys):
        buf = self.local_cpu_backend.allocate(
            self.meta_shape, self.meta_dtype, self.meta_fmt
        )
        memory_objs.append(buf)
        buf_tensor = buf.tensor
        if buf is not None and buf_tensor is not None:
            valid_idx.append(i)

            # Prepare the argument lists for the C++ call
            key_strs.append(keys[i].to_string())
            buffer_ptrs.append(get_data_ptr(buf_tensor))
            buffer_sizes.append(get_numel(buf_tensor) * get_element_size(buf_tensor))

    if not valid_idx:
        logger.warning("Batch-get aborted: unable to allocate any buffers.")
        return [None] * len(keys)

    try:
        # Single RPC call for multiple chunks
        logger.debug(f"Calling batch_get_into with {len(key_strs)} keys")
        bytes_read_list = await asyncio.to_thread(
            self.store.batch_get_into, key_strs, buffer_ptrs, buffer_sizes
        )
        logger.debug(f"batch_get_into returned: {bytes_read_list}")

        # Assemble the final result list
        results: list[Optional[MemoryObj]] = [None] * len(keys)

        for i, n_read in zip(valid_idx, bytes_read_list, strict=False):
            if n_read <= 0:
                logger.warning(
                    f"batch_get_into failed for key {keys[i]} (code={n_read})"
                )
                memory_objs[i].ref_count_down()  # type: ignore
                continue

            try:
                results[i] = self.reshape_partial_chunk(
                    memory_objs[i],  # type: ignore
                    n_read,
                )
            except Exception as exc:
                logger.error(f"Reshape failed for key {keys[i]}: {exc}")
                memory_objs[i].ref_count_down()  # type: ignore

        return results

    except Exception as exc:
        logger.error(f"batch_get_into threw exception: {str(exc)}")
        # Release any buffers we successfully allocated
        for i in valid_idx:
            memory_objs[i].ref_count_down()  # type: ignore
        return [None] * len(keys)


async def MooncakeStoreConnector__put_without_metadata(
    self, key_str: str, memory_obj: MemoryObj
):
    """
    Zero-copy put using put_from when metadata is not stored remotely.
    This is used when save_chunk_meta=False (matches _batch_get_into).
    """
    try:
        tensor = memory_obj.tensor
        assert tensor is not None
        buffer_ptr = get_data_ptr(tensor)
        buffer_size = get_numel(tensor) * get_element_size(tensor)

        await asyncio.wait_for(
            asyncio.to_thread(
                self.store.put_from,
                key_str,
                buffer_ptr,
                buffer_size,
                self.replica_config,
            ),
            timeout=self.config.transfer_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"Timeout when putting key {key_str} using put_from. "
            "Decode instance may redo prefill."
        )
    except Exception as e:
        logger.error(
            f"Failed to put key {key_str} using put_from: {type(e).__name__}: {str(e)}"
        )
        raise
