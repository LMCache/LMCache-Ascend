# SPDX-License-Identifier: Apache-2.0
"""Independent synchronous state saves through managed local storage."""

# Third Party
from lmcache.logging import init_logger
import torch

# First Party
from lmcache_ascend.v1.state_checkpoint import (
    CheckpointRef,
    StateBlockBinding,
    StateOperation,
    state_checkpoint_key,
)
from lmcache_ascend.v1.state_memory import allocate_state_checkpoint
from lmcache_ascend.v1.state_transfer import transfer_state


logger = init_logger(__name__)


class StateCache:
    """Copy completed endpoints before publication; no success registry or lookup."""

    def __init__(self, storage_manager, token_database, chunk_size):
        self.storage_manager = storage_manager
        self.token_database = token_database
        self.chunk_size = chunk_size

    def save(self, execution, layouts, kv_caches, store_stream, producer_event):
        blocks = execution.save_blocks(self.chunk_size)
        if not blocks:
            return
        manager = self.storage_manager
        cpu = manager.storage_backends["LocalCPUBackend"]
        if manager.allocator_backend is not cpu:
            raise ValueError("State save requires the local CPU allocator backend")
        if not cpu.use_hot:
            return  # Disk publication is added by the state disk integration.
        key = None
        for _, end, candidate in self.token_database.process_tokens(
            list(execution.token_ids), request_configs=execution.request_configs
        ):
            if end == execution.end:
                key = candidate
                break
        if key is None:
            raise ValueError("State endpoint has no complete prefix chunk key")
        by_group = {layout.group_index: layout for layout in layouts}
        for group, block_id in blocks:
            layout = by_group[group]
            checkpoint = CheckpointRef.from_chunk(
                key,
                chunk_end=execution.end,
                boundary=execution.end,
                chunk_size=self.chunk_size,
                group_index=group,
            )
            state_key = state_checkpoint_key(checkpoint)
            if cpu.contains(state_key):
                continue
            runtime = StateBlockBinding(
                tuple(tuple(kv_caches[name]) for name in layout.layer_names), block_id
            )
            try:
                buffer = allocate_state_checkpoint(layout, cpu, busy_loop=False)
            except MemoryError:
                logger.warning(
                    "State allocation exhausted: request=%s rank=%s "
                    "group=%s boundary=%s",
                    execution.req_id,
                    key.worker_id,
                    group,
                    execution.end,
                )
                continue
            with buffer:
                operation = StateOperation(checkpoint, runtime, buffer, "store")
                with torch.npu.stream(store_stream):
                    store_stream.wait_event(producer_event)
                    transfer_state(operation)
                # The manager consumes this extra reference only on normal return.
                # One CPU-only submission uses the original allocator, so there are
                # no copied allocations or partially consumed reference batches.
                obj = buffer.memory_obj
                obj.ref_count_up()
                try:
                    manager.batched_put([state_key], [obj], location="LocalCPUBackend")
                except BaseException:
                    obj.ref_count_down()
                    # A backend may raise after admitting the fully copied object;
                    # its retained reference belongs to the backend, not this owner.
                    raise
