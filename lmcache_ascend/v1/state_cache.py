# SPDX-License-Identifier: Apache-2.0
"""Independent synchronous state saves through managed local storage."""

# Standard
from time import perf_counter

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
from lmcache_ascend.v1.state_memory import (
    allocate_state_checkpoint,
    state_checkpoint_metadata,
)
from lmcache_ascend.v1.state_transfer import transfer_state
from lmcache_ascend.v1.storage_backend.storage_manager import state_store_locations

logger = init_logger(__name__)


class StateLoadError(ValueError):
    """A detectable missing/incompatible restore input, without recovery claims."""

    def __init__(self, reason, group="all"):
        super().__init__(reason)
        self.group = group


def _validate_load_device(runtime):
    devices = {tensor.device for entry in runtime.tensors for tensor in entry}
    if len(devices) != 1 or next(iter(devices)).type != "npu":
        raise ValueError("State load requires runtime planes on one NPU device")


class StateCache:
    """Copy completed endpoints before publication; no success registry or lookup."""

    def __init__(self, storage_manager, token_database, chunk_size):
        self.storage_manager = storage_manager
        self.token_database = token_database
        self.chunk_size = chunk_size

    def prepare_load(self, execution, boundary, layouts, kv_caches, selection):
        """Borrow the retained lookup buffers; preflight every group before copying.

        The caller holds the engine lock until every synchronous load completes,
        and releases the selection through lookup_unpin in its finally block.
        There is no second get, adoption or allocation here.
        """
        if execution is None or execution.start != boundary:
            raise StateLoadError("Missing execution or mismatched restore boundary")
        if selection is None or selection.boundary != boundary:
            raise StateLoadError("Missing or expired selected checkpoint")
        blocks = dict(execution.load_blocks(boundary))
        layouts = tuple(layouts)
        required = {layout.group_index for layout in layouts}
        if not required or len(required) != len(layouts):
            raise StateLoadError("Missing or duplicate required state layouts")
        if set(blocks) != required:
            raise StateLoadError("Missing pre-movement state block mapping")
        if len(execution.token_ids) < boundary:
            raise StateLoadError("Incomplete restore prefix")
        key = None
        for _, end, candidate in self.token_database.process_tokens(
            list(execution.token_ids[:boundary]),
            request_configs=execution.request_configs,
        ):
            if end == boundary:
                key = candidate
                break
        if key is None:
            raise StateLoadError("Restore boundary has no complete prefix key")
        operations = []
        for layout in layouts:
            group = layout.group_index
            buffer = selection.buffers.get(group)
            if buffer is None:
                raise StateLoadError("Missing selected state buffer", group)
            if (
                buffer._released
                or not buffer.memory_obj.is_valid()
                or buffer.layout != layout
                or getattr(buffer.memory_obj.meta, "state_checkpoint", None)
                != state_checkpoint_metadata(layout)
            ):
                raise StateLoadError("Incompatible selected state metadata", group)
            if any(name not in kv_caches for name in layout.layer_names):
                raise StateLoadError("Missing runtime state tensors", group)
            checkpoint = CheckpointRef.from_chunk(
                key,
                chunk_end=boundary,
                boundary=boundary,
                chunk_size=self.chunk_size,
                group_index=group,
            )
            try:
                runtime = StateBlockBinding(
                    tuple(tuple(kv_caches[name]) for name in layout.layer_names),
                    blocks[group],
                )
                _validate_load_device(runtime)
                operations.append(StateOperation(checkpoint, runtime, buffer, "load"))
            except Exception:
                logger.exception(
                    "Hybrid load failed: request=%s rank=%s group=%s R=%s "
                    "reason=state input validation",
                    execution.req_id,
                    key.worker_id,
                    group,
                    boundary,
                )
                raise
        return operations

    def save(
        self, execution, layouts, kv_caches, store_stream, producer_event, location=None
    ):
        blocks = execution.save_blocks(self.chunk_size)
        if not blocks:
            return
        start = perf_counter()
        manager = self.storage_manager
        cpu = manager.storage_backends["LocalCPUBackend"]
        if manager.allocator_backend is not cpu:
            raise ValueError("State save requires the local CPU allocator backend")
        locations = state_store_locations(manager, location)
        if not locations:
            return
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
        stored_groups = []
        stored_bytes = 0
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
            missing_locations = [
                name
                for name in locations
                if not manager.storage_backends[name].contains(state_key)
            ]
            if not missing_locations:
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
                # Each manager call consumes its extra reference on normal return.
                # All supported tiers share this allocator; there are no copies or
                # partially consumed reference batches. Async disk owns its own ref.
                obj = buffer.memory_obj
                for name in missing_locations:
                    obj.ref_count_up()
                    try:
                        manager.batched_put([state_key], [obj], location=name)
                    except BaseException:
                        obj.ref_count_down()
                        # Earlier complete backend entries retain their own refs.
                        raise
                stored_groups.append(group)
                stored_bytes += sum(plane.nbytes for plane in layout.planes)
        if stored_groups:
            elapsed = perf_counter() - start
            size_gb = stored_bytes / 1024**3
            # transfer_state waits for the copy. Backend submission does
            # not imply that an asynchronous disk write has completed.
            logger.info(
                "[req_id=%s] Stored state checkpoint: rank=%s, boundary=%s, "
                "groups=%s, size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s;",
                execution.req_id,
                key.worker_id,
                execution.end,
                stored_groups,
                size_gb,
                elapsed * 1000,
                size_gb / elapsed if elapsed > 0 else 0,
            )
