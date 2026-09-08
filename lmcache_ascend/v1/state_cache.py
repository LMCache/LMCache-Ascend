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
from lmcache_ascend.v1.state_memory import (
    allocate_state_checkpoint,
    state_checkpoint_metadata,
)
from lmcache_ascend.v1.state_transfer import transfer_state


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
