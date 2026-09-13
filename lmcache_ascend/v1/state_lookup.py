# SPDX-License-Identifier: Apache-2.0
"""Sparse checkpoint selection over the existing synchronous lookup transport."""

# Standard
from dataclasses import dataclass, field
import json
import threading

# Third Party
from lmcache.logging import init_logger

# First Party
from lmcache_ascend.v1.state_checkpoint import CheckpointRef, state_checkpoint_key
from lmcache_ascend.v1.state_memory import adopt_state_checkpoint

logger = init_logger(__name__)
OP_KEY = "lmcache.internal.state_lookup"


def validate_state_lookup_client(client, required_world_size):
    if not hasattr(client, "transport") or not hasattr(client, "token_database"):
        raise ValueError("State lookup requires the synchronous RPC client")
    config = client.config
    if (
        required_world_size <= 0
        or client.transport.world_size != required_world_size
        or config.enable_blending
        or config.enable_async_loading
        or config.use_layerwise
        or config.hit_miss_ratio is not None
        or config.enable_chunk_statistics
    ):
        raise ValueError(
            "State lookup requires synchronous, unwrapped, all-worker lookup"
        )
    workers = config.get_lookup_server_worker_ids(False, required_world_size)
    if workers and sorted(workers) != list(range(required_world_size)):
        raise ValueError("State lookup does not support worker subsets")


def _exchange(client, hashes, offsets, lookup_id, configs, operation):
    wire = dict(configs or {})
    if OP_KEY in wire:
        raise ValueError("Request configs collide with reserved state lookup operation")
    wire[OP_KEY] = operation
    return client.transport.send_and_recv_all(
        [hashes, offsets, lookup_id, json.dumps(wire)]
    )


def cancel_state_lookup(client, lookup_id):
    """Release remote probes/selection; ordinary clear_lookup_status is local only.

    Unreachable workers expire their selection lease. A subsequent probe replaces
    any previous selection even when cancellation could not reach that worker.
    """
    try:
        _exchange(client, [], [], lookup_id, None, {"op": "release"})
    finally:
        client.clear_lookup_status(lookup_id)


def lookup_state(
    client,
    token_ids,
    lookup_id,
    local_cached,
    upper,
    required_world_size,
    request_configs=None,
):
    """Return only an equal candidate from every worker; never cache by req ID.

    Upper is inclusive. The scheduling caller applies full-hit/logits and
    minimum retrieval constraints BEFORE this selection. Worker objects remain
    protected until consumption, explicit release, replacement, or lease expiry.
    """
    validate_state_lookup_client(client, required_world_size)
    if OP_KEY in (request_configs or {}):
        raise ValueError("Request configs collide with reserved state lookup operation")
    token_ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if not 0 <= local_cached <= len(token_ids) or not 0 <= upper <= len(token_ids):
        raise ValueError("Invalid state lookup boundaries")
    hashes, offsets = [], []
    for start, end, key in client.token_database.process_tokens(
        token_ids, make_key=False, request_configs=request_configs
    ):
        hashes.append(key)
        offsets.append(end - start)
    selected = False
    try:
        while upper > local_cached:
            responses = _exchange(
                client,
                hashes,
                offsets,
                lookup_id,
                request_configs,
                {"op": "probe", "C": local_cached, "upper": upper},
            )
            if len(responses) != required_world_size or any(
                not isinstance(resp, bytes) or len(resp) != 4 for resp in responses
            ):
                return 0
            hits = [int.from_bytes(resp, "big") for resp in responses]
            if any(hit <= local_cached or hit > upper for hit in hits):
                return 0
            if len(set(hits)) == 1:
                selected = True
                return hits[0]
            upper = min(hits)
        return 0
    finally:
        if not selected:
            try:
                cancel_state_lookup(client, lookup_id)
            except Exception:
                logger.exception(
                    "State lookup cancellation failed: request=%s", lookup_id
                )


@dataclass
class StateLookupSelection:
    boundary: int
    buffers: dict = field(default_factory=dict)
    attention_pins: dict = field(default_factory=dict)
    timer: object = None

    def close(self, manager):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        error = None
        pins, self.attention_pins = self.attention_pins, {}
        buffers, self.buffers = self.buffers, {}
        for location, keys in pins.items():
            try:
                manager.batched_unpin(keys, [location])
            except BaseException as exc:
                error = error or exc
        for buffer in buffers.values():
            try:
                buffer.close()
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error


def local_candidate(
    manager, chunks, layouts, chunk_size, local_cached, upper, locations
):
    """Acquire readable state refs and pin external Attention above local C."""
    layouts = tuple(layouts)
    if not layouts or len({x.group_index for x in layouts}) != len(layouts):
        raise ValueError("State lookup requires all distinct registered state groups")
    locations = list(locations or ("LocalCPUBackend", "LocalDiskBackend"))
    if any(x not in ("LocalCPUBackend", "LocalDiskBackend") for x in locations):
        raise ValueError("State lookup supports only local CPU/disk")
    for _, boundary, prefix_key in reversed(chunks):
        if not local_cached < boundary <= upper or boundary % chunk_size:
            continue
        selection = StateLookupSelection(boundary)
        valid = False
        try:
            for layout in layouts:
                key = state_checkpoint_key(
                    CheckpointRef.from_chunk(
                        prefix_key,
                        chunk_end=boundary,
                        boundary=boundary,
                        chunk_size=chunk_size,
                        group_index=layout.group_index,
                    )
                )
                for location in locations:
                    backend = manager.storage_backends.get(location)
                    obj = None if backend is None else backend.get_blocking(key)
                    if obj is None:
                        continue
                    try:
                        selection.buffers[layout.group_index] = adopt_state_checkpoint(
                            layout, obj
                        )
                    except ValueError:
                        continue  # Adoption already released its owned get reference.
                    break
                if layout.group_index not in selection.buffers:
                    break
            if len(selection.buffers) != len(layouts):
                continue
            covered = True
            for start, end, key in chunks:
                if end <= local_cached or end > boundary:
                    continue
                location = manager.contains(key, locations, pin=True)
                if location is None:
                    covered = False
                    break
                selection.attention_pins.setdefault(location, []).append(key)
            if covered:
                valid = True
                return selection
        finally:
            if not valid:
                selection.close(manager)
    return None


def release_engine_selection(engine, lookup_id):
    selections = getattr(engine, "_state_lookup_selections", {})
    selection = selections.pop(lookup_id, None)
    if selection is not None:
        engine.lookup_pins.pop(lookup_id, None)
        selection.close(engine.storage_manager)


def dispatch_state_lookup(engine, lookup_id, operation, configs, **inputs):
    """Called with engine lock held; control fields never enter token identity."""
    release_engine_selection(engine, lookup_id)
    if not isinstance(operation, dict) or not lookup_id:
        raise ValueError("Invalid state lookup operation")
    if operation == {"op": "release"}:
        return 0
    if set(operation) != {"op", "C", "upper"} or operation["op"] != "probe":
        raise ValueError("Invalid state lookup operation")
    c, upper = operation["C"], operation["upper"]
    if type(c) is not int or type(upper) is not int or not 0 <= c < upper:
        raise ValueError("Invalid state probe boundaries")
    if (
        engine.use_layerwise
        or engine.config.enable_blending
        or engine.config.enable_async_loading
        or engine.config.pin_timeout_sec <= 0
    ):
        raise ValueError("Unsupported hybrid lookup configuration")
    if not engine.is_healthy():
        return 0
    chunks = list(
        engine.token_database.process_tokens(request_configs=configs, **inputs)
    )
    selection = local_candidate(
        engine.storage_manager,
        chunks,
        engine.state_layouts,
        engine.config.chunk_size,
        c,
        upper,
        engine.retrieve_locations,
    )
    if selection is None:
        return 0
    if not hasattr(engine, "_state_lookup_selections"):
        engine._state_lookup_selections = {}
    engine._state_lookup_selections[lookup_id] = selection
    engine.lookup_pins[lookup_id] = selection.attention_pins

    def expire():
        with engine._engine_state_lock:
            if (
                engine._state_lookup_selections.get(lookup_id) is selection
                and selection.timer is not None
            ):
                release_engine_selection(engine, lookup_id)

    selection.timer = threading.Timer(engine.config.pin_timeout_sec, expire)
    selection.timer.daemon = True
    try:
        selection.timer.start()
    except BaseException:
        release_engine_selection(engine, lookup_id)
        raise
    return selection.boundary
