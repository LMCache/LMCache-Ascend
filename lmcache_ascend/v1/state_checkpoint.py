# SPDX-License-Identifier: Apache-2.0
"""Checkpoint identity and structural operations; no availability or execution."""

# Standard
from copy import copy
from dataclasses import dataclass
from typing import Literal

# Third Party
from lmcache.utils import CacheEngineKey
import torch

# First Party
from lmcache_ascend.v1.state_layout import StateGroupLayout
from lmcache_ascend.v1.state_memory import StateCheckpointBuffer


@dataclass(frozen=True)
class CheckpointRef:
    """Identity of S(R), not proof that its payload exists or is ready.

    Construct via from_chunk; the raw dataclass initializer does not validate
    the prefix boundary contract. Do not replace identity fields directly.
    The captured key must be treated as read-only, like existing cache keys.
    Payload compatibility belongs to its layout, not this logical identity.
    """

    key: CacheEngineKey
    boundary: int
    group_index: int

    @classmethod
    def from_chunk(
        cls,
        key: CacheEngineKey,
        *,
        chunk_end: int,
        boundary: int,
        chunk_size: int,
        group_index: int,
    ) -> "CheckpointRef":
        """Reuse a default chained chunk key whose prefix ends exactly at R.

        The caller supplies (chunk_end, key) from ChunkedTokenDatabase. This
        checks their boundary contract, not the existence of a state snapshot.
        Segment/blending keys are not valid inputs. No hash is recomputed.
        """
        if chunk_size <= 0 or boundary <= 0 or boundary != chunk_end:
            raise ValueError("Checkpoint boundary must equal the positive chunk end")
        if boundary % chunk_size:
            raise ValueError("Checkpoint boundary must end at a complete chunk")
        # Capture identity fields without tracking later changes to the caller's key.
        return cls(copy(key), boundary, group_index)


@dataclass(frozen=True)
class StateBlockBinding:
    """Borrowed contiguous tensors in layer/plane order and one group block ID.

    Binding does not pin the block or establish the state boundary/readiness.
    Those conditions belong to the future transfer caller.
    """

    tensors: tuple[tuple[torch.Tensor, ...], ...]
    block_id: int

    def validate(self, layout: StateGroupLayout) -> None:
        """Check shape, dtype, continuity and block bounds for the HMA copy path."""
        if len(self.tensors) != len(layout.layer_names):
            raise ValueError("State binding layer count does not match layout")
        for entry in self.tensors:
            if len(entry) != len(layout.planes):
                raise ValueError("State binding plane count does not match layout")
            for tensor, plane in zip(entry, layout.planes, strict=True):
                if (
                    tuple(tensor.shape[1:]) != plane.shape[1:]
                    or tensor.dtype != plane.dtype
                    or not tensor.is_contiguous()
                ):
                    raise ValueError("State binding does not match runtime layout")
                if not 0 <= self.block_id < tensor.shape[0]:
                    raise ValueError("State block ID is out of range")


@dataclass(frozen=True)
class StateOperation:
    """Describe a load or save without copying data or claiming S(R) is available.

    Store copies runtime to buffer; load copies buffer to runtime.
    The buffer layout defines the expected runtime geometry and payload format.
    E/K belong to the scheduling caller, not this transfer description.
    Callers retain buffer ownership and protect runtime sources before execution.
    """

    checkpoint: CheckpointRef
    runtime: StateBlockBinding
    buffer: StateCheckpointBuffer
    direction: Literal["store", "load"]

    def __post_init__(self) -> None:
        if self.direction not in ("store", "load"):
            raise ValueError("State operation direction must be store or load")
        if self.checkpoint.group_index != self.buffer.layout.group_index:
            raise ValueError("Checkpoint group does not match buffer layout")
        _ = self.buffer.planes  # Verify the borrowed buffer is still owned and valid.
        self.runtime.validate(self.buffer.layout)


_STATE_TAGS = ("state.domain", "state.group", "state.boundary")


def state_checkpoint_key(checkpoint: CheckpointRef) -> CacheEngineKey:
    """Derive an independent backend object key without changing the prefix key."""
    key = checkpoint.key
    configs = dict(key.request_configs or {})
    if any("lmcache.tag." + name in configs for name in _STATE_TAGS) or any(
        name in _STATE_TAGS for name, _ in (key.tags or ())
    ):
        raise ValueError("State key input contains reserved internal tags")
    if checkpoint.group_index < 0 or checkpoint.boundary <= 0:
        raise ValueError("State key requires a nonnegative group and positive boundary")
    # Preserve captured tag order and values, including keys decoded from a backend.
    result = copy(key)
    internal = tuple(
        zip(
            _STATE_TAGS,
            ("checkpoint", str(checkpoint.group_index), str(checkpoint.boundary)),
            strict=True,
        )
    )
    configs.update(("lmcache.tag." + name, value) for name, value in internal)
    result.request_configs = configs
    result.tags = tuple(key.tags or ()) + internal
    return result
