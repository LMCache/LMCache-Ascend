# SPDX-License-Identifier: Apache-2.0
"""Checkpoint identity and structural operations; no availability or execution."""

# Standard
from dataclasses import dataclass, field

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
    """

    model_name: str
    world_size: int
    worker_id: int
    prefix_hash: int
    key_dtype: torch.dtype
    tags: tuple | None
    boundary: int
    group_index: int
    layout_signature: tuple
    kind: str = field(default="gdn_state", init=False)

    @classmethod
    def from_chunk(
        cls,
        key: CacheEngineKey,
        *,
        chunk_end: int,
        boundary: int,
        chunk_size: int,
        layout: StateGroupLayout,
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
        return cls(
            key.model_name,
            key.world_size,
            key.worker_id,
            key.chunk_hash,
            key.dtype,
            key.tags,
            boundary,
            layout.group_index,
            layout.signature,
        )


@dataclass(frozen=True)
class StateBlockBinding:
    """Borrowed runtime tensors in layout layer/plane order and one group block ID.

    Binding does not pin the block or establish the state boundary/readiness.
    Those conditions belong to the future transfer caller.
    """

    tensors: tuple[tuple[torch.Tensor, ...], ...]
    block_id: int

    def validate(self, layout: StateGroupLayout) -> None:
        """Reject out-of-range blocks or tensors incompatible with source geometry."""
        if len(self.tensors) != len(layout.layer_names):
            raise ValueError("State binding layer count does not match layout")
        for layer_index, entry in enumerate(self.tensors):
            if len(entry) != len(layout.planes):
                raise ValueError("State binding plane count does not match layout")
            for tensor, plane in zip(entry, layout.planes, strict=True):
                if (
                    tuple(tensor.shape[1:]) != plane.shape[1:]
                    or tensor.dtype != plane.dtype
                    or tuple(tensor.stride()) != plane.source_strides[layer_index]
                ):
                    raise ValueError("State binding does not match runtime layout")
                if not 0 <= self.block_id < tensor.shape[0]:
                    raise ValueError("State block ID is out of range")


@dataclass(frozen=True)
class StateOperation:
    """Describe a load or save without copying data or claiming S(R) is available.

    Exactly one endpoint is runtime state and the other is a managed buffer.
    E/K are optional context, never substitutes for the checkpoint's explicit R.
    Callers retain buffer ownership and protect runtime sources before execution.
    """

    checkpoint: CheckpointRef
    layout: StateGroupLayout
    source: StateBlockBinding | StateCheckpointBuffer
    target: StateBlockBinding | StateCheckpointBuffer
    executed_end: int | None = None
    attention_end: int | None = None

    def __post_init__(self) -> None:
        if (
            self.checkpoint.group_index != self.layout.group_index
            or self.checkpoint.layout_signature != self.layout.signature
        ):
            raise ValueError("Checkpoint does not match operation layout")
        if isinstance(self.source, StateBlockBinding) and isinstance(
            self.target, StateCheckpointBuffer
        ):
            runtime, buffer = self.source, self.target
        elif isinstance(self.target, StateBlockBinding) and isinstance(
            self.source, StateCheckpointBuffer
        ):
            runtime, buffer = self.target, self.source
        else:
            raise ValueError("State operation requires runtime and buffer endpoints")
        if (
            buffer.layout.group_index != self.layout.group_index
            or buffer.layout.signature != self.layout.signature
        ):
            raise ValueError("Checkpoint buffer does not match operation layout")
        _ = buffer.planes  # Verify the borrowed buffer is still owned and valid.
        runtime.validate(self.layout)
        if any(
            end is not None and end < 0
            for end in (self.executed_end, self.attention_end)
        ):
            raise ValueError("Execution/save endpoints cannot be negative")
