# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AscendLMCacheEngine._sharded_broadcast_and_load.

Tests verify the sharded broadcast protocol where rank 0 sends KV cache
chunks in configurable-size shards and non-rank-0 ranks receive each shard,
immediately load to GPU, and release NPU tensors before the next shard.

Multi-rank is simulated in-process with shared-state broadcast functions.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_device_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch torch_dev and torch_device_type so tests work without GPU/NPU.

    The implementation uses ``torch_dev.device_count()`` to compute
    ``local_rank`` and ``torch_device_type`` to construct device strings.
    Without real hardware these resolve to ``torch.cuda`` / ``"cuda"`` which
    may fail (e.g. ``device_count() == 0`` or ``.to("cuda:0")`` errors).

    We patch ``torch_device_type`` to ``"cpu"`` and wrap ``torch.Tensor.to``
    to strip device indices (``"cpu:0"`` -> ``"cpu"``) since PyTorch does not
    accept indexed CPU device strings.
    """
    import lmcache_ascend.v1.cache_engine as _ce_mod

    mock_dev = MagicMock()
    mock_dev.device_count.return_value = 1
    mock_dev.Stream = MagicMock

    monkeypatch.setattr(_ce_mod, "torch_dev", mock_dev)
    monkeypatch.setattr(_ce_mod, "torch_device_type", "cpu")

    # torch.npu may not exist on non-Ascend systems; stub empty_cache.
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", MagicMock())

    # Normalize device strings: "cpu:0" -> "cpu", "cpu:1" -> "cpu"
    _original_to = torch.Tensor.to

    def _patched_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args and isinstance(args[0], str) and args[0].startswith("cpu:"):
            args = ("cpu",) + args[1:]
        return _original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _patched_to)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(
    shape: tuple[int, ...] = (256, 2, 128),
    dtype: torch.dtype = torch.bfloat16,
) -> MemoryObjMetadata:
    """Create a MemoryObjMetadata with sensible defaults for testing."""
    phy_size = 1
    for s in shape:
        phy_size *= s
    phy_size *= dtype.itemsize
    return MemoryObjMetadata(
        shape=torch.Size(shape),
        dtype=dtype,
        address=0,
        phy_size=phy_size,
        ref_count=1,
    )


def _make_memory_obj(
    shape: tuple[int, ...] = (256, 2, 128),
    dtype: torch.dtype = torch.bfloat16,
) -> TensorMemoryObj:
    """Create a TensorMemoryObj backed by a CPU tensor."""
    meta = _make_metadata(shape, dtype)
    raw = torch.empty(meta.get_size(), dtype=torch.uint8)
    # Fill with deterministic pattern so we can verify round-trip.
    torch.arange(raw.numel(), out=raw, dtype=torch.uint8)
    return TensorMemoryObj(raw_data=raw, metadata=meta, parent_allocator=None)


def _make_engine(
    worker_id: int,
    shard_size: int,
) -> MagicMock:
    """Create a minimal mock AscendLMCacheEngine for broadcast testing.

    Only the attributes accessed by ``_sharded_broadcast_and_load`` are set.
    The method will be invoked directly on the engine mock via its real
    implementation from AscendLMCacheEngine.
    """
    from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

    engine = MagicMock(spec=AscendLMCacheEngine)
    engine._broadcast_shard_size = shard_size

    meta = MagicMock()
    meta.worker_id = worker_id
    meta.first_rank = 0
    meta.is_first_rank.return_value = (worker_id == 0)
    engine.metadata = meta

    gpu_connector = MagicMock()
    engine.gpu_connector = gpu_connector

    return engine


def _run_sharded_broadcast(
    sender_chunks: list[tuple],
    shard_size: int,
    total_num_tokens: int = 1024,
) -> tuple[list, torch.Tensor]:
    """Simulate a two-rank sharded broadcast in-process.

    Returns ``(receiver_chunks, ret_mask)`` from the receiver's perspective.
    """
    from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

    shared: dict[str, Any] = {}

    def sender_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
        shared["tensor"] = tensor.clone()

    def receiver_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
        if "tensor" in shared:
            tensor.copy_(shared["tensor"])

    def sender_broadcast_object_fn(obj: Any, src: int) -> Any:
        shared["obj"] = obj
        return obj

    def receiver_broadcast_object_fn(obj: Any, src: int) -> Any:
        return shared.get("obj")

    sender = _make_engine(worker_id=0, shard_size=shard_size)
    receiver = _make_engine(worker_id=1, shard_size=shard_size)

    sender.broadcast_fn = sender_broadcast_fn
    sender.broadcast_object_fn = sender_broadcast_object_fn
    receiver.broadcast_fn = receiver_broadcast_fn
    receiver.broadcast_object_fn = receiver_broadcast_object_fn

    sender_ret_mask = torch.zeros(total_num_tokens, dtype=torch.bool)

    # Run sender side
    AscendLMCacheEngine._sharded_broadcast_and_load(
        sender, sender_chunks, sender_ret_mask
    )

    receiver_chunks: list = []
    receiver_ret_mask = torch.zeros(total_num_tokens, dtype=torch.bool)

    # Run receiver side
    AscendLMCacheEngine._sharded_broadcast_and_load(
        receiver, receiver_chunks, receiver_ret_mask
    )

    return receiver_chunks, receiver_ret_mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShardedBroadcast:
    """Tests for _sharded_broadcast_and_load covering all plan.md Step 5
    scenarios."""

    def test_shard_equals_total(self) -> None:
        """shard_size >= total_chunks: behavior equivalent to full broadcast."""
        chunks: list[tuple] = []
        for i in range(3):
            mem_obj = _make_memory_obj()
            chunks.append((MagicMock(), mem_obj, i * 256, (i + 1) * 256))

        receiver_chunks, ret_mask = _run_sharded_broadcast(
            sender_chunks=chunks,
            shard_size=10,  # larger than total_chunks (3)
            total_num_tokens=768,
        )

        # All 3 chunks received
        assert len(receiver_chunks) == 3
        # ret_mask marks all positions True
        assert ret_mask[:768].all()
        # Verify data integrity for each chunk
        for i, (recv_key, recv_obj, recv_start, recv_end) in enumerate(
            receiver_chunks
        ):
            _, send_obj, send_start, send_end = chunks[i]
            assert recv_start == send_start
            assert recv_end == send_end
            assert recv_obj.raw_tensor is not None
            torch.testing.assert_close(
                recv_obj.raw_tensor, send_obj.raw_tensor
            )

    def test_multiple_shards_partial_last(self) -> None:
        """Multiple shards with last shard not full: verify all chunks
        received correctly."""
        # 5 chunks, shard_size=2 => shards of [2, 2, 1]
        chunks: list[tuple] = []
        for i in range(5):
            mem_obj = _make_memory_obj()
            chunks.append((MagicMock(), mem_obj, i * 256, (i + 1) * 256))

        receiver_chunks, ret_mask = _run_sharded_broadcast(
            sender_chunks=chunks,
            shard_size=2,
            total_num_tokens=1280,
        )

        assert len(receiver_chunks) == 5
        assert ret_mask[:1280].all()
        for i in range(5):
            _, send_obj, _, _ = chunks[i]
            _, recv_obj, _, _ = receiver_chunks[i]
            assert recv_obj.raw_tensor is not None
            torch.testing.assert_close(
                recv_obj.raw_tensor, send_obj.raw_tensor
            )

    def test_single_chunk(self) -> None:
        """shard_size=1, total=1: minimal case."""
        mem_obj = _make_memory_obj()
        chunks = [(MagicMock(), mem_obj, 0, 256)]

        receiver_chunks, ret_mask = _run_sharded_broadcast(
            sender_chunks=chunks,
            shard_size=1,
            total_num_tokens=256,
        )

        assert len(receiver_chunks) == 1
        assert ret_mask[:256].all()
        _, recv_obj, _, _ = receiver_chunks[0]
        assert recv_obj.raw_tensor is not None
        torch.testing.assert_close(recv_obj.raw_tensor, mem_obj.raw_tensor)

    def test_empty_chunks(self) -> None:
        """Rank 0 with no cache hits: broadcast 0 chunks, nothing crashes."""
        chunks: list[tuple] = []

        receiver_chunks, ret_mask = _run_sharded_broadcast(
            sender_chunks=chunks,
            shard_size=4,
            total_num_tokens=1024,
        )

        assert len(receiver_chunks) == 0
        assert not ret_mask.any()

    def test_receiver_calls_batched_to_gpu_per_shard(self) -> None:
        """Verify batched_to_gpu is called once per shard on the receiver."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        # 4 chunks, shard_size=2 => 2 shards
        chunks: list[tuple] = []
        for i in range(4):
            mem_obj = _make_memory_obj()
            chunks.append((MagicMock(), mem_obj, i * 256, (i + 1) * 256))

        shared: dict[str, Any] = {}

        def sender_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            shared["tensor"] = tensor.clone()

        def receiver_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            if "tensor" in shared:
                tensor.copy_(shared["tensor"])

        def sender_broadcast_object_fn(obj: Any, src: int) -> Any:
            shared["obj"] = obj
            return obj

        def receiver_broadcast_object_fn(obj: Any, src: int) -> Any:
            return shared.get("obj")

        sender = _make_engine(worker_id=0, shard_size=2)
        receiver = _make_engine(worker_id=1, shard_size=2)
        sender.broadcast_fn = sender_broadcast_fn
        sender.broadcast_object_fn = sender_broadcast_object_fn
        receiver.broadcast_fn = receiver_broadcast_fn
        receiver.broadcast_object_fn = receiver_broadcast_object_fn

        sender_ret_mask = torch.zeros(1024, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            sender, chunks, sender_ret_mask
        )

        receiver_chunks: list = []
        receiver_ret_mask = torch.zeros(1024, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            receiver, receiver_chunks, receiver_ret_mask
        )

        # batched_to_gpu should have been called 2 times (2 shards of 2)
        assert receiver.gpu_connector.batched_to_gpu.call_count == 2

        # Verify each call had the right number of memory objects
        for call in receiver.gpu_connector.batched_to_gpu.call_args_list:
            call_args = call[0]
            memory_objs = call_args[0]
            assert len(memory_objs) == 2

    def test_receiver_ref_counts_down_per_shard(self) -> None:
        """Verify ref_count_down is called on each received memory object."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        chunks: list[tuple] = []
        for i in range(3):
            mem_obj = _make_memory_obj()
            chunks.append((MagicMock(), mem_obj, i * 256, (i + 1) * 256))

        shared: dict[str, Any] = {}

        def sender_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            shared["tensor"] = tensor.clone()

        def receiver_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            if "tensor" in shared:
                tensor.copy_(shared["tensor"])

        def sender_broadcast_object_fn(obj: Any, src: int) -> Any:
            shared["obj"] = obj
            return obj

        def receiver_broadcast_object_fn(obj: Any, src: int) -> Any:
            return shared.get("obj")

        receiver = _make_engine(worker_id=1, shard_size=2)
        sender = _make_engine(worker_id=0, shard_size=2)
        sender.broadcast_fn = sender_broadcast_fn
        sender.broadcast_object_fn = sender_broadcast_object_fn
        receiver.broadcast_fn = receiver_broadcast_fn
        receiver.broadcast_object_fn = receiver_broadcast_object_fn

        sender_ret_mask = torch.zeros(768, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            sender, chunks, sender_ret_mask
        )

        receiver_chunks: list = []
        receiver_ret_mask = torch.zeros(768, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            receiver, receiver_chunks, receiver_ret_mask
        )

        # 3 chunks => shard 1 (2 chunks) + shard 2 (1 chunk)
        # All received memory objects should have been ref_count_down'd
        for _, recv_obj, _, _ in receiver_chunks:
            # ref_count starts at 1, ref_count_down called once per obj
            assert recv_obj.meta.ref_count == 0

    def test_data_integrity_with_varying_chunk_sizes(self) -> None:
        """Chunks of different sizes all round-trip correctly."""
        shapes: list[tuple[int, ...]] = [
            (256, 2, 128),
            (128, 2, 64),
            (64, 2, 32),
        ]
        chunks: list[tuple] = []
        offset = 0
        for shape in shapes:
            mem_obj = _make_memory_obj(shape=shape)
            num_tokens = shape[0]
            chunks.append((MagicMock(), mem_obj, offset, offset + num_tokens))
            offset += num_tokens

        receiver_chunks, ret_mask = _run_sharded_broadcast(
            sender_chunks=chunks,
            shard_size=2,
            total_num_tokens=offset,
        )

        assert len(receiver_chunks) == 3
        assert ret_mask[:offset].all()
        for i in range(3):
            _, send_obj, _, _ = chunks[i]
            _, recv_obj, _, _ = receiver_chunks[i]
            assert recv_obj.raw_tensor is not None
            torch.testing.assert_close(
                recv_obj.raw_tensor, send_obj.raw_tensor
            )


class TestOOMFallback:
    """Test OOM degradation behavior during broadcast receive."""

    def test_oom_skips_chunk_no_crash(self) -> None:
        """When torch.empty raises OOM, the chunk is skipped (ret_mask stays
        False) and broadcast continues without crash."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        chunks: list[tuple] = []
        for i in range(3):
            mem_obj = _make_memory_obj()
            chunks.append((MagicMock(), mem_obj, i * 256, (i + 1) * 256))

        shared: dict[str, Any] = {}

        def sender_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            shared["tensor"] = tensor.clone()

        def receiver_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            if "tensor" in shared:
                tensor.copy_(shared["tensor"])

        def sender_broadcast_object_fn(obj: Any, src: int) -> Any:
            shared["obj"] = obj
            return obj

        def receiver_broadcast_object_fn(obj: Any, src: int) -> Any:
            return shared.get("obj")

        sender = _make_engine(worker_id=0, shard_size=3)
        receiver = _make_engine(worker_id=1, shard_size=3)
        sender.broadcast_fn = sender_broadcast_fn
        sender.broadcast_object_fn = sender_broadcast_object_fn
        receiver.broadcast_fn = receiver_broadcast_fn
        receiver.broadcast_object_fn = receiver_broadcast_object_fn

        # Patch torch.empty on the receiver module to raise OOM on the 2nd
        # call (chunk index 1). We need to intercept the call within
        # _sharded_broadcast_and_load.
        original_empty = torch.empty
        call_count = [0]

        def mock_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return original_empty(*args, **kwargs)

        sender_ret_mask = torch.zeros(768, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            sender, chunks, sender_ret_mask
        )

        receiver_chunks: list = []
        receiver_ret_mask = torch.zeros(768, dtype=torch.bool)

        with pytest.MonkeyPatch.context() as m:
            m.setattr("torch.empty", mock_empty)
            # Should not raise despite OOM
            AscendLMCacheEngine._sharded_broadcast_and_load(
                receiver, receiver_chunks, receiver_ret_mask
            )

        # Chunk 1 (index 1) should be skipped due to OOM
        assert len(receiver_chunks) == 2
        # ret_mask: chunk 0 and chunk 2 received, chunk 1 skipped
        assert receiver_ret_mask[0:256].all()
        assert not receiver_ret_mask[256:512].any()
        assert receiver_ret_mask[512:768].all()

    def test_oom_must_still_call_broadcast_fn(self) -> None:
        """OOM fallback must still call broadcast_fn to maintain HCCL sync."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        mem_obj = _make_memory_obj()
        chunks = [(MagicMock(), mem_obj, 0, 256)]

        shared: dict[str, Any] = {}
        broadcast_fn_calls: list[int] = []

        def sender_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            shared["tensor"] = tensor.clone()

        def receiver_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            broadcast_fn_calls.append(tensor.shape[0])
            if "tensor" in shared:
                tensor.copy_(shared["tensor"])

        def sender_broadcast_object_fn(obj: Any, src: int) -> Any:
            shared["obj"] = obj
            return obj

        def receiver_broadcast_object_fn(obj: Any, src: int) -> Any:
            return shared.get("obj")

        sender = _make_engine(worker_id=0, shard_size=1)
        receiver = _make_engine(worker_id=1, shard_size=1)
        sender.broadcast_fn = sender_broadcast_fn
        sender.broadcast_object_fn = sender_broadcast_object_fn
        receiver.broadcast_fn = receiver_broadcast_fn
        receiver.broadcast_object_fn = receiver_broadcast_object_fn

        original_empty = torch.empty

        def mock_empty_oom(*args: Any, **kwargs: Any) -> torch.Tensor:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        sender_ret_mask = torch.zeros(256, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            sender, chunks, sender_ret_mask
        )

        receiver_chunks: list = []
        receiver_ret_mask = torch.zeros(256, dtype=torch.bool)

        with pytest.MonkeyPatch.context() as m:
            m.setattr("torch.empty", mock_empty_oom)
            AscendLMCacheEngine._sharded_broadcast_and_load(
                receiver, receiver_chunks, receiver_ret_mask
            )

        # broadcast_fn was still called (on the CPU fallback tensor)
        assert len(broadcast_fn_calls) == 1
        # No chunks were successfully received
        assert len(receiver_chunks) == 0
        assert not receiver_ret_mask.any()


class TestSenderBehavior:
    """Tests for the sender (rank 0) side of _sharded_broadcast_and_load."""

    def test_sender_ret_mask_unchanged(self) -> None:
        """Rank 0's ret_mask should not be modified by
        _sharded_broadcast_and_load."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        chunks: list[tuple] = []
        for i in range(3):
            mem_obj = _make_memory_obj()
            chunks.append((MagicMock(), mem_obj, i * 256, (i + 1) * 256))

        engine = _make_engine(worker_id=0, shard_size=2)
        engine.broadcast_fn = MagicMock()
        engine.broadcast_object_fn = MagicMock(return_value=None)

        ret_mask = torch.zeros(768, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            engine, chunks, ret_mask
        )

        # Sender does not modify ret_mask; it was pre-populated by
        # _process_tokens_internal in the real flow.
        assert not ret_mask.any()

    def test_sender_broadcasts_total_then_shard_counts(self) -> None:
        """Verify sender broadcasts total_chunks, then shard_count per
        shard."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        chunks: list[tuple] = []
        for i in range(5):
            mem_obj = _make_memory_obj()
            chunks.append((MagicMock(), mem_obj, i * 256, (i + 1) * 256))

        broadcast_object_calls: list[Any] = []
        broadcast_tensor_calls: list[torch.Tensor] = []

        def mock_broadcast_object_fn(obj: Any, src: int) -> Any:
            broadcast_object_calls.append(obj)
            return obj

        def mock_broadcast_fn(tensor: torch.Tensor, src: int) -> None:
            broadcast_tensor_calls.append(tensor)

        engine = _make_engine(worker_id=0, shard_size=2)
        engine.broadcast_fn = mock_broadcast_fn
        engine.broadcast_object_fn = mock_broadcast_object_fn

        ret_mask = torch.zeros(1280, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            engine, chunks, ret_mask
        )

        # First broadcast_object_fn call: total_chunks = 5
        assert broadcast_object_calls[0] == 5

        # Then shard_counts: 2, 2, 1 (for shards of [2, 2, 1])
        shard_counts = broadcast_object_calls[1::6]  # every 6th after first
        # More precisely: after total_chunks, the pattern repeats:
        # shard_count, metadata0, metadata1, ... then tensors
        # Let's extract shard_counts from the object calls
        obj_idx = 1  # start after total_chunks
        extracted_shard_counts: list[int] = []
        while obj_idx < len(broadcast_object_calls):
            shard_count = broadcast_object_calls[obj_idx]
            extracted_shard_counts.append(shard_count)
            obj_idx += 1 + shard_count  # skip metadata calls

        assert extracted_shard_counts == [2, 2, 1]

    def test_sender_empty_chunks_broadcasts_zero(self) -> None:
        """Sender with 0 chunks broadcasts total_chunks=0 and returns."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        broadcast_object_calls: list[Any] = []

        def mock_broadcast_object_fn(obj: Any, src: int) -> Any:
            broadcast_object_calls.append(obj)
            return obj

        engine = _make_engine(worker_id=0, shard_size=4)
        engine.broadcast_fn = MagicMock()
        engine.broadcast_object_fn = mock_broadcast_object_fn

        ret_mask = torch.zeros(1024, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            engine, [], ret_mask
        )

        # Only one call: total_chunks = 0
        assert len(broadcast_object_calls) == 1
        assert broadcast_object_calls[0] == 0


class TestReceiverBehavior:
    """Tests for the receiver (non-rank 0) side."""

    def test_receiver_none_total_chunks(self) -> None:
        """If broadcast_object_fn returns None for total_chunks, receiver
        returns immediately."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        engine = _make_engine(worker_id=1, shard_size=4)
        engine.broadcast_fn = MagicMock()
        engine.broadcast_object_fn = MagicMock(return_value=None)

        chunks: list = []
        ret_mask = torch.zeros(1024, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            engine, chunks, ret_mask
        )

        assert len(chunks) == 0
        assert not ret_mask.any()

    def test_receiver_none_shard_count_aborts(self) -> None:
        """If shard_count is None mid-broadcast, receiver aborts."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        call_count = [0]

        def mock_broadcast_object_fn(obj: Any, src: int) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return 2  # total_chunks
            if call_count[0] == 2:
                return None  # shard_count = None => abort
            return None

        engine = _make_engine(worker_id=1, shard_size=2)
        engine.broadcast_fn = MagicMock()
        engine.broadcast_object_fn = mock_broadcast_object_fn

        chunks: list = []
        ret_mask = torch.zeros(512, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            engine, chunks, ret_mask
        )

        assert len(chunks) == 0
        assert not ret_mask.any()

    def test_receiver_none_metadata_aborts(self) -> None:
        """If combined_metadata is None mid-shard, receiver aborts."""
        from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine

        call_count = [0]

        def mock_broadcast_object_fn(obj: Any, src: int) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return 1  # total_chunks
            if call_count[0] == 2:
                return 1  # shard_count
            # call 3: metadata = None => abort
            return None

        engine = _make_engine(worker_id=1, shard_size=1)
        engine.broadcast_fn = MagicMock()
        engine.broadcast_object_fn = mock_broadcast_object_fn

        chunks: list = []
        ret_mask = torch.zeros(256, dtype=torch.bool)
        AscendLMCacheEngine._sharded_broadcast_and_load(
            engine, chunks, ret_mask
        )

        assert len(chunks) == 0
        assert not ret_mask.any()
