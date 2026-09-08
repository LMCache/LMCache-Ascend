# SPDX-License-Identifier: Apache-2.0
"""GDN copy regressions using the real NPU operator and registered host pool."""

# Standard
from math import prod

# Third Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import PinMemoryAllocator
import pytest
import torch

# First Party
from lmcache_ascend.v1.state_checkpoint import (
    CheckpointRef,
    StateBlockBinding,
    StateOperation,
)
from lmcache_ascend.v1.state_layout import build_state_group_layout
from lmcache_ascend.v1.state_memory import allocate_state_checkpoint
from lmcache_ascend.v1.state_transfer import transfer_state
import lmcache_ascend.c_ops as lmc_ops


@pytest.fixture
def pool():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU required")
    allocator = PinMemoryAllocator(8 * 1024 * 1024)
    yield allocator
    try:
        torch.npu.current_stream().synchronize()
        assert allocator.allocator.num_active_allocations == 0
        assert allocator.memcheck()
    finally:
        allocator.close()


def runtime_tensors(
    shapes=((3, 128), (2, 128, 128)),
    dtypes=(torch.bfloat16, torch.float32),
    layers=2,
):
    entries = []
    for layer in range(layers):
        planes = []
        for plane, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True)):
            values = torch.arange(4 * prod(shape), device="npu").reshape(4, *shape)
            values = values % 17 + layer * 32 + plane * 8
            for block in range(4):
                values[block] += block * 2
            planes.append(values.to(dtype))
        entries.append(tuple(planes))
    return tuple(entries)


def group_layout(entries):
    return build_state_group_layout(
        0, tuple(f"gdn.{i}" for i in range(len(entries))), entries
    )


def plane_major(entries):
    return [entry[plane] for plane in range(2) for entry in entries]


def assert_bytes_equal(actual, expected):
    # Snapshot CPU payload before expected.cpu() can synchronize the NPU stream.
    actual_bytes = actual.cpu().contiguous().view(torch.uint8).clone()
    assert torch.equal(actual_bytes, expected.cpu().contiguous().view(torch.uint8))


def operation(entries, buffer, direction, block=1, boundary=16):
    ref = CheckpointRef.from_chunk(
        CacheEngineKey("qwen-test", 1, 0, 123, torch.bfloat16),
        chunk_end=boundary,
        boundary=boundary,
        chunk_size=16,
        group_index=buffer.layout.group_index,
    )
    return StateOperation(ref, StateBlockBinding(entries, block), buffer, direction)


def assert_payload_matches(buffer, entries, block):
    for plane, payload in enumerate(buffer.planes):
        for layer, entry in enumerate(entries):
            assert_bytes_equal(payload[layer], entry[plane][block])


@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_kernel_round_trip(pool, dtypes):
    source = runtime_tensors(dtypes=dtypes)
    target = tuple(tuple(torch.full_like(t, -1) for t in entry) for entry in source)
    layout = group_layout(source)
    with allocate_state_checkpoint(layout, pool) as buffer:
        raw = buffer.memory_obj.raw_tensor
        base = lmc_ops.get_device_ptr(raw.data_ptr())
        assert base != 0
        for plane, view in zip(layout.planes, buffer.planes, strict=True):
            assert lmc_ops.get_device_ptr(view.data_ptr()) == base + plane.offset

        lmc_ops.multi_layer_gdn_state_transfer(
            list(buffer.planes), plane_major(source), 1, True
        )
        torch.npu.current_stream().synchronize()
        for plane, payload in enumerate(buffer.planes):
            for layer, entry in enumerate(source):
                assert_bytes_equal(payload[layer], entry[plane][1])

        lmc_ops.multi_layer_gdn_state_transfer(
            list(buffer.planes), plane_major(target), 2, False
        )
        torch.npu.current_stream().synchronize()
        for source_entry, target_entry in zip(source, target, strict=True):
            for source_plane, target_plane in zip(
                source_entry, target_entry, strict=True
            ):
                assert_bytes_equal(target_plane[2], source_plane[1])
                for untouched in (0, 1, 3):
                    assert torch.all(target_plane[untouched].cpu() == -1)


@pytest.mark.parametrize(
    "invalid",
    [
        "dtype",
        "block",
        "noncontiguous",
        "cpu_runtime",
        "unregistered",
        "plane_count",
        "state_count",
        "shape",
        "mixed_device",
    ],
)
def test_kernel_rejects_invalid_inputs_before_copy(pool, invalid):
    source = runtime_tensors()
    with allocate_state_checkpoint(group_layout(source), pool) as buffer:
        memories = list(buffer.planes)
        states = plane_major(source)
        block = 1
        if invalid == "dtype":
            # Reject the second plane before submitting the valid first plane.
            memories[1] = torch.empty_like(memories[1], dtype=torch.float16)
            states[2:] = [t.to(torch.float16) for t in states[2:]]
        elif invalid == "block":
            block = source[0][0].shape[0]
        elif invalid == "noncontiguous":
            states[-1] = states[-1].transpose(-1, -2)
        elif invalid == "cpu_runtime":
            states = [t.cpu() for t in states]
        elif invalid == "unregistered":
            memories[1] = torch.empty_like(memories[1])
            assert lmc_ops.get_device_ptr(memories[1].data_ptr()) == 0
        elif invalid == "plane_count":
            memories = memories[:1]
        elif invalid == "state_count":
            states = states[:-1]
        elif invalid == "shape":
            states[-1] = torch.empty((4, 1, 128, 128), device="npu")
        else:
            states[-1] = states[-1].cpu()
        for payload in buffer.planes:
            payload.fill_(-7)
        with pytest.raises(RuntimeError):
            lmc_ops.multi_layer_gdn_state_transfer(memories, states, block, True)
        torch.npu.current_stream().synchronize()
        for payload in buffer.planes:
            assert torch.all(payload == -7)


@pytest.mark.parametrize("source_block, target_block", [(0, 3), (1, 2)])
def test_operation_round_trip(pool, source_block, target_block):
    source = runtime_tensors()
    target = tuple(tuple(torch.full_like(t, -1) for t in entry) for entry in source)
    with allocate_state_checkpoint(group_layout(source), pool) as buffer:
        transfer_state(operation(source, buffer, "store", source_block))
        # Read registered CPU payload immediately: transfer_state must have waited.
        assert_payload_matches(buffer, source, source_block)
        transfer_state(operation(target, buffer, "load", target_block))
        for source_entry, target_entry in zip(source, target, strict=True):
            for source_plane, target_plane in zip(
                source_entry, target_entry, strict=True
            ):
                assert_bytes_equal(
                    target_plane[target_block], source_plane[source_block]
                )
                for block in set(range(4)) - {target_block}:
                    assert torch.all(target_plane[block].cpu() == -1)


@pytest.mark.parametrize("invalid", ["released", "noncontiguous", "block"])
def test_operation_revalidates_borrowed_inputs(pool, invalid):
    source = runtime_tensors()
    with allocate_state_checkpoint(group_layout(source), pool) as buffer:
        store = operation(source, buffer, "store")
        if invalid == "released":
            buffer.close()
            with pytest.raises(RuntimeError, match="released"):
                transfer_state(store)
        else:
            for payload in buffer.planes:
                payload.fill_(-7)
            if invalid == "noncontiguous":
                source[0][1].transpose_(-1, -2)
            else:
                source[0][0].resize_(1, *source[0][0].shape[1:])
            with pytest.raises(ValueError, match="runtime layout|out of range"):
                transfer_state(store)
            for payload in buffer.planes:
                assert torch.all(payload == -7)


@pytest.mark.parametrize(
    "conv_numel, ssm_numel, layers",
    [
        pytest.param(8192, 4096, 1, id="16KiB"),
        pytest.param(16384, 8192, 5, id="32KiB-five-layers"),
        pytest.param(8208, 4104, 1, id="tile-plus-32-byte-tail"),
        pytest.param(8193, 4097, 1, id="probe-unaligned-tail"),
        pytest.param(3, 4, 1, id="probe-small-payload-with-padding"),
        pytest.param(8193, 4097, 3, id="unaligned-tail-three-layers"),
        pytest.param(3, 4, 3, id="small-payload-three-layers"),
    ],
)
def test_transfer_tile_boundaries(pool, conv_numel, ssm_numel, layers):
    # The probes must produce evidence; do not xfail or silently skip failures.
    source = runtime_tensors(((1, conv_numel), (ssm_numel, 1)), layers=layers)
    target = tuple(tuple(torch.full_like(t, -1) for t in entry) for entry in source)
    layout = group_layout(source)
    with allocate_state_checkpoint(layout, pool) as buffer:
        raw = buffer.memory_obj.raw_tensor
        raw.fill_(91)
        transfer_state(operation(source, buffer, "store", block=3))
        assert torch.npu.current_stream().query()
        assert_payload_matches(buffer, source, 3)
        for left, right in zip(layout.planes, layout.planes[1:], strict=False):
            assert torch.all(raw[left.offset + left.nbytes : right.offset] == 91)
        transfer_state(operation(target, buffer, "load", block=0))
        assert torch.npu.current_stream().query()
        for original, restored in zip(source, target, strict=True):
            for src, dst in zip(original, restored, strict=True):
                assert_bytes_equal(dst[0], src[3])
                assert torch.all(dst[1:].cpu() == -1)


def test_transfer_completion_and_reuse(pool):
    source = runtime_tensors()
    layout = group_layout(source)
    current = torch.npu.current_stream()
    producer = torch.npu.Stream()
    producer.wait_stream(current)
    # A simultaneously owned checkpoint must survive repeated pool reuse.
    with allocate_state_checkpoint(layout, pool) as retained:
        for payload in retained.planes:
            payload.fill_(-9)
        for iteration in range(3):
            with allocate_state_checkpoint(layout, pool) as buffer:
                with torch.npu.stream(producer):
                    for layer, entry in enumerate(source):
                        for plane, tensor in enumerate(entry):
                            tensor.fill_(iteration * 10 + layer * 2 + plane)
                current.wait_stream(producer)
                transfer_state(operation(source, buffer, "store"))
                assert current.query()
                for plane, payload in enumerate(buffer.planes):
                    for layer in range(len(source)):
                        assert torch.all(
                            payload[layer] == iteration * 10 + layer * 2 + plane
                        )
                for payload in retained.planes:
                    assert torch.all(payload == -9)
            # The previous call completed before its allocation was released.


def test_transfer_drains_failed_submission(pool, monkeypatch):
    source = runtime_tensors()
    original = lmc_ops.multi_layer_gdn_state_transfer
    submitted = torch.npu.Event()

    def submit_then_fail(*args):
        original(*args)
        submitted.record()
        raise RuntimeError("injected post-submission failure")

    monkeypatch.setattr(lmc_ops, "multi_layer_gdn_state_transfer", submit_then_fail)
    with allocate_state_checkpoint(group_layout(source), pool) as buffer:
        with pytest.raises(RuntimeError, match="injected post-submission failure"):
            transfer_state(operation(source, buffer, "store"))
        assert submitted.query()
        # Failure is propagated; this buffer is released, never published.


def test_kernel_rejects_cross_npu_runtime(pool):
    if torch.npu.device_count() < 2:
        pytest.skip("Two NPUs required for cross-device rejection")
    source = runtime_tensors()
    with allocate_state_checkpoint(group_layout(source), pool) as buffer:
        states = plane_major(source)
        other = (states[0].device.index + 1) % torch.npu.device_count()
        states[-1] = states[-1].to(f"npu:{other}")
        with pytest.raises(RuntimeError, match="same NPU"):
            lmc_ops.multi_layer_gdn_state_transfer(list(buffer.planes), states, 1, True)
