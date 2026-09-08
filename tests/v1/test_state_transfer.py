# SPDX-License-Identifier: Apache-2.0
"""GDN copy regressions using the real NPU operator and registered host pool."""

# Standard
from math import prod

# Third Party
from lmcache.v1.memory_management import PinMemoryAllocator
import pytest
import torch

# First Party
from lmcache_ascend.v1.state_layout import build_state_group_layout
from lmcache_ascend.v1.state_memory import allocate_state_checkpoint
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
    assert torch.equal(
        actual.cpu().contiguous().view(torch.uint8),
        expected.cpu().contiguous().view(torch.uint8),
    )


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
