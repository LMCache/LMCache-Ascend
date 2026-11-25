# SPDX-License-Identifier: Apache-2.0
"""P1 CacheBlend data-plane tests.

These tests launch Ascend kernels and are skipped unless an NPU is available.
They intentionally exercise the flat LMCache ABI used by the MP planner.
"""

# Future
from __future__ import annotations

# Standard
import sys
import threading

# Third Party
import numpy as np
import pytest
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU required",
)

FMT2 = int(lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS)
FMT16 = int(lmc_ops.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS)


def _empty_table() -> np.ndarray:
    return np.empty((0, 4), dtype=np.int64)


def _kv_layers(
    *,
    device: torch.device,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    block_stride_elems: int,
    fill: float,
    dtype: torch.dtype = torch.float16,
    separate_planes: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Canonical [NB,2,BS,NH,HS] views over interleaved or split K/V storage."""
    storages: list[torch.Tensor] = []
    views: list[torch.Tensor] = []
    hidden = nh * hs
    for _ in range(nl):
        storage = torch.full(
            (nb * block_stride_elems * (2 if separate_planes else 1),),
            fill,
            dtype=dtype,
            device=device,
        )
        view = torch.as_strided(
            storage,
            (nb, 2, bs, nh, hs),
            (
                block_stride_elems,
                nb * block_stride_elems if separate_planes else bs * hidden,
                hidden,
                hs,
                1,
            ),
        )
        storages.append(storage)
        views.append(view)
    return storages, views


def _pointer_table(layers, engine_format, device):
    pointers = (
        [layer.data_ptr() for layer in layers]
        if engine_format == FMT2
        else [
            plane.data_ptr() for layer in layers for plane in (layer[:, 0], layer[:, 1])
        ]
    )
    return torch.tensor(pointers, dtype=torch.int64, device=device)


def _group_spec(
    *,
    pointer_table: torch.Tensor,
    temp: torch.Tensor,
    slot_mapping: torch.Tensor,
    nl: int,
    slot_tokens: int,
    hidden: int,
    page_buffer_size: int,
    block_size: int,
    head_size: int,
    block_stride_elems: int,
    cos_sin_cache: torch.Tensor | None = None,
    is_neox: bool = True,
    engine_format: int = FMT2,
    temp_slots: list[torch.Tensor] | None = None,
) -> object:
    scalar_type = {torch.float16: 5, torch.bfloat16: 15, torch.float32: 6}[temp.dtype]
    return lmc_ops.CBGroupSpec(
        pointer_table.data_ptr(),
        (
            [slot.data_ptr() for slot in temp_slots]
            if temp_slots is not None
            else [temp.data_ptr()]
        ),
        nl,
        slot_tokens,
        hidden,
        temp.element_size(),
        engine_format,
        page_buffer_size,
        block_size,
        head_size,
        slot_mapping.data_ptr(),
        slot_mapping.numel(),
        0 if cos_sin_cache is None else cos_sin_cache.data_ptr(),
        head_size,
        hidden // head_size,
        head_size,
        scalar_type,
        is_neox,
        0,
        block_stride_elems,
        scalar_type,
    )


def _rope_reference(
    key: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    """Independent CPU RoPE from pre-RoPE K, with no native re-RoPE calls.

    key is [layers, tokens, heads, head_size]. Compute each target position
    directly in float64 rather than undoing the stored rotation.
    """
    key = key.to(device="cpu", dtype=torch.float64)
    angles = positions.to(dtype=torch.float64)[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]
    if is_neox:
        first, second = key.chunk(2, dim=-1)
        return torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
    first, second = key[..., ::2], key[..., 1::2]
    return torch.stack(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    ).flatten(-2)


def _rope_cache(head_size, max_position, dtype, device):
    inv_freq = 10000.0 ** (
        -torch.arange(0, head_size, 2, dtype=torch.float64) / head_size
    )
    positions = torch.arange(max_position, dtype=torch.float64)
    angles = positions[:, None] * inv_freq[None, :]
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(
        device=device, dtype=dtype
    )
    return cache, inv_freq


def _reference_tolerance(dtype):
    # Reusing low-precision K incurs old-K quantization plus two rotations
    # with a quantized trig table; the independent oracle rotates raw K once.
    return {
        torch.float16: dict(rtol=3e-3, atol=5e-3),
        torch.bfloat16: dict(rtol=2e-2, atol=5e-2),
        torch.float32: dict(rtol=1e-5, atol=1e-5),
    }[dtype]


def test_flat_plan_rejects_bad_table_shape_without_launch() -> None:
    with pytest.raises(RuntimeError, match=r"staging table must be \[n, 4\]"):
        lmc_ops.execute_cb_retrieve_plan_flat(
            torch.device("npu:0"),
            4096,
            [],
            np.empty((0, 3), dtype=np.int64),
            _empty_table(),
            _empty_table(),
            np.empty((0, 3), dtype=np.int64),
        )


@requires_npu
@pytest.mark.parametrize(
    "operation,row,error",
    [
        ("rope", [0, -1, 0, 1], "CBRopeVar.slot_idx"),
        ("rope", [0, 1, 0, 1], "CBRopeVar.slot_idx"),
        ("scatter", [0, -1, 0, 1], "CBScatterVar.slot_idx"),
        ("scatter", [0, 1, 0, 1], "CBScatterVar.slot_idx"),
        ("scatter", [0, 0, 0, -1], "CBScatterVar.n_tok"),
        ("scatter", [0, 0, 0, 2], "CBScatterVar.n_tok"),
        ("scatter", [0, 0, -1, 1], "CBScatterVar slot_mapping slice"),
        ("scatter", [0, 0, 1, 1], "CBScatterVar slot_mapping slice"),
    ],
)
def test_cb_plan_upstream_boundaries(operation, row, error) -> None:
    device = torch.device("npu:0")
    temp = torch.zeros((2, 1, 1, 32), dtype=torch.float16, device=device)
    pointers = torch.zeros(1, dtype=torch.int64, device=device)
    mapping = torch.zeros(1, dtype=torch.int64, device=device)
    cache, _ = _rope_cache(32, 2, temp.dtype, device)
    spec = _group_spec(
        pointer_table=pointers,
        temp=temp,
        slot_mapping=mapping,
        nl=1,
        slot_tokens=1,
        hidden=32,
        page_buffer_size=4,
        block_size=4,
        head_size=32,
        block_stride_elems=256,
        cos_sin_cache=cache,
    )
    rows = np.array([row], dtype=np.int64)
    ropes = rows if operation == "rope" else _empty_table()
    scatters = rows if operation == "scatter" else _empty_table()
    offsets = np.array([[0, len(ropes), len(scatters)]], dtype=np.int64)
    with pytest.raises(RuntimeError, match=error):
        lmc_ops.execute_cb_retrieve_plan_flat(
            device,
            4096,
            [spec],
            _empty_table(),
            ropes,
            scatters,
            offsets,
        )


@requires_npu
def test_cb_plan_skips_disabled_rope_and_unmatched_groups() -> None:
    device = torch.device("npu:0")
    temp = torch.zeros((2, 1, 1, 32), dtype=torch.float16, device=device)
    pointers = torch.zeros(1, dtype=torch.int64, device=device)
    mapping = torch.zeros(1, dtype=torch.int64, device=device)
    spec = _group_spec(
        pointer_table=pointers,
        temp=temp,
        slot_mapping=mapping,
        nl=1,
        slot_tokens=1,
        hidden=32,
        page_buffer_size=4,
        block_size=4,
        head_size=32,
        block_stride_elems=256,
    )
    # CUDA skips disabled RoPE before checking its slot, and ignores rows
    # whose group index does not match any registered group.
    ropes = np.array([[0, -1, -1, -1], [1, -1, -1, -1]], dtype=np.int64)
    scatters = np.array([[-1, -1, -1, -1], [1, -1, -1, -1]], dtype=np.int64)
    lmc_ops.execute_cb_retrieve_plan_flat(
        device,
        4096,
        [spec],
        _empty_table(),
        ropes,
        scatters,
        np.array([[0, 2, 2]], dtype=np.int64),
    )
    torch.npu.synchronize()
    assert torch.count_nonzero(temp).item() == 0


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "layout", ["format2", "format16_interleaved", "format16_split"]
)
def test_cb_plan_serial_staging_scatter_cross_block_and_padding(
    dtype: torch.dtype, layout: str
) -> None:
    device = torch.device("npu:0")
    nl, nb, bs, nh, hs = 2, 3, 4, 2, 16
    hidden = nh * hs
    separate_planes = layout == "format16_split"
    engine_format = FMT2 if layout == "format2" else FMT16
    tight_stride = (1 if separate_planes else 2) * bs * hidden
    block_stride = tight_stride + 64
    slot_tokens = 3  # capacity exceeds the two tokens written per step
    n_tok = 2
    sentinel = -7.0

    storages, layers = _kv_layers(
        device=device,
        nl=nl,
        nb=nb,
        bs=bs,
        nh=nh,
        hs=hs,
        block_stride_elems=block_stride,
        fill=sentinel,
        dtype=dtype,
        separate_planes=separate_planes,
    )
    pointer_table = _pointer_table(layers, engine_format, device)
    slot_mapping = torch.tensor([3, 4, 5, 8], dtype=torch.int64, device=device)
    temp = torch.zeros((2, nl, slot_tokens, hidden), dtype=dtype, device=device)
    host_slots = [
        (torch.arange(temp.numel(), dtype=dtype) + 10).reshape(temp.shape),
        (torch.arange(temp.numel(), dtype=dtype) + 1000).reshape(temp.shape),
    ]
    spec = _group_spec(
        pointer_table=pointer_table,
        temp=temp,
        slot_mapping=slot_mapping,
        nl=nl,
        slot_tokens=slot_tokens,
        hidden=hidden,
        page_buffer_size=nb * bs,
        block_size=bs,
        head_size=hs,
        block_stride_elems=block_stride,
        engine_format=engine_format,
    )

    staging = np.array(
        [[temp.data_ptr(), host.data_ptr(), host.nbytes, 0] for host in host_slots],
        dtype=np.int64,
    )
    scatters = np.array([[0, 0, 0, 2], [0, 0, 2, 2]], dtype=np.int64)
    offsets = np.array([[1, 0, 1], [2, 0, 2]], dtype=np.int64)
    lmc_ops.execute_cb_retrieve_plan_flat(
        device, 4096, [spec], staging, _empty_table(), scatters, offsets
    )
    torch.npu.synchronize()

    expected_storages, expected_layers = _kv_layers(
        device=torch.device("cpu"),
        nl=nl,
        nb=nb,
        bs=bs,
        nh=nh,
        hs=hs,
        block_stride_elems=block_stride,
        fill=sentinel,
        dtype=dtype,
        separate_planes=separate_planes,
    )
    for step, host in enumerate(host_slots):
        for token in range(n_tok):
            slot = int(slot_mapping[step * n_tok + token].cpu())
            block, in_block = divmod(slot, bs)
            for plane in range(2):
                for layer in range(nl):
                    expected_layers[layer][block, plane, in_block].copy_(
                        host[plane, layer, token].reshape(nh, hs)
                    )
    for actual, expected in zip(storages, expected_storages, strict=True):
        # Flat-storage equality also verifies that padding remains untouched.
        assert torch.equal(actual.cpu(), expected)


@requires_npu
@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("engine_format", [FMT2, FMT16])
def test_cb_plan_range_rerope_matches_explicit_positions(
    is_neox: bool, dtype: torch.dtype, engine_format: int
) -> None:
    device = torch.device("npu:0")
    nl, nb, bs, nh, hs = 2, 2, 4, 2, 32
    hidden = nh * hs
    slot_tokens = 3
    old_start, new_start = 2, 11
    block_stride = 2 * bs * hidden

    _, layers = _kv_layers(
        device=device,
        nl=nl,
        nb=nb,
        bs=bs,
        nh=nh,
        hs=hs,
        block_stride_elems=block_stride,
        fill=0,
        dtype=dtype,
    )
    pointer_table = _pointer_table(layers, engine_format, device)
    slot_mapping = torch.arange(slot_tokens, dtype=torch.int64, device=device)
    temp = torch.randn((2, nl, slot_tokens, hidden), dtype=dtype, device=device)
    original_v = temp[1].clone()

    max_position = 32
    freq = torch.arange(hs // 2, dtype=torch.float32) / (hs // 2)
    angles = torch.arange(max_position, dtype=torch.float32)[:, None] * freq[None]
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(
        device=device, dtype=dtype
    )
    expected_k = temp[0].clone().reshape(nl * slot_tokens, hidden)
    old_positions = torch.arange(
        old_start, old_start + slot_tokens, dtype=torch.int64, device=device
    ).repeat(nl)
    new_positions = torch.arange(
        new_start, new_start + slot_tokens, dtype=torch.int64, device=device
    ).repeat(nl)
    lmc_ops.rotary_embedding_k_fused(
        old_positions,
        new_positions,
        expected_k,
        hs,
        cos_sin_cache,
        is_neox,
    )

    spec = _group_spec(
        pointer_table=pointer_table,
        temp=temp,
        slot_mapping=slot_mapping,
        nl=nl,
        slot_tokens=slot_tokens,
        hidden=hidden,
        page_buffer_size=nb * bs,
        block_size=bs,
        head_size=hs,
        block_stride_elems=block_stride,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        engine_format=engine_format,
    )
    ropes = np.array([[0, 0, old_start, new_start]], dtype=np.int64)
    # Test the full ordering: staging -> re-RoPE -> transfer, not RoPE alone.
    host = temp.cpu()
    temp.zero_()
    staging = np.array(
        [[temp.data_ptr(), host.data_ptr(), host.nbytes, 0]], dtype=np.int64
    )
    scatters = np.array([[0, 0, 0, slot_tokens]], dtype=np.int64)
    offsets = np.array([[1, 1, 1]], dtype=np.int64)
    lmc_ops.execute_cb_retrieve_plan_flat(
        device, 4096, [spec], staging, ropes, scatters, offsets
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        temp[0].reshape_as(expected_k),
        expected_k,
        rtol=1e-2 if dtype == torch.bfloat16 else 1e-3,
        atol=1e-1 if dtype == torch.bfloat16 else 5e-2,
    )
    assert torch.equal(temp[1], original_v)

    for layer in range(nl):
        for token in range(slot_tokens):
            torch.testing.assert_close(
                layers[layer][0, 0, token],
                expected_k.reshape(nl, slot_tokens, nh, hs)[layer, token],
                rtol=1e-2 if dtype == torch.bfloat16 else 1e-3,
                atol=1e-1 if dtype == torch.bfloat16 else 5e-2,
            )
            assert torch.equal(
                layers[layer][0, 1, token],
                original_v[layer, token].reshape(nh, hs),
            )


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cb_plan_random_multigroup_rerope_double_slot_reuse(dtype) -> None:
    """Full pipeline, three waves, slots 0/1/0, mixed formats and RoPE modes."""
    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(354)
    nl, nb, bs, nh, hs = 2, 4, 4, 2, 32
    capacity, n_tok = 5, 3
    hidden = nh * hs
    stride = 2 * bs * hidden + 64
    cos_sin_cache, inv_freq = _rope_cache(hs, 64, dtype, device)
    formats = [FMT16, FMT2]
    styles = [True, False]
    specs, owners, golden = [], [], []
    for group in range(2):
        storages, layers = _kv_layers(
            device=device,
            nl=nl,
            nb=nb,
            bs=bs,
            nh=nh,
            hs=hs,
            block_stride_elems=stride,
            fill=-7,
            dtype=dtype,
        )
        expected_storage, expected_layers = _kv_layers(
            device=torch.device("cpu"),
            nl=nl,
            nb=nb,
            bs=bs,
            nh=nh,
            hs=hs,
            block_stride_elems=stride,
            fill=-7,
            dtype=dtype,
        )
        pointers = _pointer_table(layers, formats[group], device)
        slots = [
            torch.zeros((2, nl, capacity, hidden), dtype=dtype, device=device)
            for _ in range(2)
        ]
        mapping_cpu = torch.randperm(nb * bs, generator=generator)[: 3 * n_tok]
        mapping = mapping_cpu.to(device)
        spec = _group_spec(
            pointer_table=pointers,
            temp=slots[0],
            slot_mapping=mapping,
            nl=nl,
            slot_tokens=capacity,
            hidden=hidden,
            page_buffer_size=nb * bs,
            block_size=bs,
            head_size=hs,
            block_stride_elems=stride,
            engine_format=formats[group],
            temp_slots=slots,
            cos_sin_cache=cos_sin_cache,
            is_neox=styles[group],
        )
        specs.append(spec)
        owners.append((storages, pointers, slots, mapping, mapping_cpu))
        golden.append((expected_storage, expected_layers))

    hosts, staging, ropes, scatters, offsets = [], [], [], [], []
    for wave in range(3):
        parity = wave % 2
        for group, (_, _, slots, _, mapping_cpu) in enumerate(owners):
            host = torch.randn(slots[parity].shape, generator=generator).to(dtype)
            raw_key = host[0].clone().reshape(nl, capacity, nh, hs)
            old_start, new_start = 2 + wave * 7 + group, 25 - wave * 5 + group
            old_positions = torch.arange(old_start, old_start + capacity)
            new_positions = torch.arange(new_start, new_start + capacity)
            stored_key = _rope_reference(
                raw_key, old_positions, inv_freq, styles[group]
            )
            host[0].copy_(stored_key.reshape(nl, capacity, hidden).to(dtype))
            expected_key = _rope_reference(
                raw_key, new_positions, inv_freq, styles[group]
            ).to(dtype)
            hosts.append(host)  # Keep CPU sources alive until stream completion.
            staging.append([slots[parity].data_ptr(), host.data_ptr(), host.nbytes, 0])
            ropes.append([group, parity, old_start, new_start])
            scatters.append([group, parity, wave * n_tok, n_tok])
            for token in range(n_tok):
                block, in_block = divmod(int(mapping_cpu[wave * n_tok + token]), bs)
                for layer in range(nl):
                    for plane in range(2):
                        golden[group][1][layer][block, plane, in_block].copy_(
                            expected_key[layer, token]
                            if plane == 0
                            else host[plane, layer, token].reshape(nh, hs)
                        )
        offsets.append([len(staging), len(ropes), len(scatters)])
    lmc_ops.execute_cb_retrieve_plan_flat(
        device,
        4096,
        specs,
        np.array(staging, dtype=np.int64),
        np.array(ropes, dtype=np.int64),
        np.array(scatters, dtype=np.int64),
        np.array(offsets, dtype=np.int64),
    )
    torch.npu.synchronize()
    for (storages, *_), (expected_storages, _) in zip(owners, golden, strict=True):
        for actual, expected in zip(storages, expected_storages, strict=True):
            actual_cpu = actual.cpu()
            torch.testing.assert_close(
                actual_cpu, expected, **_reference_tolerance(dtype)
            )
            untouched = expected == -7
            assert torch.equal(actual_cpu[untouched], expected[untouched])
    for group, (_, _, slots, _, _) in enumerate(owners):
        # After three waves slot 0 holds wave 2 and slot 1 holds wave 1.
        for parity, last_wave in ((0, 2), (1, 1)):
            assert torch.equal(slots[parity][1].cpu(), hosts[last_wave * 2 + group][1])


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize(
    "old_start,new_start",
    [(2, 11), (13, 1), (7, 7)],
    ids=["forward", "backward", "unchanged"],
)
def test_cb_plan_rerope_matches_target_position_recompute(
    dtype, is_neox, old_start, new_start
) -> None:
    device = torch.device("npu:0")
    nl, capacity, nh, hs = 2, 5, 2, 32
    generator = torch.Generator().manual_seed(354)
    raw_key = torch.randn((nl, capacity, nh, hs), generator=generator)
    cos_sin_cache, inv_freq = _rope_cache(hs, 32, dtype, device)
    stored_key = _rope_reference(
        raw_key, torch.arange(old_start, old_start + capacity), inv_freq, is_neox
    ).to(dtype)
    expected_key = _rope_reference(
        raw_key, torch.arange(new_start, new_start + capacity), inv_freq, is_neox
    ).to(dtype)
    original_v = torch.randn(raw_key.shape, generator=generator).to(dtype)
    temp = (
        torch.stack((stored_key, original_v))
        .reshape(2, nl, capacity, nh * hs)
        .to(device)
    )
    # Supply the usual descriptor; this step executes RoPE only.
    _, layers = _kv_layers(
        device=device,
        nl=nl,
        nb=2,
        bs=4,
        nh=nh,
        hs=hs,
        block_stride_elems=2 * 4 * nh * hs,
        fill=-7,
        dtype=dtype,
    )
    pointers = _pointer_table(layers, FMT16, device)
    mapping = torch.arange(capacity, dtype=torch.int64, device=device)
    spec = _group_spec(
        pointer_table=pointers,
        temp=temp,
        slot_mapping=mapping,
        nl=nl,
        slot_tokens=capacity,
        hidden=nh * hs,
        page_buffer_size=8,
        block_size=4,
        head_size=hs,
        block_stride_elems=2 * 4 * nh * hs,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        engine_format=FMT16,
    )
    lmc_ops.execute_cb_retrieve_plan_flat(
        device,
        4096,
        [spec],
        _empty_table(),
        np.array([[0, 0, old_start, new_start]], dtype=np.int64),
        _empty_table(),
        np.array([[0, 1, 0]], dtype=np.int64),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        temp[0].cpu().reshape_as(expected_key),
        expected_key,
        **_reference_tolerance(dtype),
    )
    assert torch.equal(temp[1].cpu().reshape_as(original_v), original_v)


@requires_npu
def test_cb_plan_releases_gil_during_native_execution() -> None:
    """Another Python thread must run before the native call returns.

    A long switch interval prevents Python scheduling before/after the call
    from giving a false positive. The observer blocks again immediately, so
    it does not delay the native thread's reacquisition of the GIL.
    """
    device = torch.device("npu:0")
    lmc_ops.execute_cb_retrieve_plan_flat(
        device,
        4096,
        [],
        _empty_table(),
        _empty_table(),
        _empty_table(),
        np.empty((0, 3), dtype=np.int64),
    )
    ready, start, progressed, stop = (threading.Event() for _ in range(4))

    def observer():
        ready.set()
        start.wait()
        progressed.set()
        stop.wait()

    thread = threading.Thread(target=observer, daemon=True)
    thread.start()
    assert ready.wait(timeout=5), "observer thread did not start"
    steps = np.zeros((200000, 3), dtype=np.int64)
    empty = _empty_table()
    previous_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(1000)
        start.set()
        lmc_ops.execute_cb_retrieve_plan_flat(
            device,
            4096,
            [],
            empty,
            empty,
            empty,
            steps,
        )
        observed_during_call = progressed.is_set()
    finally:
        stop.set()
        sys.setswitchinterval(previous_interval)
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert observed_during_call, "native executor held the GIL for its entire call"


@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.xfail(
    run=False, strict=True, reason="RFC #354 P4: FP8 ramp not implemented"
)
def test_cb_plan_fp8_ramp_deferred(is_neox) -> None:
    """Explicit deferred coverage, not an FP8 numerical test or a passing UT."""
