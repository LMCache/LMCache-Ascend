# SPDX-License-Identifier: Apache-2.0
"""Direct correctness tests for the block-level MP-mode AscendC kernel.

Bypasses the MP server. One engine builder; layouts/hosts/directions are
parametrizations of a few contracts, not separate tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import torch

import lmcache.lmcache_native as native
import lmcache_ascend.c_ops as lmc_ops
from lmcache_ascend.v1.shape_desc import (
    attach_tuple_block_strides,
    attach_tuple_planes,
)

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU required",
)

KG0_FMT = native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS
NH_CS_FMT = native.EngineKVFormat.NL_X_NB_BS_NH_CS
SEP_KV_FMT = native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS
D2H = native.TransferDirection.D2H
H2D = native.TransferDirection.H2D


def _planes(item: Any) -> tuple[torch.Tensor, ...]:
    return item if isinstance(item, tuple) else (item,)


def _shape_desc(
    *,
    kv_size: int,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    dtype: torch.dtype,
    block_stride_elems: int = 0,
    plane_slot_bytes: tuple[int, ...] | None = None,
    plane_block_stride_bytes: tuple[int, ...] | None = None,
) -> object:
    desc = lmc_ops.PageBufferShapeDesc()
    desc.kv_size = kv_size
    desc.nl = nl
    desc.nb = nb
    desc.bs = bs
    desc.nh = nh
    desc.hs = hs
    desc.element_size = torch.empty((), dtype=dtype).element_size()
    desc.block_stride_elems = block_stride_elems
    desc.dtype = dtype
    if plane_slot_bytes is not None:
        attach_tuple_planes(desc, plane_slot_bytes)
    if plane_block_stride_bytes is not None:
        attach_tuple_block_strides(desc, plane_block_stride_bytes)
    return desc


def _pointer_table(layers: Sequence[Any], device: torch.device) -> torch.Tensor:
    ptrs = [int(t.data_ptr()) for item in layers for t in _planes(item)]
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def _zero_engine(layers: Sequence[Any]) -> None:
    for tensor in (t for item in layers for t in _planes(item)):
        tensor.zero_()


def _clone_engine(layers: Sequence[Any]) -> list[Any]:
    return [
        tuple(t.clone() for t in item) if isinstance(item, tuple) else item.clone()
        for item in layers
    ]


def _engine_nb(layers: Sequence[Any]) -> int:
    return int(_planes(layers[0])[0].shape[0])


def _transfer(
    paged: object,
    obj_ptrs: list[int],
    block_ids: torch.Tensor,
    device: torch.device,
    direction: object,
    desc: object,
    chunk: int,
    fmt: object,
    skip: int = 0,
) -> None:
    lmc_ops.multi_layer_block_kv_transfer(
        paged, obj_ptrs, block_ids, device, direction, desc, chunk, fmt, skip
    )
    torch.npu.synchronize()


def _run(fn: Callable[[], None], *, affinity: bool) -> None:
    if not affinity:
        fn()
        return
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    pool = AffinityThreadPool(max_workers=1, thread_name_prefix="lmcache")
    try:
        pool.submit(fn, affinity_key=0).result(timeout=120)
    finally:
        pool.shutdown(wait=True)


@contextmanager
def _host_object(
    kind: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Iterator[tuple[list[int], torch.Tensor | None]]:
    nbytes = int(torch.empty(shape, dtype=dtype).nbytes)
    if kind == "npu":
        obj = torch.zeros(shape, dtype=dtype, device=device)
        yield [int(obj.data_ptr())], obj
        return
    if kind in ("cpu", "affinity"):
        obj = torch.zeros(shape, dtype=dtype)
        yield [int(obj.data_ptr())], obj
        return
    if kind == "pinned":
        ptr = int(lmc_ops.alloc_pinned_ptr(nbytes, 0))
        try:
            yield [ptr], None
        finally:
            lmc_ops.free_pinned_ptr(ptr)
        return
    raise ValueError(kind)


@contextmanager
def _host_objects(
    kind: str,
    shapes: Sequence[tuple[int, ...]],
    dtype: torch.dtype,
    device: torch.device,
) -> Iterator[tuple[list[int], list[torch.Tensor | None]]]:
    with ExitStack() as stack:
        allocated = [
            stack.enter_context(_host_object(kind, shape, dtype, device))
            for shape in shapes
        ]
        ptrs: list[int] = []
        tensors: list[torch.Tensor | None] = []
        for part, tensor in allocated:
            ptrs.extend(part)
            tensors.append(tensor)
        yield ptrs, tensors


def _transferred_block_ids(ids: list[int], num_objects: int, skip: int) -> set[int]:
    bpo = len(ids) // num_objects
    return {
        ids[obj_i * bpo + local]
        for obj_i in range(num_objects)
        for local in range(skip, bpo)
    }


def _assert_close_or_equal(got: torch.Tensor, exp: torch.Tensor) -> None:
    if exp.dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(got, exp)
    else:
        assert torch.equal(got.cpu(), exp.cpu())


def _assert_block_zero(layers: Sequence[Any], bid: int) -> None:
    for tensor in (t for item in layers for t in _planes(item)):
        assert torch.count_nonzero(tensor[bid]) == 0


def _assert_block_equal(left: Sequence[Any], right: Sequence[Any], bid: int) -> None:
    for a, b in zip(left, right, strict=True):
        for ta, tb in zip(_planes(a), _planes(b), strict=True):
            _assert_close_or_equal(ta[bid], tb[bid])


def _assert_engine_equal(left: Sequence[Any], right: Sequence[Any]) -> None:
    for bid in range(_engine_nb(left)):
        _assert_block_equal(left, right, bid)


def _packed_block(
    latent: torch.Tensor, scale: torch.Tensor, bid: int, bs: int
) -> torch.Tensor:
    return torch.cat(
        [
            latent[bid].reshape(bs, 128),
            scale[bid].contiguous().view(torch.uint8).reshape(bs, 2),
        ],
        dim=-1,
    )


def _expected_host_block(layers: Sequence[Any], bid: int, bs: int, kv_leading: bool) -> Any:
    first = layers[0]
    if kv_leading:
        return [(key[bid], value[bid]) for key, value in layers]
    if isinstance(first, tuple):
        return [_packed_block(lat, scale, bid, bs) for lat, scale in layers]
    return [tensor[bid] for tensor in layers]


def _assert_d2h_object(
    objects: Sequence[torch.Tensor],
    layers: Sequence[Any],
    ids: list[int],
    *,
    skip: int,
    bs: int,
    num_objects: int,
    kv_leading: bool,
) -> None:
    bpo = len(ids) // num_objects
    for obj_i, obj in enumerate(objects):
        for local in range(bpo):
            sl = slice(local * bs, (local + 1) * bs)
            bid = ids[obj_i * bpo + local]
            if local < skip:
                prefix = obj[:, :, sl] if kv_leading else obj[:, sl]
                assert torch.count_nonzero(prefix) == 0
                continue
            expected = _expected_host_block(layers, bid, bs, kv_leading)
            if kv_leading:
                for layer, (key, value) in enumerate(expected):
                    torch.testing.assert_close(obj[0, layer, sl].reshape_as(key), key)
                    torch.testing.assert_close(obj[1, layer, sl].reshape_as(value), value)
            else:
                for layer, exp in enumerate(expected):
                    got = obj[layer, sl].cpu().reshape_as(exp.cpu())
                    _assert_close_or_equal(got, exp.cpu())


def _assert_h2d_engine(
    layers: Sequence[Any],
    golden: Sequence[Any],
    ids: list[int],
    *,
    skip: int,
    num_objects: int,
) -> None:
    transferred = _transferred_block_ids(ids, num_objects, skip)
    for bid in range(_engine_nb(layers)):
        if bid in transferred:
            _assert_block_equal(layers, golden, bid)
        else:
            _assert_block_zero(layers, bid)


def _fill_arange(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, offset: float) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32, device=device)
    return (values.reshape(shape) + offset).to(dtype)


def _kg0_planes(
    *,
    nl: int,
    nb: int,
    bs: int,
    device: torch.device,
    independent: bool,
) -> tuple[list[Any], list[torch.Tensor] | None, int, tuple[int, int]]:
    latent_w = 128
    if independent:
        k_row, v_row = bs * latent_w + 64, bs * 2 + 32
        pools = None
        layers: list[Any] = []
        for layer_i in range(nl):
            k_pool = torch.zeros(nb, k_row, dtype=torch.uint8, device=device)
            v_pool = torch.zeros(nb, v_row, dtype=torch.uint8, device=device)
            latent = k_pool[:, : bs * latent_w].view(nb, bs, latent_w)
            scale = v_pool[:, : bs * 2].view(torch.float16).view(nb, bs, 1)
            latent.copy_(
                (torch.arange(nb * bs * latent_w, device=device, dtype=torch.int32) % 251)
                .to(torch.uint8)
                .view(nb, bs, latent_w)
                + layer_i
            )
            scale.copy_(
                torch.arange(nb * bs, device=device, dtype=torch.float16).view(nb, bs, 1)
                + layer_i
            )
            layers.append((latent, scale))
        return layers, pools, 0, (k_row, v_row)
    stride = bs * 130
    pools = []
    layers = []
    for layer_i in range(nl):
        pool = torch.zeros(nb, stride, dtype=torch.uint8, device=device)
        latent = pool[:, : bs * latent_w].view(nb, bs, latent_w)
        scale = pool[:, bs * latent_w :].view(torch.float16).view(nb, bs, 1)
        latent.copy_(
            (torch.arange(nb * bs * latent_w, device=device, dtype=torch.int32) % 251)
            .to(torch.uint8)
            .view(nb, bs, latent_w)
            + layer_i
        )
        scale.copy_(
            torch.arange(nb * bs, device=device, dtype=torch.float16).view(nb, bs, 1)
            + layer_i
        )
        pools.append(pool)
        layers.append((latent, scale))
    return layers, pools, stride, (stride, stride)


def _build(
    layout: str,
    device: torch.device,
    *,
    padded: bool = False,
    nl: int | None = None,
    nb: int | None = None,
    bs: int | None = None,
    with_planes: bool | None = None,
) -> dict[str, Any]:
    """Build engine layers + PageBufferShapeDesc for one named layout."""
    if layout == "sep_kv":
        nl, nb, bs, nh, hs = nl or 2, nb or 8, bs or 4, 2, 8
        hidden = nh * hs
        layers = [
            (
                _fill_arange((nb, bs + int(padded), nh, hs), torch.float16, device, 10_000 * i)[
                    :, :bs
                ],
                _fill_arange(
                    (nb, bs + int(padded), nh, hs), torch.float16, device, 10_000 * i + 5_000
                )[:, :bs],
            )
            for i in range(nl)
        ]
        stride = (bs + int(padded)) * hidden
        return dict(
            fmt=SEP_KV_FMT,
            desc=_shape_desc(
                kv_size=2,
                nl=nl,
                nb=nb,
                bs=bs,
                nh=nh,
                hs=hs,
                dtype=torch.float16,
                block_stride_elems=stride if padded else 0,
            ),
            layers=layers,
            pools=None,
            table=_pointer_table(layers, device),
            obj_dtype=torch.float16,
            obj_width=hidden,
            kv_leading=True,
            packed=False,
            bs=bs,
            nl=nl,
            pad_width=None,
        )
    if layout in ("nh_cs", "nh_cs_fmt17", "nh_cs_padded"):
        padded_layout = layout == "nh_cs_padded" or padded
        nl = nl or 2
        if padded_layout:
            nb, bs, hs, stride = nb or 4, bs or 2, 2048, 8192
            dtype = torch.float32
            pools, layers = [], []
            for i in range(nl):
                pool = torch.full((nb, stride), 999.0, dtype=dtype, device=device)
                view = pool[:, : bs * hs].view(nb, bs, 1, hs)
                view.copy_(_fill_arange((nb, bs, 1, hs), dtype, device, 1000 * i))
                pools.append(pool)
                layers.append(view)
            desc = _shape_desc(
                kv_size=1, nl=nl, nb=nb, bs=bs, nh=1, hs=hs, dtype=dtype,
                block_stride_elems=stride,
            )
            return dict(
                fmt=NH_CS_FMT,
                desc=desc,
                layers=layers,
                pools=pools,
                table=_pointer_table(layers, device),
                obj_dtype=dtype,
                obj_width=hs,
                kv_leading=False,
                packed=False,
                bs=bs,
                nl=nl,
                pad_width=bs * hs,
            )
        nb, bs, hs = nb or 32, bs or 32, 512
        dtype = torch.bfloat16
        layers = [
            _fill_arange((nb, bs, 1, hs), dtype, device, 1000 * i) for i in range(nl)
        ]
        use_planes = True if with_planes is None else with_planes
        desc = _shape_desc(
            kv_size=1,
            nl=nl,
            nb=nb,
            bs=bs,
            nh=1,
            hs=hs,
            dtype=dtype,
            plane_slot_bytes=(hs * 2,) if use_planes else None,
            plane_block_stride_bytes=(bs * hs * 2,) if use_planes else None,
        )
        return dict(
            fmt=KG0_FMT if layout == "nh_cs_fmt17" else NH_CS_FMT,
            desc=desc,
            layers=layers,
            pools=None,
            table=_pointer_table(layers, device),
            obj_dtype=dtype,
            obj_width=hs,
            kv_leading=False,
            packed=False,
            bs=bs,
            nl=nl,
            pad_width=None,
        )
    # Packed MLA (fmt 17): kg0 / kg0_bs16 / kg0_bs64 / kg0_indep
    nl = nl or 2
    bs = bs or {"kg0_bs16": 16, "kg0_bs64": 64}.get(layout, 32)
    nb = nb or 32
    independent = layout == "kg0_indep"
    layers, pools, block_stride, plane_strides = _kg0_planes(
        nl=nl, nb=nb, bs=bs, device=device, independent=independent
    )
    return dict(
        fmt=KG0_FMT,
        desc=_shape_desc(
            kv_size=1,
            nl=nl,
            nb=nb,
            bs=bs,
            nh=1,
            hs=130,
            dtype=torch.int8,
            block_stride_elems=block_stride,
            plane_slot_bytes=(128, 2),
            plane_block_stride_bytes=plane_strides,
        ),
        layers=layers,
        pools=pools,
        table=_pointer_table(layers, device),
        obj_dtype=torch.uint8,
        obj_width=130,
        kv_leading=False,
        packed=True,
        bs=bs,
        nl=nl,
        pad_width=None,
    )


def _obj_shape(spec: dict[str, Any], chunk: int) -> tuple[int, ...]:
    if spec["kv_leading"]:
        return (2, spec["nl"], chunk, spec["obj_width"])
    return (spec["nl"], chunk, spec["obj_width"])


def _fill_src_object(spec: dict[str, Any], device: torch.device) -> torch.Tensor:
    nl, chunk, width = spec["nl"], spec["bs"], spec["obj_width"]
    if spec["packed"]:
        obj = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
        obj[:, :, :128] = (
            torch.arange(nl * chunk * 128, dtype=torch.int32, device=device)
            .to(torch.uint8)
            .view(nl, chunk, 128)
        )
        obj[:, :, 128:] = (
            torch.arange(nl * chunk, device=device, dtype=torch.float16)
            .view(nl, chunk, 1)
            .view(torch.uint8)
        )
        return obj
    return _fill_arange((nl, chunk, width), spec["obj_dtype"], device, 0)


def _launch(spec: dict[str, Any], paged: object, ptrs: list[int], ids: torch.Tensor, device: torch.device, direction: object, chunk: int, skip: int = 0) -> None:
    _transfer(paged, ptrs, ids, device, direction, spec["desc"], chunk, spec["fmt"], skip)


# ---------------------------------------------------------------------------
# Contract: D2H then H2D restores unskipped selected blocks only.
# ---------------------------------------------------------------------------

_HOSTS = ("npu", "cpu", "pinned", "affinity")


def _roundtrip_cases() -> list[Any]:
    rows: list[tuple] = [
        ("sep_kv", "npu", False, 1, 2, 2, [1, 3, 4, 6], "sep-kv-tight-skip"),
        ("sep_kv", "npu", True, 1, 2, 2, [1, 3, 4, 6], "sep-kv-padded-skip"),
        ("kg0", "npu", False, 1, 2, 1, [1, 3], "kg0-npu-skip"),
        ("kg0", "npu", False, 0, 21, 2, list(range(64)), "kg0-npu-2chunk"),
        ("nh_cs", "npu", False, 0, 21, 1, list(range(32)), "nh_cs-npu-1chunk"),
        ("nh_cs_fmt17", "npu", False, 0, 2, 1, [0], "nh_cs-fmt17"),
        ("kg0_indep", "npu", False, 0, 2, 1, [0], "kg0-indep-mini"),
        ("kg0_bs16", "npu", False, 0, 2, 1, [0], "kg0-bs16-chunk16"),
        ("kg0_bs64", "npu", False, 0, 2, 1, [0, 1], "kg0-bs64-chunk128"),
    ]
    for host in _HOSTS:
        rows.append(("kg0", host, False, 0, 2, 1, [0], f"kg0-{host}-mini"))
        rows.append(("kg0", host, False, 0, 21, 1, list(range(32)), f"kg0-{host}-live"))
    return [pytest.param(*row[:-1], id=row[-1]) for row in rows]


@requires_npu
@pytest.mark.parametrize("layout,host,padded,skip,nl,num_objects,block_ids", _roundtrip_cases())
def test_roundtrip_restores_unskipped_selected_blocks(
    layout: str,
    host: str,
    padded: bool,
    skip: int,
    nl: int,
    num_objects: int,
    block_ids: list[int],
) -> None:
    device = torch.device("npu:0")
    nb = None if layout == "sep_kv" else max(32, max(block_ids) + 1)
    spec = _build(layout, device, padded=padded, nl=nl, nb=nb)
    bs, bpo = spec["bs"], len(block_ids) // num_objects
    chunk = bpo * bs
    ids = torch.tensor(block_ids, dtype=torch.int64, device=device)
    golden = _clone_engine(spec["layers"])
    with _host_objects(host, [_obj_shape(spec, chunk)] * num_objects, spec["obj_dtype"], device) as (
        ptrs,
        obj_tensors,
    ):
        _run(lambda: _launch(spec, spec["table"], ptrs, ids, device, D2H, chunk, skip), affinity=host == "affinity")
        visible = [t for t in obj_tensors if t is not None]
        if len(visible) == num_objects:
            _assert_d2h_object(
                visible, spec["layers"], block_ids, skip=skip, bs=bs,
                num_objects=num_objects, kv_leading=spec["kv_leading"],
            )
        _zero_engine(spec["layers"])
        _run(lambda: _launch(spec, spec["table"], ptrs, ids, device, H2D, chunk, skip), affinity=host == "affinity")
    _assert_h2d_engine(spec["layers"], golden, block_ids, skip=skip, num_objects=num_objects)


# ---------------------------------------------------------------------------
# Contract: native kernel matches torch_ops; nested list ≡ pointer table.
# ---------------------------------------------------------------------------

_SMALL = {
    "kg0": dict(nl=2, nb=4, bs=32),
    "nh_cs": dict(nl=2, nb=4, bs=4, with_planes=False),
    "nh_cs_padded": dict(nl=2, nb=4, bs=2),
}


@requires_npu
@pytest.mark.parametrize("layout", ["kg0", "nh_cs", "nh_cs_padded"])
@pytest.mark.parametrize("direction_d2h", [True, False], ids=["d2h", "h2d"])
@pytest.mark.parametrize("nested", [False, True], ids=["ptrs", "nested"])
def test_native_matches_reference(layout: str, direction_d2h: bool, nested: bool) -> None:
    device = torch.device("npu:0")
    a = _build(layout, device, **_SMALL[layout])
    b = _build(layout, device, **_SMALL[layout])
    chunk, ids = a["bs"], torch.tensor([0 if layout == "kg0" else 1], dtype=torch.int64, device=device)
    direction = D2H if direction_d2h else H2D
    if direction_d2h:
        obj_a = torch.zeros(_obj_shape(a, chunk), dtype=a["obj_dtype"], device=device)
        obj_b = obj_a.clone()
    else:
        src = _fill_src_object(a, device)
        obj_a, obj_b = src.clone(), src.clone()
        _zero_engine(a["layers"])
        _zero_engine(b["layers"])

    paged_a = a["layers"] if nested else a["table"]
    _launch(a, paged_a, [int(obj_a.data_ptr())], ids, device, direction, chunk)

    if nested:
        _launch(b, b["table"], [int(obj_b.data_ptr())], ids, device, direction, chunk)
    else:
        torch_ops = pytest.importorskip("lmcache.v1.platform.torch_ops")
        torch_ops.multi_layer_block_kv_transfer(
            b["layers"], [obj_b], ids, device, direction, b["desc"], chunk, b["fmt"], 0,
        )
        torch.npu.synchronize()

    if direction_d2h:
        assert torch.equal(obj_a.cpu(), obj_b.cpu())
        if a["pad_width"] is not None:
            assert not torch.any(obj_a.cpu() == 999.0)
    else:
        _assert_engine_equal(a["layers"], b["layers"])
    if a["pools"] is not None and a["pad_width"] is not None:
        for pool in a["pools"]:
            assert torch.all(pool[:, a["pad_width"] :] == 999.0)


@requires_npu
def test_torch_check_raises_python_exception() -> None:
    device = torch.device("npu:0")
    spec = _build("kg0", device, nl=2, nb=4, bs=32)
    obj = torch.zeros((2, 32, 130), dtype=torch.uint8, device=device)
    with pytest.raises(RuntimeError, match="paged_buffer_ptrs_tensor"):
        lmc_ops.multi_layer_block_kv_transfer(
            torch.tensor([obj.data_ptr()], dtype=torch.int64, device=device),
            [obj.data_ptr()],
            torch.tensor([0], dtype=torch.int64, device=device),
            device, D2H, spec["desc"], 32, KG0_FMT, 0,
        )


@requires_npu
@pytest.mark.parametrize("direction_d2h", [True, False], ids=["d2h", "h2d"])
def test_object_group_plan_matches_direct_launches(direction_d2h: bool) -> None:
    device = torch.device("npu:0")
    kw = dict(nl=2, nb=4, bs=32)
    pairs = [
        (_build("kg0", device, **kw), _build("kg0", device, **kw)),
        (_build("nh_cs", device, with_planes=False, **kw), _build("nh_cs", device, with_planes=False, **kw)),
    ]
    direction = D2H if direction_d2h else H2D
    chunk = 32
    ids = torch.tensor([0], dtype=torch.int64, device=device)
    objs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for direct, planned in pairs:
        if direction_d2h:
            d = torch.zeros(_obj_shape(direct, chunk), dtype=direct["obj_dtype"], device=device)
            p = d.clone()
        else:
            src = _fill_src_object(direct, device)
            d, p = src.clone(), src.clone()
            _zero_engine(direct["layers"])
            _zero_engine(planned["layers"])
        objs.append((d, p))
        _launch(direct, direct["table"], [int(d.data_ptr())], ids, device, direction, chunk)

    specs = [
        lmc_ops.KernelGroupSpec(
            planned["table"].data_ptr(), [p.data_ptr()], planned["desc"], chunk,
            int(planned["fmt"]), ids.data_ptr(), ids.numel(),
        )
        for (_, planned), (_, p) in zip(pairs, objs)
    ]
    step = lmc_ops.BatchStep([], [lmc_ops.LaunchVar(i, 0, 1, 1, 0) for i in range(len(pairs))])
    lmc_ops.execute_object_group_transfer(int(direction), device, 1 << 26, specs, [step])
    torch.npu.synchronize()
    for (direct, planned), (d, p) in zip(pairs, objs):
        if direction_d2h:
            assert torch.equal(d.cpu(), p.cpu())
        else:
            _assert_engine_equal(direct["layers"], planned["layers"])


@requires_npu
@pytest.mark.parametrize("affinity", [False, True], ids=["main", "affinity"])
@pytest.mark.parametrize("num_objects", [1, 2], ids=["1obj", "2obj"])
def test_plan_staging_d2h_matches_direct(affinity: bool, num_objects: int) -> None:
    device = torch.device("npu:0")
    spec = _build("kg0", device, nl=2, nb=max(4, num_objects), bs=32)
    chunk = 32
    ids = torch.arange(num_objects, dtype=torch.int64, device=device)
    goldens = [
        torch.zeros(_obj_shape(spec, chunk), dtype=torch.uint8, device=device)
        for _ in range(num_objects)
    ]
    temp = goldens[0].clone()
    hosts = [torch.zeros(_obj_shape(spec, chunk), dtype=torch.uint8) for _ in range(num_objects)]
    _launch(spec, spec["table"], [int(g.data_ptr()) for g in goldens], ids, device, D2H, chunk)
    kspec = lmc_ops.KernelGroupSpec(
        spec["table"].data_ptr(), [int(temp.data_ptr())], spec["desc"], chunk,
        int(KG0_FMT), ids.data_ptr(), ids.numel(),
    )
    steps = [
        lmc_ops.BatchStep(
            [lmc_ops.StagingCopy(int(host.data_ptr()), int(temp.data_ptr()), host.nbytes, 0)],
            [lmc_ops.LaunchVar(0, obj_i, 1, 1, 0)],
        )
        for obj_i, host in enumerate(hosts)
    ]

    def _store() -> None:
        lmc_ops.execute_object_group_transfer(int(D2H), device, 1 << 26, [kspec], steps)
        torch.npu.synchronize()

    _run(_store, affinity=affinity)
    for host, golden in zip(hosts, goldens, strict=True):
        assert torch.equal(host, golden.cpu())


@requires_npu
def test_lmcache_memcpy_async_host_roundtrip() -> None:
    device = torch.device("npu:0")
    nbytes = 64 * 1024
    host = torch.arange(nbytes, dtype=torch.uint8)
    dev = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    lmc_ops.lmcache_memcpy_async(
        int(dev.data_ptr()), int(host.data_ptr()), nbytes, lmc_ops.TransferDirection.H2D, 0, 4096,
    )
    torch.npu.synchronize()
    assert torch.equal(dev.cpu(), host)
    host_back = torch.zeros(nbytes, dtype=torch.uint8)
    lmc_ops.lmcache_memcpy_async(
        int(host_back.data_ptr()), int(dev.data_ptr()), nbytes, lmc_ops.TransferDirection.D2H, 0, 4096,
    )
    torch.npu.synchronize()
    assert torch.equal(host_back, host)
