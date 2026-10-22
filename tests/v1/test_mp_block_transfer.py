# SPDX-License-Identifier: Apache-2.0
"""Direct correctness tests for the block-level MP-mode AscendC kernel.

Bypasses the MP server. One engine builder; layouts/hosts/directions are
parametrizations of a few contracts, not separate tests.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any, TypedDict

# Third Party
import lmcache.lmcache_native as native
import pytest
import torch

# First Party
from lmcache_ascend.v1.shape_desc import (
    attach_tuple_block_strides,
    attach_tuple_planes,
)
import lmcache_ascend.c_ops as lmc_ops

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU required",
)

KG0_FMT = native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS
NH_CS_FMT = native.EngineKVFormat.NL_X_NB_BS_NH_CS
SEP_KV_FMT = native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS
# One [NB, 2, BS, NH, HS] tensor per layer. fused_kv is semantic K then V.
# interleaved_attn_* reuses it for the two halves of an opaque page.
FUSED_KV_FMT = native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
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


def _factory_shape_desc(
    kv_caches: Sequence[Any],
    engine_kv_format: object,
    *,
    nl: int,
    nb: int,
    bs: int,
    block_stride_elems: int | None = None,
) -> object:
    """Build a descriptor through upstream's production factory."""
    # Third Party
    from lmcache.v1.gpu_connector.utils import make_page_buffer_shape_desc

    return make_page_buffer_shape_desc(
        kv_caches,
        engine_kv_format,
        layer_idx=0,
        num_layers_in_group=nl,
        num_blocks=nb,
        block_size=bs,
        block_stride_elems=block_stride_elems,
    )


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


def _assert_pointer_table_contract(spec: dict[str, Any]) -> None:
    """Pointer-table width and plane offsets expected per engine format."""
    table = spec["table"]
    desc = spec["desc"]
    nl = spec["nl"]
    fmt = spec["fmt"]
    if fmt == FUSED_KV_FMT:
        # Format 2 leaves plane offsets unset. prepare_group derives
        # (0, bs * nh * hs * element_size) from the scalar descriptor.
        assert table.numel() == nl
        assert int(desc.num_planes) == 0
        return
    if fmt == SEP_KV_FMT:
        assert table.numel() == nl * int(desc.num_planes)
        return
    if fmt == KG0_FMT and int(desc.num_planes) > 0:
        assert table.numel() == nl * int(desc.num_planes)


def _paged_ptr_tensor(paged: Any, device: torch.device) -> torch.Tensor:
    """Native host wants a 1-D int64 pointer table, not nested layer tensors."""
    if isinstance(paged, torch.Tensor):
        return paged
    ptrs: list[int] = []
    for layer in paged:
        if isinstance(layer, (tuple, list)):
            ptrs.extend(int(plane.data_ptr()) for plane in layer)
        else:
            ptrs.append(int(layer.data_ptr()))
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


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
        _paged_ptr_tensor(paged, device),
        obj_ptrs,
        block_ids,
        device,
        direction,
        desc,
        chunk,
        fmt,
        skip,
    )
    torch.npu.synchronize()


def _run(fn: Callable[[], None], *, affinity: bool) -> None:
    if not affinity:
        fn()
        return
    # Third Party
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


def _assert_block_equal(left: Sequence[Any], right: Sequence[Any], bid: int) -> None:
    for a, b in zip(left, right, strict=True):
        for ta, tb in zip(_planes(a), _planes(b), strict=True):
            _assert_close_or_equal(ta[bid], tb[bid])


def _assert_engine_equal(left: Sequence[Any], right: Sequence[Any]) -> None:
    for bid in range(_engine_nb(left)):
        _assert_block_equal(left, right, bid)


def _packed_block(planes: Sequence[torch.Tensor], bid: int, bs: int) -> torch.Tensor:
    cols = [
        plane[bid].contiguous().view(torch.uint8).reshape(bs, -1) for plane in planes
    ]
    return torch.cat(cols, dim=-1)


def _expected_host_block(
    layers: Sequence[Any], bid: int, bs: int, kv_leading: bool
) -> Any:
    first = layers[0]
    if kv_leading:
        return [(key[bid], value[bid]) for key, value in layers]
    if isinstance(first, tuple):
        return [_packed_block(planes, bid, bs) for planes in layers]
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
                    torch.testing.assert_close(
                        obj[1, layer, sl].reshape_as(value), value
                    )
            else:
                for layer, exp in enumerate(expected):
                    got = obj[layer, sl].cpu().reshape_as(exp.cpu())
                    _assert_close_or_equal(got, exp.cpu())


def _fill_arange(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    offset: float,
) -> torch.Tensor:
    values = torch.arange(
        int(torch.tensor(shape).prod()), dtype=torch.float32, device=device
    )
    return (values.reshape(shape) + offset).to(dtype)


# fmt17 production plane shapes: DS MLA (latent+rope), DSA 3-plane, and the
# DSv4 DSA-C8 4-plane mixed-dtype bundle.
_PACKED_PLANE_SPECS: dict[str, tuple[tuple[int, torch.dtype], ...]] = {
    "mla_bf16": ((512, torch.bfloat16), (64, torch.bfloat16)),
    "mla_bf16_bs128": ((512, torch.bfloat16), (64, torch.bfloat16)),
    "dsa3": (
        (512, torch.bfloat16),
        (64, torch.bfloat16),
        (128, torch.bfloat16),
    ),
    "dsa_c8_4": (
        (512, torch.bfloat16),
        (64, torch.bfloat16),
        (128, torch.int8),
        (1, torch.float16),
    ),
    # 2B single-plane row: the thinnest legal payload (a bare fp16 scale).
    # With bs=4096 the unclamped fit would be 128KB/32B = 4096 rows, one
    # past the DataCopyPad blockCount limit (4095).
    "thin2b_bs4096": ((1, torch.float16),),
}


def _plane_row_bytes(plane_specs: tuple[tuple[int, torch.dtype], ...]) -> int:
    return sum(w * torch.empty((), dtype=d).element_size() for w, d in plane_specs)


def _packed_planes_desc(
    *,
    nl: int,
    nb: int,
    bs: int,
    plane_specs: tuple[tuple[int, torch.dtype], ...],
) -> object:
    slot_bytes = tuple(
        w * torch.empty((), dtype=d).element_size() for w, d in plane_specs
    )
    element_size = max(torch.empty((), dtype=d).element_size() for _, d in plane_specs)
    return _shape_desc(
        kv_size=1,
        nl=nl,
        nb=nb,
        bs=bs,
        nh=1,
        hs=_plane_row_bytes(plane_specs) // element_size,
        dtype=torch.bfloat16,
        plane_slot_bytes=slot_bytes,
        plane_block_stride_bytes=tuple(bs * b for b in slot_bytes),
    )


def _multi_plane_layers(
    *,
    nl: int,
    nb: int,
    bs: int,
    plane_specs: tuple[tuple[int, torch.dtype], ...],
    device: torch.device,
) -> list[tuple[torch.Tensor, ...]]:
    layers: list[tuple[torch.Tensor, ...]] = []
    for layer_i in range(nl):
        planes: list[torch.Tensor] = []
        for plane_i, (width, dtype) in enumerate(plane_specs):
            salt = 1000 * layer_i + 97 * plane_i
            numel = nb * bs * width
            if dtype in (torch.int8, torch.uint8):
                values = (
                    torch.arange(numel, device=device, dtype=torch.int32) + salt
                ) % 251
            else:
                values = torch.arange(
                    numel, device=device, dtype=torch.float32
                ) + float(salt)
            planes.append(values.to(dtype).view(nb, bs, 1, width))
        layers.append(tuple(planes))
    return layers


def _packed_object_from_planes(
    *,
    nl: int,
    chunk: int,
    plane_specs: tuple[tuple[int, torch.dtype], ...],
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for layer_i in range(nl):
        columns: list[torch.Tensor] = []
        for plane_i, (width, dtype) in enumerate(plane_specs):
            salt = 1000 * layer_i + 97 * plane_i
            numel = chunk * width
            if dtype in (torch.int8, torch.uint8):
                values = (
                    torch.arange(numel, device=device, dtype=torch.int32) + salt
                ) % 251
            else:
                values = torch.arange(
                    numel, device=device, dtype=torch.float32
                ) + float(salt)
            typed = values.to(dtype).view(chunk, width)
            columns.append(typed.contiguous().view(torch.uint8).view(chunk, -1))
        rows.append(torch.cat(columns, dim=-1))
    return torch.stack(rows)


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
                (
                    torch.arange(nb * bs * latent_w, device=device, dtype=torch.int32)
                    % 251
                )
                .to(torch.uint8)
                .view(nb, bs, latent_w)
                + layer_i
            )
            scale.copy_(
                torch.arange(nb * bs, device=device, dtype=torch.float16).view(
                    nb, bs, 1
                )
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


def _fused_kv_page_spec(
    *,
    dtype: torch.dtype,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    layers: Sequence[Any],
    fused_layers: Sequence[torch.Tensor],
    block_stride_elems: int,
    obj_width: int,
    storages: Sequence[torch.Tensor] | None = None,
    prefix_elems: int = 0,
) -> dict[str, Any]:
    """Register one fused page per layer: one base pointer, planes via offsets.

    Plane arrays stay empty. ``prepare_group`` derives them from ``bs``,
    ``nh``, ``hs``, ``element_size``, and ``block_stride_elems``.
    """
    spec = dict(
        fmt=FUSED_KV_FMT,
        desc=_shape_desc(
            kv_size=2,
            nl=nl,
            nb=nb,
            bs=bs,
            nh=nh,
            hs=hs,
            dtype=dtype,
            block_stride_elems=block_stride_elems,
        ),
        layers=layers,
        fused_layers=fused_layers,
        pools=None,
        table=torch.tensor(
            [int(fused.data_ptr()) for fused in fused_layers],
            dtype=torch.int64,
            device=fused_layers[0].device,
        ),
        obj_dtype=dtype,
        obj_width=obj_width,
        kv_leading=True,
        packed=False,
        bs=bs,
        nl=nl,
        pad_width=None,
    )
    if prefix_elems:
        spec["storages"] = list(storages or ())
        spec["prefix_elems"] = prefix_elems
    _assert_pointer_table_contract(spec)
    return spec


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
    if layout == "fused_kv":
        # Semantic K then V in one [NB, 2, BS, NH, HS] tensor per layer.
        # Padded mode adds a storage prefix and a tail gap inside each page.
        nl, nb, bs, nh, hs = nl or 2, nb or 8, bs or 4, 2, 8
        hidden = nh * hs
        prefix = 32 if padded else 0
        pad_elems = 16 if padded else 0
        page_elems = 2 * bs * hidden + pad_elems
        dtype = torch.float16
        fused_layers, layers, storages = [], [], []
        for i in range(nl):
            storage = torch.full(
                (prefix + nb * page_elems,), -1, dtype=dtype, device=device
            )
            fused = storage.as_strided(
                (nb, 2, bs, nh, hs),
                (page_elems, bs * hidden, hidden, hs, 1),
                storage_offset=prefix,
            )
            fused[:, 0].copy_(_fill_arange((nb, bs, nh, hs), dtype, device, 10_000 * i))
            fused[:, 1].copy_(
                _fill_arange((nb, bs, nh, hs), dtype, device, 10_000 * i + 5_000)
            )
            fused_layers.append(fused)
            layers.append((fused[:, 0], fused[:, 1]))
            storages.append(storage)
        return _fused_kv_page_spec(
            dtype=dtype,
            nl=nl,
            nb=nb,
            bs=bs,
            nh=nh,
            hs=hs,
            layers=layers,
            fused_layers=fused_layers,
            block_stride_elems=page_elems if padded else 0,
            obj_width=hidden,
            storages=storages,
            prefix_elems=prefix,
        )
    if layout in ("interleaved_attn_tp4", "interleaved_attn_tp1"):
        # v0.30 opaque page: [NB, 2, 1024, 1, HS]. The two slabs are half-pages,
        # not K and V. HS is 256 at TP4 and 1024 at TP1.
        nl = nl or 2
        nb = nb or 4
        bs = bs or 1024
        hs = 256 if layout == "interleaved_attn_tp4" else 1024
        dtype = torch.bfloat16
        page_elems = 2 * bs * hs
        prefix = 64 if padded else 0
        fused_layers, layers, storages = [], [], []
        for i in range(nl):
            storage = torch.full(
                (prefix + nb * page_elems,), -1, dtype=dtype, device=device
            )
            pages = _fill_arange((nb, page_elems), dtype, device, 10_000 * i)
            storage[prefix:].copy_(pages.reshape(-1))
            fused = storage.as_strided(
                (nb, 2, bs, 1, hs),
                (page_elems, page_elems // 2, hs, hs, 1),
                storage_offset=prefix,
            )
            fused_layers.append(fused)
            layers.append((fused[:, 0], fused[:, 1]))
            storages.append(storage)
        spec = _fused_kv_page_spec(
            dtype=dtype,
            nl=nl,
            nb=nb,
            bs=bs,
            nh=1,
            hs=hs,
            layers=layers,
            fused_layers=fused_layers,
            block_stride_elems=0,
            obj_width=hs,
            storages=storages,
            prefix_elems=prefix,
        )
        return spec
    if layout == "sep_kv":
        nl, nb, bs, nh, hs = nl or 2, nb or 8, bs or 4, 2, 8
        hidden = nh * hs
        layers = [
            (
                _fill_arange(
                    (nb, bs + int(padded), nh, hs), torch.float16, device, 10_000 * i
                )[:, :bs],
                _fill_arange(
                    (nb, bs + int(padded), nh, hs),
                    torch.float16,
                    device,
                    10_000 * i + 5_000,
                )[:, :bs],
            )
            for i in range(nl)
        ]
        stride = (bs + int(padded)) * hidden
        slot = nh * hs * 2
        spec = dict(
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
                plane_slot_bytes=(slot, slot),
                plane_block_stride_bytes=(stride * 2, stride * 2),
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
        _assert_pointer_table_contract(spec)
        return spec
    if layout in _PACKED_PLANE_SPECS:
        plane_specs = _PACKED_PLANE_SPECS[layout]
        nl = nl or 2
        # bs=128 crosses the built-in 128KB floor (multi-segment split when
        # the UB query falls back); the 2B thin plane with bs=4096 always
        # splits via the 4095-row blockCount clamp.
        bs = {"mla_bf16_bs128": 128, "thin2b_bs4096": 4096}.get(layout, bs or 16)
        nb = nb or 32
        layers = _multi_plane_layers(
            nl=nl, nb=nb, bs=bs, plane_specs=plane_specs, device=device
        )
        row = _plane_row_bytes(plane_specs)
        spec = dict(
            fmt=KG0_FMT,
            desc=_packed_planes_desc(nl=nl, nb=nb, bs=bs, plane_specs=plane_specs),
            layers=layers,
            pools=None,
            table=_pointer_table(layers, device),
            obj_dtype=torch.uint8,
            obj_width=row,
            kv_leading=False,
            packed=True,
            packed_planes=plane_specs,
            bs=bs,
            nl=nl,
            pad_width=None,
        )
        _assert_pointer_table_contract(spec)
        return spec
    if layout in ("nh_cs_factory", "nh_cs_factory_padded"):
        padded_layout = layout == "nh_cs_factory_padded"
        nl, nb, bs, nh, hs = nl or 2, nb or 4, bs or 2, 8, 64
        dtype = torch.float32
        row_elems = nh * hs
        block_stride = bs * row_elems + (8 if padded_layout else 0)
        padded_pools: list[torch.Tensor] = []
        pools: list[torch.Tensor] | None = padded_pools if padded_layout else None
        factory_layers: list[torch.Tensor] = []
        for layer_idx in range(nl):
            if padded_layout:
                pool = torch.full((nb, block_stride), 999.0, dtype=dtype, device=device)
                layer = pool[:, : bs * row_elems].view(nb, bs, nh, hs)
                padded_pools.append(pool)
            else:
                layer = torch.empty((nb, bs, nh, hs), dtype=dtype, device=device)
            layer.copy_(_fill_arange((nb, bs, nh, hs), dtype, device, 1000 * layer_idx))
            factory_layers.append(layer)
        desc = _factory_shape_desc(
            factory_layers,
            NH_CS_FMT,
            nl=nl,
            nb=nb,
            bs=bs,
            block_stride_elems=block_stride if padded_layout else None,
        )
        return dict(
            fmt=NH_CS_FMT,
            desc=desc,
            layers=factory_layers,
            pools=pools,
            table=_pointer_table(factory_layers, device),
            obj_dtype=dtype,
            obj_width=row_elems,
            kv_leading=False,
            packed=False,
            bs=bs,
            nl=nl,
            pad_width=bs * row_elems if padded_layout else None,
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
                kv_size=1,
                nl=nl,
                nb=nb,
                bs=bs,
                nh=1,
                hs=hs,
                dtype=dtype,
                block_stride_elems=stride,
                plane_slot_bytes=(hs * 4,),
                plane_block_stride_bytes=(stride * 4,),
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
    layers, kg0_pools, block_stride, plane_strides = _kg0_planes(
        nl=nl, nb=nb, bs=bs, device=device, independent=independent
    )
    spec = dict(
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
        pools=kg0_pools,
        table=_pointer_table(layers, device),
        obj_dtype=torch.uint8,
        obj_width=130,
        kv_leading=False,
        packed=True,
        bs=bs,
        nl=nl,
        pad_width=None,
    )
    _assert_pointer_table_contract(spec)
    return spec


def _obj_shape(spec: dict[str, Any], chunk: int) -> tuple[int, ...]:
    if spec["kv_leading"]:
        return (2, spec["nl"], chunk, spec["obj_width"])
    return (spec["nl"], chunk, spec["obj_width"])


def _fill_src_object(spec: dict[str, Any], device: torch.device) -> torch.Tensor:
    nl, chunk, width = spec["nl"], spec["bs"], spec["obj_width"]
    if spec.get("packed_planes"):
        return _packed_object_from_planes(
            nl=nl, chunk=chunk, plane_specs=spec["packed_planes"], device=device
        )
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
    if spec["kv_leading"]:
        return _fill_arange((2, nl, chunk, width), spec["obj_dtype"], device, 0)
    return _fill_arange((nl, chunk, width), spec["obj_dtype"], device, 0)


def _launch(
    spec: dict[str, Any],
    paged: object,
    ptrs: list[int],
    ids: torch.Tensor,
    device: torch.device,
    direction: object,
    chunk: int,
    skip: int = 0,
) -> None:
    _transfer(
        paged, ptrs, ids, device, direction, spec["desc"], chunk, spec["fmt"], skip
    )


# ---------------------------------------------------------------------------
# Contract: D2H then H2D restores unskipped selected blocks only.
# ---------------------------------------------------------------------------

_HOSTS = ("npu", "cpu", "pinned", "affinity")


def _roundtrip_cases() -> list[Any]:
    rows: list[tuple] = [
        ("sep_kv", "npu", False, 1, 2, 2, [1, 3, 4, 6], "sep-kv-tight-skip"),
        ("sep_kv", "npu", True, 1, 2, 2, [1, 3, 4, 6], "sep-kv-padded-skip"),
        ("fused_kv", "npu", False, 1, 2, 2, [1, 3, 4, 6], "fused-kv-skip"),
        ("fused_kv", "npu", True, 1, 2, 2, [1, 3, 4, 6], "fused-kv-padded"),
        ("interleaved_attn_tp4", "npu", True, 0, 2, 1, [0, 2], "interleaved-attn-tp4"),
        ("interleaved_attn_tp1", "npu", True, 0, 2, 1, [0, 2], "interleaved-attn-tp1"),
        ("kg0", "npu", False, 1, 2, 1, [1, 3], "kg0-npu-skip"),
        ("kg0", "npu", False, 0, 21, 2, list(range(64)), "kg0-npu-2chunk"),
        # Format 13 is a single fused-K/V [NB, BS, NH, CS] plane.  Exercise
        # both dense blocks and a dim-0-padded view with non-monotonic engine
        # block IDs, so a tight-stride implementation cannot accidentally pass.
        ("nh_cs", "npu", False, 0, 2, 1, [3, 1], "fmt13-tight"),
        ("nh_cs_padded", "npu", False, 0, 2, 1, [3, 1], "fmt13-padded"),
        (
            "nh_cs_factory",
            "npu",
            False,
            0,
            2,
            1,
            [3, 1],
            "fmt13-factory-tight-nh8",
        ),
        (
            "nh_cs_factory_padded",
            "npu",
            False,
            0,
            2,
            1,
            [3, 1],
            "fmt13-factory-padded-nh8",
        ),
        ("nh_cs_fmt17", "npu", False, 0, 2, 1, [0], "nh_cs-fmt17"),
        ("kg0_indep", "npu", False, 0, 2, 1, [0], "kg0-indep-mini"),
        ("kg0_bs16", "npu", False, 0, 2, 1, [0], "kg0-bs16-chunk16"),
        ("kg0_bs64", "npu", False, 0, 2, 1, [0, 1], "kg0-bs64-chunk128"),
        ("mla_bf16", "npu", False, 0, 2, 1, [0, 1], "mla-bf16-bs16"),
        ("mla_bf16_bs128", "npu", False, 0, 2, 1, [0, 1], "mla-bf16-bs128-segsplit"),
        ("dsa3", "npu", False, 0, 2, 1, [0, 1], "dsa3-bs16"),
        ("dsa_c8_4", "npu", False, 0, 2, 1, [0, 1], "dsa-c8-4plane"),
        ("dsa_c8_4", "npu", False, 1, 2, 1, [0, 1], "dsa-c8-4plane-skip"),
        # skip == blocks_per_object: host must skip the launch (<<<0>>> is
        # invalid) and leave every byte untouched.
        ("kg0", "npu", False, 2, 2, 1, [1, 3], "kg0-npu-skip-all-blocks"),
        ("kg0", "cpu", False, 2, 2, 1, [1, 3], "kg0-cpu-skip-all-blocks"),
        # 2B row fit = 4096; bs=4096 must split 4095+1 instead of overflowing
        # DataCopyPad blockCount.
        ("thin2b_bs4096", "npu", False, 0, 2, 1, [0], "thin2b-bs4096-rowclamp"),
    ]
    for host in _HOSTS:
        rows.append(("kg0", host, False, 0, 2, 1, [0], f"kg0-{host}-mini"))
        rows.append(("kg0", host, False, 0, 21, 1, list(range(32)), f"kg0-{host}-live"))
    return [pytest.param(*row[:-1], id=row[-1]) for row in rows]


@requires_npu
@pytest.mark.parametrize(
    "layout,host,padded,skip,nl,num_objects,block_ids", _roundtrip_cases()
)
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
    nb = (
        None
        if layout
        in ("sep_kv", "fused_kv", "interleaved_attn_tp4", "interleaved_attn_tp1")
        else max(32, max(block_ids) + 1)
    )
    spec = _build(layout, device, padded=padded, nl=nl, nb=nb)
    bs, bpo = spec["bs"], len(block_ids) // num_objects
    chunk = bpo * bs
    ids = torch.tensor(block_ids, dtype=torch.int64, device=device)
    golden = _clone_engine(spec["layers"])
    prefix_elems = spec.get("prefix_elems") or 0
    storages = spec.get("storages")
    prefixes = (
        [storage[:prefix_elems].clone() for storage in storages]
        if storages and prefix_elems
        else None
    )
    with _host_objects(
        host, [_obj_shape(spec, chunk)] * num_objects, spec["obj_dtype"], device
    ) as (
        ptrs,
        obj_tensors,
    ):
        _run(
            lambda: _launch(spec, spec["table"], ptrs, ids, device, D2H, chunk, skip),
            affinity=host == "affinity",
        )
        visible = [t for t in obj_tensors if t is not None]
        if len(visible) == num_objects:
            _assert_d2h_object(
                visible,
                spec["layers"],
                block_ids,
                skip=skip,
                bs=bs,
                num_objects=num_objects,
                kv_leading=spec["kv_leading"],
            )
        for bid in _transferred_block_ids(block_ids, num_objects, skip):
            for tensor in (t for item in spec["layers"] for t in _planes(item)):
                tensor[bid].zero_()
        _run(
            lambda: _launch(spec, spec["table"], ptrs, ids, device, H2D, chunk, skip),
            affinity=host == "affinity",
        )
    _assert_engine_equal(spec["layers"], golden)
    if prefixes is not None:
        assert storages is not None
        for storage, prefix in zip(storages, prefixes, strict=True):
            assert torch.equal(storage[:prefix_elems], prefix)


# ---------------------------------------------------------------------------
# Contract: native kernel matches torch_ops; nested list ≡ pointer table.
# ---------------------------------------------------------------------------


class _SmallParams(TypedDict):
    nl: int
    nb: int
    bs: int


_SMALL: dict[str, _SmallParams] = {
    "kg0": {"nl": 2, "nb": 4, "bs": 32},
    "mla_bf16": {"nl": 2, "nb": 4, "bs": 16},
    "dsa3": {"nl": 2, "nb": 4, "bs": 16},
    "dsa_c8_4": {"nl": 2, "nb": 4, "bs": 16},
}


@requires_npu
@pytest.mark.parametrize("layout", ["kg0", "mla_bf16", "dsa3", "dsa_c8_4"])
@pytest.mark.parametrize("direction_d2h", [True, False], ids=["d2h", "h2d"])
@pytest.mark.parametrize("nested", [False, True], ids=["ptrs", "nested"])
def test_native_matches_reference(
    layout: str, direction_d2h: bool, nested: bool
) -> None:
    device = torch.device("npu:0")
    a = _build(layout, device, **_SMALL[layout])
    b = _build(layout, device, **_SMALL[layout])
    chunk, ids = a["bs"], torch.tensor([0], dtype=torch.int64, device=device)
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
            b["layers"],
            [obj_b],
            ids,
            device,
            direction,
            b["desc"],
            chunk,
            b["fmt"],
            0,
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
            device,
            D2H,
            spec["desc"],
            32,
            KG0_FMT,
            0,
        )


@requires_npu
def test_fused_kv_rejects_one_pointer_per_plane() -> None:
    """Fused pages take one base pointer per layer, not one per K and V."""
    device = torch.device("npu:0")
    spec = _build("fused_kv", device, nl=2, nb=4, bs=4)
    obj = torch.zeros(
        _obj_shape(spec, spec["bs"]), dtype=spec["obj_dtype"], device=device
    )
    two_per_layer = torch.tensor(
        [int(plane.data_ptr()) for pair in spec["layers"] for plane in pair],
        dtype=torch.int64,
        device=device,
    )
    with pytest.raises(RuntimeError, match="paged_buffer_ptrs_tensor"):
        lmc_ops.multi_layer_block_kv_transfer(
            two_per_layer,
            [int(obj.data_ptr())],
            torch.tensor([0], dtype=torch.int64, device=device),
            device,
            D2H,
            spec["desc"],
            spec["bs"],
            spec["fmt"],
            0,
        )


@requires_npu
def test_factory_plane_geometry_contract() -> None:
    device = torch.device("npu:0")
    fmt13 = _build("nh_cs_factory", device, nl=2, nb=4, bs=2)
    desc13 = fmt13["desc"]
    assert desc13.num_planes == 0
    assert desc13.nh == 8
    assert desc13.block_stride_elems == 0
    assert tuple(desc13.plane_slot_bytes) == ()
    assert tuple(desc13.plane_block_stride_bytes) == ()

    padded13 = _build("nh_cs_factory_padded", device, nl=2, nb=4, bs=2)
    padded_desc13 = padded13["desc"]
    assert padded_desc13.num_planes == 0
    assert padded_desc13.nh == 8
    assert padded_desc13.block_stride_elems == padded13["layers"][0].stride(0)

    for layout, expected_planes in (("sep_kv", 2), ("mla_bf16", 2)):
        spec = _build(layout, device, nl=2, nb=4)
        desc = _factory_shape_desc(
            spec["layers"],
            spec["fmt"],
            nl=spec["nl"],
            nb=4,
            bs=spec["bs"],
        )
        assert desc.num_planes == expected_planes
        assert len(desc.plane_slot_bytes) == expected_planes
        assert len(desc.plane_block_stride_bytes) == expected_planes


@requires_npu
def test_format13_rejects_multiple_planes() -> None:
    device = torch.device("npu:0")
    spec = _build("nh_cs_factory", device, nl=2, nb=4, bs=2)
    desc = spec["desc"]
    payload = int(desc.nh) * int(desc.hs) * int(desc.element_size)
    block_stride = int(desc.bs) * payload
    attach_tuple_planes(desc, (payload, payload))
    attach_tuple_block_strides(desc, (block_stride, block_stride))
    obj = torch.zeros(
        _obj_shape(spec, spec["bs"]), dtype=spec["obj_dtype"], device=device
    )
    ids = torch.tensor([0], dtype=torch.int64, device=device)

    with pytest.raises(RuntimeError, match="format 13 requires exactly 1"):
        _launch(
            spec,
            spec["table"],
            [int(obj.data_ptr())],
            ids,
            device,
            D2H,
            spec["bs"],
        )


@requires_npu
@pytest.mark.parametrize("layout", ["sep_kv", "nh_cs_fmt17"], ids=["fmt16", "fmt17"])
def test_tuple_formats_reject_unfilled_plane_geometry(layout: str) -> None:
    device = torch.device("npu:0")
    spec = _build(layout, device, nl=2, nb=4)
    spec["desc"].num_planes = 0
    obj = torch.zeros(
        _obj_shape(spec, spec["bs"]), dtype=spec["obj_dtype"], device=device
    )
    ids = torch.tensor([0], dtype=torch.int64, device=device)

    with pytest.raises(RuntimeError, match="num_planes must be in"):
        _launch(
            spec,
            spec["table"],
            [int(obj.data_ptr())],
            ids,
            device,
            D2H,
            spec["bs"],
        )


@requires_npu
@pytest.mark.parametrize("direction_d2h", [True, False], ids=["d2h", "h2d"])
def test_object_group_plan_matches_direct_launches(direction_d2h: bool) -> None:
    device = torch.device("npu:0")
    pairs = [
        (
            _build("kg0", device, nl=2, nb=4, bs=32),
            _build("kg0", device, nl=2, nb=4, bs=32),
        ),
        (
            _build("dsa_c8_4", device, nl=2, nb=4, bs=16),
            _build("dsa_c8_4", device, nl=2, nb=4, bs=16),
        ),
        (
            _build("fused_kv", device, padded=True, nl=2, nb=4, bs=4),
            _build("fused_kv", device, padded=True, nl=2, nb=4, bs=4),
        ),
    ]
    direction = D2H if direction_d2h else H2D
    ids = torch.tensor([0], dtype=torch.int64, device=device)
    objs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for direct, planned in pairs:
        chunk = direct["bs"]
        if direction_d2h:
            d = torch.zeros(
                _obj_shape(direct, chunk), dtype=direct["obj_dtype"], device=device
            )
            p = d.clone()
        else:
            src = _fill_src_object(direct, device)
            d, p = src.clone(), src.clone()
            _zero_engine(direct["layers"])
            _zero_engine(planned["layers"])
        objs.append((d, p))
        _launch(
            direct, direct["table"], [int(d.data_ptr())], ids, device, direction, chunk
        )

    specs = [
        lmc_ops.KernelGroupSpec(
            planned["table"].data_ptr(),
            [p.data_ptr()],
            planned["desc"],
            planned["bs"],
            int(planned["fmt"]),
            ids.data_ptr(),
            ids.numel(),
        )
        for (_, planned), (_, p) in zip(pairs, objs, strict=True)
    ]
    step = lmc_ops.BatchStep(
        [], [lmc_ops.LaunchVar(i, 0, 1, 1, 0) for i in range(len(pairs))]
    )
    lmc_ops.execute_object_group_transfer(
        int(direction), device, 1 << 26, specs, [step]
    )
    torch.npu.synchronize()
    for (direct, planned), (d, p) in zip(pairs, objs, strict=True):
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
    hosts = [
        torch.zeros(_obj_shape(spec, chunk), dtype=torch.uint8)
        for _ in range(num_objects)
    ]
    golden_ptrs = [int(g.data_ptr()) for g in goldens]
    _launch(spec, spec["table"], golden_ptrs, ids, device, D2H, chunk)
    kspec = lmc_ops.KernelGroupSpec(
        spec["table"].data_ptr(),
        [int(temp.data_ptr())],
        spec["desc"],
        chunk,
        int(KG0_FMT),
        ids.data_ptr(),
        ids.numel(),
    )
    steps = [
        lmc_ops.BatchStep(
            [
                lmc_ops.StagingCopy(
                    int(host.data_ptr()), int(temp.data_ptr()), host.nbytes, 0
                )
            ],
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
        int(dev.data_ptr()),
        int(host.data_ptr()),
        nbytes,
        lmc_ops.TransferDirection.H2D,
        0,
        4096,
    )
    torch.npu.synchronize()
    assert torch.equal(dev.cpu(), host)
    host_back = torch.zeros(nbytes, dtype=torch.uint8)
    lmc_ops.lmcache_memcpy_async(
        int(host_back.data_ptr()),
        int(dev.data_ptr()),
        nbytes,
        lmc_ops.TransferDirection.D2H,
        0,
        4096,
    )
    torch.npu.synchronize()
    assert torch.equal(host_back, host)
