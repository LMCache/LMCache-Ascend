# SPDX-License-Identifier: Apache-2.0
"""E2E: LMCacheDriven store/retrieve across processes on Ascend NPU.

The worker process owns paged MLA-tuple KV planes, exports them through the
plane-aggregating NpuIPCWrapper (one wrapper per layer) plus interprocess
events; the parent drives the real server-side LMCacheDrivenTransferModule
(fake storage bus, real cache context, real transfers, real events).
"""

# Standard
from contextlib import closing
import math
import multiprocessing as mp
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch
import torch_npu  # noqa: F401

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
import lmcache_ascend  # noqa: F401, E402  (applies plugin patches)

# First Party
import lmcache.lmcache_native as lmcache_native  # noqa: E402
from lmcache.utils import EngineType  # noqa: E402
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402
from lmcache.v1.gpu_connector.utils import LayoutHints  # noqa: E402
from lmcache.v1.multiprocess.custom_types import KVCache  # noqa: E402
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer  # noqa: E402
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (  # noqa: E402
    LMCacheDrivenTransferContext,
    create_transfer_context,
)
from lmcache.v1.platform.npu.ipc_wrapper import NpuIPCWrapper  # noqa: E402
from tests.v1.multiprocess.test_custom_types import (  # noqa: E402
    get_customized_decoder,
    get_customized_encoder,
)

NL = 4
NB = 16
BS = 16
CHUNK = 256  # BS * NB
CHUNK_KEY = ObjectKey(chunk_hash=b"e2e0", model_name="mla-e2e", kv_rank=0)
W_LATENT = 128
W_ROPE = 16
HIDDEN = W_LATENT + W_ROPE

requires_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU hardware is required"
)


def _worker(device_index: int, conn) -> None:
    torch.npu.set_device(device_index)
    device = f"npu:{device_index}"
    wrappers: list[NpuIPCWrapper] = []
    planes: list[torch.Tensor] = []
    for layer in range(NL):
        latent = torch.zeros(NB, BS, 1, W_LATENT, device=device)
        rope = torch.zeros(NB, BS, 1, W_ROPE, device=device)
        for block in range(NB):
            latent[block].fill_(float(layer * 1000 + block) % 251.0)
            rope[block].fill_(float((layer * 7 + block) % 13.0))
        planes.extend([latent, rope])
        # One plane-aggregating wrapper per layer, exactly what
        # wrap_kv_caches produces for vLLM-Ascend's tuple registration.
        wrappers.append(NpuIPCWrapper.wrap((latent, rope)))
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for plane in planes:
            plane.mul_(1.0)  # materialize writes on the worker stream
    # The parent imports these handles from another process; this CANN
    # build rejects same-process ``from_ipc_handle`` re-imports, so the
    # worker exports one producer event per consumer call.
    producer_a = torch.npu.Event(enable_timing=False, interprocess=True)
    producer_a.record(stream)
    producer_b = torch.npu.Event(enable_timing=False, interprocess=True)
    producer_b.record(stream)
    encoder = get_customized_encoder(type=list[NpuIPCWrapper])
    conn.send(
        {
            "wrappers": encoder.encode(wrappers),
            "producer_a": producer_a.ipc_handle(),
            "producer_b": producer_b.ipc_handle(),
            "device_index": device_index,
        }
    )
    conn.recv()  # hold the mappings until the parent finishes


class _FakeMemoryObj:
    """MemoryObj stand-in hitting the plain-tensor memcpy branch."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.raw_tensor = tensor
        # Object-group staging reads ``meta.address`` as the host-pool offset.
        # A dedicated tensor is the whole allocation, so the offset is 0.
        self.meta = SimpleNamespace(address=0)
        self.metadata = self.meta

    def get_size(self) -> int:
        return self.raw_tensor.numel() * self.raw_tensor.element_size()

    def parent(self) -> None:
        return None

    @property
    def data_ptr(self) -> int:
        return self.raw_tensor.data_ptr()


class _FakeReadWindow:
    def __init__(self, objects: list[_FakeMemoryObj]) -> None:
        self._objects = objects

    def __enter__(self) -> list[_FakeMemoryObj]:
        return self._objects

    def __exit__(self, *args: object) -> None:
        pass


class _FakeStorageManager:
    def __init__(self) -> None:
        self.objects: dict[Any, _FakeMemoryObj] = {}

    def _bytes(self, layout_desc: Any) -> int:
        return sum(
            math.prod(shape) * dtype.itemsize
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )

    def reserve_write(
        self, keys: list[Any], layout_desc: Any, mode: str
    ) -> dict[Any, _FakeMemoryObj]:
        reserved = {}
        for key in keys:
            tensor = torch.empty(
                self._bytes(layout_desc), dtype=torch.uint8, device="cpu"
            )
            obj = _FakeMemoryObj(tensor)
            reserved[key] = obj
            self.objects[key] = obj
        return reserved

    def finish_write(self, keys: list[Any], read_locks: int = 0) -> None:
        pass

    def finish_read_prefetched(self, keys: list[Any], read_locks: int = 0) -> None:
        pass

    def read_prefetched_results(self, keys: list[Any]) -> "_FakeReadWindow":
        return _FakeReadWindow([self.objects[k] for k in keys])


def _expected_staging() -> torch.Tensor:
    """Rank-3 [L, NB*BS, W] expectation: latent then rope plane per layer."""
    expected = torch.empty(NL, NB * BS, HIDDEN, dtype=torch.float32)
    for layer in range(NL):
        for block in range(NB):
            expected[layer, block * BS : (block + 1) * BS, :W_LATENT] = (
                float(layer * 1000 + block) % 251.0
            )
            expected[layer, block * BS : (block + 1) * BS, W_LATENT:] = float(
                (layer * 7 + block) % 13.0
            )
    return expected


@requires_npu
def test_forced_mode_selects_lmcache_driven_context() -> None:
    """Explicit mode routes NPU tensors to the handle-transfer context."""
    tensors = {
        f"layer-{i}": torch.zeros(NB, BS, 1, W_LATENT, device="npu:0")
        for i in range(NL)
    }
    tensors.update(
        {
            f"layer-rope-{i}": torch.zeros(NB, BS, 1, W_ROPE, device="npu:0")
            for i in range(NL)
        }
    )
    context = create_transfer_context(tensors, mode="lmcache_driven")
    assert isinstance(context, LMCacheDrivenTransferContext)
    context.close()


@requires_npu
@pytest.mark.parametrize("device_index", [0, 1])
def test_lmcache_driven_store_and_retrieve_roundtrip(
    device_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if device_index >= torch.npu.device_count():
        pytest.skip("single-device box")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=_worker, args=(device_index, child_conn))
    process.start()
    try:
        message = parent_conn.recv()
        decoder = get_customized_decoder(type=list[NpuIPCWrapper])
        # One wrapper per layer; to_tensor() yields the (latent, rope)
        # plane tuple, so discovery classifies the MLA tuple layout from
        # the in-band structure alone -- no regroup hint involved.
        kv_caches: KVCache = list(decoder.decode(message["wrappers"]))

        from lmcache.v1.platform.npu import NpuDeviceSpec
        from lmcache.v1.platform.npu.event_ipc import NpuEventIPCBackend

        cache_context = NpuDeviceSpec().create_cache_context(
            kv_caches,
            CHUNK,
            layout_hints=LayoutHints(kv_layout="NHD"),
            engine_group_infos=(),
            engine_type=EngineType.VLLM,
        )
        with closing(cache_context):
            assert (
                cache_context.get_engine_kv_format(0)
                == lmcache_native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS
            )
            event_backend = NpuEventIPCBackend()
            storage_manager = _FakeStorageManager()

            module = lmcache_driven_transfer.LMCacheDrivenTransferModule(
                SimpleNamespace(
                    chunk_size=CHUNK,
                    storage_manager=storage_manager,
                    event_bus=SimpleNamespace(
                        publish=lambda event: None,
                        publish_on_stream=lambda stream, event: None,
                        has_subscribers=lambda event_type: False,
                    ),
                    resolve_obj_keys=lambda key, group_ids: [[CHUNK_KEY]],
                )
            )
            with closing(module):
                entry = lmcache_driven_transfer.ContextEntry(
                    cache_context=cache_context,
                    model_name="mla-e2e",
                    world_size=1,
                    event_backend=event_backend,
                )
                monkeypatch.setattr(
                    module, "get_and_touch_context_entry", lambda iid: entry
                )
                key = SimpleNamespace(
                    request_id="e2e",
                    cache_salt="",
                    worker_id=0,
                    token_ids=list(range(CHUNK)),
                    start=0,
                    end=CHUNK,
                )

                block_ids = [list(range(NB))]
                # The producer events were exported by the worker process; the
                # module imports them cross-process (the supported path on CANN).
                store_handle, stored = module.store(
                    key, 1, block_ids, message["producer_a"]
                )
                assert stored is True
                assert isinstance(store_handle, bytes) and len(store_handle) > 0

                # The module returns after the completion submit; drain the
                # transfer stream before reading the staged host bytes.
                cache_context.stream.synchronize()

                # Stored bytes equal the worker's plane contents in [L, tokens, W]
                # staging order (latent plane then rope plane per layer).
                stored_obj = storage_manager.objects[CHUNK_KEY]
                host = stored_obj.raw_tensor.view(torch.float32)
                assert torch.allclose(
                    host.view(NL, NB * BS, HIDDEN), _expected_staging()
                )

                # Retrieve: mutate host bytes, scatter back into the same blocks.
                stored_obj.raw_tensor.view(torch.float32).add_(1.0)
                retrieve_handle, retrieved = module.retrieve(
                    key, 1, block_ids, message["producer_b"]
                )
                assert retrieved is True
                assert isinstance(retrieve_handle, bytes) and len(retrieve_handle) > 0
                cache_context.stream.synchronize()

                # Every imported plane element now carries the +1 mutation. The
                # context's own views are the tensors the scatter wrote into.
                expected = _expected_staging()
                kv_tensors = cache_context.kv_tensors
                for layer in range(NL):
                    latent = kv_tensors[layer][0]
                    rope = kv_tensors[layer][1]
                    for block in range(NB):
                        exp_latent = (
                            expected[layer, block * BS : (block + 1) * BS, :W_LATENT]
                            + 1.0
                        )
                        exp_rope = (
                            expected[layer, block * BS : (block + 1) * BS, W_LATENT:]
                            + 1.0
                        )
                        assert torch.allclose(
                            latent[block, :, 0, :].cpu(), exp_latent
                        )
                        assert torch.allclose(rope[block, :, 0, :].cpu(), exp_rope)
    finally:
        parent_conn.send("done")
        process.join(timeout=60)
    # Producer-side CANN aborts (SIGABRT) when imported NPU IPC storages
    # are released after the worker; the roundtrip above is the contract.
    if process.exitcode not in (0, -6):
        pytest.fail(f"worker exitcode {process.exitcode}")
