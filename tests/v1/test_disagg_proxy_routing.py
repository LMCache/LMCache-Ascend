# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
import asyncio

# Third Party
import pytest

# First Party
from tests.v1.disagg_proxy_test_utils import load_proxy_server

proxy = load_proxy_server()


def _client_info(name: str):
    return proxy.ClientInfo(
        client=object(),
        host="127.0.0.1",
        init_port=[7100],
        alloc_port=[7200],
        name=name,
        base_url=f"http://{name}",
    )


def test_weighted_semaphore_blocks_until_slots_are_released():
    async def scenario():
        semaphore = proxy.WeightedSemaphore(3)
        await semaphore.acquire(2)

        waiter = asyncio.create_task(semaphore.acquire(2))
        await asyncio.sleep(0)

        assert semaphore.available == 1
        assert not waiter.done()

        await semaphore.release(2)
        await asyncio.wait_for(waiter, timeout=1)

        assert semaphore.available == 1
        await semaphore.release(2)
        assert semaphore.available == semaphore.capacity == 3

    asyncio.run(scenario())


def test_select_and_release_prefiller_updates_load_and_metrics():
    async def scenario():
        busier = proxy.PrefillerState(
            client_info=_client_info("prefiller-0"),
            name="prefiller-0",
            host="127.0.0.1",
            port=8000,
            active_prefill_requests=1,
        )
        selected_state = proxy.PrefillerState(
            client_info=_client_info("prefiller-1"),
            name="prefiller-1",
            host="127.0.0.1",
            port=8001,
            active_prefill_tokens=10,
        )
        proxy.app.state.prefiller_states = [busier, selected_state]
        proxy.app.state.prefiller_lock = asyncio.Lock()
        proxy.app.state.prefiller_select_seq = 0

        selected, route_info = await proxy.select_prefiller(64)

        assert selected is selected_state
        assert selected.active_prefill_tokens == 74
        assert selected.active_prefill_requests == 1
        assert selected.total_prefill_tokens == 64
        assert selected.total_prefill_requests == 1
        assert route_info["selected_prefiller"] == "prefiller-1"
        assert len(route_info["candidate_prefiller_loads"]) == 2

        snapshot = await proxy.release_prefiller(
            selected,
            64,
            success=True,
            prefill_ms=20.0,
        )

        assert snapshot["active_prefill_tokens"] == 10
        assert snapshot["active_prefill_requests"] == 0
        assert snapshot["last_prefill_ms"] == 20.0
        assert snapshot["prefill_ms_ewma"] == 20.0
        assert snapshot["failed_prefill_requests"] == 0

    asyncio.run(scenario())


def test_select_and_release_decoder_updates_load_and_metrics():
    async def scenario():
        busier = proxy.DecoderState(
            client_info=_client_info("decoder-0"),
            name="decoder-0",
            host="127.0.0.1",
            port=8100,
            init_port=[7100],
            alloc_port=[7200],
            active_decode_tokens=100,
        )
        selected_state = proxy.DecoderState(
            client_info=_client_info("decoder-1"),
            name="decoder-1",
            host="127.0.0.1",
            port=8101,
            init_port=[7101],
            alloc_port=[7201],
            active_decode_tokens=10,
        )
        proxy.app.state.decoder_states = [busier, selected_state]
        proxy.app.state.decoder_lock = asyncio.Lock()
        proxy.app.state.decoder_select_seq = 0

        selected, route_info = await proxy.select_decoder(32)

        assert selected is selected_state
        assert selected.active_decode_tokens == 42
        assert selected.active_decode_requests == 1
        assert selected.total_decode_tokens == 32
        assert selected.total_decode_requests == 1
        assert route_info["selected_decoder"] == "decoder-1"
        assert len(route_info["candidate_decoder_loads"]) == 2

        snapshot = await proxy.release_decoder(
            selected,
            32,
            success=False,
            error="decode failed",
        )

        assert snapshot["active_decode_tokens"] == 10
        assert snapshot["active_decode_requests"] == 0
        assert snapshot["failed_decode_requests"] == 1
        assert snapshot["last_error"] == "decode failed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("transfer_mode", "admission_enabled"),
    [
        (proxy.PD_TRANSFER_MODE_PUSH, True),
        (proxy.PD_TRANSFER_MODE_EAGER_PULL, True),
        (proxy.PD_TRANSFER_MODE_DELAY_PULL, False),
    ],
)
def test_pd_buffer_admission_matches_transfer_mode(
    transfer_mode,
    admission_enabled,
):
    async def scenario():
        semaphore = proxy.WeightedSemaphore(4) if admission_enabled else None
        decoder = proxy.DecoderState(
            client_info=_client_info("decoder-0"),
            name="decoder-0",
            host="127.0.0.1",
            port=8100,
            init_port=[7100],
            alloc_port=[7200],
            pd_buffer_semaphore=semaphore,
            pd_transfer_mode=transfer_mode,
        )
        proxy.global_args = SimpleNamespace(chunk_size=16)

        slots, wait_ms, acquired = await proxy.acquire_pd_buffer_slots(decoder, 17)

        if admission_enabled:
            assert (slots, acquired) == (2, True)
            assert wait_ms >= 0
            assert semaphore.available == 2
            await proxy.release_pd_buffer_slots(decoder, slots)
            assert semaphore.available == semaphore.capacity == 4
        else:
            assert (slots, wait_ms, acquired) == (0, 0.0, False)

    asyncio.run(scenario())
