# SPDX-License-Identifier: Apache-2.0
# Standard
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import asyncio
import json

# Third Party
import pytest

# First Party
from tests.v1.disagg_proxy_test_utils import (
    FakeRequest,
    FakeResponse,
    collect_streaming_response,
    load_proxy_server,
)

proxy = load_proxy_server()


def _client_info(name: str, *, init_port: list[int] | None = None):
    return proxy.ClientInfo(
        client=name,
        host="127.0.0.1",
        init_port=init_port or [7100],
        alloc_port=[7200],
        name=name,
        base_url=f"http://{name}",
    )


def _prefiller_state():
    return proxy.PrefillerState(
        client_info=_client_info("prefiller"),
        name="prefiller",
        host="127.0.0.1",
        port=8000,
    )


def _decoder_state():
    return proxy.DecoderState(
        client_info=_client_info("decoder", init_port=[7100]),
        name="decoder",
        host="127.0.0.1",
        port=8100,
        init_port=[7100],
        alloc_port=[7200],
    )


def _prefill_response():
    return {
        "id": "cmpl-prefill",
        "object": "text_completion",
        "created": 1,
        "model": "MiniMax-M2.7",
        "choices": [
            {
                "index": 0,
                "text": "A",
                "logprobs": None,
                "finish_reason": "length",
                "stop_reason": None,
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        },
        "kv_transfer_params": {"first_tok": 30},
    }


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_completion_endpoint_handles_prefill_only_and_decode_paths(
    monkeypatch,
    max_tokens,
):
    async def scenario():
        tokenization_client = _client_info("tokenizer")
        prefiller = _prefiller_state()
        decoder = _decoder_state()
        service_calls = []
        decode_requests = []

        async def send_request(client, endpoint, req_data):
            service_calls.append((client, endpoint, deepcopy(req_data)))
            if endpoint == "/tokenize":
                return FakeResponse({"tokens": [10, 20]})
            assert client == "prefiller"
            assert endpoint == "/v1/completions"
            return FakeResponse(_prefill_response())

        async def stream_response(client, endpoint, req_data):
            assert client == "decoder"
            assert endpoint == "/v1/completions"
            decode_requests.append(deepcopy(req_data))
            yield b"data: [DONE]\n\n"

        release_prefiller = AsyncMock(return_value={})
        release_decoder = AsyncMock(return_value={})
        select_decoder = AsyncMock(return_value=(decoder, {}))
        monkeypatch.setattr(proxy, "counter", 0)
        monkeypatch.setattr(proxy, "stats_calculator", SimpleNamespace(add=Mock()))
        monkeypatch.setattr(
            proxy,
            "pick_up_tokenization_client",
            lambda _request: tokenization_client,
        )
        monkeypatch.setattr(proxy, "send_request_to_service", send_request)
        monkeypatch.setattr(proxy, "stream_service_response", stream_response)
        monkeypatch.setattr(
            proxy,
            "select_prefiller",
            AsyncMock(return_value=(prefiller, {})),
        )
        monkeypatch.setattr(proxy, "release_prefiller", release_prefiller)
        monkeypatch.setattr(proxy, "select_decoder", select_decoder)
        monkeypatch.setattr(proxy, "release_decoder", release_decoder)
        monkeypatch.setattr(
            proxy,
            "acquire_pd_buffer_slots",
            AsyncMock(return_value=(0, 0.0, False)),
        )
        monkeypatch.setattr(proxy, "wait_decode_kv_ready", AsyncMock())
        monkeypatch.setattr(proxy, "log_route_event", Mock())

        request = FakeRequest(
            {
                "model": "MiniMax-M2.7",
                "prompt": "hello",
                "max_tokens": max_tokens,
                "stream_options": {"include_usage": True},
            }
        )
        response = await proxy.handle_completions(request)
        body = await collect_streaming_response(response)

        assert service_calls[0] == (
            "tokenizer",
            "/tokenize",
            {"prompt": "hello"},
        )
        release_prefiller.assert_awaited_once()

        if max_tokens == 1:
            select_decoder.assert_not_awaited()
            release_decoder.assert_not_awaited()
            assert not decode_requests
            assert b'"text":"A"' in body
            assert b'"finish_reason":"length"' in body
            assert b'"completion_tokens":1' in body
        else:
            select_decoder.assert_awaited_once_with(2)
            release_decoder.assert_awaited_once()
            assert decode_requests == [
                {
                    "model": "MiniMax-M2.7",
                    "prompt": [10, 20, 30],
                    "max_tokens": 3,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ]
            assert b'"text":"A"' in body

        assert body.endswith(b"data: [DONE]\n\n")

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [True, False])
def test_chat_endpoint_preserves_native_stream_and_nonstream_responses(
    monkeypatch,
    stream,
):
    async def scenario():
        render_client = _client_info("renderer")
        prefiller = _prefiller_state()
        decoder = _decoder_state()
        prefill_requests = []
        decode_requests = []
        tool_call_chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ],
        }
        nonstream_body = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [{"message": {"tool_calls": [{"id": "call-1"}]}}],
            }
        ).encode()

        async def send_request(client, endpoint, req_data):
            if endpoint == "/v1/chat/completions/render":
                assert client == "renderer"
                return FakeResponse(
                    {
                        "token_ids": [10, 20],
                        "sampling_params": {"max_tokens": 32},
                    }
                )
            if client == "prefiller":
                assert endpoint == "/v1/completions"
                prefill_requests.append(deepcopy(req_data))
                return FakeResponse({"choices": [{"text": "discarded"}]})
            assert client == "decoder"
            assert endpoint == "/v1/chat/completions"
            decode_requests.append(deepcopy(req_data))
            return FakeResponse(content=nonstream_body)

        async def stream_response(client, endpoint, req_data):
            assert client == "decoder"
            assert endpoint == "/v1/chat/completions"
            decode_requests.append(deepcopy(req_data))
            yield proxy.encode_sse_data(tool_call_chunk)
            yield b"data: [DONE]\n\n"

        release_decoder = AsyncMock(return_value={})
        monkeypatch.setattr(proxy, "counter", 0)
        monkeypatch.setattr(proxy, "stats_calculator", SimpleNamespace(add=Mock()))
        monkeypatch.setattr(proxy.app.state, "prefill_clients", [render_client])
        monkeypatch.setattr(proxy, "send_request_to_service", send_request)
        monkeypatch.setattr(proxy, "stream_service_response", stream_response)
        monkeypatch.setattr(
            proxy,
            "select_prefiller",
            AsyncMock(return_value=(prefiller, {})),
        )
        monkeypatch.setattr(proxy, "release_prefiller", AsyncMock(return_value={}))
        monkeypatch.setattr(
            proxy,
            "select_decoder",
            AsyncMock(return_value=(decoder, {})),
        )
        monkeypatch.setattr(proxy, "release_decoder", release_decoder)
        monkeypatch.setattr(
            proxy,
            "acquire_pd_buffer_slots",
            AsyncMock(return_value=(0, 0.0, False)),
        )
        monkeypatch.setattr(proxy, "wait_decode_kv_ready", AsyncMock())
        monkeypatch.setattr(proxy, "log_route_event", Mock())

        request_data = {
            "model": "MiniMax-M2.7",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": stream,
        }
        response = await proxy.handle_chat_completions(FakeRequest(request_data))

        if stream:
            release_decoder.assert_not_awaited()
            body = await collect_streaming_response(response)
            assert b'"tool_calls"' in body
            assert body.endswith(b"data: [DONE]\n\n")
        else:
            assert response.body == nonstream_body
            assert response.status_code == 200

        assert len(prefill_requests) == 1
        assert prefill_requests[0]["prompt"] == [10, 20]
        assert prefill_requests[0]["max_tokens"] == 1
        assert prefill_requests[0]["stream"] is False
        assert "disagg_spec" in prefill_requests[0]["kv_transfer_params"]
        assert decode_requests == [request_data]
        assert "kv_transfer_params" not in decode_requests[0]
        release_decoder.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_stage", ["prefill", "cancelled_prefill", "stream"])
def test_request_failure_and_cancellation_release_resources_once(
    monkeypatch,
    failure_stage,
):
    async def scenario():
        tokenization_client = _client_info("tokenizer")
        prefiller = _prefiller_state()
        decoder = _decoder_state()

        async def send_request(client, endpoint, req_data):
            if endpoint == "/tokenize":
                return FakeResponse({"tokens": [10, 20]})
            if failure_stage == "prefill":
                raise RuntimeError("prefill failed")
            if failure_stage == "cancelled_prefill":
                raise asyncio.CancelledError
            return FakeResponse(_prefill_response())

        async def cancelled_stream(_client, _endpoint, _req_data):
            raise asyncio.CancelledError
            yield b""  # pragma: no cover

        release_prefiller = AsyncMock(return_value={})
        release_decoder = AsyncMock(return_value={})
        release_slots = AsyncMock()
        monkeypatch.setattr(proxy, "counter", 0)
        monkeypatch.setattr(proxy, "stats_calculator", SimpleNamespace(add=Mock()))
        monkeypatch.setattr(
            proxy,
            "pick_up_tokenization_client",
            lambda _request: tokenization_client,
        )
        monkeypatch.setattr(proxy, "send_request_to_service", send_request)
        monkeypatch.setattr(proxy, "stream_service_response", cancelled_stream)
        monkeypatch.setattr(
            proxy,
            "select_prefiller",
            AsyncMock(return_value=(prefiller, {})),
        )
        monkeypatch.setattr(proxy, "release_prefiller", release_prefiller)
        monkeypatch.setattr(
            proxy,
            "select_decoder",
            AsyncMock(return_value=(decoder, {})),
        )
        monkeypatch.setattr(proxy, "release_decoder", release_decoder)
        monkeypatch.setattr(
            proxy,
            "acquire_pd_buffer_slots",
            AsyncMock(return_value=(2, 0.0, True)),
        )
        monkeypatch.setattr(proxy, "release_pd_buffer_slots", release_slots)
        monkeypatch.setattr(proxy, "wait_decode_kv_ready", AsyncMock())
        monkeypatch.setattr(proxy, "log_route_event", Mock())

        request = FakeRequest(
            {
                "model": "MiniMax-M2.7",
                "prompt": "hello",
                "max_tokens": 4,
            }
        )

        if failure_stage == "prefill":
            with pytest.raises(RuntimeError, match="prefill failed"):
                await proxy.handle_completions(request)
        elif failure_stage == "cancelled_prefill":
            with pytest.raises(asyncio.CancelledError):
                await proxy.handle_completions(request)
        else:
            response = await proxy.handle_completions(request)
            with pytest.raises(asyncio.CancelledError):
                await collect_streaming_response(response)

        release_prefiller.assert_awaited_once()
        release_decoder.assert_awaited_once()
        release_slots.assert_awaited_once_with(decoder, 2)
        assert release_prefiller.await_args.kwargs["success"] is (
            failure_stage == "stream"
        )
        assert release_decoder.await_args.kwargs["success"] is False

    asyncio.run(scenario())


@pytest.mark.parametrize("exit_error", [False, True])
def test_lifespan_closes_idle_zmq_socket_and_can_restart(monkeypatch, exit_error):
    async def scenario():
        sockets = []
        clients = []
        receiving = asyncio.Event()

        async def recv():
            receiving.set()
            await asyncio.Event().wait()

        def make_socket(_socket_type):
            socket = SimpleNamespace(bind=Mock(), recv=recv, close=Mock())
            sockets.append(socket)
            return socket

        def make_client(**_kwargs):
            client = SimpleNamespace(aclose=AsyncMock())
            clients.append(client)
            return client

        monkeypatch.setattr(proxy.app, "state", SimpleNamespace())
        monkeypatch.setattr(proxy, "run_proxy", True)
        monkeypatch.setattr(proxy, "zmq_ctx", SimpleNamespace(socket=make_socket))
        monkeypatch.setattr(proxy.httpx, "AsyncClient", make_client)
        monkeypatch.setattr(
            proxy,
            "global_args",
            SimpleNamespace(
                prefiller_host=["localhost"],
                prefiller_port=[8000],
                num_prefillers=1,
                decoder_host=["localhost"],
                decoder_port=[8100],
                num_decoders=1,
                decoder_init_port=[7100],
                decoder_alloc_port=[7200],
                pd_transfer_mode="delay_pull",
                pd_buffer_size=1024,
                proxy_host="localhost",
                proxy_port=9999,
            ),
            raising=False,
        )

        async def enter_and_exit():
            async with proxy.lifespan(proxy.app):
                await receiving.wait()
                if exit_error:
                    raise RuntimeError("lifespan body failed")

        for _ in range(2):
            receiving.clear()
            if exit_error:
                with pytest.raises(RuntimeError, match="lifespan body failed"):
                    await asyncio.wait_for(enter_and_exit(), timeout=1)
            else:
                await asyncio.wait_for(enter_and_exit(), timeout=1)
            assert proxy.app.state.zmq_task.done()
            assert not proxy.run_proxy
            sockets[-1].close.assert_called_once()

        assert len(sockets) == 2
        assert len(clients) == 4
        for client in clients:
            client.aclose.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("transfer_mode", "stream", "cancel_stage"),
    [
        ("eager_pull", False, "kv_ready"),
        ("eager_pull", False, "decode"),
        ("eager_pull", True, "prefill"),
        ("delay_pull", False, "kv_ready"),
    ],
)
def test_chat_cancellation_releases_accounting_and_permits(
    monkeypatch, transfer_mode, stream, cancel_stage
):
    async def scenario():
        prefiller = _prefiller_state()
        decoder = _decoder_state()
        decoder.pd_transfer_mode = transfer_mode
        if transfer_mode == "eager_pull":
            decoder.pd_buffer_semaphore = proxy.WeightedSemaphore(2)
        reached = asyncio.Event()

        async def block_until_cancelled():
            reached.set()
            await asyncio.Event().wait()

        async def send_request(_client, endpoint, _data):
            if endpoint.endswith("/render"):
                return FakeResponse(
                    {"token_ids": [10, 20], "sampling_params": {"max_tokens": 32}}
                )
            if endpoint == "/v1/completions":
                if cancel_stage == "prefill":
                    await block_until_cancelled()
                return FakeResponse({"choices": [{"text": "discarded"}]})
            assert endpoint == "/v1/chat/completions"
            await block_until_cancelled()

        async def wait_ready(*_args):
            if cancel_stage == "kv_ready":
                await block_until_cancelled()

        release_prefiller = AsyncMock(wraps=proxy.release_prefiller)
        release_decoder = AsyncMock(wraps=proxy.release_decoder)
        release_slots = AsyncMock(wraps=proxy.release_pd_buffer_slots)
        monkeypatch.setattr(
            proxy.app,
            "state",
            SimpleNamespace(
                prefill_clients=[prefiller.client_info],
                prefiller_states=[prefiller],
                decoder_states=[decoder],
                prefiller_lock=asyncio.Lock(),
                decoder_lock=asyncio.Lock(),
                prefiller_select_seq=0,
                decoder_select_seq=0,
            ),
        )
        monkeypatch.setattr(
            proxy, "global_args", SimpleNamespace(chunk_size=1), raising=False
        )
        monkeypatch.setattr(proxy, "stats_calculator", SimpleNamespace(add=Mock()))
        monkeypatch.setattr(proxy, "send_request_to_service", send_request)
        monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait_ready)
        monkeypatch.setattr(proxy, "release_prefiller", release_prefiller)
        monkeypatch.setattr(proxy, "release_decoder", release_decoder)
        monkeypatch.setattr(proxy, "release_pd_buffer_slots", release_slots)
        monkeypatch.setattr(proxy, "log_route_event", Mock())
        request = FakeRequest(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            }
        )
        task = asyncio.create_task(proxy.handle_chat_completions(request))
        try:
            await asyncio.wait_for(reached.wait(), timeout=1)
            assert decoder.active_decode_requests == 1
            assert decoder.active_decode_tokens == 2
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert decoder.active_decode_requests == decoder.active_decode_tokens == 0
        assert prefiller.active_prefill_requests == prefiller.active_prefill_tokens == 0
        release_decoder.assert_awaited_once()
        assert release_decoder.await_args.kwargs["success"] is False
        release_prefiller.assert_awaited_once()
        assert release_prefiller.await_args.kwargs["success"] is (
            cancel_stage != "prefill"
        )
        if transfer_mode == "eager_pull":
            release_slots.assert_awaited_once_with(decoder, 2)
            assert decoder.pd_buffer_semaphore.available == 2
        else:
            release_slots.assert_not_awaited()

    asyncio.run(scenario())
