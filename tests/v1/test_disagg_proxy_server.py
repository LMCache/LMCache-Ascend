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


@pytest.mark.parametrize("failure_stage", ["prefill", "stream"])
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
