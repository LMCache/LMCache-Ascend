# SPDX-License-Identifier: Apache-2.0
# Standard
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import asyncio
import json

# Third Party
import anyio
import httpx
import pytest

# First Party
from tests.v1.disagg_proxy_test_utils import (
    FakeByteStream,
    FakeRequest,
    FakeResponse,
    collect_streaming_response,
    load_proxy_server,
    mock_streaming_service,
)

proxy = load_proxy_server()
real_stream_service_response = proxy.stream_service_response


def test_completion_usage_and_logprobs_preserve_inputs():
    # Third Party
    from disagg_proxy_response import adjust_completion_usage, merge_completion

    assert adjust_completion_usage(None) is None
    assert adjust_completion_usage({}) == {}
    prefill = {
        "id": "p",
        "created": 1,
        "model": "m",
        "choices": [{"text": "A", "logprobs": {"tokens": ["A"], "text_offset": [0]}}],
    }
    decoded = {
        "choices": [
            {
                "text": "B",
                "finish_reason": "stop",
                "logprobs": {"tokens": ["B"], "text_offset": [0]},
            }
        ],
        "usage": None,
    }
    original = deepcopy(decoded)
    merged = merge_completion(prefill, decoded)
    assert merged["choices"][0]["logprobs"] == {
        "tokens": ["A", "B"],
        "text_offset": [0, 1],
    }
    assert merged["choices"][0]["finish_reason"] == "stop"
    assert merged["usage"] is None
    assert decoded == original


@pytest.fixture
def backend(monkeypatch):
    prefill = {
        "id": "cmpl-p",
        "created": 1,
        "model": "model",
        "object": "text_completion",
        "choices": [
            {"index": 0, "text": "A", "finish_reason": "length", "logprobs": None}
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "kv_transfer_params": {"first_tok": 30},
    }
    decoded = {
        "id": "cmpl-d",
        "created": 2,
        "model": "model",
        "object": "text_completion",
        "choices": [
            {"index": 0, "text": "\u4e2dB", "finish_reason": "stop", "stop_reason": 99}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    p = SimpleNamespace(name="p", client_info=SimpleNamespace(client="p"))
    d = SimpleNamespace(
        name="d",
        client_info=SimpleNamespace(
            client="d", host="localhost", init_port=[7100], alloc_port=[7200]
        ),
    )
    state = SimpleNamespace(
        prefill=prefill, decoded=decoded, calls=[], chunks=None, p=p, d=d
    )

    async def send(client, endpoint, data):
        state.calls.append((client, deepcopy(data)))
        return FakeResponse(deepcopy(state.prefill if client == "p" else state.decoded))

    async def stream(client, endpoint, data):
        state.calls.append((client, deepcopy(data)))
        chunks = state.chunks
        if chunks is None:
            chunks = [proxy.encode_sse_data(state.decoded), b"data: [DONE]\n\n"]
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(proxy, "stats_calculator", SimpleNamespace(add=Mock()))
    monkeypatch.setattr(proxy, "log_route_event", Mock())
    monkeypatch.setattr(proxy, "select_prefiller", AsyncMock(return_value=(p, {})))
    monkeypatch.setattr(proxy, "select_decoder", AsyncMock(return_value=(d, {})))
    monkeypatch.setattr(proxy, "release_prefiller", AsyncMock(return_value={}))
    monkeypatch.setattr(proxy, "release_decoder", AsyncMock(return_value={}))
    monkeypatch.setattr(
        proxy, "acquire_pd_buffer_slots", AsyncMock(return_value=(2, 0.0, True))
    )
    monkeypatch.setattr(proxy, "release_pd_buffer_slots", AsyncMock())
    monkeypatch.setattr(proxy, "wait_decode_kv_ready", AsyncMock())
    monkeypatch.setattr(proxy, "send_request_to_service", send)
    monkeypatch.setattr(
        proxy, "stream_service_response", mock_streaming_service(stream)
    )
    return state


@pytest.mark.parametrize("stream", [None, False, True])
@pytest.mark.parametrize("path", ["decode", "one_token", "stop"])
def test_completion_response_mode_and_prefill_stop(backend, stream, path):
    async def scenario():
        if path == "stop":
            backend.prefill["choices"][0].update(finish_reason="stop", stop_reason=99)
            backend.prefill.pop("kv_transfer_params")
        data = {"prompt": [10, 20], "max_tokens": 1 if path == "one_token" else 4}
        if stream is not None:
            data["stream"] = stream
        data["stream_options"] = {"include_usage": True}
        response = await proxy.handle_completions(FakeRequest(data))
        if stream:
            assert response.media_type == "text/event-stream"
            body = await collect_streaming_response(response)
            assert body.count(b"data: [DONE]") == 1
            events = [
                json.loads(part[6:])
                for part in body.decode().split("\n\n")
                if part.startswith("data: ") and part != "data: [DONE]"
            ]
            text = "".join(c["text"] for e in events for c in e.get("choices", []))
            usage = next(e["usage"] for e in reversed(events) if e.get("usage"))
            reasons = [
                c.get("finish_reason") for e in events for c in e.get("choices", [])
            ]
            assert reasons[-1] == ("length" if path == "one_token" else "stop")
            assert len({e["id"] for e in events}) == 1
        else:
            assert response.media_type == "application/json"
            body = json.loads(response.body)
            text, usage = body["choices"][0]["text"], body["usage"]
            assert "kv_transfer_params" not in body
        assert text == ("A\u4e2dB" if path == "decode" else "A")
        assert usage == {
            "prompt_tokens": 2,
            "completion_tokens": 3 if path == "decode" else 1,
            "total_tokens": 5 if path == "decode" else 3,
        }
        d_calls = [data for client, data in backend.calls if client == "d"]
        assert len(d_calls) == (1 if path == "decode" else 0)
        if d_calls:
            assert d_calls[0]["stream"] is bool(stream)
            if not stream:
                assert "stream_options" not in d_calls[0]
        if path != "one_token":
            proxy.wait_decode_kv_ready.assert_awaited_once()
            proxy.release_pd_buffer_slots.assert_awaited_once_with(backend.d, 2)
            proxy.release_decoder.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("split", ["bytes", "coalesced", "crlf", "multiline"])
def test_completion_sse_handles_network_boundaries(backend, split):
    async def scenario():
        payload = json.dumps(backend.decoded, ensure_ascii=False).encode()
        wire = b": comment\ndata: " + payload + b"\n\ndata: [DONE]\n\n"
        if split == "crlf":
            wire = wire.replace(b"\n", b"\r\n")
        if split == "multiline":
            wire = wire.replace(b', "created"', b',\ndata: "created"')
        backend.chunks = [wire] if split == "coalesced" else [bytes([b]) for b in wire]
        response = await proxy.handle_completions(
            FakeRequest({"prompt": [10, 20], "stream": True})
        )
        body = await collect_streaming_response(response)
        events = [
            json.loads(p[6:])
            for p in body.decode().split("\n\n")
            if p.startswith("data: ") and p != "data: [DONE]"
        ]
        assert events[-1]["choices"][0]["text"] == "\u4e2dB"
        assert events[-1]["usage"] == {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "wire", [b'data: {"choices": []}\n\n', b"data: {invalid}\n\ndata: [DONE]\n\n"]
)
def test_completion_sse_does_not_hide_protocol_errors(backend, wire):
    async def scenario():
        backend.chunks = [wire]
        response = await proxy.handle_completions(
            FakeRequest({"prompt": [10, 20], "stream": True})
        )
        with pytest.raises(ValueError):
            await collect_streaming_response(response)
        assert proxy.release_decoder.await_args.kwargs["success"] is False

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "exit_mode", ["normal", "cancel", "upstream_error", "close_timeout"]
)
def test_service_stream_owns_http_response(monkeypatch, exit_mode):
    async def scenario():
        monkeypatch.setattr(proxy, "STREAM_CLOSE_TIMEOUT", 0.01)
        closed = []

        async def chunks():
            yield b"data: [DONE]\n\n"

        class Stream(FakeByteStream):
            async def aclose(self):
                await asyncio.sleep(0)
                closed.append(True)
                if exit_mode == "close_timeout":
                    await asyncio.Event().wait()
                await super().aclose()

        async def handle(request):
            assert request.url.path == "/v1/completions"
            return httpx.Response(
                400 if exit_mode == "upstream_error" else 200, stream=Stream(chunks())
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), base_url="http://d"
        ) as client:

            async def consume():
                with anyio.CancelScope() as scope:
                    async with real_stream_service_response(
                        client, "/v1/completions", {}
                    ) as response:
                        assert await anext(response.aiter_lines()) == "data: [DONE]"
                        if exit_mode == "cancel":
                            scope.cancel()

            if exit_mode == "upstream_error":
                with pytest.raises(proxy.UpstreamServiceError):
                    await consume()
            elif exit_mode == "close_timeout":
                with pytest.raises(TimeoutError):
                    await consume()
            else:
                await consume()
        assert closed == [True]

    asyncio.run(scenario())
