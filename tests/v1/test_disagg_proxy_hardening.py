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
real_wait_decode_kv_ready = proxy.wait_decode_kv_ready
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
    monkeypatch.setattr(
        proxy,
        "wait_decode_kv_ready",
        AsyncMock(side_effect=proxy.app.state.kv_waiters.pop),
    )
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


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("routed", [False, True])
def test_route_error_logging_requires_selected_route(
    backend, monkeypatch, chat, routed
):
    async def scenario():
        async def send(client, endpoint, data):
            if routed and endpoint.endswith("/render"):
                return FakeResponse(
                    {"token_ids": [10, 20], "sampling_params": {"max_tokens": 4}}
                )
            raise ValueError("service failed")

        monkeypatch.setattr(proxy, "send_request_to_service", send)
        preprocessing_client = SimpleNamespace(client="p", name="p")
        monkeypatch.setattr(
            proxy.app.state, "prefill_clients", [preprocessing_client], raising=False
        )
        monkeypatch.setattr(
            proxy,
            "pick_up_tokenization_client",
            Mock(return_value=preprocessing_client),
        )
        handler = proxy.handle_chat_completions if chat else proxy.handle_completions
        payload = (
            {"messages": [{"role": "user", "content": "hi"}]}
            if chat
            else {"prompt": [10, 20] if routed else "hello"}
        )
        with pytest.raises(ValueError, match="service failed"):
            await handler(FakeRequest(payload))
        errors = [
            call.args[1]
            for call in proxy.log_route_event.call_args_list
            if call.args[0] == "proxy_route_error"
        ]
        assert len(errors) == int(routed)
        if routed:
            assert errors[0]["chosen_decoder"] == "d"
            assert errors[0]["prefiller_state_after_release"] == {}
            assert errors[0]["decoder_state_after_release"] == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("disconnect", ["before_body", "during_body", "receive"])
def test_stream_disconnect_releases_resources(backend, monkeypatch, disconnect):
    async def scenario():
        body_sent = asyncio.Event()

        async def send(message):
            if disconnect == "before_body":
                raise OSError("connection closed before body")
            if message["type"] == "http.response.body":
                body_sent.set()
                if disconnect == "during_body":
                    raise OSError("connection closed during body")

        async def receive():
            await body_sent.wait()
            return {"type": "http.disconnect"}

        # Make the native ASGI receive-disconnect cancel an actual waiting stream.
        if disconnect == "receive":

            async def wait_ready(*args):
                await asyncio.Event().wait()

            monkeypatch.setattr(proxy, "wait_decode_kv_ready", wait_ready)
        response = await proxy.handle_completions(
            FakeRequest({"prompt": [10, 20], "stream": True})
        )
        scope = {
            "type": "http",
            "asgi": {"spec_version": "2.0" if disconnect == "receive" else "2.4"},
        }
        try:
            await asyncio.wait_for(response(scope, receive, send), 1)
        except OSError:
            assert disconnect != "receive"
        except Exception as exc:
            assert disconnect != "receive"
            assert type(exc).__name__ == "ClientDisconnect"
        proxy.release_prefiller.assert_awaited_once()
        proxy.release_decoder.assert_awaited_once()
        proxy.release_pd_buffer_slots.assert_awaited_once()
        assert proxy.release_decoder.await_args.kwargs["success"] is False

    asyncio.run(scenario())


@pytest.mark.parametrize("chat", [False, True])
def test_decoder_stream_is_closed_before_disconnect_returns(backend, monkeypatch, chat):
    # Third Party
    from starlette.requests import ClientDisconnect

    async def scenario():
        closed = []
        retained = []

        async def upstream_chunks():
            try:
                yield proxy.encode_sse_data(backend.decoded)
                await asyncio.Event().wait()
            finally:
                closed.append(True)

        def stream(*args):
            chunks = upstream_chunks()
            retained.append(chunks)
            return chunks

        original_events = proxy.completion_decode_events

        def events(*args):
            iterator = original_events(*args)
            retained.append(iterator)
            return iterator

        monkeypatch.setattr(proxy, "completion_decode_events", events)
        monkeypatch.setattr(
            proxy, "stream_service_response", mock_streaming_service(stream)
        )
        if chat:
            original_send = proxy.send_request_to_service

            async def send_request(client, endpoint, data):
                if endpoint.endswith("/render"):
                    return FakeResponse(
                        {"token_ids": [10, 20], "sampling_params": {"max_tokens": 4}}
                    )
                return await original_send(client, endpoint, data)

            monkeypatch.setattr(proxy, "send_request_to_service", send_request)
            monkeypatch.setattr(
                proxy.app.state,
                "prefill_clients",
                [SimpleNamespace(client="p", name="p")],
            )
        handler = proxy.handle_chat_completions if chat else proxy.handle_completions
        request = (
            {"messages": [{"role": "user", "content": "hi"}]}
            if chat
            else {"prompt": [10, 20]}
        )
        response = await handler(FakeRequest(dict(request, stream=True)))
        bodies = 0

        async def send(message):
            nonlocal bodies
            if message["type"] == "http.response.body":
                bodies += 1
                if bodies == (1 if chat else 2):
                    raise OSError("disconnect during decoder output")

        with pytest.raises(ClientDisconnect):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send
            )
        assert closed == [True]
        proxy.release_decoder.assert_awaited_once()
        assert not proxy.app.state.kv_waiters

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


def test_cleanup_continues_after_one_release_fails(backend, monkeypatch):
    async def scenario():
        monkeypatch.setattr(
            proxy,
            "send_request_to_service",
            AsyncMock(side_effect=ValueError("prefill failed")),
        )
        monkeypatch.setattr(
            proxy,
            "release_prefiller",
            AsyncMock(side_effect=RuntimeError("release failed")),
        )
        with pytest.raises(ValueError, match="prefill failed"):
            await proxy.handle_completions(FakeRequest({"prompt": [10, 20]}))
        proxy.release_decoder.assert_awaited_once()
        proxy.release_pd_buffer_slots.assert_awaited_once()

    asyncio.run(scenario())


def test_cleanup_retries_only_unreleased_resources(backend, monkeypatch):
    async def scenario():
        release_slots = AsyncMock(side_effect=[RuntimeError("release failed"), None])
        monkeypatch.setattr(proxy, "release_pd_buffer_slots", release_slots)
        resources = proxy.RequestResources(
            prompt_token_count=2, decoder_state=backend.d, slots=2, acquired=True
        )
        assert await resources.cleanup(error="disconnect") is False
        assert not resources.pd_slots_released
        assert resources.decoder_released
        assert await resources.cleanup(error="retry") is True
        assert resources.pd_slots_released
        assert release_slots.await_count == 2
        proxy.release_decoder.assert_awaited_once()
        assert await resources.cleanup(error="already released") is True
        assert release_slots.await_count == 2

    asyncio.run(scenario())


def test_local_cleanup_waits_for_lock_instead_of_abandoning_permits(monkeypatch):
    # First Party
    from tests.v1.test_disagg_proxy_server import _decoder_state

    async def scenario():
        # Network close deadlines must not abort local accounting under contention.
        monkeypatch.setattr(proxy, "STREAM_CLOSE_TIMEOUT", 0.01)
        decoder = _decoder_state()
        decoder.active_decode_requests = 1
        decoder.active_decode_tokens = 2
        decoder.pd_buffer_semaphore = proxy.WeightedSemaphore(2)
        await decoder.pd_buffer_semaphore.acquire(2)
        monkeypatch.setattr(
            proxy.app.state, "decoder_lock", asyncio.Lock(), raising=False
        )
        resources = proxy.RequestResources(
            prompt_token_count=2, decoder_state=decoder, slots=2, acquired=True
        )
        async with decoder.pd_buffer_semaphore._lock:
            task = asyncio.create_task(resources.cleanup(error="disconnect"))
            await asyncio.sleep(0.03)
            pending = not task.done()
        await asyncio.wait_for(task, 1)
        assert pending
        assert decoder.pd_buffer_semaphore.available == 2
        assert decoder.active_decode_requests == decoder.active_decode_tokens == 0
        assert resources.pd_slots_released

    asyncio.run(scenario())


def test_cleanup_survives_cancel_scope_and_repeated_task_cancel(backend, monkeypatch):
    async def scenario():
        started, finish = asyncio.Event(), asyncio.Event()

        async def release(*args, **kwargs):
            started.set()
            await finish.wait()
            return {}

        monkeypatch.setattr(proxy, "release_decoder", AsyncMock(side_effect=release))
        resources = proxy.RequestResources(
            prompt_token_count=2, decoder_state=backend.d, slots=2, acquired=True
        )

        async def run():
            with anyio.CancelScope() as scope:
                scope.cancel()
                await resources.cleanup(error="cancelled")

        task = asyncio.create_task(run())
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await resources.cleanup(error="again")
        proxy.release_decoder.assert_awaited_once()
        proxy.release_pd_buffer_slots.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("expected", [0, 2])
def test_kv_ready_accepts_early_notifications_and_ignores_late(monkeypatch, expected):
    async def scenario():
        monkeypatch.setattr(proxy.app.state, "kv_waiters", {}, raising=False)
        monkeypatch.setattr(
            proxy, "global_args", SimpleNamespace(kv_ready_timeout=0.05), raising=False
        )
        waiter = proxy.register_kv_ready("request", expected)
        proxy.notify_kv_ready("unknown")
        for _ in range(expected):
            proxy.notify_kv_ready("request")
        await proxy.wait_decode_kv_ready("request")
        assert waiter.event.is_set()
        assert not proxy.app.state.kv_waiters
        proxy.notify_kv_ready("request")
        assert not proxy.app.state.kv_waiters

    asyncio.run(scenario())


@pytest.mark.parametrize("exit_mode", ["timeout", "cancel", "success"])
def test_kv_ready_pending_wait_cleans_registration(monkeypatch, exit_mode):
    async def scenario():
        monkeypatch.setattr(proxy.app.state, "kv_waiters", {}, raising=False)
        monkeypatch.setattr(
            proxy, "global_args", SimpleNamespace(kv_ready_timeout=0.01), raising=False
        )
        proxy.register_kv_ready("request", 2)
        task = asyncio.create_task(proxy.wait_decode_kv_ready("request"))
        await asyncio.sleep(0)
        proxy.notify_kv_ready("request")
        assert not task.done()
        if exit_mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif exit_mode == "timeout":
            with pytest.raises(proxy.KVReadyTimeout):
                await asyncio.wait_for(task, 1)
        else:
            proxy.notify_kv_ready("request")
            await task
        assert not proxy.app.state.kv_waiters

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_kv_ready_timeout_rejects_invalid_values(value):
    # Standard
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        proxy.positive_timeout(value)


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_kv_ready_timeout_is_reported_and_releases_endpoint_resources(
    backend, monkeypatch, chat, stream
):
    async def scenario():
        monkeypatch.setattr(proxy.app.state, "kv_waiters", {}, raising=False)
        monkeypatch.setattr(
            proxy.app.state, "prefill_clients", [SimpleNamespace(client="p", name="p")]
        )
        monkeypatch.setattr(
            proxy, "global_args", SimpleNamespace(kv_ready_timeout=0.01), raising=False
        )
        monkeypatch.setattr(proxy, "wait_decode_kv_ready", real_wait_decode_kv_ready)
        original_send = proxy.send_request_to_service

        async def send(client, endpoint, data):
            if endpoint.endswith("/render"):
                return FakeResponse(
                    {"token_ids": [10, 20], "sampling_params": {"max_tokens": 4}}
                )
            if client == "p":
                req_id = data["kv_transfer_params"]["disagg_spec"]["req_id"]
                assert req_id in proxy.app.state.kv_waiters
            return await original_send(client, endpoint, data)

        monkeypatch.setattr(proxy, "send_request_to_service", send)
        payload = (
            {"messages": [{"role": "user", "content": "hi"}]}
            if chat
            else {"prompt": [10, 20]}
        )
        payload["stream"] = stream
        endpoint = "/v1/chat/completions" if chat else "/v1/completions"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy"
        ) as client:
            response = await asyncio.wait_for(client.post(endpoint, json=payload), 1)
        assert response.status_code == (200 if stream else 504)
        if stream:
            assert '"code":"kv_ready_timeout"' in response.text
            assert "[DONE]" not in response.text
        else:
            assert response.json()["error"]["code"] == "kv_ready_timeout"
        assert not [call for call in backend.calls if call[0] == "d"]
        assert not proxy.app.state.kv_waiters
        proxy.release_decoder.assert_awaited_once()
        proxy.release_pd_buffer_slots.assert_awaited_once()
        assert proxy.release_decoder.await_args.kwargs["success"] is False

    asyncio.run(scenario())
