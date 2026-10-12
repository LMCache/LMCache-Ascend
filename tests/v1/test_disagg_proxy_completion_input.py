# SPDX-License-Identifier: Apache-2.0
# Standard
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
import asyncio
import importlib.util
import logging
import sys

# Third Party
import httpx
import pytest

# First Party
from tests.v1.disagg_proxy_test_utils import mock_streaming_service


@pytest.fixture
def proxy(monkeypatch):
    # The HTTP tests do not need LMCache tensor or transport initialization.
    for name in ("lmcache", "lmcache.v1", "lmcache.v1.storage_backend"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    log = ModuleType("lmcache.logging")
    log.init_logger = logging.getLogger
    pd = ModuleType("lmcache.v1.storage_backend.pd_backend")
    pd.PDMsg = object
    pd.ProxyNotif = type("ProxyNotif", (), {})
    monkeypatch.setitem(sys.modules, log.__name__, log)
    monkeypatch.setitem(sys.modules, pd.__name__, pd)
    directory = Path(__file__).parents[2] / "examples" / "disagg_prefill"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "proxy_input_test", directory / "disagg_proxy_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.stats_calculator = SimpleNamespace(add=Mock())
    module.log_route_event = Mock()
    return module


@pytest.mark.parametrize("prompt", ["hello", [2047, 374, 3899]])
@pytest.mark.parametrize("max_tokens", [1, 4])
def test_completion_prompt_paths(proxy, prompt, max_tokens):
    async def scenario():
        tokens = [2047, 374, 3899]
        calls, decode_requests = [], []
        prefill = SimpleNamespace(name="p", client_info=SimpleNamespace(client="p"))
        decode = SimpleNamespace(
            name="d",
            client_info=SimpleNamespace(
                client="d",
                host="localhost",
                init_port=[7300],
                alloc_port=[7400],
            ),
        )
        proxy.pick_up_tokenization_client = Mock(
            return_value=SimpleNamespace(client="t", name="t")
        )
        proxy.select_prefiller = AsyncMock(return_value=(prefill, {}))
        proxy.select_decoder = AsyncMock(return_value=(decode, {}))
        proxy.release_prefiller = AsyncMock(return_value={})
        proxy.release_decoder = AsyncMock(return_value={})
        proxy.acquire_pd_buffer_slots = AsyncMock(return_value=(0, 0.0, False))
        proxy.wait_decode_kv_ready = AsyncMock(
            side_effect=proxy.app.state.kv_waiters.pop
        )

        async def send(client, endpoint, data):
            calls.append((client, endpoint, deepcopy(data)))
            if endpoint == "/tokenize":
                return SimpleNamespace(json=lambda: {"tokens": tokens})
            return SimpleNamespace(
                json=lambda: {
                    "id": "test",
                    "object": "text_completion",
                    "created": 1,
                    "model": "model",
                    "choices": [{"index": 0, "text": "A", "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                    "kv_transfer_params": {"first_tok": 42},
                }
            )

        async def stream(client, endpoint, data):
            decode_requests.append(deepcopy(data))
            yield b"data: [DONE]\n\n"

        proxy.send_request_to_service = send
        proxy.stream_service_response = mock_streaming_service(stream)
        payload = {
            "model": "model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
        }
        original = deepcopy(payload)
        request = SimpleNamespace(json=AsyncMock(return_value=payload))
        response = await proxy.handle_completions(request)
        body = b"".join([chunk async for chunk in response.body_iterator])
        assert b"[DONE]" in body
        assert payload == original
        if isinstance(prompt, str):
            proxy.pick_up_tokenization_client.assert_called_once()
            assert calls[0] == ("t", "/tokenize", {"prompt": prompt})
        else:
            proxy.pick_up_tokenization_client.assert_not_called()
            assert all(endpoint != "/tokenize" for _, endpoint, _ in calls)
        prefill_requests = [
            data for _, endpoint, data in calls if endpoint == "/v1/completions"
        ]
        assert len(prefill_requests) == 1
        assert prefill_requests[0]["prompt"] == tokens
        assert prefill_requests[0]["max_tokens"] == 1
        proxy.select_prefiller.assert_awaited_once_with(3)
        proxy.release_prefiller.assert_awaited_once()
        if max_tokens == 1:
            proxy.select_decoder.assert_not_awaited()
            proxy.acquire_pd_buffer_slots.assert_not_awaited()
            assert not decode_requests
        else:
            proxy.select_decoder.assert_awaited_once_with(3)
            proxy.acquire_pd_buffer_slots.assert_awaited_once_with(decode, 3)
            assert decode_requests[0]["prompt"] == tokens + [42]
            assert decode_requests[0]["max_tokens"] == 3
            assert (
                decode_requests[0]["kv_transfer_params"]["lmcache.pd_handoff_id"]
                == (prefill_requests[0]["kv_transfer_params"]["disagg_spec"]["req_id"])
            )
            proxy.release_decoder.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload", [{}, {"prompt": []}, {"prompt": [True]}, {"prompt": [[1]]}]
)
def test_invalid_prompt_returns_400_before_backend_or_admission(proxy, payload):
    async def scenario():
        proxy.pick_up_tokenization_client = Mock()
        proxy.send_request_to_service = AsyncMock()
        proxy.select_prefiller = AsyncMock()
        proxy.select_decoder = AsyncMock()
        proxy.acquire_pd_buffer_slots = AsyncMock()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "prompt"
        proxy.pick_up_tokenization_client.assert_not_called()
        for mock in (
            proxy.send_request_to_service,
            proxy.select_prefiller,
            proxy.select_decoder,
            proxy.acquire_pd_buffer_slots,
        ):
            mock.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize(
    "body", [b"", b"   ", b"{", b"\xff", b"null", b"[]", b'"hello"', b"123"]
)
def test_invalid_request_body_returns_400_before_backend(proxy, endpoint, body):
    async def scenario():
        proxy.pick_up_tokenization_client = Mock()
        proxy.send_request_to_service = AsyncMock()
        proxy.select_prefiller = AsyncMock()
        proxy.select_decoder = AsyncMock()
        proxy.acquire_pd_buffer_slots = AsyncMock()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy"
        ) as client:
            response = await client.post(endpoint, content=body)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "body"
        assert error["code"] is None
        assert "Request body must" in error["message"]
        proxy.pick_up_tokenization_client.assert_not_called()
        for mock in (
            proxy.send_request_to_service,
            proxy.select_prefiller,
            proxy.select_decoder,
            proxy.acquire_pd_buffer_slots,
        ):
            mock.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload", [{}, {"messages": [{"role": "user", "content": "hello"}]}]
)
def test_parse_request_body_preserves_json_objects(proxy, payload):
    request = SimpleNamespace(json=AsyncMock(return_value=payload))
    assert asyncio.run(proxy.parse_request_body(request)) is payload


def test_parse_request_body_does_not_mask_unrelated_errors(proxy):
    request = SimpleNamespace(json=AsyncMock(side_effect=RuntimeError("read failed")))
    with pytest.raises(RuntimeError, match="read failed"):
        asyncio.run(proxy.parse_request_body(request))


def test_tokenize_error_preserves_upstream_response(proxy):
    async def scenario():
        body = b'{"error":{"message":"invalid model"}}'
        upstream = SimpleNamespace(
            post=AsyncMock(
                return_value=httpx.Response(
                    400,
                    content=body,
                    headers={"content-type": "application/json"},
                    request=httpx.Request("POST", "http://vllm/tokenize"),
                )
            )
        )
        proxy.pick_up_tokenization_client = Mock(
            return_value=SimpleNamespace(client=upstream, name="t")
        )
        proxy.select_prefiller = AsyncMock()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy"
        ) as client:
            response = await client.post("/v1/completions", json={"prompt": "hello"})
        assert response.status_code == 400
        assert response.content == body
        proxy.select_prefiller.assert_not_awaited()

    asyncio.run(scenario())
