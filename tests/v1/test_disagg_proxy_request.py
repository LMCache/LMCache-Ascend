# SPDX-License-Identifier: Apache-2.0
# Standard
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

# Third Party
import pytest

EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "disagg_prefill"
sys.path.insert(0, str(EXAMPLE_DIR))

# Third Party
import disagg_proxy_request as request_helpers  # noqa: E402


@pytest.mark.parametrize(
    "request_data",
    [
        {"tools": []},
        {"tools": [], "tool_choice": None},
        {"tools": [], "tool_choice": "none"},
    ],
)
def test_normalize_chat_request_removes_semantically_empty_tools(request_data):
    request_data.update(
        {
            "model": "MiniMax-M2.7",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    original = deepcopy(request_data)

    normalized = request_helpers.normalize_chat_request(request_data)

    assert "tools" not in normalized
    assert request_data == original


@pytest.mark.parametrize("tool_choice", ["auto", "required", {"type": "function"}])
def test_normalize_chat_request_preserves_invalid_tool_choice_for_validation(
    tool_choice,
):
    request_data = {"tools": [], "tool_choice": tool_choice}

    normalized = request_helpers.normalize_chat_request(request_data)

    assert normalized == request_data
    assert normalized is not request_data


def test_chat_decode_preserves_native_request_and_full_output_budget():
    request_data = {
        "model": "MiniMax-M2.7",
        "messages": [{"role": "user", "content": "hello"}],
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
        "max_completion_tokens": 150,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7,
    }
    original = deepcopy(request_data)

    prefill_request, decode_request = request_helpers.build_chat_phase_requests(
        request_data,
        [10, 20, 30],
    )

    assert request_data == original
    assert "max_completion_tokens" not in prefill_request
    assert "stream_options" not in prefill_request
    assert prefill_request["prompt"] == [10, 20, 30]
    assert prefill_request["max_tokens"] == 1
    assert prefill_request["stream"] is False
    assert decode_request == original
    assert decode_request is not request_data
    assert "prompt" not in decode_request
    assert "kv_transfer_params" not in decode_request


def test_chat_decode_leaves_omitted_output_limit_for_vllm_to_resolve():
    request_data = {
        "model": "MiniMax-M2.7",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    prefill_request, decode_request = request_helpers.build_chat_phase_requests(
        request_data,
        [10, 20],
    )

    assert prefill_request["max_tokens"] == 1
    assert "max_completion_tokens" not in decode_request
    assert "max_tokens" not in decode_request
    assert decode_request["stream"] is False
    assert "kv_transfer_params" not in decode_request


def test_parse_chat_render_output_returns_tokens_and_effective_budget():
    prompt_token_ids, resolved_max_tokens = request_helpers.parse_chat_render_output(
        {
            "token_ids": [10, 20, 30],
            "sampling_params": {"max_tokens": 196569, "temperature": 1.0},
        }
    )

    assert prompt_token_ids == [10, 20, 30]
    assert resolved_max_tokens == 196569


def test_upstream_service_error_preserves_response_details():
    response = SimpleNamespace(
        status_code=400,
        request=SimpleNamespace(
            url="http://7.150.2.142:7100/v1/chat/completions/render"
        ),
        content=b'{"error":{"message":"Input length exceeds model limit"}}',
        headers={"content-type": "application/json"},
    )

    error = request_helpers.upstream_service_error_from_response(response)

    assert error.status_code == 400
    assert error.url == "http://7.150.2.142:7100/v1/chat/completions/render"
    assert error.body == response.content
    assert error.content_type == "application/json"
    assert "Input length exceeds model limit" in str(error)


def test_completion_without_max_tokens_uses_vllm_default():
    request_data = {
        "model": "MiniMax-M2.7",
        "prompt": "hello",
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    original = deepcopy(request_data)

    prefill_request, decode_request = request_helpers.build_phase_requests(
        request_data,
        [10, 20],
        is_chat=False,
    )

    assert request_data == original
    assert prefill_request == {
        "model": "MiniMax-M2.7",
        "prompt": [10, 20],
        "ignore_eos": True,
        "max_tokens": 1,
        "stream": False,
    }
    assert decode_request == {
        "model": "MiniMax-M2.7",
        "prompt": [10, 20],
        "ignore_eos": True,
        "max_tokens": 15,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_completion_single_token_builds_zero_token_decode_budget():
    request_data = {
        "model": "MiniMax-M2.7",
        "prompt": "hello",
        "max_tokens": 1,
        "stream_options": {"include_usage": True},
    }

    prefill_request, decode_request = request_helpers.build_phase_requests(
        request_data,
        [10, 20],
        is_chat=False,
    )

    assert prefill_request["max_tokens"] == 1
    assert prefill_request["stream"] is False
    assert "stream_options" not in prefill_request
    assert decode_request["max_tokens"] == 0
    assert decode_request["stream"] is True
    assert decode_request["stream_options"] == {"include_usage": True}
