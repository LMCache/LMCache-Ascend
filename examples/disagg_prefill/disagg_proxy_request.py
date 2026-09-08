# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

DEFAULT_COMPLETION_MAX_TOKENS = 16


class UpstreamServiceError(Exception):
    """An HTTP error response returned by a proxied vLLM service."""

    def __init__(
        self,
        *,
        status_code: int,
        url: str,
        body: bytes,
        content_type: str | None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.body = body
        self.content_type = content_type
        self.body_text = body.decode("utf-8", errors="replace")
        super().__init__(
            f"Upstream service returned {status_code} for {url}: {self.body_text}"
        )


def upstream_service_error_from_response(response: Any) -> UpstreamServiceError:
    """Capture an upstream HTTP response before its details are discarded."""
    return UpstreamServiceError(
        status_code=response.status_code,
        url=str(response.request.url),
        body=response.content,
        content_type=response.headers.get("content-type"),
    )


def normalize_chat_request(request_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize semantically empty tool configuration for vLLM chat APIs."""
    normalized = request_data.copy()
    if normalized.get("tools") == [] and normalized.get("tool_choice") in (
        None,
        "none",
    ):
        normalized.pop("tools")
    return normalized


def parse_chat_render_output(
    render_output: dict[str, Any],
) -> tuple[list[int], int]:
    """Extract rendered prompt tokens and the effective output budget."""
    return (
        list(render_output["token_ids"]),
        render_output["sampling_params"]["max_tokens"],
    )


def build_chat_phase_requests(
    request_data: dict[str, Any],
    prompt_token_ids: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an internal prefill request and a native Chat decode request."""
    prefill_request = request_data.copy()
    prefill_request.pop("max_completion_tokens", None)
    prefill_request.pop("stream_options", None)
    prefill_request["prompt"] = list(prompt_token_ids)
    prefill_request["max_tokens"] = 1
    prefill_request["stream"] = False

    decode_request = request_data.copy()

    return prefill_request, decode_request


def build_phase_requests(
    request_data: dict[str, Any],
    prompt_token_ids: list[int],
    *,
    is_chat: bool,
    resolved_max_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build independent Completion payloads for the prefill and decode phases."""
    normalized = request_data.copy()

    if is_chat:
        normalized.pop("max_completion_tokens", None)
        if resolved_max_tokens is None:
            raise ValueError("Chat requests require a resolved max_tokens value")
        max_tokens = resolved_max_tokens
    else:
        max_tokens = normalized.get("max_tokens", DEFAULT_COMPLETION_MAX_TOKENS)

    stream_options = normalized.pop("stream_options", None)
    normalized["prompt"] = list(prompt_token_ids)

    prefill_request = normalized.copy()
    prefill_request["prompt"] = list(prompt_token_ids)
    prefill_request["max_tokens"] = 1
    prefill_request["stream"] = False

    decode_request = normalized.copy()
    decode_request["prompt"] = list(prompt_token_ids)
    decode_request["max_tokens"] = None if max_tokens is None else max_tokens - 1
    decode_request["stream"] = True
    if stream_options is not None:
        decode_request["stream_options"] = stream_options

    return prefill_request, decode_request
