# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import AsyncIterator
from copy import deepcopy
import json

# Third Party
from fastapi.responses import JSONResponse, StreamingResponse


def encode_sse_data(data: dict) -> bytes:
    return ("data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode()


def adjust_completion_usage(usage: dict | None) -> dict | None:
    if usage is None:
        return None
    result = usage.copy()
    if result.get("prompt_tokens") is not None:
        result["prompt_tokens"] -= 1
    if result.get("completion_tokens") is not None:
        result["completion_tokens"] += 1
    return result


def public_completion(output: dict) -> dict:
    result = deepcopy(output)
    result.pop("kv_transfer_params", None)
    return result


def completion_head(prefill: dict) -> dict:
    result = public_completion(prefill)
    result["usage"] = None
    result["choices"][0].update(finish_reason=None, stop_reason=None)
    return result


def merge_completion(prefill: dict, decoded: dict) -> dict:
    result = public_completion(decoded)
    for key in ("id", "created", "model"):
        result[key] = prefill[key]
    first, choice = prefill["choices"][0], result["choices"][0]
    choice["text"] = first["text"] + choice["text"]
    # Preserve available token-level fields without changing decoder finish reasons.
    if first.get("logprobs") is not None and choice.get("logprobs") is not None:
        for key, values in first["logprobs"].items():
            tail = choice["logprobs"].get(key)
            if isinstance(values, list) and isinstance(tail, list):
                if key == "text_offset":
                    tail = [offset + len(first["text"]) for offset in tail]
                choice["logprobs"][key] = values + tail
    if "usage" in result:
        result["usage"] = adjust_completion_usage(result["usage"])
    return result


async def completion_decode_events(lines: AsyncIterator[str], prefill: dict):
    data = []
    async for line in lines:
        if line:
            if line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
            continue
        if not data:
            continue
        payload = "\n".join(data)
        data.clear()
        if payload == "[DONE]":
            yield b"data: [DONE]\n\n"
            return
        event = json.loads(payload)
        if not isinstance(event, dict):
            raise ValueError("Decoder SSE data must be a JSON object")
        if "error" in event:
            raise ValueError(f"Decoder stream failed: {event['error']}")
        for key in ("id", "created", "model"):
            event[key] = prefill[key]
        if "usage" in event:
            event["usage"] = adjust_completion_usage(event["usage"])
        for choice in event.get("choices", []):
            logprobs = choice.get("logprobs")
            if logprobs and logprobs.get("text_offset") is not None:
                logprobs["text_offset"] = [
                    offset + len(prefill["choices"][0]["text"])
                    for offset in logprobs["text_offset"]
                ]
        yield encode_sse_data(event)
    raise ValueError("Decoder SSE stream ended before [DONE]")


async def stream_prefill_only_completion_response(
    prefill_output: dict, include_usage: bool
):
    choice = prefill_output["choices"][0]
    base_chunk = {
        "id": prefill_output["id"],
        "object": "text_completion",
        "created": prefill_output["created"],
        "model": prefill_output["model"],
    }
    yield encode_sse_data(
        {
            **base_chunk,
            "choices": [
                {
                    "index": 0,
                    "text": choice["text"],
                    "logprobs": choice.get("logprobs"),
                    "finish_reason": None,
                    "stop_reason": None,
                }
            ],
            "usage": None,
        }
    )
    yield encode_sse_data(
        {
            **base_chunk,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "logprobs": None,
                    "finish_reason": choice.get("finish_reason"),
                    "stop_reason": choice.get("stop_reason"),
                }
            ],
            "usage": None,
        }
    )
    if include_usage and prefill_output.get("usage") is not None:
        yield encode_sse_data(
            {
                **base_chunk,
                "choices": [],
                "usage": prefill_output["usage"],
            }
        )
    yield b"data: [DONE]\n\n"


def prefill_completion_response(prefill_output: dict, req_data: dict):
    if not req_data.get("stream", False):
        return JSONResponse(public_completion(prefill_output))
    include_usage = bool((req_data.get("stream_options") or {}).get("include_usage"))
    return StreamingResponse(
        stream_prefill_only_completion_response(prefill_output, include_usage),
        media_type="text/event-stream",
    )
