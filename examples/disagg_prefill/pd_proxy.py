# SPDX-License-Identifier: Apache-2.0
"""Thin P/D entry proxy for the MP+PD connector (1P1D static deployments).

Adapted from upstream LMCache
``examples/disagg_prefill_mp/disagg_proxy_server.py`` (Apache-2.0) with
the HTTP telemetry gate removed: this repo's MP+PD connector is
store-through, so the prefill instance's HTTP response is itself the
"KV arrived at the decoder" signal. The proxy only sequences two legs:

1. Prefill leg: the request with ``max_tokens=1`` (output discarded).
   Timeouts and failures are logged and swallowed -- the decoder
   recomputes the prefix, so service never hard-fails.
2. Decode leg: the original request forwarded verbatim (streaming or
   not).

No ``kv_transfer_params`` injection, no request-ID correlation, and no
engine-side telemetry env vars are involved. Run it in its own venv
(``pip install fastapi uvicorn httpx``)::

    python pd_proxy.py --host 0.0.0.0 --port 9100 \\
        --prefiller-host 10.50.85.46 --prefiller-port 8019 \\
        --decoder-host 10.50.85.45 --decoder-port 8019
"""

# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass
import argparse
import itertools
import logging
import os
import time
import uuid

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

logger = logging.getLogger("pd_proxy")

global_args = None


def pair_hosts_and_ports(hosts, ports, count=None):
    """Flexible host-port pairing (same semantics as the upstream proxy)."""
    if not isinstance(hosts, list):
        hosts = [hosts]
    if not isinstance(ports, list):
        ports = [ports]
    if len(hosts) == 1 and len(ports) == 1:
        if count is None or count <= 1:
            return [(hosts[0], ports[0])]
        return [(hosts[0], ports[0] + i) for i in range(count)]
    if len(hosts) == 1:
        return [(hosts[0], p) for p in ports]
    if len(ports) == 1:
        return [(h, ports[0]) for h in hosts]
    if len(hosts) != len(ports):
        raise ValueError("Length mismatch between hosts and ports lists")
    return list(zip(hosts, ports, strict=False))


@dataclass
class ClientInfo:
    client: httpx.AsyncClient

    async def aclose(self):
        await self.client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pref_pairs = pair_hosts_and_ports(
        global_args.prefiller_host,
        global_args.prefiller_port,
        global_args.num_prefillers,
    )
    for host, port in pref_pairs:
        base = f"http://{host}:{int(port)}"
        app.state.prefill_clients.append(
            ClientInfo(httpx.AsyncClient(timeout=None, base_url=base))
        )

    dec_pairs = pair_hosts_and_ports(
        global_args.decoder_host,
        global_args.decoder_port,
        global_args.num_decoders,
    )
    for host, port in dec_pairs:
        base = f"http://{host}:{int(port)}"
        app.state.decode_clients.append(
            ClientInfo(httpx.AsyncClient(timeout=None, base_url=base))
        )
    app.state.total_clients = (
        app.state.prefill_clients + app.state.decode_clients
    )
    logger.info(
        "PD entry proxy up: %d prefill, %d decode backends",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )

    yield

    for client in app.state.total_clients:
        await client.aclose()


app = FastAPI(lifespan=lifespan)
app.state.prefill_clients = []
app.state.decode_clients = []
app.state.total_clients = []


def csv_ints(s):
    return [int(x) for x in s.split(",")]


def csv_strs(s):
    return [x.strip() for x in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--prefiller-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--prefiller-port", type=csv_ints, default=[8100])
    parser.add_argument("--num-prefillers", type=int, default=1)
    parser.add_argument("--decoder-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--decoder-port", type=csv_ints, default=[8200])
    parser.add_argument("--num-decoders", type=int, default=1)
    parser.add_argument(
        "--prefill-timeout",
        type=float,
        default=300.0,
        help=(
            "Seconds to wait for the prefill leg. On timeout the decode"
            " leg is sent anyway; the decoder's pending wait covers"
            " late-arriving KV, then falls back to recompute."
        ),
    )
    return parser.parse_args()


round_robin_counter = itertools.count()


def pick_clients() -> tuple[ClientInfo, ClientInfo]:
    idx = next(round_robin_counter)
    prefill = app.state.prefill_clients[idx % len(app.state.prefill_clients)]
    decode = app.state.decode_clients[idx % len(app.state.decode_clients)]
    return prefill, decode


def _build_headers(**extra: str) -> dict[str, str]:
    """Build common HTTP headers, including auth if OPENAI_API_KEY is set."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra)
    return headers


async def send_prefill_leg(
    client: httpx.AsyncClient, endpoint: str, req_data: dict, request_id: str
):
    headers = _build_headers(**{"X-Request-Id": request_id})
    response = await client.post(
        endpoint,
        json=req_data,
        headers=headers,
        timeout=global_args.prefill_timeout,
    )
    response.raise_for_status()
    return response


async def send_request_to_decoder(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    headers = _build_headers()
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def stream_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    headers = _build_headers()
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_disagg_request(request: Request, endpoint: str):
    """Common handler: prefill leg (max_tokens=1) then decode leg."""
    req_data = await request.json()

    request_id = uuid.uuid4().hex[:16]
    logger.info("Received %s request, id=%s", endpoint, request_id)

    prefill_client, decode_client = pick_clients()

    prefill_req = req_data.copy()
    prefill_req["max_tokens"] = 1
    if "max_completion_tokens" in prefill_req:
        prefill_req["max_completion_tokens"] = 1
    prefill_req["stream"] = False
    prefill_req.pop("stream_options", None)

    send_time = time.monotonic()
    try:
        # Output is discarded; with the store-through connector the
        # response returning means the KV transfer already completed.
        await send_prefill_leg(
            prefill_client.client, endpoint, prefill_req, request_id
        )
    except Exception:
        logger.exception(
            "Prefill leg failed for %s; decoder will recompute", request_id
        )
    prefill_done = time.monotonic()
    logger.info(
        "Request %s: prefill leg took %.4fs",
        request_id,
        prefill_done - send_time,
    )

    if req_data.get("stream", False):

        async def generate_stream():
            first_chunk = True
            async for chunk in stream_service_response(
                decode_client.client, endpoint, req_data
            ):
                if first_chunk:
                    logger.info(
                        "Request %s: decode first chunk +%.2fms after"
                        " prefill leg",
                        request_id,
                        (time.monotonic() - prefill_done) * 1000,
                    )
                    first_chunk = False
                yield chunk

        return StreamingResponse(
            generate_stream(), media_type="application/json"
        )
    response = await send_request_to_decoder(
        decode_client.client, endpoint, req_data
    )
    return JSONResponse(content=response.json())


@app.post("/v1/completions")
async def handle_completions(request: Request):
    """Handle /v1/completions requests."""
    return await _handle_disagg_request(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    """Handle /v1/chat/completions requests."""
    return await _handle_disagg_request(request, "/v1/chat/completions")


@app.get("/v1/models")
async def handle_models():
    """Handle /v1/models by forwarding to the first prefiller."""
    prefill_client = app.state.prefill_clients[0]
    try:
        response = await prefill_client.client.get(
            "/v1/models", headers=_build_headers()
        )
        response.raise_for_status()
        return JSONResponse(content=response.json())
    except Exception as e:
        logger.error("Error in /v1/models endpoint: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    global_args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Third Party
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
