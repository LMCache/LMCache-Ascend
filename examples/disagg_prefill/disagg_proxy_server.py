# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional
import argparse
import asyncio
import itertools
import json
import math
import os
import time

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.pd_backend import (
    PDMsg,
    ProxyNotif,
)
import httpx
import msgspec
import numpy as np
import zmq
import zmq.asyncio

logger = init_logger(__name__)

PREFILL_REQUEST_ALPHA = 256
PD_TRANSFER_MODE_PUSH = "push"
PD_TRANSFER_MODE_EAGER_PULL = "eager_pull"
PD_TRANSFER_MODE_DELAY_PULL = "delay_pull"
PD_BUFFER_ADMISSION_MODES = {
    PD_TRANSFER_MODE_PUSH,
    PD_TRANSFER_MODE_EAGER_PULL,
}


class WeightedSemaphore:
    """Async semaphore with variable-weight acquire.

    Limits in-flight PD token usage: each request holds ceil(L/chunk_size)
    slots until decoding starts, preventing decoder buffer exhaustion deadlocks
    in push and eager-pull modes.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._available = capacity
        self._lock = asyncio.Condition()

    async def acquire(self, slots: int) -> None:
        """Acquire *slots* from the semaphore, blocking until available.

        Args:
            slots: Number of slots to acquire (must be <= capacity).

        Raises:
            ValueError: If slots exceeds total capacity (would block forever).
        """
        if slots > self._capacity:
            raise ValueError(
                f"Requested {slots} slots exceeds total capacity {self._capacity}"
            )
        async with self._lock:
            await self._lock.wait_for(lambda: self._available >= slots)
            self._available -= slots

    async def release(self, slots: int) -> None:
        """Return *slots* to the semaphore and wake waiting acquirers.

        Args:
            slots: Number of slots to release. No-op if <= 0.
        """
        if slots <= 0:
            return
        async with self._lock:
            self._available += slots
            self._lock.notify_all()

    @property
    def available(self) -> int:
        """Number of slots currently available."""
        return self._available

    @property
    def capacity(self) -> int:
        """Total slot capacity."""
        return self._capacity


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize clients
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.total_clients = []
    app.state.prefiller_states = []
    app.state.prefiller_select_seq = 0
    app.state.decoder_states = []
    app.state.decoder_select_seq = 0

    # Build prefill clients with CSV-based broadcast pairing
    pref_hosts = global_args.prefiller_host
    pref_ports = global_args.prefiller_port

    def pair_hosts_and_ports(hosts, ports, count=None):
        """
        Flexible host-port pairing with expansion strategies.

        Multiple pairing strategies:
        1. Single host + single port + count: Generate incremental ports on same host
        2. Single host + multiple ports: Pair the host with each port
        3. Multiple hosts + single port: Pair each host with the same port
        4. Multiple hosts + multiple ports: Strict one-to-one pairing
           (must have same length)
        """
        # Ensure lists
        if not isinstance(hosts, list):
            hosts = [hosts]
        if not isinstance(ports, list):
            ports = [ports]
        # Single host/port with count -> incremental ports
        if len(hosts) == 1 and len(ports) == 1:
            if count is None or count <= 1:
                return [(hosts[0], ports[0])]
            else:
                return [(hosts[0], ports[0] + i) for i in range(count)]
        # Expand single host to multiple ports
        if len(hosts) == 1:
            return [(hosts[0], p) for p in ports]
        # Expand single port to multiple hosts
        if len(ports) == 1:
            return [(h, ports[0]) for h in hosts]
        # Strict one-to-one pairing when both lists are provided
        if len(hosts) != len(ports):
            raise ValueError(
                "Length mismatch between hosts and ports lists for pairing"
            )
        return list(zip(hosts, ports, strict=False))

    prefill_pairs = pair_hosts_and_ports(
        pref_hosts, pref_ports, global_args.num_prefillers
    )
    for i, (host, port) in enumerate(prefill_pairs):
        prefiller_base_url = f"http://{host}:{int(port)}"
        prefill_client = httpx.AsyncClient(timeout=None, base_url=prefiller_base_url)
        client_info = ClientInfo(
            prefill_client,
            host=host,
            name=f"prefiller-{i}",
            base_url=prefiller_base_url,
        )
        app.state.prefill_clients.append(client_info)
        app.state.prefiller_states.append(
            PrefillerState(
                client_info=client_info,
                name=client_info.name or f"prefiller-{i}",
                host=host,
                port=int(port),
            )
        )

    # Build decoder clients with CSV-based broadcast pairing
    dec_hosts = global_args.decoder_host
    dec_ports = global_args.decoder_port

    decoder_pairs = pair_hosts_and_ports(dec_hosts, dec_ports, global_args.num_decoders)

    # Whether the ports increase per instances
    # (only when using single host/port with num_decoders > 1)
    incremental_mode = (
        len(dec_hosts) == 1 and len(dec_ports) == 1 and global_args.num_decoders > 1
    )

    pd_buffer_admission_enabled = (
        global_args.pd_transfer_mode in PD_BUFFER_ADMISSION_MODES
    )
    kv_bytes_per_token = None
    pd_capacity_slots = None
    if pd_buffer_admission_enabled:
        kv_bytes_per_token = compute_kv_bytes_per_token(global_args.model)
        pd_capacity_slots = global_args.pd_buffer_size // (
            kv_bytes_per_token * global_args.chunk_size
        )

    for i, (host, port) in enumerate(decoder_pairs):
        decoder_base_url = f"http://{host}:{int(port)}"
        decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)
        if incremental_mode:
            init_ports = [p + i for p in global_args.decoder_init_port]
            alloc_ports = [p + i for p in global_args.decoder_alloc_port]
        else:
            # Use the provided ports as-is
            # (suitable when different hosts can reuse same port numbers)
            init_ports = list(global_args.decoder_init_port)
            alloc_ports = list(global_args.decoder_alloc_port)

        client_info = ClientInfo(
            decode_client,
            host=host,
            init_port=init_ports,
            alloc_port=alloc_ports,
            name=f"decoder-{i}",
            base_url=decoder_base_url,
        )
        app.state.decode_clients.append(client_info)
        app.state.decoder_states.append(
            DecoderState(
                client_info=client_info,
                name=client_info.name or f"decoder-{i}",
                host=host,
                port=int(port),
                init_port=init_ports,
                alloc_port=alloc_ports,
                pd_buffer_semaphore=(
                    WeightedSemaphore(pd_capacity_slots)
                    if pd_capacity_slots is not None
                    else None
                ),
                pd_transfer_mode=global_args.pd_transfer_mode,
            )
        )

    app.state.total_clients = app.state.prefill_clients + app.state.decode_clients

    app.state.zmq_task = asyncio.create_task(zmq_pull_server())

    if pd_buffer_admission_enabled:
        logger.info(
            "Per-decoder PD buffer semaphore: transfer_mode=%s capacity=%d"
            " slots per decoder (%d bytes / (%d bytes/tok * %d chunk_size))"
            " for model %s.",
            global_args.pd_transfer_mode,
            pd_capacity_slots,
            global_args.pd_buffer_size,
            kv_bytes_per_token,
            global_args.chunk_size,
            global_args.model,
        )
    else:
        logger.info(
            "Per-decoder PD buffer admission disabled for transfer_mode=%s;"
            " --pd-buffer-size=%d is ignored by proxy admission control.",
            global_args.pd_transfer_mode,
            global_args.pd_buffer_size,
        )

    yield

    # Shutdown: Close clients
    for client in app.state.prefill_clients:
        await client.client.aclose()
    for client in app.state.decode_clients:
        await client.client.aclose()

    global run_proxy
    run_proxy = False
    await app.state.zmq_task  # Wait for background task to finish


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


class StatsCalculator:
    def __init__(self):
        self._stats = []
        self._last_log_time = time.time()

    def add(self, value):
        self._stats.append(value)
        if time.time() - self._last_log_time > 5:
            self._log_stats()
            self._last_log_time = time.time()

    def _log_stats(self):
        # Print average, median, and 99th percentile
        np_arr = np.array(self._stats) * 1000
        output_str = (
            f"\nNum requests: {len(self._stats)}"
            + "\nPrefill node TTFT stats:"
            + f"\n - Average (ms): {np.mean(np_arr)}"
            + f"\n - Median (ms): {np.median(np_arr)}"
            + f"\n - 99th Percentile (ms): {np.percentile(np_arr, 99)}\n"
        )
        print(
            "===============================",
            output_str,
            "===============================",
        )


stats_calculator = StatsCalculator()
counter = 0


def csv_ints(s):
    return [int(x) for x in s.split(",")]


def csv_strs(s):
    return [x.strip() for x in s.split(",")]


def compute_kv_bytes_per_token(model_name: str) -> int:
    """Return the number of KV cache bytes per token for *model_name*.

    Reads num_hidden_layers, num_key_value_heads, head_dim, and torch_dtype
    from the HuggingFace config without downloading model weights.

    Args:
        model_name: HuggingFace model id or local path.

    Returns:
        Bytes per token across all layers and both K/V tensors.
    """
    # Third Party
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    num_layers: int = cfg.num_hidden_layers
    num_kv_heads: int = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim: int = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    # 4 bytes for float32, 2 bytes for float16/bfloat16 (the common default)
    torch_dtype = str(getattr(cfg, "torch_dtype", "bfloat16"))
    dtype_bytes = 4 if "float32" in torch_dtype else 2
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--prefiller-port", type=csv_ints, default=[8100])
    parser.add_argument("--num-prefillers", type=int, default=1)
    parser.add_argument("--decoder-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--decoder-port", type=csv_ints, default=[8200])
    parser.add_argument("--decoder-init-port", type=csv_ints, default=[8300])
    parser.add_argument("--decoder-alloc-port", type=csv_ints, default=[8400])

    parser.add_argument("--num-decoders", type=int, default=1)
    parser.add_argument("--proxy-host", type=str, default="localhost")
    parser.add_argument("--proxy-port", type=int, default=8500)

    # PD buffer concurrency limiting. In push and eager-pull modes, a weighted
    # semaphore caps in-flight chunk slots to prevent decoder buffer exhaustion.
    # Delay-pull mode does not pre-allocate the full decoder-side KV payload, so
    # this buffer-size based admission control is disabled in that mode.
    # capacity_slots = pd_buffer_size // (kv_bytes_per_token * chunk_size)
    # kv_bytes_per_token is derived from the model config automatically.
    parser.add_argument(
        "--pd-transfer-mode",
        type=str,
        choices=[
            PD_TRANSFER_MODE_PUSH,
            PD_TRANSFER_MODE_EAGER_PULL,
            PD_TRANSFER_MODE_DELAY_PULL,
        ],
        default=PD_TRANSFER_MODE_PUSH,
        help=(
            "PD transfer mode used by LMCache-Ascend. push and eager_pull"
            " enable proxy-side PD buffer admission control; delay_pull uses"
            " decoder load balancing only and ignores --pd-buffer-size for"
            " proxy admission. Default: push."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help=(
            "HuggingFace model name or local path. Used to derive"
            " kv_bytes_per_token for the PD buffer semaphore capacity when"
            " proxy-side PD buffer admission is enabled."
        ),
    )
    parser.add_argument(
        "--pd-buffer-size",
        type=int,
        default=2 * 1024 * 1024 * 1024,  # 2 GB
        help=(
            "PD transfer buffer size in bytes (must match the decoder's"
            " LMCache config). Used to derive the in-flight slot capacity in"
            " push and eager_pull modes. Ignored by proxy admission control in"
            " delay_pull mode. Default: 2 GB."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="LMCache chunk size in tokens (must match the LMCache config).",
    )

    args = parser.parse_args()
    return args


@dataclass
class ClientInfo:
    client: httpx.AsyncClient
    host: Optional[str] = None
    init_port: Optional[list[int]] = None
    alloc_port: Optional[list[int]] = None
    name: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class PrefillerState:
    client_info: ClientInfo
    name: str
    host: str
    port: int
    active_prefill_tokens: int = 0
    active_prefill_requests: int = 0
    total_prefill_tokens: int = 0
    total_prefill_requests: int = 0
    failed_prefill_requests: int = 0
    last_error: Optional[str] = None
    last_prefill_ms: Optional[float] = None
    prefill_ms_ewma: Optional[float] = None  # Smoothed recent prefill latency in ms.
    last_success_ts: Optional[float] = None
    last_selected_seq: int = 0

    @property
    def load_score(self) -> int:
        return (
            self.active_prefill_tokens
            + self.active_prefill_requests * PREFILL_REQUEST_ALPHA
        )

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "active_prefill_tokens": self.active_prefill_tokens,
            "active_prefill_requests": self.active_prefill_requests,
            "load_score": self.load_score,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_prefill_requests": self.total_prefill_requests,
            "failed_prefill_requests": self.failed_prefill_requests,
            "last_error": self.last_error,
            "last_prefill_ms": self.last_prefill_ms,
            "prefill_ms_ewma": self.prefill_ms_ewma,
        }


@dataclass
class DecoderState:
    client_info: ClientInfo
    name: str
    host: str
    port: int
    init_port: list[int]
    alloc_port: list[int]
    pd_buffer_semaphore: Optional[WeightedSemaphore] = None
    pd_transfer_mode: str = PD_TRANSFER_MODE_PUSH
    active_decode_tokens: int = 0
    active_decode_requests: int = 0
    total_decode_tokens: int = 0
    total_decode_requests: int = 0
    failed_decode_requests: int = 0
    last_error: Optional[str] = None
    last_decode_ms: Optional[float] = None
    decode_ms_ewma: Optional[float] = None  # Smoothed recent decode latency in ms.
    last_success_ts: Optional[float] = None
    last_selected_seq: int = 0

    @property
    def load_score(self) -> int:
        return self.active_decode_tokens

    def snapshot(self) -> dict:
        pd_buffer_admission_enabled = self.pd_buffer_semaphore is not None
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "init_port": self.init_port,
            "alloc_port": self.alloc_port,
            "pd_transfer_mode": self.pd_transfer_mode,
            "pd_buffer_admission_enabled": pd_buffer_admission_enabled,
            "active_decode_tokens": self.active_decode_tokens,
            "active_decode_requests": self.active_decode_requests,
            "load_score": self.load_score,
            "total_decode_tokens": self.total_decode_tokens,
            "total_decode_requests": self.total_decode_requests,
            "failed_decode_requests": self.failed_decode_requests,
            "last_error": self.last_error,
            "last_decode_ms": self.last_decode_ms,
            "decode_ms_ewma": self.decode_ms_ewma,
            "pd_slots_available": (
                self.pd_buffer_semaphore.available
                if pd_buffer_admission_enabled
                else None
            ),
            "pd_slots_capacity": (
                self.pd_buffer_semaphore.capacity
                if pd_buffer_admission_enabled
                else None
            ),
        }


# Initialize variables to hold the persistent clients
app.state.prefill_clients = []
app.state.decode_clients = []
app.state.total_clients = []
app.state.prefiller_states = []
app.state.prefiller_lock = asyncio.Lock()
app.state.prefiller_select_seq = 0
app.state.decoder_states = []
app.state.decoder_lock = asyncio.Lock()
app.state.decoder_select_seq = 0

"""
client_request and tokenization client map
key:   str    - unique id for requests across same conversation
value: ClientInfo - tokenization client only
"""
app.state.bound_clients = {}

# Keep finished reqs
app.state.finished_reqs = defaultdict(int)


zmq_ctx = zmq.asyncio.Context()
run_proxy = True  # Shutdown flag


async def zmq_pull_server():
    socket = zmq_ctx.socket(zmq.PULL)
    proxy_url = f"{global_args.proxy_host}:{global_args.proxy_port}"
    try:
        socket.bind(f"tcp://{proxy_url}")
    except zmq.ZMQError:
        logger.exception("ZMQ proxy server failed to bind on %s", proxy_url)
        return
    logger.info("ZMQ proxy server started on %s", proxy_url)

    while run_proxy:
        try:
            msg_bytes = await socket.recv()
        except zmq.Again:
            await asyncio.sleep(0.01)  # Avoid busy loop
            continue
        except zmq.ZMQError as exc:
            if exc.errno in (zmq.ETERM, zmq.ENOTSOCK):
                break
            logger.warning("ZMQ recv error: %s", exc)
            await asyncio.sleep(0.05)
            continue

        try:
            msg = msgspec.msgpack.decode(msg_bytes, type=PDMsg)
        except msgspec.DecodeError as exc:
            logger.warning("ZMQ received non-PD message: %s", exc)
            continue
        except Exception as exc:
            logger.exception("ZMQ message decode failed: %s", exc)
            continue

        if not isinstance(msg, ProxyNotif):
            logger.debug("ZMQ ignored message type: %s", type(msg).__name__)
            continue

        req_id = msg.req_id
        app.state.finished_reqs[req_id] += 1
        logger.debug("Prefill of req %s done.", req_id)

    socket.close()
    logger.info("ZMQ PULL server stopped.")


async def send_request_to_service(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Send a request to a service using a persistent client.
    """

    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def stream_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Asynchronously stream the response from a service using a persistent client.
    """
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


def round_robin_pick_client(clients, idx):
    if not clients:
        raise ValueError("No clients configured")
    return clients[idx % len(clients)]


tokenization_round_robin_counter = itertools.count()


BOUND_CLIENTS_MAX_NUM = 1024 * 1024

BOUND_CLIENT = os.getenv("CLIENT_BOUND", "false").lower() == "true"
# CLIENT_BOUND_KEY, the field name of the client uid in http request
CLIENT_BOUND_KEY = os.getenv("CLIENT_BOUND_KEY", "session-id")


def pick_up_bound_tokenization_client(client_id: str) -> ClientInfo:
    if client_id not in app.state.bound_clients:
        if len(app.state.bound_clients) >= BOUND_CLIENTS_MAX_NUM:
            # Here simply clear the bound_clients if full
            app.state.bound_clients.clear()
        idx = next(tokenization_round_robin_counter)
        app.state.bound_clients[client_id] = round_robin_pick_client(
            app.state.total_clients, idx
        )
    return app.state.bound_clients[client_id]


def pick_up_tokenization_client(request: Request) -> ClientInfo:
    bound_client_id = request.headers.get(CLIENT_BOUND_KEY) if BOUND_CLIENT else None
    if bound_client_id:
        # keeps tokenizer binding. Prefiller selection is load-aware.
        return pick_up_bound_tokenization_client(bound_client_id)
    idx = next(tokenization_round_robin_counter)
    return round_robin_pick_client(app.state.total_clients, idx)


async def select_decoder(
    prompt_token_count: int,
) -> tuple[DecoderState, dict]:
    if not app.state.decoder_states:
        raise ValueError("No decoder clients configured")

    async with app.state.decoder_lock:
        candidate_loads = [state.snapshot() for state in app.state.decoder_states]
        selected = min(
            app.state.decoder_states,
            key=lambda state: (
                state.load_score,
                state.last_selected_seq,
                state.name,
            ),
        )
        selected_load_before = selected.load_score
        app.state.decoder_select_seq += 1
        selected.last_selected_seq = app.state.decoder_select_seq
        selected.active_decode_tokens += prompt_token_count
        selected.active_decode_requests += 1
        selected.total_decode_tokens += prompt_token_count
        selected.total_decode_requests += 1

        return selected, {
            "decoder_policy": "min_active_decode_tokens",
            "decode_score": prompt_token_count,
            "pd_transfer_mode": selected.pd_transfer_mode,
            "pd_buffer_admission_enabled": (selected.pd_buffer_semaphore is not None),
            "selected_decoder": selected.name,
            "selected_decoder_load_before": selected_load_before,
            "selected_decoder_state_after": selected.snapshot(),
            "candidate_decoder_loads": candidate_loads,
        }


async def select_prefiller(
    prompt_token_count: int,
) -> tuple[PrefillerState, dict]:
    if not app.state.prefiller_states:
        raise ValueError("No prefiller clients configured")

    async with app.state.prefiller_lock:
        candidate_loads = [state.snapshot() for state in app.state.prefiller_states]
        selected = min(
            app.state.prefiller_states,
            key=lambda state: (
                state.load_score,
                state.last_selected_seq,
                state.name,
            ),
        )
        selected_load_before = selected.load_score
        app.state.prefiller_select_seq += 1
        selected.last_selected_seq = app.state.prefiller_select_seq
        selected.active_prefill_tokens += prompt_token_count
        selected.active_prefill_requests += 1
        selected.total_prefill_tokens += prompt_token_count
        selected.total_prefill_requests += 1

        return selected, {
            "request_alpha": PREFILL_REQUEST_ALPHA,
            "route_reason": "min_active_prefill_tokens_plus_requests",
            "prefill_score": prompt_token_count,
            "selected_prefiller": selected.name,
            "selected_prefiller_load_before": selected_load_before,
            "selected_prefiller_state_after": selected.snapshot(),
            "candidate_prefiller_loads": candidate_loads,
        }


async def release_prefiller(
    prefiller_state: PrefillerState,
    prompt_token_count: int,
    success: bool,
    prefill_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> dict:
    async with app.state.prefiller_lock:
        prefiller_state.active_prefill_tokens = max(
            0, prefiller_state.active_prefill_tokens - prompt_token_count
        )
        prefiller_state.active_prefill_requests = max(
            0, prefiller_state.active_prefill_requests - 1
        )
        if success:
            prefiller_state.last_error = None
            prefiller_state.last_success_ts = time.time()
            if prefill_ms is not None:
                prefiller_state.last_prefill_ms = prefill_ms
                if prefiller_state.prefill_ms_ewma is None:
                    prefiller_state.prefill_ms_ewma = prefill_ms
                else:
                    prefiller_state.prefill_ms_ewma = (
                        prefiller_state.prefill_ms_ewma * 0.8 + prefill_ms * 0.2
                    )
        else:
            prefiller_state.failed_prefill_requests += 1
            prefiller_state.last_error = error
        return prefiller_state.snapshot()


async def release_decoder(
    decoder_state: DecoderState,
    prompt_token_count: int,
    success: bool,
    decode_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> dict:
    async with app.state.decoder_lock:
        decoder_state.active_decode_tokens = max(
            0, decoder_state.active_decode_tokens - prompt_token_count
        )
        decoder_state.active_decode_requests = max(
            0, decoder_state.active_decode_requests - 1
        )
        if success:
            decoder_state.last_error = None
            decoder_state.last_success_ts = time.time()
            if decode_ms is not None:
                decoder_state.last_decode_ms = decode_ms
                if decoder_state.decode_ms_ewma is None:
                    decoder_state.decode_ms_ewma = decode_ms
                else:
                    decoder_state.decode_ms_ewma = (
                        decoder_state.decode_ms_ewma * 0.8 + decode_ms * 0.2
                    )
        else:
            decoder_state.failed_decode_requests += 1
            decoder_state.last_error = error
        return decoder_state.snapshot()


def _format_metric(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_prefiller_active(state: dict) -> str:
    if not state:
        return "-"
    return (
        f"{state.get('active_prefill_requests', '-')}"
        f"req/{state.get('active_prefill_tokens', '-')}tok"
    )


def _format_prefiller_candidate(state: dict) -> str:
    name = state.get("name", "-")
    host = state.get("host", "-")
    port = state.get("port", "-")
    return (
        f"{name}@{host}:{port}"
        f"(load={state.get('load_score', '-')},"
        f"active={_format_prefiller_active(state)},"
        f"total={state.get('total_prefill_requests', '-')}req/"
        f"{state.get('total_prefill_tokens', '-')}tok,"
        f"failed={state.get('failed_prefill_requests', '-')},"
        f"last_ms={_format_metric(state.get('last_prefill_ms'))},"
        f"ewma_ms={_format_metric(state.get('prefill_ms_ewma'))})"
    )


def _format_candidate_prefillers(candidate_loads: list[dict]) -> str:
    if not candidate_loads:
        return "[]"
    return (
        "["
        + "; ".join(_format_prefiller_candidate(state) for state in candidate_loads)
        + "]"
    )


def _format_decoder_active(state: dict) -> str:
    if not state:
        return "-"
    return (
        f"{state.get('active_decode_requests', '-')}"
        f"req/{state.get('active_decode_tokens', '-')}tok"
    )


def _format_pd_slots(state: dict) -> str:
    if not state:
        return "-"
    if not state.get("pd_buffer_admission_enabled", False):
        return "disabled"
    return (
        f"{state.get('pd_slots_available', '-')}/{state.get('pd_slots_capacity', '-')}"
    )


def _format_pd_buffer_admission(value) -> str:
    if value is None:
        return "-"
    return "enabled" if value else "disabled"


def _format_decoder_candidate(state: dict) -> str:
    name = state.get("name", "-")
    host = state.get("host", "-")
    port = state.get("port", "-")
    return (
        f"{name}@{host}:{port}"
        f"(load={state.get('load_score', '-')},"
        f"active={_format_decoder_active(state)},"
        f"pd_slots={_format_pd_slots(state)},"
        f"total={state.get('total_decode_requests', '-')}req/"
        f"{state.get('total_decode_tokens', '-')}tok,"
        f"failed={state.get('failed_decode_requests', '-')},"
        f"last_ms={_format_metric(state.get('last_decode_ms'))},"
        f"ewma_ms={_format_metric(state.get('decode_ms_ewma'))})"
    )


def _format_candidate_decoders(candidate_loads: list[dict]) -> str:
    if not candidate_loads:
        return "[]"
    return (
        "["
        + "; ".join(_format_decoder_candidate(state) for state in candidate_loads)
        + "]"
    )


def _format_route_summary(event: str, payload: dict) -> str:
    prefiller_selected_state = payload.get("selected_prefiller_state_after") or {}
    prefiller_released_state = payload.get("prefiller_state_after_release") or {}
    prefiller_active_state = prefiller_released_state or prefiller_selected_state
    decoder_selected_state = payload.get("selected_decoder_state_after") or {}
    decoder_released_state = payload.get("decoder_state_after_release") or {}
    decoder_active_state = decoder_released_state or decoder_selected_state

    lines = [
        "===============================",
        f"Proxy route event: {event}",
        f" - req_id: {payload.get('req_id', '-')}",
        f" - endpoint: {payload.get('endpoint', '-')}",
        f" - chosen_prefiller: {payload.get('chosen_prefiller', '-')}",
        f" - chosen_decoder: {payload.get('chosen_decoder', '-')}",
        f" - prompt_token_count: {payload.get('prompt_token_count', '-')}",
        f" - prefill_ms: {_format_metric(payload.get('prefill_ms'))}",
        f" - pd_transfer_mode: {payload.get('pd_transfer_mode', '-')}",
        " - pd_buffer_admission: "
        f"{_format_pd_buffer_admission(payload.get('pd_buffer_admission_enabled'))}",
        f" - pd_slot_count: {payload.get('pd_slot_count', '-')}",
        f" - pd_slot_wait_ms: {_format_metric(payload.get('pd_slot_wait_ms'))}",
        " - prefiller_load_before: "
        f"{payload.get('selected_prefiller_load_before', '-')}",
        " - prefiller_load_after_select: "
        f"{prefiller_selected_state.get('load_score', '-')}",
        " - prefiller_load_after_release: "
        f"{prefiller_released_state.get('load_score', '-')}",
        f" - decoder_load_before: {payload.get('selected_decoder_load_before', '-')}",
        " - decoder_load_after_select: "
        f"{decoder_selected_state.get('load_score', '-')}",
        " - decoder_load_after_release: "
        f"{decoder_released_state.get('load_score', '-')}",
        " - prefiller_active_current: "
        f"{_format_prefiller_active(prefiller_active_state)}",
        f" - decoder_active_current: {_format_decoder_active(decoder_active_state)}",
        " - candidate_prefiller_loads: "
        f"{_format_candidate_prefillers(payload.get('candidate_prefiller_loads', []))}",
        " - candidate_decoder_loads: "
        f"{_format_candidate_decoders(payload.get('candidate_decoder_loads', []))}",
    ]

    if payload.get("kv_ready_wait_ms") is not None:
        lines.append(
            f" - kv_ready_wait_ms: {_format_metric(payload['kv_ready_wait_ms'])}"
        )
    if payload.get("decode_stream_ms") is not None:
        lines.append(
            f" - decode_stream_ms: {_format_metric(payload['decode_stream_ms'])}"
        )
    if payload.get("total_ms") is not None:
        lines.append(f" - total_ms: {_format_metric(payload['total_ms'])}")
    if payload.get("error") is not None:
        lines.append(f" - error: {payload['error'] or '-'}")

    lines.append(" ===============================")
    return "\n" + "\n".join(lines)


def log_route_event(event: str, payload: dict):
    try:
        logger.info(_format_route_summary(event, payload))
        logger.debug(
            "%s_full %s",
            event,
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )
    except TypeError:
        logger.info("%s %s", event, payload)


async def wait_decode_kv_ready(req_id: str, num_tp_rank: int):
    while app.state.finished_reqs[req_id] < num_tp_rank:
        await asyncio.sleep(0.0001)  # sleep for 0.1 ms
    logger.debug(f"Prefill node signaled kv ready for req {req_id}")
    app.state.finished_reqs.pop(req_id)


async def acquire_pd_buffer_slots(
    decoder_state: DecoderState,
    prompt_token_count: int,
) -> tuple[int, float, bool]:
    """Acquire decoder PD buffer slots when mode-aware admission is enabled."""
    if decoder_state.pd_buffer_semaphore is None:
        return 0, 0.0, False

    slots = math.ceil(prompt_token_count / global_args.chunk_size)
    pd_slot_wait_start = time.time()
    await decoder_state.pd_buffer_semaphore.acquire(slots)
    pd_slot_wait_ms = (time.time() - pd_slot_wait_start) * 1000
    return slots, pd_slot_wait_ms, True


async def release_pd_buffer_slots(
    decoder_state: DecoderState,
    slots: int,
) -> None:
    """Release decoder PD buffer slots if mode-aware admission is enabled."""
    if decoder_state.pd_buffer_semaphore is None:
        return
    await decoder_state.pd_buffer_semaphore.release(slots)


@app.post("/v1/completions")
async def handle_completions(request: Request):
    global counter, stats_calculator
    counter += 1
    req_id = str(counter)  # we use counter as req_id

    st = time.time()
    slots = 0  # slots to release on error; set after successful acquire only
    acquired = False
    pd_slots_released = False
    prompt_token_count = 0
    prefiller_state = None
    prefiller_released = False
    decoder_state = None
    decoder_released = False
    route_info = {}
    try:
        req_data = await request.json()

        tokenization_client = pick_up_tokenization_client(request)

        tokenize_output = await send_request_to_service(
            tokenization_client.client, "/tokenize", {"prompt": req_data["prompt"]}
        )
        tokenize_output = tokenize_output.json()
        prompt_token_count = len(tokenize_output["tokens"])

        decoder_state, decoder_info = await select_decoder(prompt_token_count)
        route_info = dict(decoder_info)
        route_info.update(
            {
                "req_id": req_id,
                "endpoint": "/v1/completions",
                "prompt_token_count": prompt_token_count,
                "chosen_decoder": decoder_state.name,
                "tokenization_client": tokenization_client.name,
            }
        )
        decode_client = decoder_state.client_info
        prefiller_state, prefiller_info = await select_prefiller(prompt_token_count)
        route_info.update(prefiller_info)
        prefill_client = prefiller_state.client_info
        route_info.update(
            {
                "chosen_prefiller": prefiller_state.name,
            }
        )
        log_route_event("proxy_route_selected", route_info)

        org_max_tokens = req_data["max_tokens"]
        req_data["prompt"] = tokenize_output["tokens"]
        req_data["max_tokens"] = 1

        # Acquire decoder PD buffer slots before prefill when mode-aware
        # admission is enabled. Delay-pull mode skips buffer-size admission.
        slots, pd_slot_wait_ms, acquired = await acquire_pd_buffer_slots(
            decoder_state,
            prompt_token_count,
        )

        disagg_spec = {
            "req_id": req_id,
            "receiver_host": decode_client.host,
            "receiver_init_port": decode_client.init_port,
            "receiver_alloc_port": decode_client.alloc_port,
        }
        num_tp_rank = len(decode_client.init_port or [])

        req_data["kv_transfer_params"] = {
            "ret_first_tok": True,
            "disagg_spec": disagg_spec,
        }

        req_data["stream"] = False
        stream_options = req_data.pop("stream_options", None)

        # Send request to prefill service, ignore the response
        prefill_start = time.time()
        prefill_output = await send_request_to_service(
            prefill_client.client, "/v1/completions", req_data
        )
        prefill_ms = (time.time() - prefill_start) * 1000
        prefiller_release_state = await release_prefiller(
            prefiller_state,
            prompt_token_count,
            success=True,
            prefill_ms=prefill_ms,
        )
        prefiller_released = True
        route_info.update(
            {
                "pd_slot_count": slots,
                "pd_slot_wait_ms": pd_slot_wait_ms,
                "prefill_ms": prefill_ms,
                "prefiller_state_after_release": prefiller_release_state,
            }
        )

        prefill_output = prefill_output.json()

        et = time.time()
        stats_calculator.add(et - st)

        req_data["max_tokens"] = org_max_tokens - 1
        req_data["prompt"].append(prefill_output["kv_transfer_params"]["first_tok"])
        req_data.pop("kv_transfer_params")
        req_data["stream"] = True
        if stream_options is not None:
            req_data["stream_options"] = stream_options

        route_log_base = dict(route_info)
        log_route_event("proxy_route_prefill_done", route_log_base)

        # Stream response from decode service
        async def generate_stream():
            nonlocal decoder_released, pd_slots_released
            kv_ready_wait_ms = None
            decode_stream_ms = None
            stream_error = None
            try:
                head_chunk = {
                    "id": prefill_output["id"],
                    "object": "text_completion",
                    "created": prefill_output["created"],
                    "model": prefill_output["model"],
                    "choices": [
                        {
                            "index": 0,
                            "text": prefill_output["choices"][0]["text"],
                            "logprobs": None,
                            "finish_reason": None,
                            "stop_reason": None,
                        }
                    ],
                    "usage": None,
                }
                yield (
                    "data: " + json.dumps(head_chunk, separators=(",", ":")) + "\n\n"
                ).encode()

                kv_ready_wait_start = time.time()
                await wait_decode_kv_ready(req_id, num_tp_rank)
                kv_ready_wait_ms = (time.time() - kv_ready_wait_start) * 1000
                if acquired:
                    await release_pd_buffer_slots(decoder_state, slots)
                    pd_slots_released = True

                decode_stream_start = time.time()
                async for chunk in stream_service_response(
                    decode_client.client, "/v1/completions", req_data
                ):
                    yield chunk
                decode_stream_ms = (time.time() - decode_stream_start) * 1000
            except BaseException as exc:
                stream_error = str(exc)
                raise
            finally:
                if acquired and not pd_slots_released:
                    await release_pd_buffer_slots(decoder_state, slots)
                    pd_slots_released = True
                decoder_release_state = None
                if decoder_state is not None and not decoder_released:
                    decoder_release_state = await release_decoder(
                        decoder_state,
                        prompt_token_count,
                        success=stream_error is None,
                        decode_ms=decode_stream_ms,
                        error=stream_error,
                    )
                    decoder_released = True
                complete_payload = dict(route_log_base)
                complete_payload.update(
                    {
                        "kv_ready_wait_ms": kv_ready_wait_ms,
                        "decode_stream_ms": decode_stream_ms,
                        "decoder_state_after_release": decoder_release_state,
                        "total_ms": (time.time() - st) * 1000,
                        "error": stream_error,
                    }
                )
                log_route_event("proxy_route_complete", complete_payload)

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        if prefiller_state is not None and not prefiller_released:
            release_state = await release_prefiller(
                prefiller_state,
                prompt_token_count,
                success=False,
                error=str(e),
            )
            route_info["prefiller_state_after_release"] = release_state
        if decoder_state is not None and not decoder_released:
            release_state = await release_decoder(
                decoder_state,
                prompt_token_count,
                success=False,
                error=str(e),
            )
            route_info["decoder_state_after_release"] = release_state
            decoder_released = True
        if decoder_state is not None and acquired and not pd_slots_released:
            await release_pd_buffer_slots(decoder_state, slots)
            pd_slots_released = True
        if route_info:
            error_payload = dict(route_info)
            error_payload.update(
                {
                    "pd_slot_count": slots,
                    "total_ms": (time.time() - st) * 1000,
                    "error": str(e),
                }
            )
            log_route_event("proxy_route_error", error_payload)
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server - completions endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    global counter, stats_calculator
    counter += 1
    req_id = str(counter)

    st = time.time()
    slots = 0  # slots to release on error; set after successful acquire only
    acquired = False
    pd_slots_released = False
    prompt_token_count = 0
    prefiller_state = None
    prefiller_released = False
    decoder_state = None
    decoder_released = False
    route_info = {}
    try:
        req_data = await request.json()

        tokenization_client = pick_up_tokenization_client(request)

        # For chat completions, we need to tokenize the messages
        tokenize_output = await send_request_to_service(
            tokenization_client.client, "/tokenize", {"messages": req_data["messages"]}
        )
        tokenize_output = tokenize_output.json()
        prompt_token_count = len(tokenize_output["tokens"])

        decoder_state, decoder_info = await select_decoder(prompt_token_count)
        route_info = dict(decoder_info)
        route_info.update(
            {
                "req_id": req_id,
                "endpoint": "/v1/chat/completions",
                "prompt_token_count": prompt_token_count,
                "chosen_decoder": decoder_state.name,
                "tokenization_client": tokenization_client.name,
            }
        )
        decode_client = decoder_state.client_info
        prefiller_state, prefiller_info = await select_prefiller(prompt_token_count)
        route_info.update(prefiller_info)
        prefill_client = prefiller_state.client_info
        route_info.update(
            {
                "chosen_prefiller": prefiller_state.name,
            }
        )
        log_route_event("proxy_route_selected", route_info)

        org_max_tokens = req_data["max_tokens"]
        req_data["prompt"] = tokenize_output["tokens"]
        req_data["max_tokens"] = 1

        org_max_completion_tokens = None
        if "max_completion_tokens" in req_data:
            org_max_completion_tokens = req_data["max_completion_tokens"]
            req_data["max_completion_tokens"] = 1

        # Acquire decoder PD buffer slots before prefill when mode-aware
        # admission is enabled. Delay-pull mode skips buffer-size admission.
        slots, pd_slot_wait_ms, acquired = await acquire_pd_buffer_slots(
            decoder_state,
            prompt_token_count,
        )

        disagg_spec = {
            "req_id": req_id,
            "receiver_host": decode_client.host,
            "receiver_init_port": decode_client.init_port,
            "receiver_alloc_port": decode_client.alloc_port,
        }

        num_tp_rank = len(decode_client.init_port or [])

        req_data["kv_transfer_params"] = {
            "ret_first_tok": True,
            "disagg_spec": disagg_spec,
        }

        req_data["stream"] = False
        stream_options = req_data.pop("stream_options", None)

        # Send request to prefill service, get the response
        prefill_start = time.time()
        prefill_output = await send_request_to_service(
            prefill_client.client, "/v1/completions", req_data
        )
        prefill_ms = (time.time() - prefill_start) * 1000
        prefiller_release_state = await release_prefiller(
            prefiller_state,
            prompt_token_count,
            success=True,
            prefill_ms=prefill_ms,
        )
        prefiller_released = True
        route_info.update(
            {
                "pd_slot_count": slots,
                "pd_slot_wait_ms": pd_slot_wait_ms,
                "prefill_ms": prefill_ms,
                "prefiller_state_after_release": prefiller_release_state,
            }
        )

        prefill_output = prefill_output.json()

        et = time.time()
        stats_calculator.add(et - st)

        req_data["max_tokens"] = org_max_tokens - 1
        if org_max_completion_tokens is not None:
            req_data["max_completion_tokens"] = org_max_completion_tokens - 1

        # Add the first token from prefill to the tokenized messages for decode
        req_data["prompt"].append(prefill_output["kv_transfer_params"]["first_tok"])

        req_data.pop("kv_transfer_params")
        req_data["stream"] = True
        if stream_options is not None:
            req_data["stream_options"] = stream_options

        route_log_base = dict(route_info)
        log_route_event("proxy_route_prefill_done", route_log_base)

        # Stream response from decode service
        async def generate_stream():
            nonlocal decoder_released, pd_slots_released
            kv_ready_wait_ms = None
            decode_stream_ms = None
            stream_error = None
            try:
                initial_chunk = {
                    "id": prefill_output["id"],
                    "object": "chat.completion.chunk",
                    "created": prefill_output["created"],
                    "model": prefill_output["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield (
                    "data: " + json.dumps(initial_chunk, separators=(",", ":")) + "\n\n"
                ).encode()

                head_chunk = {
                    "id": prefill_output["id"],
                    "object": "chat.completion.chunk",
                    "created": prefill_output["created"],
                    "model": prefill_output["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": prefill_output["choices"][0]["text"]},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield (
                    "data: " + json.dumps(head_chunk, separators=(",", ":")) + "\n\n"
                ).encode()

                kv_ready_wait_start = time.time()
                await wait_decode_kv_ready(req_id, num_tp_rank)
                kv_ready_wait_ms = (time.time() - kv_ready_wait_start) * 1000
                if acquired:
                    await release_pd_buffer_slots(decoder_state, slots)
                    pd_slots_released = True

                decode_stream_start = time.time()
                # Stream and convert completion format chunks to chat completion format
                async for chunk in stream_service_response(
                    decode_client.client, "/v1/completions", req_data
                ):
                    chunk_str = chunk.decode("utf-8")
                    if chunk_str.startswith("data: ") and not chunk_str.startswith(
                        "data: [DONE]"
                    ):
                        try:
                            json_str = chunk_str[6:].strip()  # Remove 'data: ' prefix
                            if json_str:
                                completion_data = json.loads(json_str)
                                chat_completion_data = {
                                    "id": completion_data["id"],
                                    "object": "chat.completion.chunk",
                                    "created": completion_data["created"],
                                    "model": completion_data["model"],
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": completion_data["choices"][
                                                    0
                                                ]["text"]
                                            },
                                            "logprobs": completion_data["choices"][
                                                0
                                            ].get("logprobs"),
                                            "finish_reason": completion_data["choices"][
                                                0
                                            ].get("finish_reason"),
                                        }
                                    ],
                                }
                                converted_chunk = (
                                    "data: "
                                    + json.dumps(
                                        chat_completion_data, separators=(",", ":")
                                    )
                                    + "\n\n"
                                ).encode()
                                yield converted_chunk
                        except (json.JSONDecodeError, KeyError):
                            yield chunk
                    else:
                        yield chunk
                decode_stream_ms = (time.time() - decode_stream_start) * 1000
            except BaseException as exc:
                stream_error = str(exc)
                raise
            finally:
                if acquired and not pd_slots_released:
                    await release_pd_buffer_slots(decoder_state, slots)
                    pd_slots_released = True
                decoder_release_state = None
                if decoder_state is not None and not decoder_released:
                    decoder_release_state = await release_decoder(
                        decoder_state,
                        prompt_token_count,
                        success=stream_error is None,
                        decode_ms=decode_stream_ms,
                        error=stream_error,
                    )
                    decoder_released = True
                complete_payload = dict(route_log_base)
                complete_payload.update(
                    {
                        "kv_ready_wait_ms": kv_ready_wait_ms,
                        "decode_stream_ms": decode_stream_ms,
                        "decoder_state_after_release": decoder_release_state,
                        "total_ms": (time.time() - st) * 1000,
                        "error": stream_error,
                    }
                )
                log_route_event("proxy_route_complete", complete_payload)

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        if prefiller_state is not None and not prefiller_released:
            release_state = await release_prefiller(
                prefiller_state,
                prompt_token_count,
                success=False,
                error=str(e),
            )
            route_info["prefiller_state_after_release"] = release_state
        if decoder_state is not None and not decoder_released:
            release_state = await release_decoder(
                decoder_state,
                prompt_token_count,
                success=False,
                error=str(e),
            )
            route_info["decoder_state_after_release"] = release_state
            decoder_released = True
        if decoder_state is not None and acquired and not pd_slots_released:
            await release_pd_buffer_slots(decoder_state, slots)
            pd_slots_released = True
        if route_info:
            error_payload = dict(route_info)
            error_payload.update(
                {
                    "pd_slot_count": slots,
                    "total_ms": (time.time() - st) * 1000,
                    "error": str(e),
                }
            )
            log_route_event("proxy_route_error", error_payload)
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(
            "Error occurred in disagg prefill proxy server  - chat completions endpoint"
        )
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    # Third Party
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
