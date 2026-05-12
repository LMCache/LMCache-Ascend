# lmcache-ascend/v1/internal_api_server/memory/memory_api.py
# SPDX-License-Identifier: Apache-2.0

"""
Memory system API.

The upper-layer memory system interacts with LMCache through this REST API:
- POST /memory/prefetch  : Prefetch KV cache to DDR
- POST /memory/evict     : Evict KV cache

chunk_hash is returned to the memory system via kv_transfer_params in vLLM
responses, no additional API needed.

Design note:
  In vLLM v1 multi-process mode, the scheduler process does not create
  LMCacheEngine (to save memory).  When the scheduler's API server
  receives a /memory/* request, it is automatically forwarded to the
  worker process's API server.
"""

from typing import List, Optional, Tuple
import asyncio
import json
import urllib.request

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)

router = APIRouter()


def _get_engine(
    request: Request,
) -> Tuple[Optional[LMCacheEngine], Optional[PlainTextResponse], bool]:
    """
    Get LMCacheEngine.

    Returns (engine, error_response, forward_to_workers):
    - engine, None, False: use engine directly
    - None, error_response, False: return error
    - None, None, True: scheduler process, forward to workers
    """
    adapter = request.app.state.lmcache_adapter
    engine = getattr(adapter, "lmcache_engine", None)
    if engine:
        return engine, None, False

    port_offset = getattr(
        request.app.state, "internal_api_server_port_offset", None
    )
    if port_offset == 0:
        return None, None, True

    return None, PlainTextResponse(
        content=json.dumps({"error": "LMCache engine not available"}),
        media_type="application/json",
        status_code=503,
    ), False


def _hashes_to_keys(
    engine: LMCacheEngine, chunk_hashes: List[str]
) -> List[CacheEngineKey]:
    world_size = (
        1 if engine.save_only_first_rank
        else engine.metadata.world_size
    )
    return [
        CacheEngineKey(
            model_name=engine.metadata.model_name,
            world_size=world_size,
            worker_id=engine.metadata.worker_id,
            chunk_hash=int(h, 16),
            dtype=engine.metadata.kv_dtype,
        )
        for h in chunk_hashes
    ]


def _get_worker_ports(request: Request) -> List[int]:
    """Get all worker API server ports.

    For non-MLA models (save_only_first_rank=False), each TP worker stores
    a different head shard of the KV cache, so we need to broadcast to all
    workers.  For MLA models (save_only_first_rank=True), only worker 0
    stores data, but broadcasting to all workers is safe (non-storing
    workers return empty results).
    """
    port_start = request.app.state.internal_api_server_port_start
    adapter = request.app.state.lmcache_adapter
    metadata = getattr(adapter, "lmcache_engine_metadata", None)
    world_size = metadata.world_size if metadata else 1
    return [port_start + 1 + i for i in range(world_size)]


async def _forward_to_all_workers(
    request: Request, path: str
) -> PlainTextResponse:
    """Forward request to all worker API servers, return merged result."""
    worker_ports = _get_worker_ports(request)
    body = await request.body()

    def _do_request(port: int):
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return port, resp.status, json.loads(resp.read())
        except Exception as e:
            return port, 503, {"error": str(e)}

    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _do_request, port)
        for port in worker_ports
    ]
    results = await asyncio.gather(*tasks)

    failures = [(p, s, r) for p, s, r in results if s != 200]
    all_ok = len(failures) == 0

    if all_ok:
        first_result = results[0][2] if results else {}
        return PlainTextResponse(
            content=json.dumps(first_result),
            media_type="application/json",
            status_code=200,
        )

    logger.error(
        "Forward %s: %d/%d workers failed: %s",
        path, len(failures), len(worker_ports),
        {p: r for p, _, r in failures},
    )
    return PlainTextResponse(
        content=json.dumps({
            "error": "Some workers failed to process request",
            "failures": {str(p): r for p, _, r in failures},
        }),
        media_type="application/json",
        status_code=500,
    )


# ==================== prefetch ====================

@router.post("/memory/prefetch")
async def prefetch(request: Request):
    """
    Prefetch KV cache to DDR.

    Called by the memory system before inference.

    Request body (JSON):
    {
        "chunk_hashes": ["a1b2c3...", "d4e5f6..."],
        "lookup_id": "req_001"
    }

    Response:
    {
        "status": "prefetch_started",
        "lookup_id": "req_001",
        "num_chunks": 2
    }
    """
    engine, err, forward = _get_engine(request)
    if forward:
        return await _forward_to_all_workers(request, "/memory/prefetch")
    if err:
        return err

    if engine.storage_manager is None:
        return PlainTextResponse(
            content=json.dumps({"error": "Storage manager not available"}),
            media_type="application/json",
            status_code=503,
        )

    try:
        body = await request.json()
        chunk_hashes = body.get("chunk_hashes", [])
        lookup_id = body.get("lookup_id")

        if not chunk_hashes or not lookup_id:
            return PlainTextResponse(
                content=json.dumps({"error": "chunk_hashes and lookup_id are required"}),
                media_type="application/json",
                status_code=400,
            )

        keys = _hashes_to_keys(engine, chunk_hashes)
        chunk_size = engine.config.chunk_size
        cum_chunk_lengths = [i * chunk_size for i in range(len(keys) + 1)]

        asyncio.run_coroutine_threadsafe(
            engine.storage_manager.async_lookup_and_prefetch(
                lookup_id=lookup_id,
                keys=keys,
                cum_chunk_lengths=cum_chunk_lengths,
                search_range=engine.retrieve_locations,
                pin=True,
                log_timing=True,
            ),
            engine.storage_manager.loop,
        )

        return PlainTextResponse(
            content=json.dumps({
                "status": "prefetch_started",
                "lookup_id": lookup_id,
                "num_chunks": len(keys),
            }),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("prefetch failed: %s", e)
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


# ==================== evict ====================

@router.post("/memory/evict")
async def evict(request: Request):
    """
    Evict KV cache.

    Called when the memory system determines certain chunks are no longer needed.

    Request body (JSON):
    {
        "chunk_hashes": ["a1b2c3..."],
        "locations": ["LocalCPUBackend"]   // optional, default all backends
    }

    Response:
    {
        "status": "success",
        "num_evicted": 1
    }
    """
    engine, err, forward = _get_engine(request)
    if forward:
        return await _forward_to_all_workers(request, "/memory/evict")
    if err:
        return err

    if engine.storage_manager is None:
        return PlainTextResponse(
            content=json.dumps({"error": "Storage manager not available"}),
            media_type="application/json",
            status_code=503,
        )

    try:
        body = await request.json()
        chunk_hashes = body.get("chunk_hashes", [])
        locations = body.get("locations")  # None = all backends

        if not chunk_hashes:
            return PlainTextResponse(
                content=json.dumps({"error": "chunk_hashes is required"}),
                media_type="application/json",
                status_code=400,
            )

        keys = _hashes_to_keys(engine, chunk_hashes)
        num_evicted = engine.storage_manager.batched_remove(keys, locations=locations)

        return PlainTextResponse(
            content=json.dumps({
                "status": "success",
                "num_evicted": num_evicted,
            }),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("evict failed: %s", e)
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )
