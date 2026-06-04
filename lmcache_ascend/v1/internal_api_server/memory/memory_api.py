# SPDX-License-Identifier: Apache-2.0
"""Memory system REST API: /memory/prefetch, /memory/evict."""

# Standard
from typing import List
import asyncio
import json
import urllib.request

# Third Party
from fastapi import APIRouter
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from starlette.requests import Request
from starlette.responses import PlainTextResponse

logger = init_logger(__name__)

router = APIRouter()


def _hashes_to_keys(
    engine: LMCacheEngine, chunk_hashes: List[str]
) -> List[CacheEngineKey]:
    world_size = 1 if engine.save_only_first_rank else engine.metadata.world_size
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


async def _forward_to_all_workers(request: Request, path: str) -> PlainTextResponse:
    adapter = request.app.state.lmcache_adapter
    port_start = adapter.config.internal_api_server_port_start
    metadata = getattr(adapter, "lmcache_engine_metadata", None)
    world_size = metadata.world_size if metadata else 1
    worker_ports = [port_start + 1 + i for i in range(world_size)]

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
                return resp.status, json.loads(resp.read())
        except Exception as e:
            return 503, {"error": str(e)}

    tasks = [
        asyncio.get_event_loop().run_in_executor(None, _do_request, port)
        for port in worker_ports
    ]
    results = await asyncio.gather(*tasks)

    failures = [(i, r) for i, (s, r) in enumerate(results) if s != 200]
    if not failures:
        return PlainTextResponse(
            content=json.dumps(results[0][1]),
            media_type="application/json",
            status_code=200,
        )

    failure_map = {worker_ports[i]: r for i, r in failures}
    logger.error(
        "Forward %s: %d/%d workers failed: %s",
        path,
        len(failures),
        len(worker_ports),
        failure_map,
    )
    return PlainTextResponse(
        content=json.dumps(
            {
                "error": "Some workers failed to process request",
                "failures": {str(k): v for k, v in failure_map.items()},
            }
        ),
        media_type="application/json",
        status_code=500,
    )


@router.post("/memory/prefetch")
async def prefetch(request: Request):
    adapter = request.app.state.lmcache_adapter
    if not adapter.lmcache_engine:
        return await _forward_to_all_workers(request, "/memory/prefetch")

    engine: LMCacheEngine = adapter.lmcache_engine
    try:
        body = await request.json()
        chunk_hashes = body.get("chunk_hashes", [])
        lookup_id = body.get("lookup_id")

        if not chunk_hashes or not lookup_id:
            return PlainTextResponse(
                content=json.dumps(
                    {"error": "chunk_hashes and lookup_id are required"}
                ),
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
                pin=False,
            ),
            engine.storage_manager.loop,
        )

        return PlainTextResponse(
            content=json.dumps(
                {
                    "status": "prefetch_started",
                    "lookup_id": lookup_id,
                    "num_chunks": len(keys),
                }
            ),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("prefetch failed: %s", e)
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


@router.post("/memory/evict")
async def evict(request: Request):
    adapter = request.app.state.lmcache_adapter
    if not adapter.lmcache_engine:
        return await _forward_to_all_workers(request, "/memory/evict")

    engine: LMCacheEngine = adapter.lmcache_engine
    try:
        body = await request.json()
        chunk_hashes = body.get("chunk_hashes", [])
        locations = body.get("locations")

        if not chunk_hashes:
            return PlainTextResponse(
                content=json.dumps({"error": "chunk_hashes is required"}),
                media_type="application/json",
                status_code=400,
            )

        keys = _hashes_to_keys(engine, chunk_hashes)
        num_evicted = engine.storage_manager.batched_remove(keys, locations=locations)

        return PlainTextResponse(
            content=json.dumps(
                {
                    "status": "success",
                    "num_evicted": num_evicted,
                }
            ),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("evict failed: %s", e)
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )
