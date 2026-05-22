# Prefetch & Evict APIs

LMCache-Ascend exposes two internal REST endpoints to control KV cache data placement
across storage tiers:

| Endpoint | Method | Purpose |
|---|---|---|
| `/memory/prefetch` | POST | Pre-load KV chunks from slow tiers (SSD / P2P) into DDR |
| `/memory/evict` | POST | Remove KV chunks from a specified storage tier |

These APIs require several configuration options to be enabled. This document
explains each option and how to use the endpoints.

---

## Required Configuration

All of the following must be enabled for the APIs to work:

### 1. `internal_api_server_enabled: true`

**What it does**: Starts LMCache's built-in HTTP server on each worker / scheduler.
Without it, no internal API endpoints exist at all.

**YAML**:
```yaml
internal_api_server_enabled: true
```

**Env**: `LMCACHE_INTERNAL_API_SERVER_ENABLED=true`

This is an upstream LMCache config. Default port start is `6999` (scheduler),
then `7000`, `7001`, ... for workers.

---

### 2. `enable_chunk_hashes_return: true`

**What it does**: Two-fold:

- Includes `chunk_hashes` in the `kv_transfer_params` of each response, so that
  callers can know which chunks were looked up for a request.
- Registers the `/memory/prefetch` and `/memory/evict` routes on the internal
  API server.

**YAML**:
```yaml
enable_chunk_hashes_return: true
```

**Env**: `LMCACHE_ENABLE_CHUNK_HASHES_RETURN=true`

**Default**: `false` (no impact on existing functionality)

---

### 3. `enable_async_loading: true`

**What it does**: Enables the asynchronous lookup-and-prefetch pipeline. The
`/memory/prefetch` endpoint calls `async_lookup_and_prefetch` internally, which
requires the async loading infrastructure (async serializer, event loop, etc.).

**YAML**:
```yaml
enable_async_loading: true
```

**Env**: `LMCACHE_ENABLE_ASYNC_LOADING=true`

This is an upstream LMCache config.

---

### 4. `lookup_hashes_cache_size` (recommended)

**What it does**: Controls the in-memory cache of `chunk_hashes` per request.
Each call to `lookup()` records the computed chunk hashes keyed by `lookup_id`.
When `request_finished` fires, the cached hashes are popped and returned via
`kv_transfer_params.chunk_hashes`.

Without a size limit, this cache would grow unboundedly if the upper layer
never calls `get_cached_hashes`.

**YAML**:
```yaml
lookup_hashes_cache_size: 1024
```

**Env**: `LMCACHE_LOOKUP_HASHES_CACHE_SIZE=1024`

**Default**: `0` (unlimited -- use `> 0` for production)

FIFO eviction: when the cache reaches the limit, the oldest entry is removed
before inserting a new one.

---

## Configuration Summary

Minimal YAML snippet:

```yaml
# Required
internal_api_server_enabled: true
enable_async_loading: true
enable_chunk_hashes_return: true

# Recommended
lookup_hashes_cache_size: 1024
```

---

## Usage

All endpoints are served on the internal API server (default scheduler port `6999`).

### Prefetch (SSD / P2P -> DDR)

Load KV chunks from slower storage tiers into CPU DDR so that subsequent
requests hit the hot cache.

**Endpoint**: `POST http://<host>:6999/memory/prefetch`

**Headers**: `Content-Type: application/json`

**Body**:
```json
{
  "chunk_hashes": ["abc123", "def456"],
  "lookup_id": "prefetch_test_001"
}
```

| Field | Type | Description |
|---|---|---|
| `chunk_hashes` | `string[]` | Hex-encoded chunk hashes (from `kv_transfer_params.chunk_hashes`) |
| `lookup_id` | `string` | A unique identifier for this prefetch session |

**Success response** (200):
```json
{
  "status": "prefetch_started",
  "lookup_id": "prefetch_test_001",
  "num_chunks": 2
}
```

**Notes**:
- The prefetch runs asynchronously. The response returns immediately after
  scheduling; actual disk reads happen in the background.
- Use the `lookup_id` to track completion (future enhancement).
- When called on the scheduler, the request is forwarded to all workers.

---

### Evict

Remove KV chunks from a specific storage tier.

**Endpoint**: `POST http://<host>:6999/memory/evict`

**Headers**: `Content-Type: application/json`

**Body**:
```json
{
  "chunk_hashes": ["abc123", "def456"],
  "locations": ["LocalCPUBackend"]
}
```

| Field | Type | Description |
|---|---|---|
| `chunk_hashes` | `string[]` | Hex-encoded chunk hashes to evict |
| `locations` | `string[]` (optional) | Storage tier name. `null` to evict from all tiers. Common value: `"LocalCPUBackend"` |

**Success response** (200):
```json
{
  "status": "success",
  "num_evicted": 2
}
```

**Notes**:
- Eviction is synchronous and blocks until complete.
- When called on the scheduler, the request is forwarded to all workers.

---

## Streaming Response Support

The `chunk_hashes` are delivered via `kv_transfer_params` in the final (finish)
chunk of the streaming response. This requires modifications to the vllm /
vllm-ascend serving layer.

### Data Flow

```
LMCache-Ascend vllm_v1_adapter.py
  request_finished() -> return_params = {"chunk_hashes": [...]}

    v
vllm scheduler.py
  _free_request() -> _connector_finished()
  -> kv_transfer_params = return value (dict)

    v
EngineCoreOutput.kv_transfer_params

    v
output_processor.py -> RequestOutput.kv_transfer_params

    v
Serving layer -> ChatCompletionStreamResponse / CompletionStreamResponse
```

### Required Modifications

vllm-ascend: `patch_glm_tool_call_parser.py` (already applied)

In the streaming loop, pass `kv_transfer_params` to the final chunk:

```python
# File: vllm-ascend/vllm_ascend/patch/platform/patch_glm_tool_call_parser.py

chunk = ChatCompletionStreamResponse(
    id=request_id,
    object=chunk_object_type,
    created=created_time,
    choices=[choice_data],
    model=model_name,
    kv_transfer_params=res.kv_transfer_params if finish_reason_ is not None else None,
)
```

**Note**: Only the last chunk (where `finish_reason_` is set) carries the
`kv_transfer_params`, avoiding unnecessary data in intermediate chunks.


