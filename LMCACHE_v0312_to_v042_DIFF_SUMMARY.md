# LMCache v0.3.12 → v0.4.2: Comprehensive Diff Summary

## Overview

- **v0.3.12**: 227 Python files
- **v0.4.2**: 322 Python files (+95 new files)
- **Total changes**: 26,238 insertions, 3,820 deletions across 178 files

---

## 1. CRITICAL: `LMCacheEngineMetadata` → `LMCacheMetadata` (Module Move + Rename)

### What changed
- **v0.3.12**: `lmcache.config.LMCacheEngineMetadata` (in `lmcache/config.py`)
- **v0.4.2**: `lmcache.v1.metadata.LMCacheMetadata` (in `lmcache/v1/metadata.py`)

### Signature changes
```python
# v0.3.12
@dataclass
class LMCacheEngineMetadata:
    model_name: str
    world_size: int
    worker_id: int
    fmt: str                    # REMOVED in v0.4.2
    kv_dtype: torch.dtype
    kv_shape: tuple[int, int, int, int, int]
    use_mla: bool = False
    role: Optional[str] = None
    first_rank = 0
    served_model_name: Optional[str] = None
    chunk_size: int = 256
    kv_layer_groups_manager: KVLayerGroupsManager = field(default_factory=...)

# v0.4.2
@dataclass
class LMCacheMetadata:
    model_name: str
    world_size: int
    local_world_size: int       # NEW
    worker_id: int
    local_worker_id: int        # NEW
    kv_dtype: torch.dtype
    kv_shape: tuple[int, int, int, int, int]
    use_mla: bool = False
    role: Optional[str] = None
    first_rank = 0
    served_model_name: Optional[str] = None
    chunk_size: int = 256
    kv_layer_groups_manager: KVLayerGroupsManager = field(default_factory=...)
    engine_id: Optional[str] = None               # NEW
    kv_connector_extra_config: Optional[dict] = None  # NEW
```

### Key differences
- **`fmt` field REMOVED** - no longer part of metadata
- **`local_world_size` and `local_worker_id` ADDED** - for multi-node support
- **`engine_id` ADDED** - for RPC path identification
- **`kv_connector_extra_config` ADDED** - extra config from KV connector
- Class moved from `lmcache/config.py` to `lmcache/v1/metadata.py`

### Impact on LMCache-Ascend
Every import of `LMCacheEngineMetadata` must change to `LMCacheMetadata` from the new module path. All construction sites must add `local_world_size` and `local_worker_id` parameters and remove `fmt`.

---

## 2. CRITICAL: `CacheEngineKey` changes (in `lmcache/utils.py`)

### What changed
- **`fmt` field REMOVED** from `CacheEngineKey` dataclass
- All `__hash__`, `__eq__`, `to_string`, `from_string`, `split_layers` updated
- `LayerCacheEngineKey` similarly updated - `fmt` removed from constructor

### Signature change
```python
# v0.3.12
@dataclass(slots=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    dtype: torch.dtype
    request_configs: Optional[dict] = None

# v0.4.2
@dataclass(slots=True)
class CacheEngineKey:
    model_name: str           # fmt removed!
    world_size: int
    worker_id: int
    chunk_hash: int
    dtype: torch.dtype
    request_configs: Optional[dict] = None
```

### Key string format change
- **v0.3.12**: `fmt@model@world_size@worker_id@chunk_hash@dtype[@tags...]`
- **v0.4.2**: `model@world_size@worker_id@chunk_hash@dtype[@tags...]`

### New: `EngineType` enum added to `lmcache/utils.py`
```python
class EngineType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    MOCK = "mock"
```

---

## 3. CRITICAL: `gpu_connector` Module Restructured (Single File → Package)

### What changed
- **v0.3.12**: Single file `lmcache/v1/gpu_connector.py`
- **v0.4.2**: Package `lmcache/v1/gpu_connector/` with multiple files:
  - `__init__.py` - `CreateGPUConnector()` factory function
  - `gpu_connectors.py` - All GPU connector classes (same interfaces)
  - `gpu_ops.py` - Helper functions `lmcache_memcpy_async_h2d`, `lmcache_memcpy_async_d2h`
  - `utils.py` - GPU KV format discovery utilities (NEW)
  - `xpu_connectors.py` - XPU connector (was `xpu_connector.py`)
  - `mock_gpu_connector.py` - Mock connector (was `mock_gpu_connector.py`)

### Import path changes
```python
# v0.3.12
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.v1.gpu_connector import SGLangGPUConnector
# etc.

# v0.4.2
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.gpu_connector.gpu_connectors import SGLangGPUConnector
# OR via __init__.py:
from lmcache.v1.gpu_connector import GPUConnectorInterface  # re-exported
from lmcache.v1.gpu_connector import CreateGPUConnector       # NEW factory
```

### New: `CreateGPUConnector` factory function
```python
def CreateGPUConnector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,   # NOTE: uses new LMCacheMetadata
    engine: EngineType
) -> GPUConnectorInterface:
```

### New: GPU KV Format discovery system (`utils.py`)
- `discover_gpu_kv_format(kv_caches, serving_engine)` - auto-detects KV format
- `get_page_buffer_size()`, `get_num_blocks()`, `get_block_size()`, etc.
- `assert_layerwise_gpu_connector()` utility
- `need_gpu_interm_buffer()` utility
- Uses C++ enum `lmc_ops.GPUKVFormat` (new in c_ops)

### Connector class signature changes
- `from_metadata()` now takes `LMCacheMetadata` instead of `LMCacheEngineMetadata`
- `batched_to_gpu()` signature loosened in interface:
  ```python
  # v0.3.12
  def batched_to_gpu(self, memory_objs, starts, ends, **kwargs)
  # v0.4.2
  def batched_to_gpu(self, memory_objs=None, starts=None, ends=None, **kwargs)
  ```

### New: `gpu_ops.py` helper functions
```python
def lmcache_memcpy_async_h2d(memory_obj: MemoryObj, gpu_buffer: torch.Tensor)
def lmcache_memcpy_async_d2h(gpu_buffer: torch.Tensor, memory_obj: MemoryObj)
```
These handle `LazyMemoryAllocator` specially.

---

## 4. CRITICAL: `xpu_connector.py` moved to `gpu_connector/xpu_connectors.py`

### What changed
- **v0.3.12**: `lmcache/v1/xpu_connector.py` (now REMOVED)
- **v0.4.2**: `lmcache/v1/gpu_connector/xpu_connectors.py`
- Class `VLLMPagedMemXPUConnectorV2` signature updated to use `LMCacheMetadata`

---

## 5. `mock_gpu_connector.py` moved into package

- **v0.3.12**: `lmcache/v1/mock_gpu_connector.py`
- **v0.4.2**: `lmcache/v1/gpu_connector/mock_gpu_connector.py`
- Import changes from `from lmcache.v1.gpu_connector import GPUConnectorInterface`

---

## 6. `cache_engine.py` Changes

### Import changes
```python
# v0.3.12
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemLayerwiseGPUConnector,
)

# v0.4.2
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.metadata import LMCacheMetadata
```

### Constructor signature change
```python
# v0.3.12
class LMCacheEngine:
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        ...
    )

# v0.4.2
class LMCacheEngine:
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,        # Changed type
        ...
    )
```

### Layerwise connector type checking
```python
# v0.3.12: explicit isinstance check
assert isinstance(
    self.gpu_connector,
    (VLLMPagedMemLayerwiseGPUConnector, VLLMBufferLayerwiseGPUConnector,
     SGLangLayerwiseGPUConnector),
)

# v0.4.2: uses utility function
assert_layerwise_gpu_connector(self.gpu_connector)
```

### New: `HealthMonitor` integration (TYPE_CHECKING import)
```python
if TYPE_CHECKING:
    from lmcache.v1.health_monitor.base import HealthMonitor
```

---

## 7. New Module: `lmcache/v1/manager.py`

This is a **new unified manager** that decouples vLLM adapter from LMCache internals:
- `LMCacheManager` class with lifecycle management
- Handles creation of `LMCacheEngine`, `GPUConnector`, lookup clients, offload servers
- Imports from `lmcache.v1.gpu_connector import CreateGPUConnector`
- Uses `LMCacheMetadata` throughout
- New `HealthMonitor` integration

---

## 8. `token_database.py` Changes

### Import change
```python
# v0.3.12
from lmcache.config import LMCacheEngineMetadata
# v0.4.2
from lmcache.v1.metadata import LMCacheMetadata
```

### NONE_HASH initialization changed
```python
# v0.3.12: NONE_HASH dynamically initialized in __init__
NONE_HASH: int  # declared as type annotation

# v0.4.2: NONE_HASH = 0  # static default
NONE_HASH = 0
```

### TokenDatabase `__init__` uses `LMCacheMetadata`
```python
# v0.3.12
def __init__(self, config=None, metadata: Optional[LMCacheEngineMetadata]=None)
# v0.4.2
def __init__(self, config=None, metadata: Optional[LMCacheMetadata]=None)
```

### `_make_key_by_hash` no longer passes `fmt`
```python
# v0.3.12
return CacheEngineKey(
    self.metadata.fmt,
    self.metadata.model_name,
    ...
)

# v0.4.2
return CacheEngineKey(
    self.metadata.model_name,   # fmt removed
    ...
)
```

---

## 9. `memory_management.py` Changes

### `MemoryObj.parent()` method added
```python
# v0.4.2 adds:
def parent(self) -> Optional["MemoryAllocatorInterface"]:
    """Get the parent allocator."""
```
This is used by `gpu_ops.py` to check if memory was allocated by `LazyMemoryAllocator`.

### Other changes are minimal - the core allocator interfaces remain the same.

---

## 10. Config Changes (`v1/config.py`)

### New config options in v0.4.2
| Config Key | Type | Default | Description |
|---|---|---|---|
| `retrieve_locations` | `Optional[list[str]]` | `None` | Locations for retrieve operations |
| `store_location` | `Optional[str]` | `None` | Location for store operations |
| `remote_storage_plugins` | `Optional[list[str]]` | `None` | Remote storage plugins |
| `min_retrieve_tokens` | `int` | `0` | Min hit tokens to perform retrieve |
| `remote_config_url` | `Optional[str]` | `None` | Remote config service URL |
| `app_id` | `Optional[str]` | `None` | App ID for remote config |

### Removed validation
- PD mode no longer asserts `remote_url is None`
- PD mode no longer asserts `save_decode_cache is False`

### New validation
- `min_retrieve_tokens` must be >= 0
- PD receiver: `store_location != "PDBackend"`, `retrieve_locations in (None, ["PDBackend"])`

### `_resolve_config_aliases` signature changed
```python
# v0.3.12 (called with 2 args)
_resolve_config_aliases(env_config, "environment variables")

# v0.4.2 (called with 5 args - matching config_base.py signature)
_resolve_config_aliases(env_config, "environment variables",
    _CONFIG_DEFINITIONS, _CONFIG_ALIASES, _DEPRECATED_CONFIGS)
```

### `_update_config_from_env` tracks user-set keys
v0.4.2 adds `_user_set_keys` set to track which config keys were explicitly set via env vars.

### `_validate_and_set_config_value` REMOVED

---

## 11. `rpc_utils.py` Changes

### `TYPE_CHECKING` import removed
```python
# v0.3.12
from typing import TYPE_CHECKING, Literal, Optional
if TYPE_CHECKING:
    from vllm.config import VllmConfig

# v0.4.2
from typing import Literal, Optional
# No TYPE_CHECKING block
```

### `get_zmq_rpc_path_lmcache` function simplified
The function signature and behavior changed to support engine_id-based paths.

---

## 12. New RPC Transport Layer: `lmcache/v1/rpc/`

Entirely new module:
- `transport.py` - `RpcClientTransport`, `RpcServerTransport` abstract interfaces
- `zmq_transport.py` - `ZmqReqRepClientTransport`, `ZmqRouterServerTransport`

---

## 13. `storage_backend/__init__.py` Changes

### Uses `LMCacheMetadata` instead of `LMCacheEngineMetadata`
All `CreateStorageBackends` and related functions updated.

### New parameters
- `skip_backends: Optional[AbstractSet[str]]` - skip specific backends
- `existing_backends: Optional[OrderedDict]` - reuse existing backends

### XPU support added
```python
elif dst_device == "xpu":
    dst_device = f"xpu:{torch.xpu.current_device()}"
```

---

## 14. New Modules in v0.4.2 Not Present in v0.3.12

### `lmcache/v1/distributed/` (28 new files)
Entirely new distributed caching subsystem:
- `l1_manager.py` - L1 cache manager
- `l2_adapters/` - L2 cache adapter pattern (fs, mock, native_connector, nixl_store, plugin)
- `storage_controller.py`, `storage_controllers/` - eviction, prefetch, store controllers
- `memory_manager.py` - distributed memory management
- `eviction.py`, `eviction_policy/` - LRU, noop eviction policies
- `config.py`, `error.py`, `api.py`, `internal_api.py`

### `lmcache/v1/health_monitor/` (4 new files)
Health monitoring subsystem:
- `base.py` - `HealthMonitor` class
- `checks/remote_backend_check.py` - remote backend health checks
- `constants.py` - ping interval defaults

### `lmcache/v1/mp_observability/` (14 new files)
Multiprocess observability:
- Prometheus metrics, stats loggers, telemetry system
- `config.py`, `prometheus_controller.py`
- Stats: `l1_stats`, `mp_server_stats`, `storage_manager_stats`, `vllm_integrator_stats`
- Telemetry: event-based tracing system

### `lmcache/v1/multiprocess/` new files
- `blend_server.py`, `blend_server_v2.py` - blend server implementations
- `config.py` - `MPServerConfig`, `HTTPFrontendConfig`
- `gpu_context.py` - GPU context management
- `http_server.py` - HTTP frontend
- `protocols/` - message protocol definitions (blend, controller, debug, engine)
- `session.py` - session management
- `token_hasher.py` - token hashing

### `lmcache/v1/multiprocess/mp_storage_manager.py` REMOVED

### `lmcache/v1/exceptions/` (new)
New exceptions package.

### `lmcache/v1/periodic_thread.py` (new)
Utility for periodic background threads.

### `lmcache/v1/standalone/manager.py` (new)
Standalone mode manager.

### `lmcache/integration/request_telemetry/` (5 new files)
Request telemetry integration:
- `base.py`, `factory.py`, `fastapi.py`, `noop.py`

### `lmcache/integration/vllm/vllm_multi_process_adapter.py` (new, 843 lines)
New multiprocess adapter for vLLM integration.

### `lmcache/v1/storage_backend/native_clients/` (new)
- `connector_client_base.py` - base class for native connector clients
- `resp_client.py` - RESP protocol client

### `lmcache/v1/storage_backend/resp_client.py` (new)
RESP client for storage backends.

### `lmcache/v1/storage_backend/plugins/rust_raw_block_backend.py` (new, 838 lines)
Rust-based raw block backend plugin.

### `lmcache/v1/utils/cache_utils.py` (new)
Cache utility functions.

---

## 15. Removed Files in v0.4.2

| File | Notes |
|---|---|
| `lmcache/v1/gpu_connector.py` | Replaced by `gpu_connector/` package |
| `lmcache/v1/xpu_connector.py` | Moved to `gpu_connector/xpu_connectors.py` |
| `lmcache/v1/mock_gpu_connector.py` | Moved to `gpu_connector/mock_gpu_connector.py` |
| `lmcache/v1/multiprocess/mp_storage_manager.py` | Removed |
| `lmcache/v1/storage_backend/full_sync_sender.py` | Moved to `cache_controller/` |
| `lmcache/v1/storage_backend/remote_monitor.py` | Removed |

---

## 16. `lmcache/integration/vllm/utils.py` Changes

### `create_lmcache_metadata` now returns `LMCacheMetadata`
```python
# v0.3.12: returns LMCacheEngineMetadata
metadata = LMCacheEngineMetadata(
    model_cfg.model, parallel_cfg.world_size, parallel_cfg.rank,
    "vllm", kv_dtype, kv_shape, use_mla, role,
    served_model_name=...
)

# v0.4.2: returns LMCacheMetadata with keyword args
metadata = LMCacheMetadata(
    model_name=model_cfg.model,
    world_size=parallel_cfg.world_size,
    local_world_size=parallel_cfg.world_size,
    worker_id=parallel_cfg.rank,
    local_worker_id=parallel_cfg.rank,
    kv_dtype=kv_dtype,
    kv_shape=kv_shape,
    use_mla=use_mla,
    role=role,
    served_model_name=...,
    engine_id=engine_id,
    kv_connector_extra_config=kv_connector_extra_config,
)
```

### New functions added
- `get_vllm_torch_dev()` - returns `(torch_dev, dev_name)` for CUDA/XPU
- `calculate_local_rank_and_world_size(vllm_config)` - multi-node support
- `hex_hash_to_int16()` - now handles arbitrary strings (not just hex)

### Remote config integration
- Calls `fetch_remote_config()` and `apply_remote_configs()` from `config_base`

---

## 17. `lmcache/integration/vllm/vllm_v1_adapter.py` Changes

Major refactoring (727 insertions, many deletions). The adapter is simplified, with logic moved to `LMCacheManager`.

---

## 18. `lmcache/integration/sglang/sglang_adapter.py` Changes

Updated to use `LMCacheMetadata` and new import paths. Uses `EngineType` enum.

---

## 19. `lmcache/utils.py` - `CacheStoreEvent` Changes

```python
# v0.4.2 adds:
lora_name: str | None     # NEW field
# lora_id retained for backwards compat but deprecated
```

### FP8 dtype string changes
```python
# v0.3.12
torch.float8_e4m3fn   → "fp8_e4m3"
torch.float8_e4m3fnuz → "fp8_e4m3"
torch.float8_e5m2fnuz → "fp8_e5m2"

# v0.4.2 (more specific)
torch.float8_e4m3fn   → "fp8_e4m3fn"
torch.float8_e4m3fnuz → "fp8_e4m3fnuz"
torch.float8_e5m2fnuz → "fp8_e5m2fnuz"
```

---

## 20. `system_detection.py` - No Changes

The file is identical between v0.3.12 and v0.4.2.

---

## Summary: What LMCache-Ascend Must Change

### Must-change items (breaking):

1. **Replace all `LMCacheEngineMetadata` → `LMCacheMetadata`**
   - Change import from `lmcache.config` to `lmcache.v1.metadata`
   - Add `local_world_size` and `local_worker_id` parameters
   - Remove `fmt` parameter from constructor calls

2. **Replace all `CacheEngineKey` constructors** - remove `fmt` parameter

3. **Update GPU connector imports**:
   - `from lmcache.v1.gpu_connector import X` → `from lmcache.v1.gpu_connector.gpu_connectors import X`
   - Or use `from lmcache.v1.gpu_connector import GPUConnectorInterface` (re-exported)
   - XPU connector: `from lmcache.v1.xpu_connector` → `from lmcache.v1.gpu_connector.xpu_connectors`
   - Mock: `from lmcache.v1.mock_gpu_connector` → `from lmcache.v1.gpu_connector.mock_gpu_connector`

4. **Use new factory function** `CreateGPUConnector(config, metadata, engine_type)` if applicable

5. **Use `EngineType` enum** from `lmcache.utils`

6. **Update `_resolve_config_aliases` calls** to pass 5 arguments

### Should-change items (recommended):

7. Use `assert_layerwise_gpu_connector()` instead of manual isinstance checks
8. Use `need_gpu_interm_buffer()` from `gpu_connector.utils`
9. Consider using `LMCacheManager` for lifecycle management
10. Update FP8 dtype string mappings if used
11. Use `gpu_ops.py` helpers for lazy memory allocator support
12. Consider `discover_gpu_kv_format()` for KV format auto-detection

### New capabilities to potentially leverage:

13. Distributed caching (`lmcache/v1/distributed/`)
14. Health monitoring (`lmcache/v1/health_monitor/`)
15. Multiprocess observability and Prometheus metrics
16. Remote config service integration
17. `retrieve_locations` / `store_location` config for asymmetric PD
18. Rust raw block backend plugin
