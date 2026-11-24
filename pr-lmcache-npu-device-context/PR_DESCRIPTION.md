# PR: Fix NPU IPC device-context races, exported-event leak, and idle-worker liveness reaping

**Repository**: LMCache (`dev` branch)
**Fixed images**: `hub.galaxy.ksyun.com/ai/lmcache-server:v3.1.2` (IPC/leak fixes), `v3.1.3` (heartbeat fix)
**Verified on**: DeepSeek-V4.1 inference service (10.184.57.8 / node-1)
**Logs**: `/data/yangfan/lmcache/dsv41_lmcache_mp/logs/`

---

## 1. Environment

### 1.1 Hardware

| Item | Value |
|---|---|
| NPU | 16 × Ascend 910B2C (64GB HBM per card) |
| Driver (HDK) | 26.1.1 (npu-smi 26.1.1, ascendhal 7.35.23) |
| OS | openEuler 24.03 SP3 (kernel 6.6.0-132.0.0.111.oe2403sp3) |

### 1.2 Software

| Component | Version |
|---|---|
| Image | hub.galaxy.ksyun.com/ai/lmcache-server:v3.1.1 (affected) → v3.1.2 / v3.1.3 (fixed) |
| LMCache | v0.5.7 (g95656e5c, /workspace/LMCache in image) |
| vLLM | 0.30.0 |
| vllm-ascend | 0.30.0 series (/vllm-workspace/vllm-ascend) |
| CANN | 9.2.1 (/usr/local/Ascend/cann-9.2.1 in image) |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| Python | 3.12.16 |

### 1.3 Model and deployment

- Model: DeepSeek-V4.1-Flash-1017-w8a8 (VL), max_model_len 1048576, quantization `ascend`
- Parallelism: TP=8 × DP=2 (single pod, 16 cards), expert parallelism on, block_size 32
- Speculative decoding: `dspark`, num_speculative_tokens 5, enforce_eager
- Engram: cpu_offload on, dp_shared_memory off
- KV transfer: LMCacheMPConnector (kv_role=kv_both, `lmcache_driven` mode)

### 1.4 Key launch commands

vLLM (reconstructed from the `non-default args` line in the startup log; exact flag
spelling depends on the launch script):

```bash
vllm serve /data/models/DeepSeek-V4.1-Flash-1017-w8a8 \
  --host 0.0.0.0 --port 13900 \
  --served-model-name deepseek-v41-cache \
  --tokenizer-mode deepseek_v41 --trust-remote-code \
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --max-model-len 1048576 \
  --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel \
  --block-size 32 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --async-scheduling \
  --quantization ascend --safetensors-load-strategy lazy \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": true}' \
  --engram-config '{"cpu_offload": true, "embedding_across_dp": false, "dp_shared_memory": false}' \
  --kv-transfer-config '{"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both", \
    "kv_buffer_device": "npu", "kv_connector_extra_config": \
    {"lmcache.mp.host": "10.184.57.8", "lmcache.mp.port": 5555, \
     "lmcache.mp.mp_transfer_mode": "lmcache_driven"}}'
```

lmcache-server (DaemonSet, hostNetwork, co-located with vLLM):

```bash
lmcache server --host ${NODE_IP} \
  --chunk-size 896 \
  --supported-transfer-mode lmcache_driven \
  --l1-size-gb 120 \
  --eviction-policy LRU --separate-object-groups \
  --disable-metrics --http-port 8090
```

---

## 2. Bug reports

### 2.1 Bug A: REGISTER_KV_CACHE fails with 507899, cascading crash of the whole pod

**Timeline (2026-11-20)**

| Time | Event |
|---|---|
| 01:37 | 16 workers send REGISTER_KV_CACHE simultaneously |
| 01:37→01:42 | no response from the server for 300 s |
| 01:42 | DP0 dies: `LMCache server did not respond to register_kv_caches within 300.0s` |
| 01:45 | the server finally reaches the request and fails with 507899 |
| 02:08 | DP1 stuck in a cross-DP AllGather for 1836 s → HCCL fftsplus timeout → every rank ERR02005 → crash |
| 09:42 | lmcache-server HDC session disconnected |

**Server-side traceback (register path)**

```
File ".../lmcache_driven_transfer.py", line 503, in register_kv_cache
  cache_context = self._create_cache_context(...)
File ".../devices/npu/cache_context.py", line 288, in _initialize
  [wrapper.to_tensor() for wrapper in kv_caches]
File ".../devices/npu/ipc_wrapper.py", line 177, in to_tensor
  storage = torch.UntypedStorage._new_shared_npu(...)
RuntimeError: NPUEvent: NPUEvent.cpp:39 NPU function error:
  acl::AclIpcOpenEventHandle(*handle, &event_), error code is 507899
EL9999: halMemImportFromShareableHandle failed, drvRetCode=42, drvDevId=0
```

**Server-side log (store path, same root cause)**

```
torch.OutOfMemoryError: createEvent: NPUEvent.cpp:266 NPU function error:
  AclrtCreateEventWithFlag(&event_, flags_), error code is 207001
Execution_Error_Invalid_Context(EE1010): Adding a host callback function to the
  stream task queue execution failed because stream does not belong to the
  current context. stream_ctx=0x7ee703fae4e0, cur_ctx=0x449302c0
```

**Root cause**

CANN's `AclIpcOpenEventHandle` / `AclrtCreateEventWithFlag` are bound to the
calling thread's current device context, not to the device passed as an
argument:

1. The server MQ thread serves registration requests for 16 devices in turn.
   The handle was exported by a worker on npu:4, but the server opens it under
   device 0's context (drvDevId=0 ≠ npu:4) → 507899.
2. On the store path, `event.record(stream)` lazily creates the driver event
   under the wrong context (EE1010; the event lands in another device's
   resource pool).
3. The exported-event liveness ring `deque(maxlen=2048)` accumulates without
   bound (a CUDA-era default). The per-card CANN event pool (hardware notify
   slots) is far smaller than 2048 → `createEvent` runs the pool dry → 207001.

**Cascade**: the server raises without replying → every worker times out after
300 s and dies → surviving ranks block in collectives → HCCL times out after
1836 s and kills all ranks → the driver session is corrupted.

### 2.2 Bug B: idle workers are reaped; caching silently stops

**Timeline (2026-11-23)**

| Time | Event |
|---|---|
| 08:32 | 16 instances register successfully (`Registered KV cache ... 54 layers`) |
| 08:32→09:32 | pod idle, zero traffic |
| 09:32 | all 16 instances reaped at once: `Reaped GPU instance <id>: silent for 3621.0s (pinged=False)` |
| after | service stays up, but stores are rejected: `Rejecting STORE for unregistered GPU instance ID`; caching silently stops and never recovers |

**Root cause**

The worker's `HeartbeatThread` (a PING every 10 s that refreshes the
server-side `last_seen`) is started lazily — only on the first
store/retrieve/event-poll, never at registration:

1. An idle worker sends no PING → it exceeds
   `worker_registration_grace_seconds=3600s` (the "never pinged" window) →
   it is reaped.
2. There is no self-healing: re-registration is hooked onto the heartbeat's
   unhealthy→healthy edge (the server-restart path), while the server's ping
   always returns True → the recovery edge never fires.

---

## 3. Fixes

### Fix 1: open IPC handles under the owning device context (fixes 507899)

`lmcache/v1/platform/devices/npu/ipc_wrapper.py`

Add a `_npu_device_ctx(device_index)` guard and wrap every `_new_shared_npu`
handle open in `to_tensor()` with it:

```python
def _npu_device_ctx(device_index: Any) -> AbstractContextManager:
    """Switch the calling thread's device context for the NPU backend. ..."""
    npu_module = getattr(torch, "npu", None)
    if npu_module is not None and hasattr(npu_module, "device"):
        return npu_module.device(device_index)
    return contextlib.nullcontext()

# inside to_tensor():
with _npu_device_ctx(device_index):
    storage = torch.UntypedStorage._new_shared_npu(device_index, *handle[1:])
```

### Fix 2: pin every event IPC op to its device (fixes EE1010 / 207001 mixing)

`lmcache/v1/platform/devices/npu/event_ipc.py`

`import_event` (which internally also calls `AclIpcOpenEventHandle`) and new
`record_event` / `wait_event` overrides run under the owning device's context:

```python
def record_event(self, event: object, stream: object) -> None:
    with self._event_module.device(stream.device):
        super().record_event(event, stream)
```

### Fix 3: shrink the exported-event ring and settle before eviction (fixes 207001 leak)

`lmcache/v1/platform/base/event_ipc.py` + `lmcache/v1/platform/devices/npu/event_ipc.py`

- Replace the unbounded `deque(maxlen=2048)` keep-alive ring with explicit
  eviction: an `exported_event_ring_size` parameter, `_retain_exported_event()`
  and an `_on_event_evicted()` hook.
- The NPU backend defaults to 128 (tunable via `LMCACHE_NPU_EVENT_RING_SIZE`)
  and calls `event.synchronize()` before dropping the last reference on
  eviction, so `aclrtDestroyEvent` reclaims deterministically.
- Rationale: CUDA events are cheap and 2048 is fine there; NPU events occupy
  hardware notify slots and the pool is far smaller than 2048. The steady-state
  in-flight window is milliseconds wide and well below 128, so behavior is
  unchanged in practice.

### Fix 4: release resources per device (closes a secondary leak)

`lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`

`_release_entries` now calls `close()` with each entry's device current and
runs `empty_cache()` / `ipc_collect()` on every released device (the old code
only reclaimed the ambient device).

### Fix 5: start the heartbeat at registration (fixes idle-worker reaping)

`lmcache/integration/vllm/vllm_multi_process_adapter.py`

```python
self._send_register_kv_caches_request(kv_caches)
# Start the heartbeat on registration: a worker that goes idle before
# its first store/retrieve would otherwise send no PING and be reaped
# after the registration grace period.
self._ensure_heartbeat_started()
```

Benefit: a live worker pings every 10 s and is never reaped; a dead worker is
cleaned up after 120 s (`worker_reap_timeout_seconds`) instead of 3600 s.

---

## 4. Tests and verification

### New tests

| File | Case | Covers |
|---|---|---|
| tests/v1/platform/devices/npu/test_npu_ipc_wrapper.py | test_to_tensor_switches_device_context_per_handle | every handle opened under its owning device context |
| tests/v1/platform/test_event_ipc.py | test_npu_backend_import_event_pins_owning_device, etc. | import/record/wait pinned to device; ring eviction and settle |
| tests/v1/test_mp_connector_kv_roles.py | test_register_kv_caches_starts_heartbeat | registration constructs and starts the HeartbeatThread |

In-image result: **101 passed, 1 skipped** (42 platform + 59 kv_roles).

### On-cluster verification (v3.1.2 / v3.1.3)

| Check | Result |
|---|---|
| REGISTER_KV_CACHE | succeeds in seconds for all 16 ranks (`Registered KV cache ... 54 layers`) |
| 507899 / 207001 / EE1010 | gone |
| Instances alive after 1 h idle | to be re-verified (expected: no more `Reaped GPU instance`) |
| Service stability | no crashes |

---

## 5. Impact and compatibility

- Behavior change 1: the NPU exported-event keep-alive window drops from 2048
  to 128. Set `LMCACHE_NPU_EVENT_RING_SIZE=2048` to restore the old behavior
  (re-opens the 207001 risk; not recommended).
- Behavior change 2: workers send a 10 s periodic heartbeat right after
  registration (one PING request per worker per 10 s — negligible overhead).
- CUDA and other backends are unaffected (the guards live in the NPU device
  layer; the default ring size is unchanged).
