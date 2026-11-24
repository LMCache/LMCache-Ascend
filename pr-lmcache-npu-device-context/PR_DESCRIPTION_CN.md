# PR: 修复 NPU IPC device context 串扰、exported event 泄漏与 worker 空闲误回收

**仓库**: LMCache（dev 分支）
**修复镜像**: `hub.galaxy.ksyun.com/ai/lmcache-server:v3.1.2`（IPC/泄漏修复）、`v3.1.3`（心跳修复）
**实测环境**: DeepSeek-V4.1 推理服务（10.184.57.8 / node-1）
**日志**: `/data/yangfan/lmcache/dsv41_lmcache_mp/logs/`

---

## 1. 环境信息

### 1.1 硬件

| 项 | 值 |
|---|---|
| NPU | 16 × Ascend 910B2C（每卡 64GB HBM） |
| 驱动 (HDK) | 26.1.1（npu-smi 26.1.1，ascendhal 7.35.23） |
| OS | openEuler 24.03 SP3（kernel 6.6.0-132.0.0.111.oe2403sp3，同集群实测） |

### 1.2 软件

| 组件 | 版本 |
|---|---|
| 镜像 | hub.galaxy.ksyun.com/ai/lmcache-server:v3.1.1（问题版本）→ v3.1.2 / v3.1.3（修复版本） |
| LMCache | v0.5.7（g95656e5c，镜像内 /workspace/LMCache） |
| vLLM | 0.30.0 |
| vllm-ascend | 0.30.0 系列（/vllm-workspace/vllm-ascend） |
| CANN | 9.2.1（镜像内 /usr/local/Ascend/cann-9.2.1） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| Python | 3.12.16 |

### 1.3 模型与部署

- 模型: DeepSeek-V4.1-Flash-1017-w8a8（VL），max_model_len 1048576，quantization ascend
- 并行: TP=8 × DP=2（单 pod 16 卡），EP 开启，block_size 32
- 投机解码: dspark，num_speculative_tokens 5，enforce_eager
- Engram: cpu_offload 开启，dp_shared_memory 关闭
- KV 传输: LMCacheMPConnector（kv_role=kv_both，lmcache_driven 模式）

### 1.4 重要启动命令

vLLM（按启动日志 non-default args 还原，flag 名以实际启动脚本为准）：

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
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --quantization ascend --safetensors-load-strategy lazy \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": true}' \
  --engram-config '{"cpu_offload": true, "embedding_across_dp": false, "dp_shared_memory": false}' \
  --kv-transfer-config '{"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both", \
    "kv_buffer_device": "npu", "kv_connector_extra_config": \
    {"lmcache.mp.host": "10.184.57.8", "lmcache.mp.port": 5555, \
     "lmcache.mp.mp_transfer_mode": "lmcache_driven"}}'
```

lmcache-server（DaemonSet，hostNetwork，与 vLLM 同节点）：

```bash
lmcache server --host ${NODE_IP} \
  --chunk-size 896 \
  --supported-transfer-mode lmcache_driven \
  --l1-size-gb 120 \
  --eviction-policy LRU --separate-object-groups \
  --disable-metrics --http-port 8090
```

---

## 2. 报错信息

### 2.1 问题 A：REGISTER_KV_CACHE 报 507899，全组级联崩溃

**现象时间线（2026-11-20）**

| 时间 | 事件 |
|---|---|
| 01:37 | 16 个 worker 同时发 REGISTER_KV_CACHE |
| 01:37→01:42 | server 300 秒无应答 |
| 01:42 | DP0 全灭：`LMCache server did not respond to register_kv_caches within 300.0s` |
| 01:45 | server 才处理到请求并报 507899 |
| 02:08 | DP1 卡在跨 DP AllGather 1836 秒，HCCL fftsplus timeout，全 rank ERR02005 崩溃 |
| 09:42 | lmcache-server HDC session disconnected |

**报错日志（server 端，register 路径）**

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

**报错日志（server 端，store 路径，同源表现）**

```
torch.OutOfMemoryError: createEvent: NPUEvent.cpp:266 NPU function error:
  AclrtCreateEventWithFlag(&event_, flags_), error code is 207001
Execution_Error_Invalid_Context(EE1010): Adding a host callback function to the
  stream task queue execution failed because stream does not belong to the
  current context. stream_ctx=0x7ee703fae4e0, cur_ctx=0x449302c0
```

**根因**

CANN 的 `AclIpcOpenEventHandle` / `AclrtCreateEventWithFlag` 绑定调用线程的
current device context，而非显式传参的 device：

1. server MQ 线程交替服务 16 个 device 的注册请求，句柄是 worker 在 npu:4 导出的，
   server 却在 device 0 的 context 下打开它（drvDevId=0 ≠ npu:4）→ 507899。
2. store 路径的 `event.record(stream)` 在错误 context 下懒创建 event（EE1010，
   event 落入别的 device 的资源池）。
3. 导出 event 保活环 `deque(maxlen=2048)` 无界囤积（CUDA 习惯值），CANN 每卡
   event 池（硬件 notify slot）远小于 2048 → `createEvent` 耗尽报 207001。

**级联**: server 抛异常不回应答 → worker 300s 超时全灭 → 存活 rank 卡集合通信 →
HCCL 1836s 超时全杀 → 驱动会话损坏。

### 2.2 问题 B：空闲 worker 被误回收，缓存静默失效

**现象时间线（2026-11-23）**

| 时间 | 事件 |
|---|---|
| 08:32 | 16 个实例注册成功（`Registered KV cache ... 54 layers`） |
| 08:32→09:32 | pod 空闲，零流量 |
| 09:32 | 16 个实例整批被回收：`Reaped GPU instance <id>: silent for 3621.0s (pinged=False)` |
| 此后 | 服务不崩溃，但 store 被拒：`Rejecting STORE for unregistered GPU instance ID`，缓存静默失效，无自动恢复 |

**根因**

worker 的 `HeartbeatThread`（10s 一 ping，刷 server 端 last_seen 保活）是懒启动——
仅在 first store/retrieve/event poll 触发，注册路径不启动：

1. 空闲 worker 零 ping → 命中 `worker_registration_grace_seconds=3600s`
   （"从未 ping"回收窗口）→ 被回收。
2. 被回收后无自愈：重注册只挂在心跳的 unhealthy→healthy 边沿（服务 server 重启
   场景），而 server 的 ping 恒返回 True → 永远不产生恢复边沿。

---

## 3. 修复

### Fix 1: IPC 句柄按 owning device 打开（治 507899）

`lmcache/v1/platform/devices/npu/ipc_wrapper.py`

新增 `_npu_device_ctx(device_index)` 守卫，`to_tensor()` 中每个
`_new_shared_npu` 句柄打开前切到句柄所属 device 的 context：

```python
def _npu_device_ctx(device_index: Any) -> AbstractContextManager:
    """Switch the calling thread's device context for the NPU backend. ..."""
    npu_module = getattr(torch, "npu", None)
    if npu_module is not None and hasattr(npu_module, "device"):
        return npu_module.device(device_index)
    return contextlib.nullcontext()

# to_tensor() 内：
with _npu_device_ctx(device_index):
    storage = torch.UntypedStorage._new_shared_npu(device_index, *handle[1:])
```

### Fix 2: event 操作全链路钉住 device（治 EE1010 / 207001 串扰）

`lmcache/v1/platform/devices/npu/event_ipc.py`

`import_event`（内部同为 `AclIpcOpenEventHandle`）、新增 `record_event` /
`wait_event` 覆写，全部在所属 device 的 context 下执行：

```python
def record_event(self, event: object, stream: object) -> None:
    with self._event_module.device(stream.device):
        super().record_event(event, stream)
```

### Fix 3: 导出 event 保活环缩容 + 淘汰前同步（治 207001 泄漏）

`lmcache/v1/platform/base/event_ipc.py` + `lmcache/v1/platform/devices/npu/event_ipc.py`

- 保活环从无界 `deque(maxlen=2048)` 改为显式淘汰：`exported_event_ring_size`
  参数 + `_retain_exported_event()` + `_on_event_evicted()` 钩子
- NPU 端默认 128（env `LMCACHE_NPU_EVENT_RING_SIZE` 可调），淘汰前
  `event.synchronize()` 确保 worker 消费完再销毁，`aclrtDestroyEvent` 确定性回收
- 理由：CUDA event 廉价可保活 2048，NPU event 占硬件 notify slot，池远小于 2048；
  稳态在途窗口毫秒级、远小于 128，功能无差别

### Fix 4: 多 device 资源释放逐卡回收（堵次级泄漏）

`lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`

`_release_entries` 按 entry 的 device 逐个 `close()`，并在每个 released device
上 `empty_cache()` / `ipc_collect()`（原实现只回收 ambient device 一个）。

### Fix 5: 注册即启动心跳（治空闲误回收）

`lmcache/integration/vllm/vllm_multi_process_adapter.py`

```python
self._send_register_kv_caches_request(kv_caches)
# Start the heartbeat on registration: a worker that goes idle before
# its first store/retrieve would otherwise send no PING and be reaped
# after the registration grace period.
self._ensure_heartbeat_started()
```

收益：live worker 每 10s ping、永不被回收；死 worker 清理时间 3600s → 120s
（`worker_reap_timeout_seconds`）。

---

## 4. 测试与验证

### 新增测试

| 文件 | 用例 | 验证点 |
|---|---|---|
| tests/v1/platform/devices/npu/test_npu_ipc_wrapper.py | test_to_tensor_switches_device_context_per_handle | 每个句柄在 owning device context 下打开 |
| tests/v1/platform/test_event_ipc.py | test_npu_backend_import_event_pins_owning_device 等 | import/record/wait 钉 device；ring 淘汰与 settle |
| tests/v1/test_mp_connector_kv_roles.py | test_register_kv_caches_starts_heartbeat | 注册即构造并启动 HeartbeatThread |

镜像内实测：**101 passed, 1 skipped**（平台 42 + kv_roles 59）。

### 实机验证（v3.1.2 / v3.1.3）

| 验证项 | 结果 |
|---|---|
| REGISTER_KV_CACHE | 全部秒级成功（16 rank，`Registered KV cache ... 54 layers`） |
| 507899 / 207001 / EE1010 | 归零 |
| 空闲 1 小时后实例存活 | 待复测（修复后应无 `Reaped GPU instance`） |
| 服务稳定性 | 无崩溃 |

---

## 5. 影响面与兼容性

- 行为变化 1：NPU 端 exported event 保活窗从 2048 降到 128，可用
  `LMCACHE_NPU_EVENT_RING_SIZE` 调回（重新暴露 207001 风险，不建议）
- 行为变化 2：worker 注册后立即发送 10s 周期心跳（每 worker 每 10s 一个 PING
  请求，开销可忽略）
- CUDA / 其他 backend 路径不受影响（守卫仅在 NPU 设备层，保活环默认值不变）
