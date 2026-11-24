# LMCache NPU IPC device-context 修复补丁集

**日期**: 2026-11-22　**目标**: 修复 lmcache-server `register_kv_cache` 507899/107017/207001/EE1010 系列失败
**基线**: LMCache dev（HEAD `aa12d34e`），叠加在 `pr-lmcache-event-leak/` 之上

## 根因（一句话）

LMCache server 端多线程/多 device 交错服务时，NPU IPC 句柄在**调用线程的当前 device context** 下被打开/记录，
而 CANN 的 `AclIpcOpenEventHandle` / `AclrtCreateEventWithFlag` 是 context 绑定的 → 每 device 的 event 资源被
错乱创建/无法回收（207001 池耗尽）或句柄打开失败（107017 invalid handle / 507899 drvRetCode=42 internal error），
并伴随 EE1010 "stream does not belong to the current context"。worker 侧 REGISTER_KV_CACHE 300s 无应答
（mq_timeout）→ DP0 全灭 → DP1 卡 world-16 AllGather → 1836s HCCL fftsplus timeout 级联崩溃。

## 补丁清单

| 文件 | 作用 | 修复的错误 |
|---|---|---|
| `0000-combined-v3.1-npu-ipc-fixes.patch` | **组合补丁（dev HEAD 基线，含下述全部 + event 泄漏修复 + 0004）** | 全部 |
| `0000-combined-v0.5.7-image-base.patch` | **镜像基线版组合补丁**（v3.1.x 镜像内 LMCache v0.5.7/g95656e5c 的精确 delta，v3.1.3 相对 v3.1.1） | 全部 |
| `0001-npu-open-ipc-storage-handles-under-owning-device-context.patch` | `NpuIPCWrapper.to_tensor` 每个 IPC handle 在 owning device context 下打开（`_npu_device_ctx` 守卫） | 507899 / 107017（register 路径） |
| `0002-npu-pin-device-context-across-event-ipc-ops.patch` | `NpuEventIPCBackend.import_event/record_event/wait_event` 钉住 owning device（与既有 `export_event` 守卫同款） | EE1010 / 207001（record 时懒创建 event 入错池） |
| `0003-mp-release-entries-under-owning-device-context.patch` | `_release_entries` 按 entry 的 device 逐个 `close()` + `empty_cache()`（原实现只回收 ambient device，多 device 批量释放时漏收） | 207001（资源不回收的次级泄漏） |
| `0004-vllm-start-heartbeat-on-register.patch` | worker `register_kv_caches` 成功后立即启动心跳（原懒启动致空闲 worker 1h 后被 grace 回收且无自愈） | idle-worker 被回收（2026-11-23 实测 16 rank 全灭） |

**已发布镜像**：
- `hub.galaxy.ksyun.com/ai/lmcache-server:v3.1.2` = v3.1.1 + 0001/0002/0003 + event 泄漏修复（2026-11-22）
- `hub.galaxy.ksyun.com/ai/lmcache-server:v3.1.3` = v3.1.2 + **0004 心跳修复**（2026-11-23，101 tests passed）

注意：0004 的 dev HEAD 上下文与镜像内 v0.5.7 有偏差，打 v3.1.x 镜像时以
`0000-combined-v0.5.7-image-base.patch` 为准（直接应用到镜像内 /workspace/LMCache）。

**event 泄漏修复**（`_exported_events` 保活环 2048→128 + 淘汰前 synchronize，治 207001 主泄漏）在
`pr-lmcache-event-leak/0001-platform-reclaim-aged-out-export-events.patch`，组合补丁 0000 已含。

## 使用方式

### A. 打内部 v3.1 修复镜像（推荐）

```bash
cd LMCache
git apply pr-lmcache-npu-device-context/0000-combined-v3.1-npu-ipc-fixes.patch
# 跑测试（Windows 本地可用 --noconftest 绕过 native 扩展）：
python -m pytest tests/v1/platform/test_event_ipc.py \
  tests/v1/platform/devices/npu/test_npu_ipc_wrapper.py --noconftest -q
# 重新构建 lmcache-server 镜像
```

### B. 拆分序列（上游 PR / 分仓提交）

应用顺序（0002 的测试 hunk 上下文含 ring 测试，须先应用 event-leak 系列）：

1. `pr-lmcache-event-leak/0001-platform-reclaim-aged-out-export-events.patch`（+ 其 `tests/` 目录的测试）
2. `0001-npu-open-ipc-storage-handles-under-owning-device-context.patch`
3. `0002-npu-pin-device-context-across-event-ipc-ops.patch`
4. `0003-mp-release-entries-under-owning-device-context.patch`

## 测试验证（已执行，2026-11-22）

`tests/v1/platform/test_event_ipc.py` + `tests/v1/platform/devices/npu/test_npu_ipc_wrapper.py`：
**42 passed, 1 skipped**（含新增 `test_npu_backend_import_event_pins_owning_device`、
`test_npu_backend_record_and_wait_pin_stream_device`、`test_to_tensor_switches_device_context_per_handle`）。

## 部署后验证判据

- lmcache-server 日志不再出现 107017 / 507899 / 207001 / EE1010
- `register_kv_caches` 秒级应答（不再 300s mq_timeout）
- DP0 不再全灭，DP1 不再 HCCL 1836s timeout

## HDK 升级定性

**不能根治，最多改善 507899 一个码。** 107017/EE1010/207001 是应用层 context 串扰与 event 泄漏，
驱动换版本不改 handle 的 context 绑定语义；CANN 9.2.1 升级已实测无效（11-17 验证）。
HDK 26.1.1（ascendhal 7.35.23）之后的版本可查 release note 是否修 `halMemImportFromShareableHandle`
（drvRetCode=42），仅作辅助；本补丁集是根治路径。
