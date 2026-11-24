# Build E：scheduler 侧前缀加载／重算策略

## 适用范围

本实现面向当前 Qwen2.5-1.5B 实验路径：LMCache-Ascend `39f2b040`、上游 LMCache 0.4.5、vLLM 0.30.0、单 KV group、非 GDN、同步 lookup、`min_retrieve_tokens=0`。默认 `OFF`，不改变原生调用。

激活任一非 OFF 模式时，上述任一前置条件不满足都会拒绝启动。实现不修改 vLLM scheduler，不修改 fused `npu_connectors.py`，也不使用 monkeypatch。

**阈值口径**：`decide()` 的阈值比较使用 lookup 报告的已缓存前缀长度 `H` 总量，**不扣除** vLLM 本地 prefix cache 已命中的 `num_computed_tokens`（该参数仅做上界校验）。本阶段实验的本地命中为 0，因此阈值标定与结论不受影响；若部署环境中 `num_computed_tokens > 0`，需要按"增量口径"（`H - num_computed_tokens`）重新标定阈值。

## 模式

通过环境变量 `LMCACHE_PREFIX_POLICY_MODE` 选择：

| 模式 | 语义 |
| --- | --- |
| `OFF` | 默认值；直接调用原生实现。 |
| `FORCED_LOAD` | 保留原生 lookup 可加载的候选；没有真实命中时不会制造命中。 |
| `FORCED_RECOMPUTE` | 保留 lookup 的 `H`，但向 scheduler 返回新增外部 token 数 0。 |
| `STATIC_H` | 单阈值或双阈值窗口（见下），阈值经 `LMCACHE_PREFIX_POLICY_STATIC_H` 与 `LMCACHE_PREFIX_POLICY_STATIC_H2` 控制。 |

`FORCED_RECOMPUTE` 复用上游 `min_retrieve_tokens` 已经存在的调度语义，而不是在 `start_load_kv` 阶段跳过 retrieve：

1. 原生 lookup 创建 `LoadSpec(lmcache_cached_tokens=H, can_load=False)`。
2. 策略将返回给 scheduler 的新增外部 token 数改为 0。
3. scheduler 因而不把外部 KV 计入 `num_computed_tokens`，后续按本地状态安排完整 prefill。
4. `update_state_after_alloc(..., num_external_tokens=0)` 保持 `can_load=False`。
5. `lmcache_cached_tokens=H` 不清零，继续供 RequestTracker 的 `num_saved_tokens` 与 store 的 `skip_leading_tokens` 使用，避免重复保存已存在前缀。

若 vLLM 本地 prefix cache 已有 `num_computed_tokens > 0`，返回 0 只取消外部 LMCache 前缀，不会取消本地命中。因此实验记录必须保留 `local_computed_tokens`，不能把该情形直接写成“实际完整重算 L 个 token”。

## STATIC_H 冻结阈值与环境变量

`STATIC_H` 模式的默认冻结阈值为 1024（`H <= 1024` 选 load，否则 recompute）。阈值可通过环境变量
`LMCACHE_PREFIX_POLICY_STATIC_H` 覆盖：

- 值必须是正整数（`int(str, 10)` 且 `>= 1`）；非整数、`0`、负数在配置解析时抛出
  `PrefixPolicyError` 并拒绝启动。
- 未设置或空值使用默认 1024。
- 该变量只在 `STATIC_H` 模式下生效，其他模式（`OFF`/`FORCED_LOAD`/`FORCED_RECOMPUTE`）忽略它。
- 启动时 `LMCACHE_PREFIX_POLICY_CONFIG` 日志会记录生效的冻结值（`static_h_threshold` 字段）。

配置解析对无效取值显式报错，不会静默回退。

## STATIC_H2 双阈值窗口（Build E v4）

`STATIC_H` 模式下通过新增环境变量 `LMCACHE_PREFIX_POLICY_STATIC_H2` 启用第二阈值 τ₂：

- 边界语义（写死并测试）：`H <= τ₁` → LOAD；`τ₁ < H < τ₂` → RECOMPUTE；`H >= τ₂` → LOAD。
  即 `H == τ₁` 走 LOAD、`H == τ₂` 走 LOAD。
- 校验：τ₁ 与 τ₂ 均为正整数；设置 τ₂ 时必须有 `τ₂ > τ₁`，否则 `PrefixPolicyError` 拒绝启动。
- **未设置 τ₂（或空值）时退化为原单阈值行为**（`H <= τ₁` → LOAD 否则 RECOMPUTE），向后兼容。
- τ₂ 只在 `STATIC_H` 模式生效；`FORCED_*` 不受影响。
- 启动日志 `LMCACHE_PREFIX_POLICY_CONFIG` 记录生效值（设置 τ₂ 时含 `static_h_threshold2` 字段）。

本阶段冻结取值：τ₁=1024、τ₂=23296（=91×256，chunk 对齐）。取值依据（如实）：E5 扫描实测第二交点
bootstrap median=23017 tokens、95%CI=[22434, 23976] tokens；τ₂ 需要落在该区间内且 256 对齐。
τ₂=23296 取的是本项目窗口实验**实际部署并完成全部测量验证**的档位；该区间内两臂 TTFT 差随
H 变化平缓（E5 实测 23552 档两臂差 +5.6ms，处于噪声量级），因此 τ₂ 在该区间内的精确取值不敏感，
未以“最接近中位数”作为选择依据。详见 `E5_REPORT.md`。

决策日志沿用 `LMCACHE_PREFIX_POLICY_DECISION` JSON 既有字段（`decision_reason` 携带窗口分支语义），
不新增字段。

## 日志边界

每次激活模式的 lookup 可输出一条 `LMCACHE_PREFIX_POLICY_DECISION` JSON 日志。该日志是**测量辅助**，
默认关闭（hot path 上不产生每请求 info 日志）；显式设置 `LMCACHE_PREFIX_POLICY_LOG_DECISION=1`
才开启。开启后字段语义：

- `h_lookup_tokens`：lookup 报告的 `H`，不是实际加载长度。
- `suffix_tokens`：`L-H`。
- `native_external_tokens`：原生实现准备返回给 scheduler 的新增外部 token 数。
- `returned_external_tokens`：策略实际返回给 scheduler 的新增外部 token 数。
- `planned_remaining_prefill_tokens`：按 scheduler 长度公式推导的计划 prefill 量，不是 worker 实测执行量。
- `policy_applied`：只有真实外部候选被改为 return 0 时才为 true；无命中不会被记成“执行了 forced recompute”。

实际加载长度、真实重算长度、首 token 正确性及 TTFT 仍须由 worker/scheduler 执行证据和 L0/L1/L2 正确性门闭环。本模块自身不宣称 `correctness_closed` 或 `deployment_ready`。
