# Skill: issue-verification-workflow

# Issue 驱动的全流程验证与修复流水线

端到端的工作流，从 GitHub Issue/PR 抓取开始，经过定位、复现验证、修复、回归验证，
最终输出结构化验证报告并提交代码。

适用场景：
- 用户说 "验证我的 PR #253"
- 用户说 "帮我修 issue #238"
- 用户说 "查一下有哪些待修的 bug，逐个处理"
- 用户提供一批 issue/commit 需要系统性验证

---

## Phase 0 — 环境感知

### 0.1 识别运行模式

判断当前是**本地开发**还是**远程服务器验证**：

| 信号 | 本地模式 | 远程模式 |
|------|---------|---------|
| SSH 地址提供 | 无 | 有 (如 `ssh root@115.120.47.8`) |
| 容器配置提供 | 无 | 有 (如 `docker exec lmcache-p2p-pd-csy`) |
| NPU 设备可用 | 本地有 NPU | 远程有 NPU |

**远程模式**下，所有测试命令通过 `ssh {host} "docker exec {container} bash -c '...'"` 执行。
代码修改优先在本地进行，然后 `rsync` 到远程。

### 0.2 收集环境信息

在远程模式下，首先收集：

```bash
# 硬件
ssh {host} "npu-smi info"                          # NPU 型号/数量
ssh {host} "docker exec {container} python --version"  # Python 版本

# 软件栈
ssh {host} "docker exec {container} pip list | grep -iE 'lmcache|vllm|torch'"

# 代码状态
ssh {host} "cd /path/to/LMCache-Ascend && git log --oneline -10"
```

### 0.3 确定目标范围

从用户输入中提取：
- **Issues**: 用 `webfetch` 获取详情
- **PR**: 查看包含哪些 commits → 提取每个 commit 对应的 issue
- **Commits**: 直接被给定的 sha 列表

---

## Phase 1 — 抓取与理解 Issue

### 1.1 获取 Issue 内容

```bash
# 通过 GitHub API 获取 issue/PR
webfetch https://github.com/LMCache/LMCache-Ascend/pull/{num}
webfetch https://github.com/LMCache/LMCache-Ascend/issues/{num}
```

### 1.2 提取关键信息

对每个 issue 提取：

| 字段 | 说明 |
|------|------|
| **标题** | 问题的一句话描述 |
| **根因** | 为什么出错 |
| **影响文件** | 哪些 .py 文件需要改 |
| **修复方式** | 代码具体改了什么 |
| **影响类型** | 正确性 / 性能 / 日志 / 清理 |

### 1.3 获取实际 Diff

```bash
git log --oneline HEAD~10         # 找到相关 commit
git show {commit} --stat           # 看改了哪些文件
git diff {base}..{commit} -- {file}  # 看具体 diff
```

### 1.4 分类 Issue

| 类型 | 验证策略 |
|------|---------|
| **性能** (host blocking, timing, throughput) | 微基准测试 + 运行时指标抓取 |
| **正确性** (索引 bug, 逻辑错误, dead state) | 源代码格式审计 + 断言 |
| **日志** (格式修正, 语义澄清) | 源码格式审计 + 运行时 grep |
| **清理** (dead code removal) | grep 确认不存在 |
| **兼容性** (API signature, version) | import 测试 + 运行时检查 |

---

## Phase 2 — 设计验证方案

### 2.1 为每个 issue 创建验证条目

每个 issue 需要以下维度的验证（至少选择一种）：

```
Issue #N:
├── 源码级 (Source Audit)
│   ├── grep 确认目标代码存在/不存在
│   └── 格式/结构检查
├── 单元级 (Unit Test)
│   ├── import + 基础调用
│   └── 断言关键属性
├── 性能级 (Benchmark)
│   ├── 微基准：循环计时，对比 before/after
│   └── 集成指标：抓取运行时 log 中的 timing 字段
├── 集成级 (Integration)
│   ├── 启动服务 → 确认无 ERROR
│   ├── 发送请求 → 确认响应正确
│   └── 检查日志 → 确认关键 log 出现
└── 回归级 (Regression)
    └── 确认修复未引入新问题
```

### 2.2 性能 Issue 专项验证模板

```python
import torch, time, inspect

# 1. 源码审计
with open("target_file.py") as f:
    src = f.read()
assert "expected_pattern" in src, "FAIL: pattern not found"

# 2. 微基准（跑在真实 NPU 上）
n_iters = 2000
# Old behavior simulation
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(n_iters):
    old_behavior()
t_old = (time.perf_counter() - t0) / n_iters * 1e6

# New behavior simulation
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(n_iters):
    new_behavior()
t_new = (time.perf_counter() - t0) / n_iters * 1e6

print(f"Old: {t_old:.1f} us/op  New: {t_new:.1f} us/op  Speedup: {t_old/t_new:.0f}x")
assert t_old / max(t_new, 0.001) > threshold, "FAIL: speedup below threshold"
```

### 2.3 正确性/日志 Issue 验证模板

```python
with open("target_file.py") as f:
    src = f.read()

# 确认旧错误模式已移除
assert "bad_pattern" not in src, "FAIL: bad pattern still present"

# 确认新正确模式已加入
assert "good_pattern" in src, "FAIL: correct pattern not found"

# 检查注释/说明
assert "NOTE(#N)" in src, "FAIL: explanatory comment missing"
```

---

## Phase 3 — 执行验证

### 3.1 同步代码到目标环境

```bash
# 远程模式：rsync 本地代码到服务器
ssh {host} "rsync -a --delete --exclude='build' --exclude='__pycache__' \
  /path/to/local/LMCache-Ascend/ /path/to/remote/LMCache-Ascend/"
```

### 3.2 依次执行验证条目

按优先级：源码级 → 单元级 → 性能级 → 集成级

每个验证条目明确输出 `PASS` 或 `FAIL` 及具体数据。

### 3.3 集成测试（如需要）

当 issue 涉及运行时行为时，需要启动完整服务：

```bash
# 1. 创建配置文件
cat > config.yaml << EOF
chunk_size: 256
local_cpu: True
max_local_cpu_size: 50
internal_api_server_enabled: True
internal_api_server_host: "0.0.0.0"
internal_api_server_port_start: 5999
EOF

# 2. 启动服务
docker exec -d {container} bash -c "
  export LMCACHE_CONFIG_FILE=/path/to/config.yaml
  export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
  export VLLM_ENABLE_V1_MULTIPROCESSING=1
  python -m vllm.entrypoints.openai.api_server \
    --port 7100 --model /path/to/model \
    --tensor-parallel-size 4 --trust-remote-code \
    --kv-transfer-config '{\"kv_connector\":\"{connector}\",...}' \
    > /path/to/logs/vllm.log 2>&1 &
"

# 3. 等待就绪（轮询 /health，最多等待 60 秒）
TIMEOUT=60
INTERVAL=5
ELAPSED=0
while ! curl -s -o /dev/null -w '%{http_code}' http://localhost:7100/health | grep -q 200; do
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "ERROR: Server failed to start within $TIMEOUT seconds"
    exit 1
  fi
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

# 4. 发送测试请求
curl -s http://localhost:7100/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"..."}],"max_tokens":10}'

# 5. 检查日志
grep -E "LMCache|offload_total_time|ERROR|Traceback" /path/to/logs/vllm.log
```

### 3.4 监控关键指标

| 指标 | 命令 | 含义 |
|------|------|------|
| `/health` 返回 | `curl -s -o /dev/null -w '%{http_code}'` | 服务是否存活 |
| HBM 使用率 | `npu-smi info -t usages -i {id}` | 模型是否加载 |
| AICore 使用率 | 同上 | 是否有推理负载 |
| ERROR 计数 | `grep -c ERROR {log}` | 应用级错误 |
| Traceback 计数 | `grep -c Traceback {log}` | Python 异常 |

---

## Phase 4 — 生成验证报告

### 4.1 报告输出路径

所有报告输出到当前工作目录下的 `output/` 目录：

```
{workspace}/output/{PR|ISSUE}_{num}_verification_report.md
{workspace}/output/{PR|ISSUE}_{num}_manual_verification_guide.md
```

### 4.2 报告结构模板

#### 验证报告 (`verification_report.md`)

```markdown
# {PR/Issue} #{num} 测试验证报告

> PR: {url}
> 标题: {title}
> 测试日期: {date}
> 测试环境: {hardware + software details}

## 一、摘要
{1-2 paragraph summary of all findings}

## 二、测试环境
{Table: HW, NPU, CANN, Container, Python, vLLM, LMCache, Model, TP}

## 三、变更清单与验证结果
{Table: commit | file | change | verification method | result}

## {For performance issues:} 性能专项验证
{Per-issue breakdown: source audit + benchmark data}

## 四、vLLM 集成测试
{Startup timeline, inference results, KV cache stats}

## 五、健康检查
{ERROR/Traceback counts, WARNING audit, NPU status}

## 六、结论
{Overall result + recommendations}
```

#### 手动验证指南 (`manual_verification_guide.md`)

```markdown
# {PR/Issue} #{num} 手动验证操作指南

## 环境
{Connection info, container, commit}

## 步骤 1: 确认代码版本
{bash commands + expected output}

## 步骤 N: 每个验证条目
{command block + expected output + pass condition}

## 验证清单
{Checklist table: #, step, item, pass condition}
```

### 4.3 数据要求

报告中所有数据必须来自**真实环境测试**，不得编造：

- ✅ 性能数据：显示具体 iteration 数、元素数、耗时
- ✅ 日志摘录：引用实际 `grep` 输出
- ✅ 推理结果：引用实际 `curl` 响应的 `content` 字段
- ✅ NPU 状态：引用 `npu-smi` 输出

---

## Phase 5 — 清理与提交

### 5.1 停止测试服务

```bash
ssh {host} "docker exec {container} pkill -9 -f vllm.entrypoints"
```

### 5.2 确认无残留

```bash
ssh {host} "docker exec {container} ps aux | grep python | grep -v grep"
```

### 5.3 提交代码（如需要）

```bash
git add {changed_files}
git commit -m "{type}({scope}): {description} (#{issue_number})"
git push origin {branch}
```

Commit message 格式遵循项目规范：
- `fix({scope}): {description} (#{issue})`
- `feat({scope}): {description}`
- `chore({scope}): {description}`

### 5.4 清理临时文件

```bash
rm -f /tmp/verify_*.py       # 本地临时脚本
rm -rf {workspace}/output/    # 报告（可选保留）
```

---

## Phase 6 — 流程总结输出

验证完成后，用简洁的表格向用户汇报：

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | Issue 抓取与分析 | ✅ |
| Phase 2 | 验证方案设计 | ✅ |
| Phase 3 | 执行验证 | ✅ N/N PASS |
| Phase 4 | 生成报告 | ✅ output/xxx.md |
| Phase 5 | 清理 | ✅ |

---

## 快速参考卡片

### 常用命令速查

```bash
# Issue 查询
webfetch https://github.com/LMCache/LMCache-Ascend/issues/{num}
webfetch https://github.com/LMCache/LMCache-Ascend/pull/{num}

# Diff 查询
git log --oneline -10
git diff {base}..{head} -- {file}
git show {commit} --stat

# 远程执行
ssh {host} "docker exec {container} bash -c '...'"
ssh {host} "docker exec {container} python3 -c '...'"

# 文件传输
scp /tmp/script.py {host}:/tmp/
ssh {host} "docker cp /tmp/script.py {container}:/tmp/"
rsync -a --exclude='build' local/ remote/

# 日志检查
grep -c "ERROR" {log}
grep -c "Traceback" {log}
grep "LMCache" {log} | grep -v WARNING
```

### Issue 类型（不局限于以下几类） → 验证方法映射

| Issue 类型 | 最小验证 | 推荐验证 | 可选验证 |
|-----------|---------|---------|---------|
| pin_memory 添加 | 源码 grep | 微基准 host blocking time | — |
| 索引 bug 修复 | 源码 grep 旧模式移除 | import + 静态检查逻辑 | 端到端推理 |
| 日志格式修正 | 源码 grep 新旧格式 | — | 运行时 grep log |
| dead code 移除 | 源码 grep 确认不存在 | — | — |
| stream sync 修复 | 源码 grep 新模式 | import 测试 | — |
| timing 修正 | 源码 grep + 注释检查 | 运行时 grep timing 值 | 基准对比 |

---

---



### Templates

| File | Purpose |
|------|---------|
| `templates/verification_report_template.md` | 验证报告标准模板 (6 章节) |
| `templates/manual_verification_guide_template.md` | 手动验证操作指南模板 |

---

Base directory for this skill: file:///Users/yhl/Project/LMCache-Ascend/.claude/skills/issue-verification-workflow
Relative paths in this skill (e.g., scripts/, reference/) are relative to this base directory.
