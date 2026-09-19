# Verification Report Template

> **PR/Issue**: #{{NUM}}
> **URL**: {{URL}}
> **标题**: {{TITLE}}
> **作者**: {{AUTHOR}}
> **测试日期**: {{DATE}}
> **测试人员**: 自动化验证 (真实 Ascend NPU 环境)

---

## 一、摘要

{{1-2 paragraph summary of what was tested and the overall result}}

---

## 二、测试环境

| 项目 | 详情 |
|------|------|
| **服务器** | `{{SSH_HOST}}` |
| **硬件平台** | {{HW_PLATFORM}} |
| **NPU** | {{NPU_SPEC}} |
| **容器** | `{{CONTAINER}}` ({{IMAGE}}) |
| **CANN** | {{CANN_VERSION}} |
| **Python** | {{PYTHON_VERSION}} |
| **vLLM** | {{VLLM_VERSION}} |
| **LMCache (上游)** | {{LMCACHE_UPSTREAM_VERSION}} |
| **LMCache-Ascend** | {{LMCACHE_ASCEND_VERSION}}, commit `{{COMMIT}}` |
| **模型** | {{MODEL}} |
| **TP 配置** | {{TP_CONFIG}} |

### LMCache 运行配置

```yaml
{{LMCACHE_CONFIG_YAML}}
```

---

## 三、变更清单与验证结果

| # | Commit | 变更文件 | 变更说明 | 验证方式 | 结果 |
|---|--------|---------|---------|---------|------|
{{CHANGES_TABLE}}

---

## {For Performance Issues} 三-B、性能相关 Issue 专项验证

### #{{ISSUE_NUM}} — {{ISSUE_TITLE}}

**问题**: {{PROBLEM_DESCRIPTION}}

**修复**: {{FIX_DESCRIPTION}}

**验证方法**: {{METHOD}}

**验证结果**:

| 指标 | 修复前 | 修复后 | 改进 |
|------|--------|--------|------|
{{METRICS_TABLE}}

---

## 四、集成测试

### 4.1 启动时间线

```
{{STARTUP_TIMELINE}}
```

### 4.2 推理测试结果

| # | 请求 | 响应 | prompt_tokens | completion_tokens | HTTP |
|---|------|------|---------------|-------------------|------|
{{INFERENCE_TABLE}}

### 4.3 LMCache 指标

| 请求 | Total Tokens | Engine Computed | LMCache Hit | Need Load |
|------|-------------|-----------------|-------------|-----------|
{{LMCACHE_METRICS}}

---

## 五、健康检查

### 5.1 错误/异常

| 类别 | 数量 | 说明 |
|------|------|------|
| **ERROR** | {{ERROR_COUNT}} | {{ERROR_DESC}} |
| **Traceback** | {{TRACEBACK_COUNT}} | {{TRACEBACK_DESC}} |
| **WARNING** | {{WARNING_COUNT}} | {{WARNING_DESC}} |

### 5.2 NPU 状态

```
{{NPU_STATUS}}
```

---

## 六、结论

### 6.1 总体评价

{{OVERALL_ASSESSMENT}}

### 6.2 建议

{{RECOMMENDATIONS}}

---

*报告由 LMCache-Ascend issue-verification-workflow 自动生成。测试时间: {{TEST_TIME}}*
