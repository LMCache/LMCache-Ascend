# Manual Verification Guide Template

# PR/Issue #{{NUM}} 手动验证操作指南

## 环境

| 项目 | 值 |
|------|-----|
| 服务器 | `ssh {{SSH_HOST}}` |
| 工作路径 | `{{WORK_PATH}}` |
| 目标容器 | `{{CONTAINER}}` |
| 待验证 commit | `{{COMMIT}}` |

---

## 步骤 1：确认代码版本

```bash
{{VERSION_CHECK_CMD}}
```

**预期输出**:
```
{{VERSION_CHECK_OUTPUT}}
```

---

## 步骤 2：同步代码到容器

```bash
{{SYNC_CMD}}
```

**预期结果**: 容器内 git log 与宿主一致，HEAD = `{{COMMIT}}`。

---

{{STEPS}}

---

## 步骤 N+1：清理

```bash
{{CLEANUP_CMD}}
```

**预期输出**: `{{CLEANUP_OUTPUT}}`

---

## 验证清单

| # | 步骤 | 验证项 | 通过条件 |
|---|------|--------|---------|
{{CHECKLIST_TABLE}}
