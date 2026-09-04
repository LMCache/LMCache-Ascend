# Review Rubric — PR 审查完整检查清单

> 本文档是 `SKILL.md` Phase 2「分层审查」的配套细则。
> 审查时按层逐项打勾；任何 ❌ 即对应一份 finding。
> 准则版本：v1（提炼自 cc-connect PR #1436/#1433/#1425/#1384/#1349 等真实实践）。

---

## 定级速查

| 级别 | 图标 | 阻塞 | 典型场景 |
|------|------|------|---------|
| Blocker | 🟥 | 是 | 数据损坏、安全漏洞、核心功能不可用 |
| Major | 🟧 | 是 | 安全类、正确性 bug、破坏既有功能、bug 修复缺回归 |
| Minor | 🟨 | 视情况 | 边界未覆盖、测试不足、命名不佳、文档缺失 |
| Nit | 🟩 | 否 | 风格、注释、微优化 |
| Info | ℹ️ | 否 | 表扬、提问、知识点 |

---

## 第 1 层：规范与元信息

### 1.1 Commit Message
- [ ] 遵循 Conventional Commits：`type(scope): description`
  - type ∈ {feat, fix, refactor, perf, chore, docs, test, style, ci, build}
  - scope 与模块名一致（core / claudecode / feishu / slack / web / api ...）
- [ ] 标题 ≤ ~72 字符，英文祈使句或中文动宾结构
- [ ] bug 修复的 commit body 说明**根因**，而非只说「修复了」
- [ ] `Fixes #N` / `Closes #N` 关联正确 issue
- [ ] 无 `WIP` / `tmp` / `final-final` 残留
- [ ] Co-Authored-By 标注真实（如 AI 协作）

### 1.2 PR 描述
- [ ] 含 Summary（这个 PR 做了什么、为什么）
- [ ] bug 修复含 Root Cause（根因分析）
- [ ] 含 Scope：显式列出 In scope / Out of scope（cc-connect 强约定）
- [ ] 含 Tests：新增/回归测试名 + 全套耗时（如 `./core/ 47.7s PASS`）
- [ ] 含 Risk：作者自评的安全/兼容/性能影响
- [ ] 含 Related：交叉引用相关 PR/issue
- [ ] 破坏性变更显式标注 `BREAKING CHANGE`
- [ ] UI 改动附截图/GIF

### 1.3 PR 卫生
- [ ] 一个 PR 一个主题（不夹带无关改动）
- [ ] requested 正确的 CODEOWNERS 审查人
- [ ] CI 全绿（或失败项与本 PR 无关并已说明）
- [ ] 无冲突，已 rebase 到最新 base
- [ ] 无调试代码（`fmt.Println` / `console.log` / `dump()`）

---

## 第 2 层：正确性与逻辑

### 2.1 通用正确性
- [ ] 边界：空值 / 零值 / 空切片 / nil map 写入 / off-by-one
- [ ] 首次与末次迭代正确（for i:=0; i<n; i++ 的边界）
- [ ] 整数溢出（int32 ↔ int64、len() 转 int32）
- [ ] error 是否被检查、是否被吞、是否 wrap（`fmt.Errorf("%w", err)`）
- [ ] defer 顺序与资源关闭（LIFO，文件/锁/连接）
- [ ] 类型断言带 `, ok` 双返回值保护
- [ ] 字符串循环拼接用 `strings.Builder` / `bytes.Buffer`
- [ ] 时间比较用 `Before/After`，不直接比 Unix 纳秒

### 2.2 Go 并发专项（高频）
- [ ] **共享变量在 goroutine 中访问时持锁**（#1436 型反模式）
- [ ] Mutex/RWMutex 保护范围覆盖**所有**读写路径（不只写路径）
- [ ] 不要在持锁时做 I/O、channel 发送、长计算（锁粒度）
- [ ] channel 关闭方唯一，发送方关闭前不再发送
- [ ] `context.Context` 正确传递与取消传播
- [ ] `sync.WaitGroup` 的 Add/Done/Wait 配对，Add 在 goroutine 外
- [ ] `select` 无 default 忙等（除非有意）
- [ ] map 并发读写必须加锁或用 `sync.Map`
- [ ] `go test -race` 在 CI 默认开启；本 PR 改动是否本地跑过 -race
- [ ] goroutine 泄漏：goroutine 是否有明确退出路径

### 2.3 错误处理专项（#1425 型）
- [ ] 不返回 OS 级模糊错误（`fork/exec ... no such file`）让用户猜
- [ ] 能判断根因处加 preflight 检查，返回业务级清晰错误
- [ ] 错误信息含：是什么 + 在哪 + 怎么修
- [ ] error chain 用 `%w` 保留原始；脱敏用 `%s`
- [ ] 库代码不 panic（除非不可恢复的编程错误）
- [ ] sentinel error（`errors.Is/As`）优先于字符串匹配

### 2.4 资源生命周期
- [ ] 文件句柄 / 网络连接 / 临时文件必关闭
- [ ] 临时文件用 `os.CreateTemp` + defer Remove
- [ ] 进程/子进程正确 wait，避免僵尸

---

## 第 3 层：安全

### 3.1 文件与权限（#1433 型）
- [ ] 新建文件 mode 最小化（配置/密钥 `0600`，普通内容按需 `0644`）
- [ ] `os.CreateTemp` 默认 `0600`，跨用户读取需显式 `Chmod`
- [ ] `run_as_user` / `setuid` 下跨用户读写权限对齐
- [ ] 路径拼接防穿越：`filepath.Clean` + 拒绝 `..` + 限定根目录
- [ ] 文件内容不含敏感信息（token / 私钥 / `.env` / 密码）

### 3.2 输入校验与输出编码
- [ ] 外部输入（用户消息、webhook、配置、URL 参数）做校验与长度限制
- [ ] SQL 用参数化查询，不字符串拼接
- [ ] Shell 用固定参数列表，不 `sh -c "拼接"`
- [ ] HTML/URL 输出转义（防 XSS / 开放重定向）
- [ ] 反序列化限制大小与类型（防 JSON 炸弹）
- [ ] 正则来源用户输入时防 ReDoS

### 3.3 鉴权与密钥
- [ ] 新增 endpoint 走鉴权中间件
- [ ] token / webhook secret 从环境变量或 secret manager 读，不硬编码
- [ ] diff 中无误提交 `.env` / `config.local.*` / `*.key` / `id_rsa`
- [ ] 日志不打印 token / password / PII（必要时打码）
- [ ] 依赖更新检查已知 CVE（`govulncheck` / `npm audit` / `pip-audit`）
- [ ] 加密算法是当前推荐的（AES-GCM、Ed25519），不用 MD5/SHA1/RC4/DES

> 安全类 finding 默认 ≥ Major，不允许「先合后修」。

---

## 第 4 层：架构与可维护性

### 4.1 分层与解耦（cc-connect 专项强约束）
- [ ] **`core/` 不硬编码平台名**（feishu/slack/discord/...）
- [ ] **`core/` 不硬编码 agent 名**（claudecode/codex/...）
- [ ] 新平台/agent 通过 adapter 注册表或接口注入（#1384、#1424 范式）
- [ ] 新增用户可见文案走 i18n，且**所有支持语言一并补齐**
- [ ] 配置项同步更新：`config.toml` schema + 默认值 + `config.example.toml` + 文档
- [ ] 公共 API 变更评估 deprecation 周期
- [ ] 不跨层调用（adapter 不直接被 core 反向依赖）

### 4.2 可维护性
- [ ] 单函数 ≤ ~80 行、圈复杂度 ≤ ~15（超出标 ⚠）
- [ ] 不重复造轮子（项目已有 `writeFileAtomic` 就别再写）
- [ ] 命名达意（`data` / `info` / `tmp` / `handler2` 要改）
- [ ] 注释解释**为什么**，不复述代码
- [ ] TODO/FIXME 带 issue 号：`TODO(#1430): ...`
- [ ] 死代码、注释掉的大段代码移除

### 4.3 性能
- [ ] 热路径避免不必要的分配（box/unbox、string↔[]byte）
- [ ] 循环内不重复计算可外提的值
- [ ] map/slice 预分配（`make(map, n)` / `make([]T, 0, n)`）
- [ ] 锁粒度：不在持锁时做 I/O 或长计算
- [ ] 请求路径无同步阻塞 I/O（应 async 或队列）
- [ ] perf 类 PR 必须带 before/after 基准数据

---

## 第 5 层：测试

### 5.1 覆盖与回归
- [ ] bug 修复 PR 必须带**回归测试**（先复现再修，#1425 范式）
- [ ] 新功能有对应单测/集测
- [ ] 回归测试名能说明被测行为（`TestSend_AfterCleanupReturnsError`）
- [ ] 测试真正断言行为，非仅检查「没 panic」
- [ ] 并发改动用 `t.Parallel()` + `-race`
- [ ] 边界用例：空输入、单元素、超长输入、非法输入

### 5.2 测试质量
- [ ] 不依赖外部环境（用 `t.TempDir()`，不硬编码 `/tmp/xxx`，#1425 修了 7 个此类）
- [ ] mock/stub 不泄漏到非测试代码
- [ ] 测试可独立运行、可重复（无测试间隐式依赖）
- [ ] 不为「让 CI 过」删除/绕过既有测试
- [ ] 表驱动测试的 case 命名达意
- [ ] PR 描述给出全套测试耗时（cc-connect 惯例）

### 5.3 测试反模式
```go
// ❌ 假测试
func TestX(t *testing.T) {
    X()  // 无断言
}

// ❌ 硬编码路径
dir := "/tmp/claudecode-test"

// ❌ 忽略错误
_, _ = riskyCall()

// ✅ 真测试
func TestSend_AfterCleanupReturnsError(t *testing.T) {
    s := newState(); s.cleanup()
    err := s.Send(ctx, msg)
    require.Error(t, err)
    require.Contains(t, err.Error(), "already cleaned up")
}
```

---

## 第 6 层：文档与变更日志

- [ ] 用户可见行为变更 → 更新 README / docs
- [ ] 新增配置项 → 更新 `config.example.toml` + 文档
- [ ] 新增命令/flag → 更新 `--help` 文本与文档
- [ ] 破坏性变更 → CHANGELOG / MIGRATION 笔记
- [ ] 新增公开 API → godoc / tsdoc 注释完整
- [ ] UI 改动 → 截图/GIF 附 PR 描述
- [ ] 删除/重命名公共 API → deprecation 提示

---

## cc-connect 项目专项约定

> 当目标仓为 chenhg5/cc-connect 时，以下为强约定，违反视同 Major。

### 提交与描述
- Commit message 用 Conventional Commits，scope 准确（core/claudecode/feishu/slack/pi/web/api/...）
- bug 修复 PR 描述必含章节：**What / Scope(In/Out) / Why / Tests / Risk / Related**
- 参考范例：#1433（描述章节最完整）、#1425（含 CUJ 影响标注）

### 代码约定
- Go 代码须通过 `go build ./...`、`go test ./...`（并发类加 `-race`）、`go vet`
- 改动触及 `core/` 时禁止硬编码任何平台名/agent 名
- 新增用户可见文案必须 i18n 并补齐所有语言
- 临时文件跨用户读取需显式 `Chmod(0o644)`（#1433 范式）
- 错误信息须业务级清晰，不暴露 OS 级 fork/exec 类消息（#1425 范式）

### 并发
- goroutine 内访问共享状态必须持锁或先捕获局部（#1436 范式）
- 清理路径置 nil 后，所有读取方必须 nil-safe

### 审查流程
- 维护者用 CODEOWNERS 自动指派；PM 不跟踪 PR 状态，由 QA lane 负责合并
- AGENTS.md 的 Pre-Commit Checklist 须满足

### 安全红线
- 不得在源码硬编码 token / webhook secret / 用户凭据
- `run_as_user` 路径下，文件权限须与跨用户读取契约一致
- diff 不得包含 `.env` / `*.key` / 私钥

---

## 审查人自检（提交报告前）

- [ ] 每条 finding 都有 `file:line` 精确定位
- [ ] 每条 finding 都给可执行建议（最好附 patch）
- [ ] 报告含至少 1 条亮点
- [ ] 区分「必须解决 / 建议 / 可选」三档
- [ ] 没有把推测写成事实（推测要标「请确认」）
- [ ] 安全/正确性类已定级 ≥ Major
- [ ] 语气对事不对人，先肯定后建议
