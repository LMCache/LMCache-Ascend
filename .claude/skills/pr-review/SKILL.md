---
name: pr-review
description: Pull Request 审查工作流，从 PR 上下文采集、分层审查（规范/正确性/安全/架构/测试），到输出结构化审查报告。覆盖 Go 并发、文件权限、错误处理、i18n、核心层解耦等高频风险领域。
license: MIT
compatibility: opencode
metadata:
  audience: code reviewers, maintainers, contributors
  output: markdown review report (file-scoped findings + verdict)
  language: Go / TypeScript (cc-connect primary stack)
---

# Skill: pr-review

# Pull Request 审查流水线

端到端的 PR 审查工作流，从采集 PR 上下文（diff / 描述 / 关联 issue / CI）
开始，经过**分层审查**（规范、正确性、安全、架构、测试、文档），
最终输出一份可执行的结构化审查报告，并给出 `Approve / Request changes / Comment`
结论与逐条 finding。

本 skill 的审查准则提炼自 cc-connect 项目的真实 PR 实践（参考 PR #1436 并发修复、
#1433 文件权限修复、#1425 错误处理、#1384 飞书卡片、#1349 会话隔离 等）。

适用场景：
- 用户说 "审查我的 PR #1436"
- 用户说 "review this pull request"
- 用户说 "帮我 review 一下这个 diff"
- 用户给出 PR URL 或本地分支名，需要系统性审查

---

## Phase 0 — 环境感知与目标确认

### 0.1 确定审查目标

从用户输入中识别以下任一形态：

| 输入形态 | 解析方式 |
|---------|---------|
| PR URL | `webfetch` 抓取 conversation / commits / files |
| PR 编号 `#1436` | 组装 URL 后 `webfetch` |
| 本地分支名 | `git log main..HEAD` + `git diff main...HEAD` |
| 已 checkout 的工作区 | 直接 `git status` + `git diff` |
| 一组 commit sha | `git show {sha}` 逐个分析 |

### 0.2 识别目标仓库的约定

在动手审查前，先确认项目约定（优先级从高到低）：

1. `AGENTS.md` / `CONTRIBUTING.md` / `REVIEW.md` — 项目自述的审查规范
2. `.github/CODEOWNERS` — 谁是必选审查人
3. `.github/pull_request_template.md` — PR 描述必填项
4. 近期已合并 PR 的 commit message 风格、描述章节
5. `.golangci.yml` / `biome.json` / `.eslintrc` — 静态检查规则

> 若目标仓是 cc-connect，约定清单见 `templates/review_rubric.md` 的「cc-connect 项目专项约定」。

### 0.3 确认审查深度

向用户确认（或按默认）审查深度：

| 深度 | 范围 | 适用 |
|------|------|------|
| **Quick** | diff 规范 + 显眼风险 | 小修小补、文档、chore |
| **Standard**（默认） | 全部六层，按风险加权 | 绝大多数功能/修复 PR |
| **Deep** | Standard + 并发/安全逐行 + 回归用例审计 | 涉及 core/、并发、鉴权、文件权限 |

---

## Phase 1 — PR 上下文采集

目标：在审查前把「这个 PR 到底改了什么、为什么改、怎么验证的」一次性吃透。

### 1.1 抓取 PR 元信息

```bash
# 远程 PR（通过 GitHub）
webfetch https://github.com/{org}/{repo}/pull/{num}            # conversation + 描述
webfetch https://github.com/{org}/{repo}/pull/{num}/files      # 改动文件清单
webfetch https://github.com/{org}/{repo}/pull/{num}/checks     # CI 状态

# 本地分支
git log --oneline {base}..HEAD
git diff --stat {base}...HEAD
git diff {base}...HEAD
```

可使用 `scripts/collect_pr_context.sh` 自动汇总以上信息。

### 1.2 提取审查关键字段

对每个 PR 建立审查卡片：

| 字段 | 说明 | 举例 |
|------|------|------|
| **类型** | feat / fix / refactor / chore / docs / perf | `fix(core)` |
| **Scope** | 改动所属模块 | core / claudecode / feishu / slack |
| **根因** | 为什么改（bug PR 必填） | goroutine 读 agentSession 未持锁 |
| **修复方式** | 代码层面具体怎么改 | 持锁捕获局部变量，nil 时返回 error |
| **In scope** | 本次改了哪些 | `writeTempAppendPromptFile` |
| **Out of scope** | 显式声明不改什么 | `ensureSharedSystemPromptFile`、daemon 路径 |
| **关联 issue** | Fixes / Refs / Related | `Fixes #1429`、Related #1072 |
| **测试** | 新增/回归测试名 + 全套耗时 | `TestNew_WorkDirDoesNotExist`；`./core/` 47.7s PASS |
| **风险声明** | 作者自评的安全/兼容影响 | chmod 0644 扩大可读范围但内容非密 |

> 任何字段缺失都不是阻塞，但要在报告中以 `⚠ 描述缺章节` 形式标注，
> 提示作者补全 —— 这是 cc-connect 维护者审查时的常见反馈点。

### 1.3 构建改动地图

把 diff 按「文件 → hunk → 性质」组织，便于分层审查：

```
PR #1436  fix(core): goroutine race
├── core/session.go
│   ├── hunk 1  [正确性]  Send() 内 agentSession 改为局部捕获
│   └── hunk 2  [正确性]  nil 检查 + 返回 error
├── core/session_test.go
│   └── hunk 1  [测试]    新增 race 场景用例
└── go.mod  (无变更)
```

性质标签取自下一节的六层模型。

---

## Phase 2 — 分层审查（核心）

按以下六层逐层过 diff。**每一层都要有结论**：✅ 通过 / ⚠ 建议改进 / ❌ 必须修复 / ➖ 不适用。
详细的逐项检查清单见 `templates/review_rubric.md`，本节给出每层的审查要点与高频反模式。

### 第 1 层：规范与元信息（Meta & Conventions）

**目标**：PR 描述、commit message、文件归属是否符合项目约定。

检查项：
- [ ] Commit message 遵循 Conventional Commits：`type(scope): description`
- [ ] bug 修复 PR 的 commit body 说明根因，而非只说「修复了」
- [ ] PR 描述含必要章节：Summary / Scope(In/Out) / Tests / Risk / Related
- [ ] `Fixes #N` / `Closes #N` 正确关联 issue
- [ ] 标题不超过 ~72 字符，使用英文祈使句或中文动宾结构
- [ ] 作者 requested 正确的 CODEOWNERS 审查人
- [ ] PR 不夹带无关改动（一个 PR 一个主题）

**反模式**：
- `update code` / `fix bug` 这类无信息 commit message
- PR 描述只有一行「如题」
- `Fixes #N` 写在标题但正文没展开根因
- 一个 PR 同时做 feat + refactor + 升级依赖

### 第 2 层：正确性与逻辑（Correctness）

**目标**：改动是否真的解决了问题、有没有引入新 bug。这是审查的**重心**。

#### 2.1 通用正确性
- [ ] 边界条件：空值、零值、空切片、off-by-one、首次/末次迭代
- [ ] 错误处理：error 是否被检查、是否被吞、是否 wrap（`fmt.Errorf("%w", err)`）
- [ ] 资源管理：`defer` 关闭、goroutine 泄漏、文件句柄
- [ ] 类型断言 / 类型转换有无保护
- [ ] 字符串拼接在循环中是否用 `strings.Builder`
- [ ] map 并发读写（Go）
- [ ] 时间比较用 `time.Time.Before/After`，不要比 int64

#### 2.2 Go 并发专项（cc-connect 高频风险）

cc-connect 大量使用 goroutine 处理消息流，并发 bug 是 P1 高发区（参考 #1436）：

- [ ] **共享变量在 goroutine 中被访问时是否持锁** —— 最高频反模式
  - 反例：`go func() { state.agentSession.Send(...) }()` 而 `cleanupInteractiveState` 会置 nil
  - 正例：持锁捕获到局部变量，goroutine 内只用局部
- [ ] `sync.Mutex` / `sync.RWMutex` 保护范围是否覆盖所有读写路径
- [ ] 是否需要 `-race` 重跑（CI 默认开 `-race`？）
- [ ] channel 关闭方与发送方是否唯一
- [ ] `context.Context` 是否正确传递与取消
- [ ] `sync.WaitGroup` 的 `Add/Done/Wait` 是否配对
- [ ] select 的 default 分支会不会忙等

```go
// ❌ 反模式（#1436 修复前）
go func() {
    state.agentSession.Send(ctx, msg)  // state.agentSession 可能已被 cleanup 置 nil
}()

// ✅ 正确
state.mu.Lock()
sess := state.agentSession
state.mu.Unlock()
if sess == nil {
    return errors.New("agent session already cleaned up")
}
go func() {
    sess.Send(ctx, msg)
}()
```

#### 2.3 错误处理专项（参考 #1425）

- [ ] 不要返回 OS 级模糊错误（如 `fork/exec ... no such file`）让用户猜根因
- [ ] 在能判断根因的位置加 preflight 检查，返回业务级清晰错误
- [ ] 错误信息包含：是什么 + 在哪 + 怎么修（如 `claudecode: work_dir "/x" does not exist`）
- [ ] error chain 用 `%w` 保留原始错误，用 `%s`/`%v` 仅在需要脱敏时
- [ ] 不要 `panic` 在库代码里，除非真的是不可恢复的编程错误

### 第 3 层：安全（Security）

**目标**：改动是否引入新的攻击面或权限/数据泄露。

#### 3.1 文件与权限（参考 #1433）
- [ ] 新建文件/目录的 mode 是否最小化
  - 临时文件默认 `os.CreateTemp` 是 `0600`，跨用户读取需显式 `Chmod(0o644)`
  - 配置/密钥类必须 `0600`
- [ ] `run_as_user` / `setuid` 场景下，跨用户读写权限是否对齐
- [ ] 路径拼接是否防目录穿越（`filepath.Clean` / 拒绝 `..`）
- [ ] 文件内容是否含敏感信息（token、私钥、`.env`）

#### 3.2 输入与输出
- [ ] 外部输入（用户消息、webhook、配置文件）是否做校验与长度限制
- [ ] SQL / Shell / HTML / URL 是否用参数化或转义
- [ ] 反序列化是否限制大小、类型
- [ ] 日志是否泄露敏感字段（token、password、PII）

#### 3.3 鉴权与密钥
- [ ] 新增 endpoint 是否走鉴权中间件
- [ ] token / webhook secret 是否从环境变量或 secret manager 读，不硬编码
- [ ] diff 中是否误提交 `.env`、`config.local.toml`、`*.key`
- [ ] 依赖更新是否引入已知 CVE（配合 `govulncheck` / `npm audit`）

> 安全类 finding **默认定级 ≥ Major**，不允许「先合后修」。

### 第 4 层：架构与可维护性（Architecture）

**目标**：改动是否破坏模块边界、是否增加未来维护成本。

#### 4.1 分层与耦合（cc-connect 专项）
- [ ] `core/` 不得硬编码具体平台名（feishu/slack/...）或 agent 名（claudecode/...）
  - 必须通过 adapter 注册表 / 接口注入
  - 参见 #1384、#1424 等 adapter PR 的做法
- [ ] 新增用户可见文案必须走 i18n，且**所有支持语言一并补齐**
- [ ] 配置项加到 `config.toml` 时同步更新 schema、默认值、文档
- [ ] 公共 API 变更是否需要 deprecation 周期

#### 4.2 可维护性
- [ ] 单函数行数 / 圈复杂度是否失控（> ~80 行 / > ~15 分支需警示）
- [ ] 是否重复造轮子（项目内已有 `writeFileAtomic` 就别再写一遍）
- [ ] 命名是否达意（`data` / `info` / `tmp` 这类无信息名要改）
- [ ] 注释是否解释 **为什么**，而非复述代码
- [ ] TODO/FIXME 是否带 issue 号（`TODO(#1430): ...`）

#### 4.3 性能
- [ ] 热路径上的分配（box/unbox、string↔[]byte、map 预分配）
- [ ] O(n²) 循环、N+1 查询、全表扫描
- [ ] 锁粒度是否过粗阻塞热路径
- [ ] 是否引入不必要的同步 I/O 在请求路径

> 性能优化类 PR 要求附带 before/after 基准数据，否则标 `⚠ 缺基准`。

### 第 5 层：测试（Tests）

**目标**：改动是否有充分的测试覆盖，且测试本身可信。

- [ ] bug 修复 PR 必须带**回归测试**（先复现再修，参考 #1425 的 `TestNew_WorkDirDoesNotExist`）
- [ ] 回归测试名能说明被测行为，不用 `TestFunc1` / `TestCase2`
- [ ] 测试是否真正断言了行为，而非只检查「没 panic」
- [ ] 并发相关改动是否用 `t.Parallel()` + `-race` 验证
- [ ] 测试不依赖外部环境（用 `t.TempDir()` 而非硬编码 `/tmp/xxx`，参考 #1425 修复的 7 个旧用例）
- [ ] mock/stub 是否泄漏到非测试代码
- [ ] PR 描述是否给出全套测试耗时（cc-connect 惯例：`./core/ 47.7s PASS`）
- [ ] 是否删除/绕过了既有测试来「让 CI 过」

**反模式**：
```go
// ❌ 假测试：只确认没崩
func TestSend(t *testing.T) {
    s := New()
    s.Send(msg)  // 无断言
}

// ❌ 硬编码路径，换机器就挂
dir := "/tmp/claudecode-test"

// ✅ 真测试
func TestSend_AfterCleanupReturnsError(t *testing.T) {
    s := newState()
    s.cleanup()
    err := s.Send(ctx, msg)
    require.Error(t, err)
    require.Contains(t, err.Error(), "already cleaned up")
}
```

### 第 6 层：文档与变更日志（Docs & Changelog）

- [ ] 用户可见行为变更是否更新 README / docs / CHANGELOG
- [ ] 新增配置项是否在 `config.example.toml` 体现
- [ ] 新增命令/flag 是否更新 `--help` 文本与文档
- [ ] 截图/GIF（如 UI 改动）是否附在 PR 描述
- [ ] 破坏性变更是否在 PR 描述显式标注 `BREAKING CHANGE`

---

## Phase 3 — 风险评估与定级

对每个 finding 定级，决定是否阻塞合并：

| 级别 | 含义 | 是否阻塞 | 处理 |
|------|------|---------|------|
| 🟥 **Blocker** | 会导致数据损坏/安全事故/核心功能不可用 | 是 | 必须 Request changes |
| 🟧 **Major** | 安全类、正确性 bug、破坏既有功能 | 是 | 必须 Request changes |
| 🟨 **Minor** | 边界未覆盖、测试不足、命名不佳 | 视情况 | 建议 Request changes 或 Approve + comment |
| 🟩 **Nit** | 风格、注释、微优化 | 否 | Approve + 顺手提建议 |
| ℹ️ **Info** | 表扬、提问、知识点分享 | 否 | Approve + comment |

**定级原则**：
1. 安全类默认 ≥ Major，不妥协
2. 正确性类按「触发概率 × 影响面」定级
3. 测试缺失：bug 修复缺回归 = Major；新功能缺测试 = Minor
4. 文档/规范类一般 Minor 及以下
5. 主分支保护范围内（如 `core/`）的架构问题从严

---

## Phase 4 — 输出审查报告

### 4.1 报告路径

```
{workspace}/output/pr_{num}_review.md
```

### 4.2 报告结构

使用 `templates/pr_review_report_template.md`，核心结构：

```markdown
# PR #{num} 审查报告

> 标题 / 作者 / 仓库 / 审查日期 / 审查深度

## 一、结论
{Approve | Request changes | Comment}  ——  一句话理由

## 二、改动概览
{文件数 / 增删行 / 类型 / scope / 关联 issue}

## 三、分层审查结论
| 层 | 结论 | 主要发现 |
|----|------|---------|
| 规范 | ✅ | ... |
| 正确性 | ⚠ | ... |
| ... | ... | ... |

## 四、Findings（逐条）
{每条：定位 file:line / 级别 / 问题 / 建议 / 可选 patch}

## 五、亮点（必有，至少 1 条）
{肯定作者做对的地方}

## 六、合并前必须解决 / 建议改进 / 可选
```

### 4.3 Finding 写作规范

每条 finding 必须可执行、可定位、可验证：

```markdown
### F1 — Send goroutine 未持锁读 agentSession 🟥 Blocker
**位置**：core/session.go:142
**问题**：goroutine 内直接读 state.agentSession，与 cleanupInteractiveState 置 nil 存在竞态。
**影响**：agent 退出时高概率 nil panic，P1。
**建议**：持锁捕获到局部变量后传入 goroutine，nil 时返回 error。
**参考**：cc-connect #1436 的同型修复。
```patch
- go func() { state.agentSession.Send(ctx, msg) }()
+ state.mu.Lock()
+ sess := state.agentSession
+ state.mu.Unlock()
+ if sess == nil { return errSessionGone }
+ go func() { sess.Send(ctx, msg) }()
```
```

写作要点：
1. **先定位**：`file:line` 必须精确
2. **先说是什么，再说为什么坏**：避免上来就评判
3. **给可执行建议**：最好附带 patch，而非「请优化」
4. **引用先例**：项目内类似修复的 PR 号、issue 号
5. **区分事实与推测**：推测要标「我推测…，请确认」

### 4.4 评论语气

- **对事不对人**：「这段循环是 O(n²)，n=10k 时会卡」，而非「你写错了」
- **先肯定后建议**：至少留 1 条亮点；大改动先肯定工作量
- **疑问优先于否定**：不确定时用「这里是否考虑过 X？」
- **尊重作者判断**：Minor 以下可加「如果你同意我再改，不同意可忽略」

---

## Phase 5 — 提交审查结果

### 5.1 回填到 GitHub（如用户授权）

```bash
# 通过 gh CLI 提交 review（需要用户确认）
gh pr review {num} \
  --request-changes \
  --body-file output/pr_{num}_review.md

# 或仅提交单条 inline comment
gh api repos/{org}/{repo}/pulls/{num}/comments \
  -f body="..." -f commit_id=... -f path=... -F line=...
```

> ⚠ **未经用户明确同意，不要执行任何 `gh pr review` / 评论 / 合并命令。**
> 默认只生成报告文件，由用户自行决定如何提交。

### 5.2 汇报给用户

审查完成后，用简洁表格向用户汇报：

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 上下文采集 | ✅ N 文件 / M 增删 |
| Phase 2 | 分层审查 | ✅ 6 层完成 |
| Phase 3 | 风险评估 | ⚠ 2 Major / 1 Minor |
| Phase 4 | 报告生成 | ✅ output/pr_1436_review.md |
| Phase 5 | 提交 | ⏸ 待用户确认 |

---

## 快速参考卡片

### 常用命令

```bash
# PR 上下文
webfetch https://github.com/{org}/{repo}/pull/{num}
git diff {base}...HEAD --stat
git log --oneline {base}..HEAD

# 本地一键采集（本 skill 提供）
bash scripts/collect_pr_context.sh {base}

# Go 专项
go build ./...
go test -race ./...
go vet ./...
gofmt -l .
govulncheck ./...

# 提交审查（需用户确认）
gh pr review {num} --approve --body-file output/pr_{num}_review.md
```

### 改动类型 → 重点审查层映射

| PR 类型 | 必查层 | 重点 |
|---------|-------|------|
| `fix(core)` 并发 | 正确性(2.2) + 测试 | 锁、goroutine、回归用例 |
| `fix` 文件/权限 | 安全(3.1) | mode、跨用户、路径穿越 |
| `fix` 错误处理 | 正确性(2.3) | 错误信息清晰度、preflight |
| `feat` 新 adapter | 架构(4.1) | core 不硬编码、i18n |
| `feat` 新 endpoint | 安全(3.3) + 测试 | 鉴权、输入校验 |
| `refactor` | 正确性 + 架构 + 测试 | 行为不变、测试不动 |
| `chore` 依赖升级 | 安全(3.3) | CVE、breaking |
| `docs` | 规范 + 文档 | 仅元信息层 |
| `perf` | 性能(4.3) + 测试 | 必须带基准数据 |

### 阻塞 vs 非阻塞 决策树

```
finding 涉及安全？ ─是─→ Blocker/Major，必须 Request changes
        │否
        ├─ 涉及正确性？ ─是─→ 按概率×影响定级，通常 Major
        │                    缺回归测试 → Major
        │否
        ├─ 涉及 core/ 主干？ ─是─→ 从严，Minor 也倾向 Request changes
        │否
        ├─ bug 修复 PR 缺测试？ ─是─→ Major
        │否
        └─ 规范/风格/文档 → Nit/Minor，Approve + comment
```

---

## Bundled Resources

### Templates

| File | Purpose |
|------|---------|
| `templates/pr_review_report_template.md` | 结构化审查报告标准模板（含分层结论 + Findings 表） |
| `templates/review_rubric.md` | 六层审查的完整检查清单 + cc-connect 项目专项约定 |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/collect_pr_context.sh` | 一键汇总本地 PR 的 diff/stat/log/CI 信息，供审查消费 |

---

## When to Use This Skill

- 用户给你一个 PR（URL / 编号 / 分支名 / diff）并要求 review
- 用户说「帮我审一下这个改动」/「这个 PR 有什么问题」
- 你作为 maintainer 需要对 contributor 的 PR 给出结构化反馈
- 团队希望统一 PR 审查的维度、定级、报告格式

## When NOT to Use

- 只是本地写代码时想 lint（用 lint 工具直接跑）
- 用户只是想理解某段代码做什么（直接讲解即可，不必走完整流水线）
- PR 仅含自动生成的文件变更（lockfile、vendor 等），走 Quick 之外的深度无意义
