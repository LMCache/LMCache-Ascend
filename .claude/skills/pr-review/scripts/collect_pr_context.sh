#!/usr/bin/env bash
# collect_pr_context.sh — 一键汇总本地 PR 的 diff/stat/log/CI 信息，供 pr-review skill 消费。
#
# 用法：
#   bash collect_pr_context.sh [BASE_REF] [HEAD_REF] [OUTPUT_DIR]
#
# 默认：
#   BASE_REF  = origin/main
#   HEAD_REF  = HEAD
#   OUTPUT_DIR = ./output/pr_context_<timestamp>
#
# 产出：
#   output/pr_context_<ts>/
#     ├── 00_summary.txt        # 一句话概览（commit 数、增删行、文件数）
#     ├── 01_commits.txt        # BASE..HEAD 的 commit 列表
#     ├── 02_stat.txt           # 每文件增删行
#     ├── 03_files.txt          # 变更文件清单
#     ├── 04_full_diff.txt      # 完整 diff（审查主输入）
#     ├── 05_messages.txt       # 各 commit 的完整 message
#     └── 06_meta.txt           # 仓库元信息 / branch / 是否 dirty
#
# 远程 PR 采集请用：
#   gh pr view {num} --json title,body,files,commits,baseRefName,headRefName
#   gh pr diff {num}
#   gh pr checks {num}
#
# 退出码：
#   0  成功
#   1  参数错误 / 不在 git 仓库
#   2  base ref 不存在

set -euo pipefail

BASE_REF="${1:-origin/main}"
HEAD_REF="${2:-HEAD}"
OUTPUT_DIR="${3:-./output/pr_context_$(date +%Y%m%d_%H%M%S)}"

log() { printf '[collect] %s\n' "$*" >&2; }
die() { printf '[collect][ERROR] %s\n' "$*" >&2; exit "${2:-1}"; }

# --- 前置检查 ---------------------------------------------------------------
command -v git >/dev/null || die "git not found"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not in a git repo"

# 检查 ref 是否存在；origin/main 可能需要 fetch
if ! git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null; then
  log "base ref '${BASE_REF}' 未找到，尝试 git fetch origin ..."
  git fetch --quiet origin 2>/dev/null || true
  git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null \
    || die "base ref '${BASE_REF}' 不存在，请先 git fetch 或指定正确的 base" 2
fi
git rev-parse --verify --quiet "${HEAD_REF}^{commit}" >/dev/null \
  || die "head ref '${HEAD_REF}' 不存在"

mkdir -p "${OUTPUT_DIR}"
log "output → ${OUTPUT_DIR}"

MERGE_BASE="$(git merge-base "${BASE_REF}" "${HEAD_REF}")"
log "merge-base = ${MERGE_BASE}"

# --- 00 summary -------------------------------------------------------------
{
  echo "repo:        $(git rev-parse --show-toplevel 2>/dev/null || echo '?')"
  echo "base ref:    ${BASE_REF}  ($(git rev-parse --short "${BASE_REF}"))"
  echo "head ref:    ${HEAD_REF}  ($(git rev-parse --short "${HEAD_REF}"))"
  echo "merge-base:  $(git rev-parse --short "${MERGE_BASE}")"
  echo "commits:     $(git rev-list --count "${MERGE_BASE}..${HEAD_REF}")"
  echo "files:       $(git diff --name-only "${MERGE_BASE}..${HEAD_REF}" | wc -l | tr -d ' ')"
  git diff --shortstat "${MERGE_BASE}..${HEAD_REF}" | sed 's/^/diff:        /'
  echo "dirty:       $(if [ -n "$(git status --porcelain)" ]; then echo yes; else echo no; fi)"
  echo "generated:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUTPUT_DIR}/00_summary.txt"

# --- 01 commits -------------------------------------------------------------
git log --oneline --no-decorate "${MERGE_BASE}..${HEAD_REF}" \
  > "${OUTPUT_DIR}/01_commits.txt"

# --- 02 stat ----------------------------------------------------------------
git diff --stat "${MERGE_BASE}..${HEAD_REF}" \
  > "${OUTPUT_DIR}/02_stat.txt"

# --- 03 files ---------------------------------------------------------------
git diff --name-status "${MERGE_BASE}..${HEAD_REF}" \
  > "${OUTPUT_DIR}/03_files.txt"

# --- 04 full diff -----------------------------------------------------------
# 约束行宽，便于审查工具读取
git diff --find-renames --find-copies "${MERGE_BASE}..${HEAD_REF}" \
  > "${OUTPUT_DIR}/04_full_diff.txt"

# --- 05 commit messages -----------------------------------------------------
{
  while IFS= read -r sha; do
    [ -z "${sha}" ] && continue
    echo "============================================================"
    git show --no-patch --format='commit %H%nAuthor: %an <%ae>%nDate:   %ad%n%n%B' "${sha}"
  done < <(git rev-list --reverse "${MERGE_BASE}..${HEAD_REF}")
} > "${OUTPUT_DIR}/05_messages.txt"

# --- 06 meta ----------------------------------------------------------------
{
  echo "## branch info"
  echo "current branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo
  echo "## remote"
  git remote -v
  echo
  echo "## top 10 recently touched files in diff (churn signal)"
  git log --name-only --pretty=format: "${MERGE_BASE}..${HEAD_REF}" \
    | grep -v '^$' | sort | uniq -c | sort -rn | head -n 10
  echo
  echo "## files by extension"
  git diff --name-only "${MERGE_BASE}..${HEAD_REF}" \
    | awk -F. 'NF>1{print $NF} NF==1{print "(no-ext)"}' \
    | sort | uniq -c | sort -rn
} > "${OUTPUT_DIR}/06_meta.txt"

log "done. files:"
ls -1 "${OUTPUT_DIR}" | sed 's/^/  /' >&2

# --- 可选：远程 PR 增强（需要 gh 且在 GitHub 仓） --------------------------
if command -v gh >/dev/null 2>&1; then
  REMOTE_URL="$(git config --get remote.origin.url 2>/dev/null || true)"
  case "${REMOTE_URL}" in
    *github.com*)
      log "检测到 GitHub remote，尝试用 gh 拉取 PR 列表（若已 push）"
      CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
      if [ -n "${CURRENT_BRANCH}" ] && [ "${CURRENT_BRANCH}" != "HEAD" ]; then
        # 找到当前分支对应的 open PR（若有）
        PR_NUM="$(gh pr list --head "${CURRENT_BRANCH}" --state open \
                    --json number --jq '.[0].number' 2>/dev/null || true)"
        if [ -n "${PR_NUM}" ]; then
          log "发现 PR #${PR_NUM}，补充远程视图"
          {
            echo "## PR #${PR_NUM} (auto-detected from branch ${CURRENT_BRANCH})"
            gh pr view "${PR_NUM}" --json title,author,body,baseRefName,headRefName,labels \
              2>/dev/null || echo "(gh pr view failed)"
            echo
            echo "## CI checks"
            gh pr checks "${PR_NUM}" 2>/dev/null || echo "(no checks)"
          } > "${OUTPUT_DIR}/07_github_pr.txt"
        fi
      fi
      ;;
  esac
fi

log "all done."
