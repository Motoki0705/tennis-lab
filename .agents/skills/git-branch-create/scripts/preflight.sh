#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${1:-main}"
TARGET_BRANCH="${2:-}"

ok_count=0
warn_count=0
fail_count=0

ok() {
  echo "[OK] $1"
  ok_count=$((ok_count + 1))
}

warn() {
  echo "[WARN] $1"
  warn_count=$((warn_count + 1))
}

fail() {
  echo "[FAIL] $1"
  fail_count=$((fail_count + 1))
}

summarize_worktree() {
  local status_output unstaged staged untracked
  status_output="$(git status --short 2>/dev/null || true)"
  unstaged="$(printf '%s\n' "${status_output}" | awk 'substr($0,1,1)!=" " && substr($0,1,1)!="?" && NF>0 {count++} END {print count+0}')"
  staged="$(printf '%s\n' "${status_output}" | awk 'substr($0,2,1)!=" " && substr($0,1,1)!="?" && NF>0 {count++} END {print count+0}')"
  untracked="$(printf '%s\n' "${status_output}" | awk 'substr($0,1,2)=="??" {count++} END {print count+0}')"
  echo "Working tree summary:"
  echo "  unstaged=${unstaged} staged=${staged} untracked=${untracked}"
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[FAIL] Not inside a git repository."
  exit 1
fi

current_branch="$(git branch --show-current)"
if [[ -n "${current_branch}" ]]; then
  ok "Current branch: ${current_branch}"
else
  fail "Could not resolve current branch."
fi

if git show-ref --verify --quiet "refs/heads/${BASE_BRANCH}"; then
  ok "Local base branch exists: ${BASE_BRANCH}"
else
  fail "Local base branch is missing: ${BASE_BRANCH}"
fi

if git show-ref --verify --quiet "refs/remotes/origin/${BASE_BRANCH}"; then
  ok "Remote-tracking base branch exists: origin/${BASE_BRANCH}"
else
  warn "Remote-tracking branch origin/${BASE_BRANCH} is missing. Run git fetch origin."
fi

if git diff --quiet && git diff --cached --quiet; then
  ok "Working tree is clean."
else
  warn "Working tree has changes. Confirm branching from this state is intentional."
fi

if [[ -n "${TARGET_BRANCH}" ]]; then
  if git show-ref --verify --quiet "refs/heads/${TARGET_BRANCH}"; then
    fail "Target branch already exists locally: ${TARGET_BRANCH}"
  else
    ok "Target branch does not exist locally: ${TARGET_BRANCH}"
  fi

  if git ls-remote --exit-code --heads origin "${TARGET_BRANCH}" >/dev/null 2>&1; then
    fail "Target branch already exists on origin: ${TARGET_BRANCH}"
  else
    ok "Target branch does not exist on origin: ${TARGET_BRANCH}"
  fi
else
  warn "Target branch name not provided. Skip branch name collision checks."
fi

echo "Branch summary:"
echo "  current=${current_branch:-<unknown>}"
echo "  base=${BASE_BRANCH}"
echo "  target=${TARGET_BRANCH:-<not-provided>}"
summarize_worktree
echo "Summary: ok=${ok_count} warn=${warn_count} fail=${fail_count}"
if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
