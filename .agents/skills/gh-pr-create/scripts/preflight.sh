#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${1:-main}"

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

if gh auth status >/dev/null 2>&1; then
  ok "gh authentication is valid."
else
  fail "gh auth status failed. Re-authenticate before creating a PR."
fi

current_branch="$(git branch --show-current)"
if [[ -n "${current_branch}" ]]; then
  ok "Current branch: ${current_branch}"
else
  fail "Could not resolve current branch."
fi

if [[ "${current_branch}" == "${BASE_BRANCH}" ]]; then
  warn "Current branch equals base branch (${BASE_BRANCH}). Confirm this is intentional."
fi

if git rev-parse --verify "${BASE_BRANCH}" >/dev/null 2>&1; then
  ok "Base branch exists locally: ${BASE_BRANCH}"
else
  fail "Base branch does not exist locally: ${BASE_BRANCH}"
fi

if [[ -f ".github/pull_request_template.md" ]]; then
  ok "PR template exists: .github/pull_request_template.md"
else
  fail "PR template is missing: .github/pull_request_template.md"
fi

if git rev-parse --verify "@{upstream}" >/dev/null 2>&1; then
  ok "Upstream branch is configured."
  upstream_ref="$(git rev-parse --abbrev-ref '@{upstream}')"
else
  warn "Upstream branch is not configured for current branch."
  upstream_ref="<none>"
fi

commit_list="$(git log --oneline "${BASE_BRANCH}..HEAD" 2>/dev/null || true)"
commit_count="$(printf '%s\n' "${commit_list}" | awk 'NF>0 {count++} END {print count+0}')"
if [[ -n "${commit_list}" ]]; then
  echo "Commits relative to ${BASE_BRANCH}:"
  echo "${commit_list}"
  ok "Found commits to include in PR."
else
  fail "No commits found relative to ${BASE_BRANCH}."
fi

echo "PR summary:"
echo "  branch=${current_branch:-<unknown>}"
echo "  base=${BASE_BRANCH}"
echo "  upstream=${upstream_ref}"
echo "  commits_relative_to_base=${commit_count}"
summarize_worktree
echo "Summary: ok=${ok_count} warn=${warn_count} fail=${fail_count}"
if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
