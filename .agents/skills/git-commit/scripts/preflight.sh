#!/usr/bin/env bash
set -euo pipefail

MESSAGE="${1:-}"

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
  warn "Could not resolve current branch."
fi

if git diff --cached --quiet; then
  fail "No staged changes found."
else
  ok "Staged changes are present."
fi

staged_files="$(git diff --cached --name-status)"
staged_file_count="$(printf '%s\n' "${staged_files}" | awk 'NF>0 {count++} END {print count+0}')"
if [[ -n "${staged_files}" ]]; then
  echo "Staged files:"
  echo "${staged_files}"
  ok "Printed staged file list."
else
  warn "Could not print staged file list."
fi

staged_stat="$(git diff --cached --stat)"
if [[ -n "${staged_stat}" ]]; then
  echo "Staged diff stat:"
  echo "${staged_stat}"
  ok "Printed staged diff stat."
else
  warn "Staged diff stat is empty."
fi

if [[ -n "${MESSAGE}" ]]; then
  if [[ "${MESSAGE}" =~ ^(feat|fix|docs|chore|refactor|style|test|ci|perf):\ [A-Z][^[:cntrl:]]+$ ]]; then
    ok "Commit message format looks valid: prefix: Subject"
  else
    warn "Commit message does not match expected format: prefix: Subject"
  fi
else
  warn "Commit message was not provided. Skip message format validation."
fi

echo "Commit summary:"
echo "  branch=${current_branch:-<unknown>}"
echo "  staged_files=${staged_file_count}"
summarize_worktree
echo "Summary: ok=${ok_count} warn=${warn_count} fail=${fail_count}"
if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
