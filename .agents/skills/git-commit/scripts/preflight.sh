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

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[FAIL] Not inside a git repository."
  exit 1
fi

if git diff --cached --quiet; then
  fail "No staged changes found."
else
  ok "Staged changes are present."
fi

staged_files="$(git diff --cached --name-status)"
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

echo "Summary: ok=${ok_count} warn=${warn_count} fail=${fail_count}"
if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi

