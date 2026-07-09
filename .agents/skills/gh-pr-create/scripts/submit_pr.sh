#!/usr/bin/env bash
# Push the current branch and create a PR in Motoki0705/tennis-lab.
#
# This script makes no decisions. Run pr_preflight.py first, resolve its
# blockers, then call this. It re-checks only the non-negotiable preconditions
# and fails loudly rather than falling back to a different behaviour.
set -euo pipefail

REPO="Motoki0705/tennis-lab"
BASE="main"
BRANCH=""
TITLE=""
BODY_FILE=""
DRAFT=""
DRY_RUN=0
LABELS=()
REVIEWERS=()

usage() {
  cat <<'USAGE'
Usage:
  submit_pr.sh --title <title> --body-file <path> [options]

Options:
  --title <t>        PR title (Japanese by default). Required.
  --body-file <p>    PR body file, filled from .github/pull_request_template.md. Required.
  --base <b>         Base branch (default: main).
  --branch <b>       Head branch (default: current branch).
  --label <l>        Label to attach. Repeatable.
  --reviewer <r>     Reviewer to request. Repeatable.
  --draft            Create the PR as a draft.
  --dry-run          Print the commands that would run, then exit.
  -h, --help         Show this message.

Exit codes:
  0 PR created   1 usage error   2 precondition failed   4 an open PR already exists
USAGE
}

fail() {
  echo "[FAIL] $1" >&2
  exit "${2:-2}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title)    TITLE="${2:?--title needs a value}"; shift 2 ;;
    --body-file) BODY_FILE="${2:?--body-file needs a value}"; shift 2 ;;
    --base)     BASE="${2:?--base needs a value}"; shift 2 ;;
    --branch)   BRANCH="${2:?--branch needs a value}"; shift 2 ;;
    --label)    LABELS+=("${2:?--label needs a value}"); shift 2 ;;
    --reviewer) REVIEWERS+=("${2:?--reviewer needs a value}"); shift 2 ;;
    --draft)    DRAFT="--draft"; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *)          usage >&2; fail "unknown argument: $1" 1 ;;
  esac
done

[[ -n "${TITLE}" ]] || { usage >&2; fail "--title is required" 1; }
[[ -n "${BODY_FILE}" ]] || { usage >&2; fail "--body-file is required" 1; }

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "not inside a git work tree"
command -v gh >/dev/null 2>&1 || fail "gh is not installed"

if [[ -z "${BRANCH}" ]]; then
  BRANCH="$(git symbolic-ref --quiet --short HEAD)" || fail "HEAD is detached; check out a branch"
fi
[[ "${BRANCH}" != "${BASE}" ]] || fail "head branch equals base branch '${BASE}'"

[[ -f "${BODY_FILE}" ]] || fail "body file not found: ${BODY_FILE}"
[[ -s "${BODY_FILE}" ]] || fail "body file is empty: ${BODY_FILE}"
! grep -q '<!--' "${BODY_FILE}" || fail "body file still contains template comments: ${BODY_FILE}"

gh auth status >/dev/null 2>&1 || fail "gh auth status failed; re-authenticate"

if ((${#LABELS[@]})); then
  KNOWN="$(gh label list --repo "${REPO}" --limit 100 --json name --jq '.[].name')"
  for label in "${LABELS[@]}"; do
    grep -Fxq "${label}" <<<"${KNOWN}" || fail "label '${label}' does not exist in ${REPO}"
  done
fi

EXISTING="$(gh pr list --repo "${REPO}" --head "${BRANCH}" --state open --json url --jq '.[0].url // empty')"
if [[ -n "${EXISTING}" ]]; then
  fail "an open PR already exists for '${BRANCH}': ${EXISTING}" 4
fi

PUSH_CMD=(git push --set-upstream origin "${BRANCH}")
CREATE_CMD=(gh pr create --repo "${REPO}" --base "${BASE}" --head "${BRANCH}"
  --title "${TITLE}" --body-file "${BODY_FILE}" --assignee "@me")
for label in "${LABELS[@]+"${LABELS[@]}"}"; do CREATE_CMD+=(--label "${label}"); done
for reviewer in "${REVIEWERS[@]+"${REVIEWERS[@]}"}"; do CREATE_CMD+=(--reviewer "${reviewer}"); done
if [[ -n "${DRAFT}" ]]; then CREATE_CMD+=("${DRAFT}"); fi

if ((DRY_RUN)); then
  printf '[dry-run]'; printf ' %q' "${PUSH_CMD[@]}"; printf '\n'
  printf '[dry-run]'; printf ' %q' "${CREATE_CMD[@]}"; printf '\n'
  exit 0
fi

"${PUSH_CMD[@]}"
"${CREATE_CMD[@]}"
