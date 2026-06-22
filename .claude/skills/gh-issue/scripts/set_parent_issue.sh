#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  set_parent_issue.sh <child-issue-number-or-url> <parent-issue-number-or-url>

Examples:
  set_parent_issue.sh 410 409
  set_parent_issue.sh https://github.com/Motoki0705/tennis-lab/issues/410 409

Environment overrides:
  GH_REPO              default: Motoki0705/tennis-lab
USAGE
}

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

parse_issue_ref() {
  local ref="$1"
  local default_repo="$2"

  if [[ "${ref}" =~ ^https://github\.com/([^/]+/[^/]+)/issues/([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
  elif [[ "${ref}" =~ ^[0-9]+$ ]]; then
    echo "${default_repo} ${ref}"
  else
    fail "Issue must be a number or https://github.com/<owner>/<repo>/issues/<number> URL: ${ref}"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 2 ]]; then
  usage >&2
  exit 1
fi

DEFAULT_REPO="${GH_REPO:-Motoki0705/tennis-lab}"
CHILD_REF="$1"
PARENT_REF="$2"

if ! command -v gh >/dev/null 2>&1; then
  fail "gh CLI is required."
fi

if ! gh auth status >/dev/null 2>&1; then
  fail "gh authentication failed. Run: gh auth login"
fi

read -r CHILD_REPO CHILD_NUMBER < <(parse_issue_ref "${CHILD_REF}" "${DEFAULT_REPO}")
read -r PARENT_REPO PARENT_NUMBER < <(parse_issue_ref "${PARENT_REF}" "${DEFAULT_REPO}")

PARENT_OWNER="${PARENT_REPO%%/*}"
PARENT_REPO_NAME="${PARENT_REPO#*/}"
CHILD_OWNER="${CHILD_REPO%%/*}"

if [[ "${CHILD_OWNER}" != "${PARENT_OWNER}" ]]; then
  fail "Child and parent issues must belong to repositories under the same owner."
fi

CHILD_ID="$(
  gh api "repos/${CHILD_REPO}/issues/${CHILD_NUMBER}" \
    --jq '.id'
)"

if [[ -z "${CHILD_ID}" ]]; then
  fail "Could not resolve child issue id for ${CHILD_REPO}#${CHILD_NUMBER}."
fi

gh api "repos/${PARENT_REPO}/issues/${PARENT_NUMBER}" --jq '.id' >/dev/null

CURRENT_PARENT_ROW="$(
  gh api "repos/${CHILD_REPO}/issues/${CHILD_NUMBER}/parent" \
    --jq '[.repository.full_name, (.number | tostring)] | @tsv' 2>/dev/null || true
)"

if [[ "${CURRENT_PARENT_ROW}" == "${PARENT_REPO}"$'\t'"${PARENT_NUMBER}" ]]; then
  echo "Parent already set: ${CHILD_REPO}#${CHILD_NUMBER} -> ${PARENT_REPO}#${PARENT_NUMBER}."
  exit 0
fi

gh api \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  "repos/${PARENT_OWNER}/${PARENT_REPO_NAME}/issues/${PARENT_NUMBER}/sub_issues" \
  -F sub_issue_id="${CHILD_ID}" \
  -F replace_parent=true \
  >/dev/null

echo "Set parent: ${CHILD_REPO}#${CHILD_NUMBER} -> ${PARENT_REPO}#${PARENT_NUMBER}."
