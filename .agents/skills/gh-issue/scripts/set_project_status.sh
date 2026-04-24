#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  set_project_status.sh <issue-number-or-url> <status>

Examples:
  set_project_status.sh 409 "In progress"
  set_project_status.sh https://github.com/Motoki0705/tennis-lab/issues/409 Done

Environment overrides:
  GH_REPO              default: Motoki0705/tennis-lab
  GH_PROJECT_OWNER     default: Motoki0705
  GH_PROJECT_NUMBER    default: 3
USAGE
}

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 2 ]]; then
  usage >&2
  exit 1
fi

REPO="${GH_REPO:-Motoki0705/tennis-lab}"
PROJECT_OWNER="${GH_PROJECT_OWNER:-Motoki0705}"
PROJECT_NUMBER="${GH_PROJECT_NUMBER:-3}"
ISSUE_REF="$1"
STATUS_NAME="$2"

if ! command -v gh >/dev/null 2>&1; then
  fail "gh CLI is required."
fi

if ! gh auth status >/dev/null 2>&1; then
  fail "gh authentication failed. Run: gh auth login"
fi

if [[ "${ISSUE_REF}" =~ ^https://github\.com/([^/]+/[^/]+)/issues/([0-9]+)$ ]]; then
  REPO="${BASH_REMATCH[1]}"
  ISSUE_NUMBER="${BASH_REMATCH[2]}"
elif [[ "${ISSUE_REF}" =~ ^[0-9]+$ ]]; then
  ISSUE_NUMBER="${ISSUE_REF}"
else
  fail "Issue must be a number or https://github.com/<owner>/<repo>/issues/<number> URL."
fi

ISSUE_URL="https://github.com/${REPO}/issues/${ISSUE_NUMBER}"

PROJECT_ID="$(
  gh project view "${PROJECT_NUMBER}" \
    --owner "${PROJECT_OWNER}" \
    --format json \
    --jq '.id'
)"

if [[ -z "${PROJECT_ID}" ]]; then
  fail "Could not resolve project ID for ${PROJECT_OWNER} project ${PROJECT_NUMBER}."
fi

FIELD_ID="$(
  gh project field-list "${PROJECT_NUMBER}" \
    --owner "${PROJECT_OWNER}" \
    --format json \
    --jq '.fields[] | select(.name == "Status") | .id'
)"

if [[ -z "${FIELD_ID}" ]]; then
  fail "Could not resolve Status field in project ${PROJECT_NUMBER}."
fi

OPTION_ROW="$(
  gh project field-list "${PROJECT_NUMBER}" \
    --owner "${PROJECT_OWNER}" \
    --format json \
    --jq '.fields[] | select(.name == "Status") | .options[] | [.name, .id] | @tsv' |
  awk -F '\t' -v target="${STATUS_NAME}" '
    BEGIN { target_lc = tolower(target) }
    tolower($1) == target_lc { print $1 "\t" $2; found = 1; exit }
    END { if (!found) exit 1 }
  '
)" || {
  echo "[FAIL] Unknown status: ${STATUS_NAME}" >&2
  echo "Available statuses:" >&2
  gh project field-list "${PROJECT_NUMBER}" \
    --owner "${PROJECT_OWNER}" \
    --format json \
    --jq '.fields[] | select(.name == "Status") | .options[].name' >&2
  exit 1
}

OPTION_NAME="${OPTION_ROW%%$'\t'*}"
OPTION_ID="${OPTION_ROW#*$'\t'}"

ITEM_ID="$(
  gh project item-list "${PROJECT_NUMBER}" \
    --owner "${PROJECT_OWNER}" \
    --query "repo:${REPO} #${ISSUE_NUMBER}" \
    --format json \
    --jq ".items[] | select(.content.url == \"${ISSUE_URL}\") | .id"
)"

if [[ -z "${ITEM_ID}" ]]; then
  fail "Issue ${ISSUE_URL} is not in ${PROJECT_OWNER} project ${PROJECT_NUMBER}."
fi

gh project item-edit \
  --id "${ITEM_ID}" \
  --project-id "${PROJECT_ID}" \
  --field-id "${FIELD_ID}" \
  --single-select-option-id "${OPTION_ID}" \
  >/dev/null

echo "Set ${ISSUE_URL} Status to ${OPTION_NAME} in ${PROJECT_OWNER} project ${PROJECT_NUMBER}."
