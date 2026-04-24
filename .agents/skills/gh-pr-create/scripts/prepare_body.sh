#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  prepare_body.sh [template]

Creates a tmp PR body file from the PR template and prints the path.

Examples:
  prepare_body.sh
  prepare_body.sh .github/pull_request_template.md
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TEMPLATE="${1:-.github/pull_request_template.md}"

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "[FAIL] Template not found: ${TEMPLATE}" >&2
  exit 1
fi

BODY_FILE="$(mktemp /tmp/pr-body-XXXXXX.md)"
cp "${TEMPLATE}" "${BODY_FILE}"
echo "${BODY_FILE}"
