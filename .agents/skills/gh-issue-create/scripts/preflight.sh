#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-Motoki0705/tennis-lab}"

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

if gh auth status >/dev/null 2>&1; then
  ok "gh authentication is valid."
else
  fail "gh auth status failed. Re-authenticate before creating or editing an issue."
fi

if gh repo view "${REPO}" >/dev/null 2>&1; then
  ok "Repository is reachable: ${REPO}"
else
  fail "Repository is not reachable: ${REPO}"
fi

templates=(
  ".github/ISSUE_TEMPLATE/01_bug.md"
  ".github/ISSUE_TEMPLATE/02_feature.md"
  ".github/ISSUE_TEMPLATE/03_research.md"
)

for template in "${templates[@]}"; do
  if [[ -f "${template}" ]]; then
    ok "Template exists: ${template}"
  else
    fail "Template is missing: ${template}"
  fi
done

required_labels=(bug enhancement documentation research question codex)
missing_labels=()
present_label_count=0

for label in "${required_labels[@]}"; do
  if gh label list --repo "${REPO}" --search "${label}" --limit 200 --json name --jq ".[].name" | grep -Fx "${label}" >/dev/null 2>&1; then
    ok "Label exists: ${label}"
    present_label_count=$((present_label_count + 1))
  else
    missing_labels+=("${label}")
  fi
done

if [[ "${#missing_labels[@]}" -gt 0 ]]; then
  warn "Missing labels: ${missing_labels[*]}"
else
  ok "All required labels are present."
fi

echo "Issue preflight summary:"
echo "  repo=${REPO}"
echo "  required_labels=${#required_labels[@]}"
echo "  present_labels=${present_label_count}"
echo "  missing_labels=${#missing_labels[@]}"
echo "Summary: ok=${ok_count} warn=${warn_count} fail=${fail_count}"
if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
