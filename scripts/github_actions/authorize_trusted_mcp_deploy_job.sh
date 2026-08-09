#!/usr/bin/env bash
# Reject every job except the owner-triggered Deploy WSL MCP workflow on main.

set -euo pipefail

readonly EXPECTED_REPOSITORY="Motoki0705/tennis-lab"
readonly EXPECTED_OWNER="Motoki0705"
readonly EXPECTED_REF="refs/heads/main"
readonly EXPECTED_WORKFLOW="Deploy WSL MCP"
readonly EXPECTED_WORKFLOW_REF="${EXPECTED_REPOSITORY}/.github/workflows/deploy-wsl-mcp.yml@${EXPECTED_REF}"

reject() {
  echo "::error::trusted MCP deploy runner rejected job: $*" >&2
  exit 78
}

require_equal() {
  local name="$1" actual="$2" expected="$3"
  if [[ "$actual" != "$expected" ]]; then
    reject "$name must be '$expected'"
  fi
}

require_equal "GITHUB_REPOSITORY" "${GITHUB_REPOSITORY:-}" "$EXPECTED_REPOSITORY"
require_equal "GITHUB_ACTOR" "${GITHUB_ACTOR:-}" "$EXPECTED_OWNER"
require_equal \
  "GITHUB_TRIGGERING_ACTOR" \
  "${GITHUB_TRIGGERING_ACTOR:-}" \
  "$EXPECTED_OWNER"
require_equal "GITHUB_REF" "${GITHUB_REF:-}" "$EXPECTED_REF"
require_equal "GITHUB_WORKFLOW" "${GITHUB_WORKFLOW:-}" "$EXPECTED_WORKFLOW"
require_equal \
  "GITHUB_WORKFLOW_REF" \
  "${GITHUB_WORKFLOW_REF:-}" \
  "$EXPECTED_WORKFLOW_REF"
require_equal "GITHUB_WORKFLOW_SHA" "${GITHUB_WORKFLOW_SHA:-}" "${GITHUB_SHA:-}"
require_equal "GITHUB_JOB" "${GITHUB_JOB:-}" "deploy"

case "${GITHUB_EVENT_NAME:-}" in
  push | workflow_dispatch) ;;
  *) reject "GITHUB_EVENT_NAME must be 'push' or 'workflow_dispatch'" ;;
esac

if [[ ! "${GITHUB_SHA:-}" =~ ^[0-9a-f]{40}$ ]]; then
  reject "GITHUB_SHA must be a full lowercase commit SHA"
fi
if [[ ! -f "${GITHUB_EVENT_PATH:-}" ]]; then
  reject "GITHUB_EVENT_PATH must name a readable event payload"
fi

/usr/bin/timeout 10 /usr/bin/python3 - "$GITHUB_EVENT_PATH" "$GITHUB_SHA" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

event_path = Path(sys.argv[1])
expected_sha = sys.argv[2]
payload = json.loads(event_path.read_text(encoding="utf-8"))

repository = payload.get("repository")
sender = payload.get("sender")
if not isinstance(repository, dict) or repository.get("full_name") != "Motoki0705/tennis-lab":
    raise SystemExit("event repository is not Motoki0705/tennis-lab")
if not isinstance(sender, dict) or sender.get("login") != "Motoki0705":
    raise SystemExit("event sender is not Motoki0705")

event_ref = payload.get("ref")
if event_ref not in {"main", "refs/heads/main"}:
    raise SystemExit("event ref is not main")
if "after" in payload and payload["after"] != expected_sha:
    raise SystemExit("push event after SHA does not match GITHUB_SHA")
PY

echo "trusted MCP deploy job authorized for ${GITHUB_SHA}"
