#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# 0. Argument Parsing & Setup
# ----------------------------------------------------------------------

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $(basename "$0")"
  echo "Runs pre-commit on currently changed files (staged + unstaged)."
  echo "If failures occur, delegates to codex sub-agent with relaxed read permissions."
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

# Log directory setup (Keep logs localized, but don't isolate auth/cache)
log_dir="${CODEX_SUBAGENT_LOG_DIR:-$repo_root/agents_workspace/sub_agents/logs}"
mkdir -p "$log_dir"

run_id="$(date +%Y%m%d_%H%M%S)"
check_log="$log_dir/pre_commit_${run_id}.log"
codex_log="$log_dir/pre_commit_codex_${run_id}.log"

# ----------------------------------------------------------------------
# 1. Identify Target Files
# ----------------------------------------------------------------------

# Get list of changed files (staged and unstaged) relative to HEAD
# This avoids --all-files causing massive unrelated changes.
mapfile -d '' -t files_to_check < <(git diff --name-only -z HEAD 2>/dev/null || true)

# Also include untracked files that are meant to be added (optional, but safer to skip if empty)
if (( ${#files_to_check[@]} == 0 )); then
  printf '{"status":"pass","fixed":false,"files_touched":[],"remaining_errors":[],"summary":"No changed files to check","needs_main":false,"message_for_main":""}\n'
  exit 0
fi

# Construct the check command targeting only changed files
# using xargs to handle file list safely is better, but simple expansion works for typical agent tasks.
# We use --no-sync to avoid network calls during uv invocation.
check_cmd="uv run --no-sync pre-commit run --show-diff-on-failure --files"
check_cmd_display="$check_cmd"
for f in "${files_to_check[@]}"; do
  check_cmd_display+=" $(printf '%q' "$f")"
done

# ----------------------------------------------------------------------
# 2. Run pre-commit (Pass 1)
# ----------------------------------------------------------------------

set +e
uv run --no-sync pre-commit run --show-diff-on-failure --files "${files_to_check[@]}" >"$check_log" 2>&1
check_ec=$?
set -e

if [[ $check_ec -eq 0 ]]; then
  printf '{"status":"pass","fixed":false,"files_touched":[],"remaining_errors":[],"summary":"pre-commit passed","needs_main":false,"message_for_main":""}\n'
  exit 0
fi

# ----------------------------------------------------------------------
# 3. Auto-fix Retry (Pass 2)
# ----------------------------------------------------------------------

# If hooks auto-modified files (e.g., ruff --fix), a second run commonly succeeds.
fixed=false
if grep -q "files were modified by this hook" "$check_log" || [[ -n "$(git diff --name-only)" ]]; then
  fixed=true
  set +e
  uv run --no-sync pre-commit run --show-diff-on-failure --files "${files_to_check[@]}" >>"$check_log" 2>&1
  check_ec=$?
  set -e
  
  if [[ $check_ec -eq 0 ]]; then
    files_touched_csv="$(git diff --name-only | paste -sd, -)"
    files_json="[]"
    if [[ -n "$files_touched_csv" ]]; then
      files_json="[\"${files_touched_csv//,/\",\"}\"]"
    fi
    printf '{"status":"pass","fixed":%s,"files_touched":%s,"remaining_errors":[],"summary":"pre-commit passed after auto-fix","needs_main":false,"message_for_main":""}\n' "$fixed" "$files_json"
    exit 0
  fi
fi

# ----------------------------------------------------------------------
# 4. Delegate to Codex (if failed)
# ----------------------------------------------------------------------

# Identify files involved in errors to define the "Modification Scope"
error_files_csv="$(
  {
    grep -oE '([A-Za-z0-9_./-]+\.(py|pyi))' "$check_log" || true
    # Always allow config files
    printf '%s\n' ".pre-commit-config.yaml" "pyproject.toml"
  } \
    | sed -E 's#^\./##' \
    | sort -u \
    | paste -sd, -
)"

schema_file="$(mktemp)"
out_file="$(mktemp)"
trap 'rm -f "$schema_file" "$out_file"' EXIT

# JSON Schema for the output
cat >"$schema_file" <<'JSON'
{
  "type": "object",
  "properties": {
    "status": {"type": "string", "enum": ["pass", "fail"]},
    "fixed": {"type": "boolean"},
    "files_touched": {"type": "array", "items": {"type": "string"}},
    "remaining_errors": {"type": "array", "items": {"type": "string"}},
    "summary": {"type": "string"},
    "needs_main": {"type": "boolean"},
    "message_for_main": {"type": "string"}
  },
  "required": [
    "status",
    "fixed",
    "files_touched",
    "remaining_errors",
    "summary",
    "needs_main",
    "message_for_main"
  ],
  "additionalProperties": false
}
JSON

# Prompt with relaxed read permissions
cat >"$out_file" <<EOF
You are a specialized sub-agent for fixing pre-commit failures (ruff, mypy, etc.) in a Python repo.

Target Files (Modify these):
${error_files_csv}

Constraints & Permissions:
1. **Modification**: You may ONLY modify the "Target Files" listed above.
2. **Read Access**: You are ALLOWED to read any file in the repository (e.g., to check function definitions, imports, or configs) if it helps resolve the error.
3. **Command**: Do not run any commands other than the provided pre-commit command.

Task:
- Analyze the logs and fix the issues so that the command passes:
  ${check_cmd_display}
- Iterate until clean.

Return JSON matching the schema.
EOF

# Check for network restriction (Fallback logic)
if [[ "${CODEX_SANDBOX_NETWORK_DISABLED:-}" == "1" ]]; then
  # Safely escape logs for JSON
  escaped_check_log="$(printf '%s' "$check_log" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  escaped_codex_log="$(printf '%s' "$codex_log" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  
  printf '{"status":"fail","fixed":%s,"files_touched":[],"remaining_errors":[],"summary":"pre-commit failed (network disabled; cannot run codex exec)","needs_main":true,"message_for_main":"Re-run in a network-enabled environment or fix manually. See logs: %s and %s"}\n' \
    "$fixed" \
    "${escaped_check_log}" \
    "${escaped_codex_log}"
  exit 0
fi

# Execute Codex Sub-agent
set +e
tmp_json="$(mktemp)"
codex_out="$(codex exec --sandbox danger-full-access --output-schema "$schema_file" - <"$out_file" 2>"$codex_log")"
codex_ec=$?
set -e

# Handle Codex failure
if [[ $codex_ec -ne 0 || -z "$codex_out" ]]; then
  escaped_check_log="$(printf '%s' "$check_log" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  escaped_codex_log="$(printf '%s' "$codex_log" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"

  printf '{"status":"fail","fixed":%s,"files_touched":[],"remaining_errors":[],"summary":"pre-commit failed (codex exec failed)","needs_main":true,"message_for_main":"Inspect logs: %s and %s"}\n' \
    "$fixed" \
    "${escaped_check_log}" \
    "${escaped_codex_log}"
  exit 0
fi

# Output the JSON from Codex
printf '%s\n' "$codex_out"