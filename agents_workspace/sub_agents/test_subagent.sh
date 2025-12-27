#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# 0. Argument Parsing & Setup
# ----------------------------------------------------------------------

 repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
 cd "$repo_root"

 sandbox_env_sh="$repo_root/agents_workspace/sub_agents/sandbox_env.sh"
 if [[ -f "$sandbox_env_sh" ]]; then
   # shellcheck disable=SC1090
   source "$sandbox_env_sh"
   codex_sandbox_env_setup
 fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $(basename "$0") [--test-cmd '...']"
  echo "Default test command: uv run --no-sync pytest -q"
  exit 0
fi

test_cmd="uv run --no-sync pytest -q"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-cmd)
      test_cmd="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

# Log directory setup (Keep logs localized, but don't isolate auth/cache)
log_dir="${CODEX_SUBAGENT_LOG_DIR:-$repo_root/agents_workspace/sub_agents/logs}"
mkdir -p "$log_dir"

run_id="$(date +%Y%m%d_%H%M%S)"
test_log="$log_dir/pytest_${run_id}.log"
codex_log="$log_dir/pytest_codex_${run_id}.log"

# ----------------------------------------------------------------------
# 1. Run Tests
# ----------------------------------------------------------------------

set +e
bash -lc "$test_cmd" >"$test_log" 2>&1
test_ec=${PIPESTATUS[0]}
set -e

if [[ $test_ec -eq 0 ]]; then
  printf '{"status":"pass","fixed":false,"files_touched":[],"remaining_failures":[],"summary":"tests passed","needs_main":false,"message_for_main":""}\n'
  exit 0
fi

# ----------------------------------------------------------------------
# 2. Delegate to Codex (if failed)
# ----------------------------------------------------------------------

# Identify files involved in errors from the log
files_csv="$(
  {
    grep -E '^FAILED ' "$test_log" | sed -E 's/^FAILED ([^: ]+).*/\1/'
    grep -oE 'File "[^"]+\.py"' "$test_log" | sed -E 's/^File "(.*)"$/\1/'
    grep -oE '^[^[:space:]]+\.py:[0-9]+:' "$test_log" | sed -E 's/:.*$//'
  } \
  | sed 's/^\s*//;s/\s*$//' \
  | grep -E '\.(py|pyi)$' \
  | awk -v root="$repo_root" '{
      p=$0;
      if (index(p, root "/") == 1) { p=substr(p, length(root)+2); }
      print p;
    }' \
  | awk 'NF>0 {print}' \
  | sort -u \
  | paste -sd, -
)"

schema_file="$(mktemp)"
out_file="$(mktemp)"
trap 'rm -f "$schema_file" "$out_file"' EXIT

cat >"$schema_file" <<'JSON'
{
  "type": "object",
  "properties": {
    "status": {"type": "string", "enum": ["pass", "fail"]},
    "fixed": {"type": "boolean"},
    "files_touched": {"type": "array", "items": {"type": "string"}},
    "remaining_failures": {"type": "array", "items": {"type": "string"}},
    "summary": {"type": "string"},
    "needs_main": {"type": "boolean"},
    "message_for_main": {"type": "string"}
  },
  "required": [
    "status",
    "fixed",
    "files_touched",
    "remaining_failures",
    "summary",
    "needs_main",
    "message_for_main"
  ],
  "additionalProperties": false
}
JSON

# Prompt with relaxed read permissions
cat >"$out_file" <<EOF
You are a specialized sub-agent for triaging and fixing failing Python tests.

Target Files (Modify these):
${files_csv}

Constraints & Permissions:
1. **Modification**: You may ONLY modify the "Target Files" listed above.
2. **Read Access**: You are ALLOWED to read any file in the repository (e.g., to check source code, fixtures, or configs) if it helps fix the test.
3. **Command**: Do not run any commands other than the provided test command.

Task:
- Fix the failures so that this command passes with exit code 0:
  ${test_cmd}
- You may re-run the command multiple times.
- If the fix is straightforward and can be done within the allowed file set, implement it.
- If you determine the root cause requires modifying files outside the allowed list, set needs_main=true and explain.
- Prefer permanent, root-cause fixes; avoid temporary suppression (e.g., skipping/xfail-ing tests, broad `# type: ignore`, loosening configs) unless there is no reasonable alternative—if unavoidable, explain and set `needs_main=true`.

Return JSON that matches the provided output schema.
EOF

# Check for network restriction (Fallback logic)
if [[ "${CODEX_SANDBOX_NETWORK_DISABLED:-}" == "1" ]]; then
  # JSON output with log paths (manual escaping for safety not strictly needed for paths, but good practice if content included)
  printf '{"status":"fail","fixed":false,"files_touched":[],"remaining_failures":[],"summary":"tests failed (network disabled; cannot run codex exec)","needs_main":true,"message_for_main":"Re-run in a network-enabled environment or fix manually. See logs: %s and %s"}\n' \
    "$test_log" \
    "$codex_log"
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
  printf '{"status":"fail","fixed":false,"files_touched":[],"remaining_failures":[],"summary":"tests failed (codex exec failed)","needs_main":true,"message_for_main":"Inspect logs: %s and %s"}\n' \
    "$test_log" \
    "$codex_log"
  exit 0
fi

printf '%s\n' "$codex_out"
