#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent-auto-common.sh
source "$SCRIPT_DIR/agent-auto-common.sh"

usage() {
  cat <<'USAGE'
Usage:
  claude-auto.sh [options] "PROMPT"
  claude-auto.sh [options] -f PROMPT_FILE
  echo "PROMPT" | claude-auto.sh [options]

Run one Claude Code task autonomously (no approval prompts) to completion.

Options:
  -f, --file FILE        Read the prompt from FILE (instead of an argument).
  -d, --dir DIR          Working directory for the run (default: current dir).
  -m, --mode MODE        Permission mode (default: bypassPermissions).
                         One of: bypassPermissions, acceptEdits, dontAsk,
                         auto, default. Use a scoped mode + --allow for
                         locked-down runs.
  -a, --allow TOOLS      --allowedTools list, e.g. "Bash(git *),Read,Edit".
      --disallow TOOLS   --disallowedTools list.
      --model MODEL      Model alias/name (e.g. opus, sonnet, haiku).
      --append-system S  Append S to the system prompt.
  -l, --log-dir DIR      Base dir for run logs (default: logs/agent-auto/claude).
      --stream           Stream live progress (stream-json) instead of buffering.
      --name NAME        Run slug (default: run).
      --dry-run          Print the claude command and exit.
  -h, --help             Show this help.

Exit codes:
  0  run completed with is_error=false
  1  run completed with is_error=true (or no result emitted)
  2  usage / setup error

Examples:
  claude-auto.sh "Run the test suite and fix any failures"
  claude-auto.sh -m dontAsk -a "Read,Bash(pytest *)" "Investigate the flaky test"
  claude-auto.sh -f task.md --stream --model sonnet
USAGE
}

PROMPT=""
PROMPT_FILE=""
WORKDIR="."
MODE="bypassPermissions"
ALLOW=""
DISALLOW=""
MODEL=""
APPEND_SYSTEM=""
LOG_BASE="logs/agent-auto/claude"
STREAM=0
RUN_NAME=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -f|--file) PROMPT_FILE="${2:?}"; shift 2 ;;
    -d|--dir) WORKDIR="${2:?}"; shift 2 ;;
    -m|--mode) MODE="${2:?}"; shift 2 ;;
    -a|--allow) ALLOW="${2:?}"; shift 2 ;;
    --disallow) DISALLOW="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --append-system) APPEND_SYSTEM="${2:?}"; shift 2 ;;
    -l|--log-dir) LOG_BASE="${2:?}"; shift 2 ;;
    --stream) STREAM=1; shift ;;
    --name) RUN_NAME="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; PROMPT="${*:-}"; break ;;
    -*) agent_auto_die "unknown option: $1 (see --help)" ;;
    *) PROMPT="$1"; shift ;;
  esac
done

agent_auto_require_command claude
agent_auto_resolve_prompt
agent_auto_validate_workdir
PYTHON="$(agent_auto_pick_python)"
agent_auto_prepare_run_dir "$LOG_BASE" "$RUN_NAME"

OUT_FORMAT="json"
OUT_FILE="$RUN_DIR/output.json"
if [[ "$STREAM" -eq 1 ]]; then
  OUT_FORMAT="stream-json"
  OUT_FILE="$RUN_DIR/stream.jsonl"
fi

CMD=(claude -p "$PROMPT"
  --permission-mode "$MODE"
  --output-format "$OUT_FORMAT")
[[ "$STREAM" -eq 1 ]] && CMD+=(--verbose --include-partial-messages)
[[ -n "$ALLOW" ]]    && CMD+=(--allowedTools "$ALLOW")
[[ -n "$DISALLOW" ]] && CMD+=(--disallowedTools "$DISALLOW")
[[ -n "$MODEL" ]]    && CMD+=(--model "$MODEL")
[[ -n "$APPEND_SYSTEM" ]] && CMD+=(--append-system-prompt "$APPEND_SYSTEM")

CMD_LINE="$(agent_auto_render_command)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  agent_auto_print_dry_run
fi

agent_auto_initialize_log
echo "[claude-auto] mode=$MODE stream=$STREAM dir=$WORKDIR"
echo "[claude-auto] log: $RUN_DIR"
agent_auto_execute

set +e
"$PYTHON" - "$OUT_FILE" "$RUN_DIR/result.txt" "$RUN_DIR/summary.txt" "$AGENT_AUTO_RC" <<'PY'
import json
import sys

source, result_path, summary_path, cli_rc = sys.argv[1:]
result = None
with open(source, encoding="utf-8") as stream:
    for line in stream:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and (
            record.get("type") == "result" or "is_error" in record
        ):
            result = record

if result is None:
    sys.exit(3)

is_error = bool(result.get("is_error", True))
with open(result_path, "w", encoding="utf-8") as output:
    output.write(str(result.get("result", "")))
    output.write("\n")
with open(summary_path, "w", encoding="utf-8") as output:
    output.write(f"is_error={str(is_error).lower()}\n")
    output.write(f"num_turns={result.get('num_turns', '?')}\n")
    output.write(f"session_id={result.get('session_id', '?')}\n")
    output.write(f"claude_rc={cli_rc}\n")

sys.exit(1 if is_error else 0)
PY
PARSE_RC=$?
set -e

if [[ "$PARSE_RC" -eq 3 ]]; then
  agent_auto_report_missing_result "claude"
fi

agent_auto_print_result "$PARSE_RC"
