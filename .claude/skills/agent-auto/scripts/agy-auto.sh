#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent-auto-common.sh
source "$SCRIPT_DIR/agent-auto-common.sh"

usage() {
  cat <<'USAGE'
Usage:
  agy-auto.sh [options] "PROMPT"
  agy-auto.sh [options] -f PROMPT_FILE
  echo "PROMPT" | agy-auto.sh [options]

Run one Antigravity (agy) task non-interactively with structured logging.

Options:
  -f, --file FILE        Read the prompt from FILE.
  -d, --dir DIR          Working directory for the run (default: current dir).
  -m, --mode MODE        Approval mode (default: yolo). Maps yolo to --dangerously-skip-permissions.
      --model MODEL      Model alias/name.
      --sandbox          Explicitly enable agy's sandbox.
  -t, --print-timeout T  Max wall time agy --print waits before aborting
                         (default: 30m). agy's own default is only 5m, which
                         truncates long autonomous runs.
  -l, --log-dir DIR      Base dir for run logs (default: logs/agent-auto/agy).
      --stream           Not natively supported by agy --print. Ignored.
      --name NAME        Run slug (default: run).
      --dry-run          Print the agy command and exit.
  -h, --help             Show this help.

Exit codes:
  0  agy emitted a successful result and exited zero
  1  agy failed, timed out, or emitted no successful result
  2  usage / setup error
USAGE
}

PROMPT=""
PROMPT_FILE=""
WORKDIR="."
MODE="yolo"
MODEL=""
SANDBOX=0
PRINT_TIMEOUT="30m"
LOG_BASE="logs/agent-auto/agy"
STREAM=0
RUN_NAME=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -f|--file) PROMPT_FILE="${2:?}"; shift 2 ;;
    -d|--dir) WORKDIR="${2:?}"; shift 2 ;;
    -m|--mode) MODE="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --sandbox) SANDBOX=1; shift ;;
    -t|--print-timeout) PRINT_TIMEOUT="${2:?}"; shift 2 ;;
    -l|--log-dir) LOG_BASE="${2:?}"; shift 2 ;;
    --stream) STREAM=1; shift ;;
    --name) RUN_NAME="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; PROMPT="${*:-}"; break ;;
    -*) agent_auto_die "unknown option: $1 (see --help)" ;;
    *) PROMPT="$1"; shift ;;
  esac
done

agent_auto_require_command agy
agent_auto_resolve_prompt
agent_auto_validate_workdir
PYTHON="$(agent_auto_pick_python)"
agent_auto_prepare_run_dir "$LOG_BASE" "$RUN_NAME"

OUT_FILE="$RUN_DIR/output.txt"

CMD=(agy --print "$PROMPT")
if [[ "$MODE" == "yolo" || "$MODE" == "auto_edit" ]]; then
  CMD+=(--dangerously-skip-permissions)
fi
[[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")
[[ "$SANDBOX" -eq 1 ]] && CMD+=(--sandbox)
[[ -n "$PRINT_TIMEOUT" ]] && CMD+=(--print-timeout "$PRINT_TIMEOUT")

CMD_LINE="$(agent_auto_render_command)"
if [[ "$DRY_RUN" -eq 1 ]]; then
  agent_auto_print_dry_run
fi

agent_auto_initialize_log
echo "[agy-auto] mode=$MODE stream=$STREAM dir=$WORKDIR"
echo "[agy-auto] log: $RUN_DIR"
agent_auto_execute

set +e
"$PYTHON" - \
  "$OUT_FILE" \
  "$RUN_DIR/stderr.log" \
  "$RUN_DIR/result.txt" \
  "$RUN_DIR/summary.txt" \
  "$AGENT_AUTO_RC" \
  "$STREAM" <<'PY'
import json
import sys
import os

source, stderr_path, result_path, summary_path, cli_rc_text, stream_text = sys.argv[1:]
cli_rc = int(cli_rc_text)
session_id = "?"
result_status = "failed"
response = ""
error = None

# Read the plain text output
if os.path.exists(source):
    with open(source, encoding="utf-8") as stream:
        response = stream.read().strip()

# Check stderr for errors
stderr_text = ""
if os.path.exists(stderr_path):
    with open(stderr_path, encoding="utf-8") as stream:
        stderr_text = stream.read().strip()

# agy does not produce structured JSON, AND it returns exit 0 even when a
# print-mode run fails: on timeout it prints "Error: timed out waiting for
# response" and on a bad invocation "Error: empty prompt ...", both with rc 0.
# So rc==0 alone is NOT sufficient — also reject these stdout error sentinels.
# https://github.com/Motoki0705/tennis-lab/pull/561 documents the observed
# behavior (agy v1.0.11).
timed_out = response.startswith("Error: timed out")
cli_error = response.startswith("Error: ")

if cli_rc == 0 and not cli_error:
    success = True
    result_status = "success"
else:
    success = False
    if timed_out:
        result_status = "timed_out"
        message = response or "agy --print exceeded its --print-timeout"
    elif cli_error:
        message = response
    else:
        message = stderr_text or "Process exited with non-zero status"
    error = {"code": cli_rc, "message": message}

with open(result_path, "w", encoding="utf-8") as output:
    output.write(response)
    output.write("\n")
with open(summary_path, "w", encoding="utf-8") as output:
    output.write(f"status={'success' if success else 'failed'}\n")
    output.write(f"provider_status={result_status}\n")
    output.write(f"session_id={session_id}\n")
    output.write(f"agy_rc={cli_rc}\n")
    if error and isinstance(error, dict) and error.get("code") is not None:
        output.write(f"error_code={error['code']}\n")

sys.exit(0 if success else 1)
PY
PARSE_RC=$?
set -e

agent_auto_print_result "$PARSE_RC"
