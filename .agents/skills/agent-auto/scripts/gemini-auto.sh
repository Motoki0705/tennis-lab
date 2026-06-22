#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent-auto-common.sh
source "$SCRIPT_DIR/agent-auto-common.sh"

usage() {
  cat <<'USAGE'
Usage:
  gemini-auto.sh [options] "PROMPT"
  gemini-auto.sh [options] -f PROMPT_FILE
  echo "PROMPT" | gemini-auto.sh [options]

Run one Gemini CLI task non-interactively with structured logging.

Options:
  -f, --file FILE        Read the prompt from FILE.
  -d, --dir DIR          Working directory for the run (default: current dir).
  -m, --mode MODE        Approval mode (default: yolo).
      --model MODEL      Model alias/name.
      --sandbox          Explicitly enable Gemini's sandbox.
      --no-skip-trust    Do not pass the session-only workspace trust flag.
  -l, --log-dir DIR      Base dir for run logs (default: logs/agent-auto/gemini).
      --stream           Stream JSONL events while retaining the run log.
      --name NAME        Run slug (default: run).
      --dry-run          Print the Gemini command and exit.
  -h, --help             Show this help.

Exit codes:
  0  Gemini emitted a successful structured result and exited zero
  1  Gemini failed or emitted no successful result
  2  usage / setup error
USAGE
}

PROMPT=""
PROMPT_FILE=""
WORKDIR="."
MODE="yolo"
MODEL=""
SANDBOX=0
SKIP_TRUST=1
LOG_BASE="logs/agent-auto/gemini"
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
    --no-skip-trust) SKIP_TRUST=0; shift ;;
    -l|--log-dir) LOG_BASE="${2:?}"; shift 2 ;;
    --stream) STREAM=1; shift ;;
    --name) RUN_NAME="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; PROMPT="${*:-}"; break ;;
    -*) agent_auto_die "unknown option: $1 (see --help)" ;;
    *) PROMPT="$1"; shift ;;
  esac
done

agent_auto_require_command gemini
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

CMD=(gemini -p "$PROMPT" --approval-mode "$MODE" --output-format "$OUT_FORMAT")
[[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")
[[ "$SANDBOX" -eq 1 ]] && CMD+=(--sandbox)
[[ "$SKIP_TRUST" -eq 1 ]] && CMD+=(--skip-trust)

CMD_LINE="$(agent_auto_render_command)"
if [[ "$DRY_RUN" -eq 1 ]]; then
  agent_auto_print_dry_run
fi

agent_auto_initialize_log
echo "[gemini-auto] mode=$MODE stream=$STREAM dir=$WORKDIR"
echo "[gemini-auto] log: $RUN_DIR"
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

source, stderr_path, result_path, summary_path, cli_rc_text, stream_text = sys.argv[1:]
cli_rc = int(cli_rc_text)
stream_mode = stream_text == "1"
session_id = "?"
result_status = "failed"
response = ""
error = None

with open(source, encoding="utf-8") as stream:
    text = stream.read()

if stream_mode:
    chunks = []
    found_result = False
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = record.get("type")
        if event_type == "init":
            session_id = record.get("session_id", "?")
        elif event_type == "message" and record.get("role") == "assistant":
            chunks.append(str(record.get("content", "")))
        elif event_type == "result":
            found_result = True
            result_status = record.get("status", "failed")
            error = record.get("error")
    response = "".join(chunks)
    success = cli_rc == 0 and found_result and result_status == "success" and not error
else:
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        try:
            with open(stderr_path, encoding="utf-8") as stream:
                record = json.load(stream)
        except (FileNotFoundError, json.JSONDecodeError):
            record = None
    if isinstance(record, dict):
        session_id = record.get("session_id", "?")
        response = str(record.get("response", ""))
        error = record.get("error")
        result_status = "success" if not error else "failed"
    success = cli_rc == 0 and isinstance(record, dict) and not error

with open(result_path, "w", encoding="utf-8") as output:
    output.write(response)
    output.write("\n")
with open(summary_path, "w", encoding="utf-8") as output:
    output.write(f"status={'success' if success else 'failed'}\n")
    output.write(f"provider_status={result_status}\n")
    output.write(f"session_id={session_id}\n")
    output.write(f"gemini_rc={cli_rc}\n")
    if isinstance(error, dict) and error.get("code") is not None:
        output.write(f"error_code={error['code']}\n")

sys.exit(0 if success else 1)
PY
PARSE_RC=$?
set -e

agent_auto_print_result "$PARSE_RC"
