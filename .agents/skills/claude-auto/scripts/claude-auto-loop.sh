#!/usr/bin/env bash
#
# claude-auto-loop.sh — drive Claude Code autonomously across multiple turns
# until the task is genuinely finished or a safety cap is hit.
#
# A single `claude -p` already runs the agent loop to completion, but a very
# large task can stop early (context limits, an interim "what next?" finish, a
# transient error). This wrapper resumes the same session and re-prompts it to
# keep going until it emits a completion SENTINEL, bounded by --max-iters.
#
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  claude-auto-loop.sh [options] "PROMPT"
  claude-auto-loop.sh [options] -f PROMPT_FILE

Run Claude Code autonomously, resuming until it signals completion.

Options:
  -f, --file FILE        Read the prompt from FILE.
  -d, --dir DIR          Working directory (default: current dir).
  -i, --max-iters N      Max turns before giving up (default: 10).
  -s, --sentinel STR     Completion marker Claude must print (default: TASK_COMPLETE).
  -m, --mode MODE        Permission mode (default: bypassPermissions).
  -a, --allow TOOLS      --allowedTools list.
      --model MODEL      Model alias/name.
  -l, --log-dir DIR      Base dir for run logs (default: logs/claude-auto).
      --name NAME        Run slug (default: timestamp).
  -h, --help             Show this help.

Exit codes:
  0  sentinel observed — task reported complete
  1  iteration cap hit without sentinel, or a turn errored
  2  usage / setup error
USAGE
}

die() { echo "[FAIL] $*" >&2; exit 2; }

pick_python() {
  if [[ -x ".venv/bin/python" ]]; then echo ".venv/bin/python";
  elif command -v python3 >/dev/null 2>&1; then echo "python3";
  elif command -v python  >/dev/null 2>&1; then echo "python";
  else die "no python found for JSON parsing"; fi
}

json_field() {  # json_field FILE FIELD  (FILE is a single JSON result object)
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    obj = json.load(fh)
val = obj.get(sys.argv[2], "")
print(val if not isinstance(val, bool) else str(val).lower())
PY
}

# --- defaults ---
PROMPT=""; PROMPT_FILE=""; WORKDIR="."; MAX_ITERS=10
SENTINEL="TASK_COMPLETE"; MODE="bypassPermissions"; ALLOW=""; MODEL=""
LOG_BASE="logs/claude-auto"; RUN_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -f|--file) PROMPT_FILE="${2:?}"; shift 2 ;;
    -d|--dir) WORKDIR="${2:?}"; shift 2 ;;
    -i|--max-iters) MAX_ITERS="${2:?}"; shift 2 ;;
    -s|--sentinel) SENTINEL="${2:?}"; shift 2 ;;
    -m|--mode) MODE="${2:?}"; shift 2 ;;
    -a|--allow) ALLOW="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    -l|--log-dir) LOG_BASE="${2:?}"; shift 2 ;;
    --name) RUN_NAME="${2:?}"; shift 2 ;;
    -*) die "unknown option: $1 (see --help)" ;;
    *) PROMPT="$1"; shift ;;
  esac
done

command -v claude >/dev/null 2>&1 || die "claude CLI not found on PATH"
PYTHON="$(pick_python)"
if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || die "prompt file not found: $PROMPT_FILE"
  PROMPT="$(cat "$PROMPT_FILE")"
fi
[[ -n "${PROMPT// /}" ]] || { usage; die "no prompt provided"; }
[[ -d "$WORKDIR" ]] || die "working dir not found: $WORKDIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "$RUN_NAME" ]]; then
  SLUG="$(echo "$RUN_NAME" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/-*$//' | cut -c1-48)"
  RUN_DIR="${LOG_BASE%/}/${TS}-${SLUG}"
else
  RUN_DIR="${LOG_BASE%/}/${TS}-loop"
fi
mkdir -p "$RUN_DIR"
printf '%s\n' "$PROMPT" > "$RUN_DIR/prompt.txt"

# Instruction appended to every turn so Claude knows the completion protocol.
PROTO="When the ENTIRE task is fully complete and verified, print this exact line on its own: ${SENTINEL}
If it is not yet complete, keep working — do not print that line until everything is done."

build_cmd() {  # build_cmd <resume_sid|"">  -> populates global CMD array
  CMD=(claude -p "$TURN_PROMPT"
    --permission-mode "$MODE"
    --output-format json)
  [[ -n "$1" ]]     && CMD+=(--resume "$1")
  [[ -n "$ALLOW" ]] && CMD+=(--allowedTools "$ALLOW")
  [[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")
  return 0  # never let a false &&-test become the function's exit status (set -e)
}

echo "[loop] sentinel='$SENTINEL' max_iters=$MAX_ITERS mode=$MODE"
echo "[loop] log: $RUN_DIR"

SID=""; STATUS="incomplete"
for (( i=1; i<=MAX_ITERS; i++ )); do
  if [[ $i -eq 1 ]]; then
    TURN_PROMPT="${PROMPT}

${PROTO}"
  else
    TURN_PROMPT="Continue the task. ${PROTO}"
  fi
  build_cmd "$SID"

  OUT="$RUN_DIR/turn-$(printf '%02d' "$i").json"
  echo "[loop] turn $i/$MAX_ITERS"
  set +e
  ( cd "$WORKDIR" && "${CMD[@]}" ) > "$OUT" 2>>"$RUN_DIR/stderr.log"
  set -e

  if ! IS_ERROR="$(json_field "$OUT" is_error 2>/dev/null)"; then
    echo "[loop] turn $i produced no result object — aborting" >&2
    break
  fi
  [[ -z "$SID" ]] && SID="$(json_field "$OUT" session_id)"
  RESULT="$(json_field "$OUT" result)"
  echo "    is_error=$IS_ERROR"

  if [[ "$IS_ERROR" != "false" ]]; then
    echo "[loop] turn $i errored: $RESULT" >&2
    STATUS="errored"; break
  fi
  if printf '%s' "$RESULT" | grep -qF "$SENTINEL"; then
    echo "[loop] sentinel observed on turn $i — task complete"
    STATUS="complete"; break
  fi
done

{
  echo "status=$STATUS"
  echo "session_id=$SID"
  echo "iterations=$i"
} > "$RUN_DIR/summary.txt"

echo "[loop] status=$STATUS session=$SID"
[[ "$STATUS" == "complete" ]] && exit 0 || exit 1
