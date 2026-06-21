#!/usr/bin/env bash
#
# claude-auto.sh — run a single Claude Code task fully autonomously (no approval
# prompts) to completion, with a budget cap, structured logging, and a
# machine-checkable result.
#
# A single `claude -p` invocation already runs the full agent loop until the
# task is done; this wrapper adds the safety rails and the success/failure
# detection that headless automation needs (Claude exits 0 even when the run
# failed, so we re-derive the exit code from the JSON `is_error` field).
#
set -euo pipefail

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
  -b, --budget USD       Hard spend cap via --max-budget-usd (default: 2.00).
  -m, --mode MODE        Permission mode (default: bypassPermissions).
                         One of: bypassPermissions, acceptEdits, dontAsk,
                         auto, default. Use a scoped mode + --allow for
                         locked-down runs.
  -a, --allow TOOLS      --allowedTools list, e.g. "Bash(git *),Read,Edit".
      --disallow TOOLS   --disallowedTools list.
      --model MODEL      Model alias/name (e.g. opus, sonnet, haiku).
      --append-system S  Append S to the system prompt.
  -l, --log-dir DIR      Base dir for run logs (default: logs/claude-auto).
      --stream           Stream live progress (stream-json) instead of buffering.
      --name NAME        Run slug (default: timestamp).
      --dry-run          Print the claude command and exit.
  -h, --help             Show this help.

Exit codes:
  0  run completed with is_error=false
  1  run completed with is_error=true (or no result emitted)
  2  usage / setup error

Examples:
  claude-auto.sh "Run the test suite and fix any failures"
  claude-auto.sh -b 5 -m dontAsk -a "Read,Bash(pytest *)" "Investigate the flaky test"
  claude-auto.sh -f task.md --stream --model sonnet
USAGE
}

die() { echo "[FAIL] $*" >&2; exit 2; }

# --- pick a python for JSON parsing (jq is not guaranteed to be installed) ---
pick_python() {
  if [[ -x ".venv/bin/python" ]]; then echo ".venv/bin/python";
  elif command -v python3 >/dev/null 2>&1; then echo "python3";
  elif command -v python  >/dev/null 2>&1; then echo "python";
  else die "no python found for JSON parsing"; fi
}

# json_field FILE FIELD  — print a top-level field from a JSON result object.
# In --stream mode FILE is JSONL; the last object with type=="result" wins.
json_field() {
  local file="$1" field="$2"
  "$PYTHON" - "$file" "$field" <<'PY'
import json, sys
path, field = sys.argv[1], sys.argv[2]
obj = None
with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and (rec.get("type") == "result" or "is_error" in rec):
            obj = rec
if obj is None:
    sys.exit(3)
val = obj.get(field, "")
print(val if not isinstance(val, bool) else str(val).lower())
PY
}

# --- defaults ---
PROMPT=""
PROMPT_FILE=""
WORKDIR="."
BUDGET="2.00"
MODE="bypassPermissions"
ALLOW=""
DISALLOW=""
MODEL=""
APPEND_SYSTEM=""
LOG_BASE="logs/claude-auto"
STREAM=0
RUN_NAME=""
DRY_RUN=0

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -f|--file) PROMPT_FILE="${2:?}"; shift 2 ;;
    -d|--dir) WORKDIR="${2:?}"; shift 2 ;;
    -b|--budget) BUDGET="${2:?}"; shift 2 ;;
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
    -*) die "unknown option: $1 (see --help)" ;;
    *) PROMPT="$1"; shift ;;
  esac
done

command -v claude >/dev/null 2>&1 || die "claude CLI not found on PATH"
PYTHON="$(pick_python)"

# --- resolve the prompt: arg > file > stdin ---
if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || die "prompt file not found: $PROMPT_FILE"
  PROMPT="$(cat "$PROMPT_FILE")"
elif [[ -z "$PROMPT" && ! -t 0 ]]; then
  PROMPT="$(cat)"
fi
[[ -n "${PROMPT// /}" ]] || { usage; die "no prompt provided"; }

[[ -d "$WORKDIR" ]] || die "working dir not found: $WORKDIR"

# --- prepare run dir ---
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "$RUN_NAME" ]]; then
  SLUG="$(echo "$RUN_NAME" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/-*$//' | cut -c1-48)"
  RUN_DIR="${LOG_BASE%/}/${TS}-${SLUG}"
else
  RUN_DIR="${LOG_BASE%/}/${TS}"
fi
# --- build the claude command ---
OUT_FORMAT="json"; OUT_FILE="$RUN_DIR/output.json"
if [[ "$STREAM" -eq 1 ]]; then OUT_FORMAT="stream-json"; OUT_FILE="$RUN_DIR/stream.jsonl"; fi

CMD=(claude -p "$PROMPT"
  --permission-mode "$MODE"
  --max-budget-usd "$BUDGET"
  --output-format "$OUT_FORMAT")
[[ "$STREAM" -eq 1 ]] && CMD+=(--verbose --include-partial-messages)
[[ -n "$ALLOW" ]]    && CMD+=(--allowedTools "$ALLOW")
[[ -n "$DISALLOW" ]] && CMD+=(--disallowedTools "$DISALLOW")
[[ -n "$MODEL" ]]    && CMD+=(--model "$MODEL")
[[ -n "$APPEND_SYSTEM" ]] && CMD+=(--append-system-prompt "$APPEND_SYSTEM")

# the exact invocation, prompt elided, for the log and for --dry-run
CMD_LINE="$( printf 'cwd: %s\ncmd:' "$WORKDIR"; printf ' %q' "${CMD[@]/$PROMPT/<PROMPT>}"; printf '\n'; )"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] run dir would be: $RUN_DIR"
  printf '%s\n' "$CMD_LINE"
  exit 0
fi

mkdir -p "$RUN_DIR"
printf '%s\n' "$PROMPT" > "$RUN_DIR/prompt.txt"
printf '%s\n' "$CMD_LINE" > "$RUN_DIR/command.txt"

echo "[claude-auto] mode=$MODE budget=\$$BUDGET stream=$STREAM dir=$WORKDIR"
echo "[claude-auto] log: $RUN_DIR"

# --- run ---
set +e
if [[ "$STREAM" -eq 1 ]]; then
  ( cd "$WORKDIR" && "${CMD[@]}" ) | tee "$OUT_FILE"
  RC=${PIPESTATUS[0]}
else
  ( cd "$WORKDIR" && "${CMD[@]}" ) > "$OUT_FILE" 2>"$RUN_DIR/stderr.log"
  RC=$?
fi
set -e

# --- parse result ---
if ! IS_ERROR="$(json_field "$OUT_FILE" is_error)"; then
  echo "[claude-auto] FAILED: no result object emitted (claude rc=$RC)" >&2
  [[ -s "$RUN_DIR/stderr.log" ]] && tail -n 20 "$RUN_DIR/stderr.log" >&2
  exit 1
fi
COST="$(json_field "$OUT_FILE" total_cost_usd || echo '?')"
TURNS="$(json_field "$OUT_FILE" num_turns || echo '?')"
SID="$(json_field "$OUT_FILE" session_id || echo '?')"
json_field "$OUT_FILE" result > "$RUN_DIR/result.txt" || true

{
  echo "is_error=$IS_ERROR"
  echo "total_cost_usd=$COST"
  echo "num_turns=$TURNS"
  echo "session_id=$SID"
  echo "claude_rc=$RC"
} > "$RUN_DIR/summary.txt"

echo "[claude-auto] is_error=$IS_ERROR cost=\$$COST turns=$TURNS session=$SID"
echo "----- result -----"
cat "$RUN_DIR/result.txt"
echo
echo "------------------"

[[ "$IS_ERROR" == "false" ]] && exit 0 || exit 1
