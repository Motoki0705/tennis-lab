#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent-auto-common.sh
source "$SCRIPT_DIR/agent-auto-common.sh"

usage() {
  cat <<'USAGE'
Usage:
  hermes-auto.sh [options] "PROMPT"
  hermes-auto.sh [options] -f PROMPT_FILE
  echo "PROMPT" | hermes-auto.sh [options]

Run one Hermes Agent task non-interactively with structured logging.

Options:
  -f, --file FILE        Read the prompt from FILE.
  -d, --dir DIR          Working directory for the run (default: current dir).
      --model MODEL      Hermes model override.
      --provider NAME    Hermes provider override.
  -t, --toolsets LIST    Comma-separated Hermes toolsets.
  -s, --skills LIST      Comma-separated Hermes skills to preload.
  -r, --resume ID        Resume an explicit Hermes session ID.
  -c, --continue [NAME]  Continue the latest session or a named session.
      --max-turns N      Maximum Hermes tool-calling iterations.
      --yolo              Bypass Hermes dangerous-command approvals (default).
      --no-yolo           Keep Hermes dangerous-command approvals enabled.
      --accept-hooks      Auto-approve unseen Hermes shell hooks.
      --worktree          Ask Hermes to use an isolated git worktree.
      --checkpoints       Enable Hermes filesystem checkpoints.
      --ignore-user-config
                          Use Hermes built-in behavioral defaults (default).
      --use-user-config   Use ~/.hermes/config.yaml instead.
      --ignore-rules      Skip repository rules and AGENTS.md injection.
  -l, --log-dir DIR       Base dir for run logs (default: logs/agent-auto/hermes).
      --name NAME         Run slug (default: run).
      --dry-run            Render the Hermes command without starting it.
  -h, --help              Show this help.

Exit codes:
  0  Hermes exited zero and emitted a response plus session_id.
  1  Hermes failed or emitted an incomplete result.
  2  Usage or setup error.

Examples:
  hermes-auto.sh "Review the repository and report risks"
  hermes-auto.sh --resume 20260727_181249_0d18c2 "Continue the previous task"
USAGE
}

PROMPT=""
PROMPT_FILE=""
WORKDIR="."
MODEL=""
PROVIDER=""
TOOLSETS=""
SKILLS=""
RESUME=""
CONTINUE_SET=0
CONTINUE_NAME=""
MAX_TURNS=""
YOLO=1
ACCEPT_HOOKS=0
WORKTREE=0
CHECKPOINTS=0
IGNORE_USER_CONFIG=1
IGNORE_RULES=0
LOG_BASE="logs/agent-auto/hermes"
RUN_NAME=""
DRY_RUN=0
STREAM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -f|--file) PROMPT_FILE="${2:?}"; shift 2 ;;
    -d|--dir) WORKDIR="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --provider) PROVIDER="${2:?}"; shift 2 ;;
    -t|--toolsets) TOOLSETS="${2:?}"; shift 2 ;;
    -s|--skills) SKILLS="${2:?}"; shift 2 ;;
    -r|--resume) RESUME="${2:?}"; shift 2 ;;
    -c|--continue)
      CONTINUE_SET=1
      shift
      if [[ $# -gt 0 && "$1" != -* ]]; then
        CONTINUE_NAME="$1"
        shift
      fi
      ;;
    --max-turns) MAX_TURNS="${2:?}"; shift 2 ;;
    --yolo) YOLO=1; shift ;;
    --no-yolo) YOLO=0; shift ;;
    --accept-hooks) ACCEPT_HOOKS=1; shift ;;
    --worktree) WORKTREE=1; shift ;;
    --checkpoints) CHECKPOINTS=1; shift ;;
    --ignore-user-config) IGNORE_USER_CONFIG=1; shift ;;
    --use-user-config) IGNORE_USER_CONFIG=0; shift ;;
    --ignore-rules) IGNORE_RULES=1; shift ;;
    -l|--log-dir) LOG_BASE="${2:?}"; shift 2 ;;
    --name) RUN_NAME="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; PROMPT="${*:-}"; break ;;
    -*) agent_auto_die "unknown option: $1 (see --help)" ;;
    *)
      [[ -z "$PROMPT" ]] || agent_auto_die "multiple prompts provided"
      PROMPT="$1"
      shift
      ;;
  esac
done

if [[ -n "$RESUME" && "$CONTINUE_SET" -eq 1 ]]; then
  agent_auto_die "--resume and --continue are mutually exclusive"
fi
if [[ -n "$MAX_TURNS" && ! "$MAX_TURNS" =~ ^[1-9][0-9]*$ ]]; then
  agent_auto_die "--max-turns must be a positive integer"
fi

agent_auto_require_command hermes
agent_auto_resolve_prompt
agent_auto_validate_workdir
PYTHON="$(agent_auto_pick_python)"
agent_auto_prepare_run_dir "$LOG_BASE" "$RUN_NAME"
OUT_FILE="$RUN_DIR/output.txt"

CMD=(hermes chat --query "$PROMPT" --quiet --pass-session-id --source tool)
[[ -n "$RESUME" ]] && CMD+=(--resume "$RESUME")
if [[ "$CONTINUE_SET" -eq 1 ]]; then
  if [[ -n "$CONTINUE_NAME" ]]; then
    CMD+=(--continue "$CONTINUE_NAME")
  else
    CMD+=(--continue)
  fi
fi
[[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")
[[ -n "$PROVIDER" ]] && CMD+=(--provider "$PROVIDER")
[[ -n "$TOOLSETS" ]] && CMD+=(--toolsets "$TOOLSETS")
[[ -n "$SKILLS" ]] && CMD+=(--skills "$SKILLS")
[[ -n "$MAX_TURNS" ]] && CMD+=(--max-turns "$MAX_TURNS")
[[ "$YOLO" -eq 1 ]] && CMD+=(--yolo)
[[ "$ACCEPT_HOOKS" -eq 1 ]] && CMD+=(--accept-hooks)
[[ "$WORKTREE" -eq 1 ]] && CMD+=(--worktree)
[[ "$CHECKPOINTS" -eq 1 ]] && CMD+=(--checkpoints)
[[ "$IGNORE_USER_CONFIG" -eq 1 ]] && CMD+=(--ignore-user-config)
[[ "$IGNORE_RULES" -eq 1 ]] && CMD+=(--ignore-rules)

CMD_LINE="$(agent_auto_render_command)"
if [[ "$DRY_RUN" -eq 1 ]]; then
  agent_auto_print_dry_run
fi

agent_auto_initialize_log
echo "[hermes-auto] yolo=$YOLO resume=${RESUME:-none} dir=$WORKDIR"
echo "[hermes-auto] log: $RUN_DIR"
agent_auto_execute

set +e
"$PYTHON" - \
  "$OUT_FILE" \
  "$RUN_DIR/stderr.log" \
  "$RUN_DIR/result.txt" \
  "$RUN_DIR/summary.txt" \
  "$AGENT_AUTO_RC" \
  "$RESUME" \
  "$CONTINUE_SET" <<'PY'
import re
import sys

source_path, stderr_path, result_path, summary_path, cli_rc_text, resume_id, continue_set = sys.argv[1:]
cli_rc = int(cli_rc_text)
with open(source_path, encoding="utf-8") as stream:
    stdout = stream.read()
with open(stderr_path, encoding="utf-8") as stream:
    stderr = stream.read()

session_ids = re.findall(r"^\s*session_id:\s*(\S+)\s*$", stderr, re.MULTILINE)
session_id = session_ids[-1] if session_ids else "?"

ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
top = re.compile(r"^\s*┌─\s*Reasoning\s*─*┐\s*$")
bottom = re.compile(r"^\s*└─+┘\s*$")
cleaned = []
in_reasoning = False
for line in stdout.splitlines(keepends=True):
    plain = ansi.sub("", line).strip()
    if not in_reasoning and top.match(plain):
        in_reasoning = True
        continue
    if in_reasoning and bottom.match(plain):
        in_reasoning = False
        continue
    if not in_reasoning:
        cleaned.append(line)
response = "".join(cleaned)

success = cli_rc == 0 and session_id != "?" and bool(response.strip()) and not in_reasoning
status = "success" if success else "failed"
with open(result_path, "w", encoding="utf-8") as output:
    output.write(response)
    if not response.endswith("\n"):
        output.write("\n")
with open(summary_path, "w", encoding="utf-8") as output:
    output.write(f"status={status}\n")
    output.write(f"session_id={session_id}\n")
    output.write(f"resumed_session={str(bool(resume_id or continue_set == '1')).lower()}\n")
    output.write(f"hermes_rc={cli_rc}\n")
    if not success:
        if cli_rc != 0:
            output.write("failure_reason=hermes_nonzero_exit\n")
        elif session_id == "?":
            output.write("failure_reason=missing_session_id\n")
        elif in_reasoning:
            output.write("failure_reason=unterminated_reasoning_display\n")
        else:
            output.write("failure_reason=empty_response\n")

sys.exit(0 if success else 1)
PY
PARSE_RC=$?
set -e

agent_auto_print_result "$PARSE_RC"
