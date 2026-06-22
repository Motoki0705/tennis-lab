#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent-auto-common.sh
source "$SCRIPT_DIR/agent-auto-common.sh"

usage() {
  cat <<'USAGE'
Usage:
  codex-auto.sh [options] "PROMPT"
  codex-auto.sh [options] -f PROMPT_FILE
  echo "PROMPT" | codex-auto.sh [options]

Run one Codex CLI task non-interactively with structured logging.

Options:
  -f, --file FILE        Read the prompt from FILE.
  -d, --dir DIR          Working directory for the run (default: current dir).
  -s, --sandbox MODE     Sandbox mode (default: workspace-write).
      --dangerous        Bypass approvals and sandboxing.
  -c, --config KEY=VALUE Add a Codex config override; repeatable.
      --model MODEL      Model name.
      --ephemeral        Do not persist Codex session files.
      --skip-git-check   Allow a working directory outside a git repository.
  -l, --log-dir DIR      Base dir for run logs (default: logs/agent-auto/codex).
      --stream           Stream JSONL events while retaining the run log.
      --name NAME        Run slug (default: run).
      --dry-run          Print the Codex command and exit.
  -h, --help             Show this help.

Exit codes:
  0  Codex emitted turn.completed without a failure event
  1  Codex failed or emitted no completed turn
  2  usage / setup error
USAGE
}

PROMPT=""
PROMPT_FILE=""
WORKDIR="."
SANDBOX="workspace-write"
DANGEROUS=0
MODEL=""
EPHEMERAL=0
SKIP_GIT_CHECK=0
LOG_BASE="logs/agent-auto/codex"
STREAM=0
RUN_NAME=""
DRY_RUN=0
CONFIG_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -f|--file) PROMPT_FILE="${2:?}"; shift 2 ;;
    -d|--dir) WORKDIR="${2:?}"; shift 2 ;;
    -s|--sandbox) SANDBOX="${2:?}"; shift 2 ;;
    --dangerous) DANGEROUS=1; shift ;;
    -c|--config) CONFIG_OVERRIDES+=("${2:?}"); shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --ephemeral) EPHEMERAL=1; shift ;;
    --skip-git-check) SKIP_GIT_CHECK=1; shift ;;
    -l|--log-dir) LOG_BASE="${2:?}"; shift 2 ;;
    --stream) STREAM=1; shift ;;
    --name) RUN_NAME="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; PROMPT="${*:-}"; break ;;
    -*) agent_auto_die "unknown option: $1 (see --help)" ;;
    *) PROMPT="$1"; shift ;;
  esac
done

agent_auto_require_command codex
agent_auto_resolve_prompt
agent_auto_validate_workdir
PYTHON="$(agent_auto_pick_python)"
agent_auto_prepare_run_dir "$LOG_BASE" "$RUN_NAME"
OUT_FILE="$RUN_DIR/stream.jsonl"

CMD=(codex exec --json --color never -C "$WORKDIR")
if [[ "$DANGEROUS" -eq 1 ]]; then
  CMD+=(--dangerously-bypass-approvals-and-sandbox)
else
  CMD+=(--sandbox "$SANDBOX" -c 'approval_policy="never"')
fi
for override in "${CONFIG_OVERRIDES[@]}"; do
  CMD+=(-c "$override")
done
[[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")
[[ "$EPHEMERAL" -eq 1 ]] && CMD+=(--ephemeral)
[[ "$SKIP_GIT_CHECK" -eq 1 ]] && CMD+=(--skip-git-repo-check)
CMD+=("$PROMPT")

CMD_LINE="$(agent_auto_render_command)"
if [[ "$DRY_RUN" -eq 1 ]]; then
  agent_auto_print_dry_run
fi

agent_auto_initialize_log
echo "[codex-auto] sandbox=$SANDBOX dangerous=$DANGEROUS dir=$WORKDIR"
echo "[codex-auto] log: $RUN_DIR"
agent_auto_execute

set +e
"$PYTHON" - "$OUT_FILE" "$RUN_DIR/result.txt" "$RUN_DIR/summary.txt" "$AGENT_AUTO_RC" <<'PY'
import json
import sys

source, result_path, summary_path, cli_rc_text = sys.argv[1:]
cli_rc = int(cli_rc_text)
thread_id = "?"
completed = False
failed = False
result = ""

with open(source, encoding="utf-8") as stream:
    for line in stream:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = record.get("type")
        if event_type == "thread.started":
            thread_id = record.get("thread_id", "?")
        elif event_type == "turn.completed":
            completed = True
        elif event_type in {"turn.failed", "error"}:
            failed = True
        elif event_type == "item.completed":
            item = record.get("item", {})
            if item.get("type") == "agent_message":
                result = item.get("text", "")

success = cli_rc == 0 and completed and not failed
with open(result_path, "w", encoding="utf-8") as output:
    output.write(result)
    output.write("\n")
with open(summary_path, "w", encoding="utf-8") as output:
    output.write(f"status={'success' if success else 'failed'}\n")
    output.write(f"turn_completed={str(completed).lower()}\n")
    output.write(f"failure_event={str(failed).lower()}\n")
    output.write(f"thread_id={thread_id}\n")
    output.write(f"codex_rc={cli_rc}\n")

sys.exit(0 if success else 1)
PY
PARSE_RC=$?
set -e

agent_auto_print_result "$PARSE_RC"
