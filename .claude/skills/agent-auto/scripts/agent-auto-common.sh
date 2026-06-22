#!/usr/bin/env bash

agent_auto_die() {
  echo "[FAIL] $*" >&2
  exit 2
}

agent_auto_require_command() {
  command -v "$1" >/dev/null 2>&1 || agent_auto_die "$1 CLI not found on PATH"
}

agent_auto_pick_python() {
  if [[ -x "$WORKDIR/.venv/bin/python" ]]; then
    echo "$WORKDIR/.venv/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    agent_auto_die "no python found for structured-output parsing"
  fi
}

agent_auto_resolve_prompt() {
  if [[ -n "$PROMPT_FILE" ]]; then
    [[ -f "$PROMPT_FILE" ]] || agent_auto_die "prompt file not found: $PROMPT_FILE"
    PROMPT="$(<"$PROMPT_FILE")"
  elif [[ -z "$PROMPT" && ! -t 0 ]]; then
    PROMPT="$(cat)"
  fi

  if [[ -z "${PROMPT//[[:space:]]/}" ]]; then
    usage
    agent_auto_die "no prompt provided"
  fi
}

agent_auto_validate_workdir() {
  [[ -d "$WORKDIR" ]] || agent_auto_die "working dir not found: $WORKDIR"
  WORKDIR="$(cd "$WORKDIR" && pwd)"
}

agent_auto_prepare_run_dir() {
  local log_base="$1"
  local run_name="$2"
  local timestamp slug

  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  slug="run"
  if [[ -n "$run_name" ]]; then
    slug="$(printf '%s' "$run_name" |
      tr -cs 'A-Za-z0-9._-' '-' |
      sed 's/^-*//; s/-*$//' |
      cut -c1-48)"
    [[ -n "$slug" ]] || slug="run"
  fi
  RUN_DIR="${log_base%/}/${timestamp}-${slug}-$$"
}

agent_auto_render_command() {
  local argument

  printf 'cwd: %q\ncmd:' "$WORKDIR"
  for argument in "${CMD[@]}"; do
    if [[ "$argument" == "$PROMPT" ]]; then
      argument="<PROMPT>"
    fi
    printf ' %q' "$argument"
  done
  printf '\n'
}

agent_auto_print_dry_run() {
  echo "[dry-run] run dir would be: $RUN_DIR"
  printf '%s\n' "$CMD_LINE"
  exit 0
}

agent_auto_initialize_log() {
  mkdir -p "$RUN_DIR"
  printf '%s\n' "$PROMPT" >"$RUN_DIR/prompt.txt"
  printf '%s\n' "$CMD_LINE" >"$RUN_DIR/command.txt"
}

agent_auto_execute() {
  set +e
  if [[ "$STREAM" -eq 1 ]]; then
    (cd "$WORKDIR" && "${CMD[@]}") | tee "$OUT_FILE"
    AGENT_AUTO_RC=${PIPESTATUS[0]}
  else
    (cd "$WORKDIR" && "${CMD[@]}") >"$OUT_FILE" 2>"$RUN_DIR/stderr.log"
    AGENT_AUTO_RC=$?
  fi
  set -e
}

agent_auto_report_missing_result() {
  local provider="$1"

  echo "[$provider-auto] FAILED: no structured result emitted (${provider} rc=$AGENT_AUTO_RC)" >&2
  if [[ -s "$RUN_DIR/stderr.log" ]]; then
    tail -n 20 "$RUN_DIR/stderr.log" >&2
  fi
  exit 1
}

agent_auto_print_result() {
  local parse_rc="$1"

  cat "$RUN_DIR/summary.txt"
  echo "----- result -----"
  cat "$RUN_DIR/result.txt"
  echo
  echo "------------------"
  [[ "$parse_rc" -eq 0 ]] && exit 0 || exit 1
}
