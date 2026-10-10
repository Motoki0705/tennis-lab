#!/usr/bin/env bash
# Run Claude Code or Codex headlessly with a prompt file and capture the final message.
#
# Callers collect data deterministically and embed it in the prompt; the agent only
# analyses and writes text. Posting results (issue/PR comments) is the caller's job.
#
# Usage:
#   agent_exec.sh --agent claude|codex --cwd DIR --prompt-file FILE --out FILE \
#                 [--mode read-only|write] [--log-dir DIR] [--model NAME]
#
# Exit status is the agent CLI's status; failures are never silently skipped.
set -euo pipefail

usage() {
  sed -n '2,10p' "$0" >&2
  exit 2
}

main() {
  local agent="" cwd="" prompt_file="" out="" mode="read-only" log_dir="" model=""
  while (($#)); do
    case "$1" in
      --agent) agent="$2"; shift 2 ;;
      --cwd) cwd="$2"; shift 2 ;;
      --prompt-file) prompt_file="$2"; shift 2 ;;
      --out) out="$2"; shift 2 ;;
      --mode) mode="$2"; shift 2 ;;
      --log-dir) log_dir="$2"; shift 2 ;;
      --model) model="$2"; shift 2 ;;
      -h|--help) usage ;;
      *) echo "agent_exec: unknown argument: $1" >&2; usage ;;
    esac
  done

  [[ -n "$agent" && -n "$cwd" && -n "$prompt_file" && -n "$out" ]] || usage
  [[ -d "$cwd" ]] || { echo "agent_exec: --cwd not a directory: $cwd" >&2; exit 2; }
  [[ -f "$prompt_file" ]] || { echo "agent_exec: prompt file missing: $prompt_file" >&2; exit 2; }
  case "$mode" in read-only|write) ;; *) echo "agent_exec: invalid --mode: $mode" >&2; exit 2 ;; esac

  log_dir="${log_dir:-${XDG_STATE_HOME:-$HOME/.local/state}/tennis-lab-agents/logs}"
  mkdir -p "$log_dir" "$(dirname "$out")"
  local log_file
  log_file="$log_dir/$(date +%Y%m%dT%H%M%S)-${agent}-$$.log"

  local -a cmd
  case "$agent" in
    claude)
      cmd=(claude -p --output-format text --permission-mode bypassPermissions)
      if [[ "$mode" == read-only ]]; then
        cmd+=(--disallowedTools Edit Write NotebookEdit)
      fi
      [[ -n "$model" ]] && cmd+=(--model "$model")
      ;;
    codex)
      cmd=(codex exec --cd "$cwd" --output-last-message "$out")
      if [[ "$mode" == read-only ]]; then
        cmd+=(--sandbox read-only)
      else
        cmd+=(--dangerously-bypass-approvals-and-sandbox)
      fi
      [[ -n "$model" ]] && cmd+=(--model "$model")
      cmd+=(-)
      ;;
    *) echo "agent_exec: unknown --agent: $agent" >&2; exit 2 ;;
  esac

  echo "agent_exec: agent=$agent mode=$mode cwd=$cwd log=$log_file" >&2
  local status=0
  if [[ "$agent" == claude ]]; then
    (cd "$cwd" && "${cmd[@]}" <"$prompt_file" >"$out" 2>"$log_file") || status=$?
  else
    (cd "$cwd" && "${cmd[@]}" <"$prompt_file" >"$log_file" 2>&1) || status=$?
  fi

  if ((status != 0)); then
    echo "agent_exec: $agent exited with status $status (see $log_file)" >&2
    return "$status"
  fi
  if [[ ! -s "$out" ]]; then
    echo "agent_exec: $agent produced an empty final message (see $log_file)" >&2
    return 1
  fi
}

main "$@"
