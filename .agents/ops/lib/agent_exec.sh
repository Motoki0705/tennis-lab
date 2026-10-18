#!/usr/bin/env bash
# Run Claude Code or Codex headlessly with a prompt file and capture the final message.
#
# Callers collect data deterministically and embed it in the prompt; the agent only
# analyses and writes text. Posting results (issue/PR comments) is the caller's job.
#
# Usage:
#   agent_exec.sh --agent claude|codex --cwd DIR --prompt-file FILE --out FILE \
#                 [--mode read-only|write] [--log-dir DIR]
#                 [--profile NAME | [--model NAME] [--effort LEVEL]]
#
# Model, effort and (codex) service tier always come from agents.conf next to this
# script, so the result never depends on which CLI config the calling environment
# happens to load. Without --profile the default pins (CLAUDE_* / CODEX_*) are used,
# optionally overridden by --model / --effort. --profile NAME uses the job pins
# <NAME>_CLAUDE_* / <NAME>_CODEX_* instead; every one of them must be set (no fallback
# to the defaults), and --model / --effort cannot be combined with it.
#
# read-only: claude gets only Read/Grep/Glob (no Bash, no MCP); codex runs in its
# read-only sandbox. write: both run without approval prompts.
#
# Exit status is the agent CLI's status; failures are never silently skipped.
set -euo pipefail

usage() {
  sed -n '2,22p' "$0" >&2
  exit 2
}

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

main() {
  local agent="" cwd="" prompt_file="" out="" mode="read-only" log_dir="" model="" effort=""
  local profile="" profile_given=0
  while (($#)); do
    case "$1" in
      --agent) agent="$2"; shift 2 ;;
      --cwd) cwd="$2"; shift 2 ;;
      --prompt-file) prompt_file="$2"; shift 2 ;;
      --out) out="$2"; shift 2 ;;
      --mode) mode="$2"; shift 2 ;;
      --log-dir) log_dir="$2"; shift 2 ;;
      --model) model="$2"; shift 2 ;;
      --effort) effort="$2"; shift 2 ;;
      --profile) profile="$2"; profile_given=1; shift 2 ;;
      -h|--help) usage ;;
      *) echo "agent_exec: unknown argument: $1" >&2; usage ;;
    esac
  done

  [[ -n "$agent" && -n "$cwd" && -n "$prompt_file" && -n "$out" ]] || usage
  [[ -d "$cwd" ]] || { echo "agent_exec: --cwd not a directory: $cwd" >&2; exit 2; }
  [[ -f "$prompt_file" ]] || { echo "agent_exec: prompt file missing: $prompt_file" >&2; exit 2; }
  case "$mode" in read-only|write) ;; *) echo "agent_exec: invalid --mode: $mode" >&2; exit 2 ;; esac
  if ((profile_given)); then
    [[ "$profile" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "agent_exec: invalid --profile: $profile" >&2; exit 2; }
    [[ -z "$model" && -z "$effort" ]] || {
      echo "agent_exec: --profile cannot be combined with --model/--effort" >&2; exit 2; }
  fi

  # Resolve every path against the caller's cwd before the agent runs in --cwd.
  cwd="$(cd "$cwd" && pwd)"
  prompt_file="$(abspath "$prompt_file")"
  out="$(abspath "$out")"
  log_dir="$(abspath "${log_dir:-${XDG_STATE_HOME:-$HOME/.local/state}/tennis-lab-agents/logs}")"
  mkdir -p "$log_dir" "$(dirname "$out")"
  rm -f "$out"

  local conf
  conf="$(dirname "$(abspath "$0")")/agents.conf"
  [[ -f "$conf" ]] || { echo "agent_exec: missing $conf" >&2; exit 2; }
  # shellcheck source=agents.conf
  source "$conf"
  if [[ "$profile" == ops_report ]]; then
    [[ "$agent" == codex && "$mode" == read-only ]] || {
      echo "agent_exec: ops_report requires codex in read-only mode" >&2; exit 2; }
    [[ "${OPS_REPORT_MAX_THREADS:-}" =~ ^[0-9]+$ ]] && ((10#$OPS_REPORT_MAX_THREADS >= 2)) || {
      echo "agent_exec: agents.conf must set OPS_REPORT_MAX_THREADS >= 2" >&2; exit 2; }
  fi

  local log_file
  log_file="$log_dir/$(date +%Y%m%dT%H%M%S)-${agent}-$$.log"

  # Pins for this run: <PROFILE>_<AGENT>_<KEY> with --profile, else <AGENT>_<KEY>.
  local pin_prefix=""
  [[ -z "$profile" ]] || pin_prefix="${profile^^}_"
  pin() {
    local name="${pin_prefix}${agent^^}_$1"
    [[ -n "${!name:-}" ]] || { echo "agent_exec: agents.conf must set $name" >&2; exit 2; }
    printf '%s\n' "${!name}"
  }

  local -a cmd
  local tier=""
  case "$agent" in
    claude)
      [[ -n "$model" ]] || model="$(pin MODEL)"
      [[ -n "$effort" ]] || effort="$(pin EFFORT)"
      cmd=(claude -p --output-format text --model "$model" --effort "$effort" --strict-mcp-config)
      if [[ "$mode" == read-only ]]; then
        cmd+=(--tools Read Grep Glob --permission-mode dontAsk)
      else
        cmd+=(--permission-mode bypassPermissions)
      fi
      ;;
    codex)
      [[ -n "$model" ]] || model="$(pin MODEL)"
      [[ -n "$effort" ]] || effort="$(pin EFFORT)"
      tier="$(pin SERVICE_TIER)"
      cmd=(codex exec --cd "$cwd" --output-last-message "$out" --model "$model"
        -c "model_reasoning_effort=\"$effort\"" -c "service_tier=\"$tier\"")
      if [[ "$profile" == ops_report ]]; then
        cmd+=(--enable multi_agent --json -c "agents.max_threads=$OPS_REPORT_MAX_THREADS" -c "agents.max_depth=1")
      fi
      if [[ "$mode" == read-only ]]; then
        cmd+=(--sandbox read-only)
      else
        cmd+=(--dangerously-bypass-approvals-and-sandbox)
      fi
      cmd+=(-)
      ;;
    *) echo "agent_exec: unknown --agent: $agent" >&2; exit 2 ;;
  esac

  echo "agent_exec: agent=$agent mode=$mode${profile:+ profile=$profile} model=$model effort=$effort${tier:+ service_tier=$tier} cwd=$cwd log=$log_file" >&2
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
