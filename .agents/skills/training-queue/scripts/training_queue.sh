#!/usr/bin/env bash
# training_queue.sh — serial background job queue for training runs (issue #523)
#
# A file-backed FIFO queue. Jobs are enqueued as tiny shell scripts and a single
# background worker runs them strictly one at a time. The worker can wait for an
# arbitrary PID (e.g. a natively-launched first training) to finish before it
# starts, so the very first run does NOT have to go through this queue.
#
# Subcommands:
#   add "<command>" [--name NAME]   Enqueue a command (runs in the current CWD).
#   start [--after-pid PID] [--idle-timeout S]
#                                   Launch the background worker (nohup-style).
#   status                          Show worker state + queued/running/done jobs.
#   list                            List pending jobs in run order.
#   stop                            Ask the worker to stop after the current job.
#   clear                           Remove all pending (not-yet-started) jobs.
#
# State dir: $TRAINING_QUEUE_DIR (default: .training_queue under the CWD).
#
# Examples:
#   training_queue.sh add "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#       python -m src.tasks.plcs.scripts.train ..." --name exp_a
#   training_queue.sh start --after-pid 12345   # wait for native run 12345 first
#   training_queue.sh status

set -uo pipefail

QUEUE_DIR="${TRAINING_QUEUE_DIR:-.training_queue}"
JOBS_DIR="$QUEUE_DIR/jobs"
LOGS_DIR="$QUEUE_DIR/logs"
DONE_DIR="$QUEUE_DIR/done"
FAILED_DIR="$QUEUE_DIR/failed"
RUNNING_DIR="$QUEUE_DIR/running"
WORKER_PID_FILE="$QUEUE_DIR/worker.pid"
WORKER_LOG="$QUEUE_DIR/worker.log"
STOP_FILE="$QUEUE_DIR/stop"

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

_ensure_dirs() {
  mkdir -p "$JOBS_DIR" "$LOGS_DIR" "$DONE_DIR" "$FAILED_DIR" "$RUNNING_DIR"
}

_worker_running() {
  [ -f "$WORKER_PID_FILE" ] || return 1
  local pid
  pid="$(cat "$WORKER_PID_FILE" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

cmd_add() {
  local name="" command=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --name) name="$2"; shift 2 ;;
      *) command="$1"; shift ;;
    esac
  done
  if [ -z "$command" ]; then
    echo "error: add requires a command string" >&2
    exit 2
  fi
  _ensure_dirs
  name="${name:-job}"
  # Sanitize name for use in a filename.
  name="$(printf '%s' "$name" | tr -c 'A-Za-z0-9._-' '_')"
  # Nanosecond + PID gives a unique, lexically-sortable (FIFO) job id without
  # needing a lock for concurrent enqueues from multiple agents.
  local jobid="$(date +%s%N)_$$_${name}"
  local tmp="$JOBS_DIR/.tmp.${jobid}.job"
  local final="$JOBS_DIR/${jobid}.job"
  {
    echo "#!/usr/bin/env bash"
    echo "# name: ${name}"
    echo "# added: $(date -Iseconds)"
    echo "cd $(printf '%q' "$PWD")"
    echo "$command"
  } > "$tmp"
  mv "$tmp" "$final"   # atomic publish
  echo "queued: ${jobid}.job"
}

# Internal: the background worker loop. Not meant to be called directly.
cmd_worker() {
  local after_pid="" idle_timeout=30
  while [ $# -gt 0 ]; do
    case "$1" in
      --after-pid) after_pid="$2"; shift 2 ;;
      --idle-timeout) idle_timeout="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  _ensure_dirs
  rm -f "$STOP_FILE"

  if [ -n "$after_pid" ]; then
    echo "[worker] waiting for PID $after_pid to finish..."
    # tail --pid returns immediately if the PID is already gone.
    tail --pid="$after_pid" -f /dev/null 2>/dev/null || true
    echo "[worker] PID $after_pid finished; starting queue."
  fi

  local idle_elapsed=0
  while true; do
    if [ -f "$STOP_FILE" ]; then
      echo "[worker] stop requested; exiting."
      rm -f "$STOP_FILE"
      break
    fi
    # Pick the lowest job id (FIFO). Ignore .tmp.* partials.
    local job
    job="$(find "$JOBS_DIR" -maxdepth 1 -name '*.job' ! -name '.tmp.*' -printf '%f\n' 2>/dev/null | sort | head -1)"
    if [ -z "$job" ]; then
      if [ "$idle_elapsed" -ge "$idle_timeout" ]; then
        echo "[worker] queue empty for ${idle_timeout}s; exiting."
        break
      fi
      sleep 2
      idle_elapsed=$((idle_elapsed + 2))
      continue
    fi
    idle_elapsed=0

    mv "$JOBS_DIR/$job" "$RUNNING_DIR/$job"
    local log="$LOGS_DIR/${job%.job}.log"
    echo "[worker] $(date -Iseconds) running: $job (log: $log)"
    local rc=0
    bash "$RUNNING_DIR/$job" > "$log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
      mv "$RUNNING_DIR/$job" "$DONE_DIR/$job"
      echo "[worker] $(date -Iseconds) done: $job"
    else
      echo "exit_code=$rc" >> "$log"
      mv "$RUNNING_DIR/$job" "$FAILED_DIR/$job"
      echo "[worker] $(date -Iseconds) FAILED (rc=$rc): $job"
    fi
  done
  rm -f "$WORKER_PID_FILE"
  echo "[worker] $(date -Iseconds) worker exited."
}

cmd_start() {
  _ensure_dirs
  if _worker_running; then
    echo "worker already running (PID $(cat "$WORKER_PID_FILE"))." >&2
    exit 1
  fi
  # Detached background worker; survives the launching shell.
  nohup bash "$SCRIPT_PATH" __worker "$@" >> "$WORKER_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$WORKER_PID_FILE"
  echo "worker started (PID $pid). log: $WORKER_LOG"
}

cmd_status() {
  _ensure_dirs
  if _worker_running; then
    echo "worker: RUNNING (PID $(cat "$WORKER_PID_FILE"))"
  else
    echo "worker: stopped"
  fi
  local q r d f
  q=$(find "$JOBS_DIR" -maxdepth 1 -name '*.job' ! -name '.tmp.*' 2>/dev/null | wc -l)
  r=$(find "$RUNNING_DIR" -maxdepth 1 -name '*.job' 2>/dev/null | wc -l)
  d=$(find "$DONE_DIR" -maxdepth 1 -name '*.job' 2>/dev/null | wc -l)
  f=$(find "$FAILED_DIR" -maxdepth 1 -name '*.job' 2>/dev/null | wc -l)
  echo "queued=$q running=$r done=$d failed=$f"
  if [ "$r" -gt 0 ]; then
    echo "running:"; find "$RUNNING_DIR" -maxdepth 1 -name '*.job' -printf '  %f\n' | sort
  fi
  if [ "$q" -gt 0 ]; then
    echo "pending (run order):"; cmd_list | sed 's/^/  /'
  fi
}

cmd_list() {
  _ensure_dirs
  find "$JOBS_DIR" -maxdepth 1 -name '*.job' ! -name '.tmp.*' -printf '%f\n' 2>/dev/null | sort
}

cmd_stop() {
  _ensure_dirs
  touch "$STOP_FILE"
  echo "stop requested; worker will exit after the current job."
}

cmd_clear() {
  _ensure_dirs
  find "$JOBS_DIR" -maxdepth 1 -name '*.job' -delete 2>/dev/null || true
  echo "pending queue cleared."
}

main() {
  local sub="${1:-}"
  [ $# -gt 0 ] && shift || true
  case "$sub" in
    add)      cmd_add "$@" ;;
    start)    cmd_start "$@" ;;
    __worker) cmd_worker "$@" ;;
    status)   cmd_status "$@" ;;
    list)     cmd_list "$@" ;;
    stop)     cmd_stop "$@" ;;
    clear)    cmd_clear "$@" ;;
    -h|--help|help|"")
      sed -n '2,40p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//' ;;
    *)
      echo "unknown subcommand: $sub (try --help)" >&2; exit 2 ;;
  esac
}

main "$@"
