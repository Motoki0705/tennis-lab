#!/usr/bin/env bash
# training_queue.sh — serial background job queue for training runs (issue #523)
#
# A file-backed FIFO queue. Jobs are enqueued as tiny shell scripts and a single
# background worker runs them strictly one at a time. The worker can wait for an
# arbitrary PID (e.g. a natively-launched first training) to finish before it
# starts, so the very first run does NOT have to go through this queue.
#
# Subcommands:
#   add "<command>" [--name NAME] [--prune-ckpt]
#                                   Enqueue a command (runs in the current CWD).
#                                   --prune-ckpt: after the run succeeds, delete
#                                   its checkpoints iff the test-split
#                                   predictions were saved (issue #533).
#   start [--after-pid PID] [--idle-timeout S]
#                                   Launch the background worker (nohup-style).
#   serve [--after-pid PID] [--idle-timeout S]
#                                   Run the worker in the foreground (systemd).
#   status                          Show worker state + queued/running/done jobs.
#   list                            List pending jobs in run order.
#   stop                            Ask the worker to stop after the current job.
#   clear                           Remove all pending (not-yet-started) jobs.
#
# State dir: $TRAINING_QUEUE_DIR (default: .training_queue under the CWD).
# Prune python: $TRAINING_QUEUE_PYTHON (default: $VIRTUAL_ENV or repo .venv).
# GPU lock: $TRAINING_QUEUE_LOCK_FILE (optional advisory flock shared with CI).
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
REPRO_DIR="$QUEUE_DIR/repro"
WORKER_PID_FILE="$QUEUE_DIR/worker.pid"
WORKER_LOCK_FILE="$QUEUE_DIR/worker.lock"
WORKER_LOG="$QUEUE_DIR/worker.log"
STOP_FILE="$QUEUE_DIR/stop"

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

_ensure_dirs() {
  mkdir -p "$JOBS_DIR" "$LOGS_DIR" "$DONE_DIR" "$FAILED_DIR" "$RUNNING_DIR" "$REPRO_DIR"
}

# Minimal JSON string escaper (backslash, double-quote, newline) so run.json is
# valid without depending on jq/python (this is the dependency-light bash tool).
_json_str() {
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g'
}

_worker_running() {
  [ -f "$WORKER_PID_FILE" ] || return 1
  local pid
  pid="$(cat "$WORKER_PID_FILE" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# Resolve a python that can import numpy (for the post-run ckpt pruner). Prefer
# an explicit override, then the active venv, then the repo's .venv (resolved via
# the *main* git dir so it works from a linked worktree too), then python3.
_queue_python() {
  if [ -n "${TRAINING_QUEUE_PYTHON:-}" ] && [ -x "${TRAINING_QUEUE_PYTHON}" ]; then
    printf '%s' "$TRAINING_QUEUE_PYTHON"; return
  fi
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    printf '%s' "${VIRTUAL_ENV}/bin/python"; return
  fi
  local common main_root
  common="$(git -C "$(dirname "$SCRIPT_PATH")" rev-parse --git-common-dir 2>/dev/null || echo '')"
  if [ -n "$common" ]; then
    main_root="$(cd "$(dirname "$common")" && pwd)"
    if [ -x "$main_root/.venv/bin/python" ]; then
      printf '%s' "$main_root/.venv/bin/python"; return
    fi
  fi
  printf 'python3'
}

# Opt-in post-run hook: when a job was added with --prune-ckpt and succeeded,
# delete its checkpoints iff its repro bundle has a verified pred_test.npz.
# Non-fatal: a prune failure never changes the job's done/failed status.
_maybe_prune_ckpt() {
  local job="$1" jobfile="$DONE_DIR/$1"
  grep -q '^# prune_ckpt: 1$' "$jobfile" 2>/dev/null || return 0
  local prune repro_root_abs repro py
  prune="$(dirname "$SCRIPT_PATH")/prune_ckpts.py"
  if [ ! -f "$prune" ]; then
    echo "[worker] prune-ckpt skipped: $prune not found"; return 0
  fi
  repro_root_abs="$(cd "$REPRO_DIR" 2>/dev/null && pwd)" || {
    echo "[worker] prune-ckpt skipped: repro dir $REPRO_DIR missing"; return 0
  }
  repro="$repro_root_abs/${job%.job}"
  py="$(_queue_python)"
  echo "[worker] $(date -Iseconds) prune-ckpt: $job (py=$py repro=$repro)"
  "$py" "$prune" --repro-dir "$repro" --delete \
    || echo "[worker] prune-ckpt non-fatal error for $job (rc=$?)"
}

cmd_add() {
  local name="" command="" provider="" session="" issue="" prune_ckpt=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --name) name="$2"; shift 2 ;;
      --provider) provider="$2"; shift 2 ;;
      --session) session="$2"; shift 2 ;;
      --issue) issue="$2"; shift 2 ;;
      --prune-ckpt) prune_ckpt=1; shift ;;
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
  # Absolute paths so the injected preamble still resolves after the job cd's
  # into its own (possibly worktree) CWD at run time.
  local queue_abs repro_job_abs
  queue_abs="$(cd "$QUEUE_DIR" && pwd)"
  repro_job_abs="$queue_abs/repro/$jobid"
  {
    echo "#!/usr/bin/env bash"
    echo "# name: ${name}"
    echo "# added: $(date -Iseconds)"
    echo "# provider: ${provider}"
    echo "# session: ${session}"
    echo "# issue: ${issue}"
    echo "# prune_ckpt: ${prune_ckpt}"
    echo "cd $(printf '%q' "$PWD")"
    echo "export TENNIS_RUN_ID=$(printf '%q' "$jobid")"
    echo "export TENNIS_REPRO_DIR=$(printf '%q' "$repro_job_abs")"
    # Capture git/repro state at *run* time, in this job's CWD (worktree-aware),
    # so commit/branch/patch reflect exactly what is executed.
    printf 'bash %q __capture-repro --dir %q --name %q --provider %q --session %q --issue %q --cmd %q\n' \
      "$SCRIPT_PATH" "$repro_job_abs" "$name" "$provider" "$session" "$issue" "$command"
    echo "$command"
  } > "$tmp"
  mv "$tmp" "$final"   # atomic publish
  echo "queued: ${jobid}.job"
}

# Internal: capture git/reproducibility state for a job into its repro dir.
# Writes run.json (commit/branch/remote/cmd/provider/session/issue), the
# uncommitted patch, and a one-shot repro.sh. Runs in the job's CWD.
cmd_capture_repro() {
  local dir="" name="" provider="" session="" issue="" cmd=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --dir) dir="$2"; shift 2 ;;
      --name) name="$2"; shift 2 ;;
      --provider) provider="$2"; shift 2 ;;
      --session) session="$2"; shift 2 ;;
      --issue) issue="$2"; shift 2 ;;
      --cmd) cmd="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  if [ -z "$dir" ]; then
    echo "error: __capture-repro requires --dir" >&2
    exit 2
  fi
  mkdir -p "$dir"

  local commit branch remote toplevel
  commit="$(git rev-parse HEAD 2>/dev/null || echo '')"
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  remote="$(git remote get-url origin 2>/dev/null || echo '')"
  toplevel="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  # Uncommitted tracked changes (apply-able patch) + a porcelain status snapshot
  # that also reveals untracked files.
  if [ -n "$commit" ]; then
    git diff HEAD > "$dir/uncommitted.patch" 2>/dev/null || : > "$dir/uncommitted.patch"
    git status --porcelain > "$dir/git_status.txt" 2>/dev/null || true
  else
    : > "$dir/uncommitted.patch"
  fi

  {
    printf '{\n'
    printf '  "run_id": "%s",\n' "$(_json_str "$(basename "$dir")")"
    printf '  "name": "%s",\n' "$(_json_str "$name")"
    printf '  "command": "%s",\n' "$(_json_str "$cmd")"
    printf '  "provider": "%s",\n' "$(_json_str "$provider")"
    printf '  "session": "%s",\n' "$(_json_str "$session")"
    printf '  "issue": "%s",\n' "$(_json_str "$issue")"
    printf '  "commit": "%s",\n' "$(_json_str "$commit")"
    printf '  "branch": "%s",\n' "$(_json_str "$branch")"
    printf '  "remote": "%s",\n' "$(_json_str "$remote")"
    printf '  "cwd": "%s",\n' "$(_json_str "$PWD")"
    printf '  "repo_root": "%s",\n' "$(_json_str "$toplevel")"
    printf '  "captured_at": "%s"\n' "$(date -Iseconds)"
    printf '}\n'
  } > "$dir/run.json"

  {
    echo "#!/usr/bin/env bash"
    echo "# Auto-generated by training_queue.sh — reproduce run: ${name}"
    echo "# provider=${provider} session=${session} issue=${issue}"
    echo "set -uo pipefail"
    echo 'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"'
    printf 'REPO="${TENNIS_REPO:-%q}"\n' "$toplevel"
    echo 'cd "$REPO" || { echo "[repro] cannot cd to $REPO" >&2; exit 1; }'
    printf 'echo "[repro] target commit: %s (branch %s)"\n' "$commit" "$branch"
    if [ -n "$commit" ]; then
      printf 'git checkout %q 2>/dev/null || echo "[repro] WARN: checkout %s failed; using current HEAD"\n' "$commit" "$commit"
    fi
    echo 'PATCH="$SCRIPT_DIR/uncommitted.patch"'
    echo 'if [ -s "$PATCH" ]; then git apply "$PATCH" 2>/dev/null || echo "[repro] WARN: patch did not apply cleanly"; fi'
    echo "# --- original training command ---"
    echo "$cmd"
  } > "$dir/repro.sh"
  chmod +x "$dir/repro.sh"
  echo "[capture-repro] wrote $dir (commit=${commit:0:12} provider=${provider} session=${session})"
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
    if [ -n "${TRAINING_QUEUE_LOCK_FILE:-}" ]; then
      mkdir -p "$(dirname "$TRAINING_QUEUE_LOCK_FILE")"
      echo "[worker] waiting for shared lock: $TRAINING_QUEUE_LOCK_FILE"
      (
        flock -x 8
        bash "$RUNNING_DIR/$job"
      ) 8> "$TRAINING_QUEUE_LOCK_FILE" > "$log" 2>&1 || rc=$?
    else
      bash "$RUNNING_DIR/$job" > "$log" 2>&1 || rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
      mv "$RUNNING_DIR/$job" "$DONE_DIR/$job"
      echo "[worker] $(date -Iseconds) done: $job"
      _maybe_prune_ckpt "$job"
    else
      echo "exit_code=$rc" >> "$log"
      mv "$RUNNING_DIR/$job" "$FAILED_DIR/$job"
      echo "[worker] $(date -Iseconds) FAILED (rc=$rc): $job"
    fi
  done
  echo "[worker] $(date -Iseconds) worker exited."
}

# Foreground worker entry point for supervisors such as systemd. The lock is
# held for the process lifetime, so start/serve cannot create duplicate workers.
cmd_serve() {
  _ensure_dirs
  exec 9> "$WORKER_LOCK_FILE"
  if ! flock -n 9; then
    echo "worker already running (lock: $WORKER_LOCK_FILE)." >&2
    exit 1
  fi

  echo "$$" > "$WORKER_PID_FILE"
  if [ -n "${TRAINING_QUEUE_START_READY_FILE:-}" ]; then
    printf '%s\n' "$$" > "$TRAINING_QUEUE_START_READY_FILE"
  fi
  _cleanup_serve() {
    if [ "$(cat "$WORKER_PID_FILE" 2>/dev/null)" = "$$" ]; then
      rm -f "$WORKER_PID_FILE"
    fi
  }
  trap _cleanup_serve EXIT
  cmd_worker "$@"
}

cmd_start() {
  _ensure_dirs
  if _worker_running; then
    echo "worker already running (PID $(cat "$WORKER_PID_FILE"))." >&2
    exit 1
  fi
  # Detached background worker. Prefer a new session so agent/CI launchers that
  # tear down their own process group do not kill long-running training jobs.
  local ready_file="$QUEUE_DIR/.worker-ready.$$.$RANDOM"
  local -a worker_cmd=(
    env "TRAINING_QUEUE_START_READY_FILE=$ready_file"
    bash "$SCRIPT_PATH" serve "$@"
  )
  if command -v setsid >/dev/null 2>&1; then
    worker_cmd=(setsid "${worker_cmd[@]}")
  fi
  nohup "${worker_cmd[@]}" >> "$WORKER_LOG" 2>&1 < /dev/null &
  local pid=$!
  local ready=0
  for _ in $(seq 1 100); do
    if [ "$(cat "$ready_file" 2>/dev/null)" = "$pid" ]; then
      ready=1
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.02
  done
  rm -f "$ready_file"
  if [ "$ready" -ne 1 ]; then
    echo "worker failed to start; inspect $WORKER_LOG" >&2
    return 1
  fi
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
    add)              cmd_add "$@" ;;
    start)            cmd_start "$@" ;;
    serve)            cmd_serve "$@" ;;
    __worker)         cmd_worker "$@" ;;
    __capture-repro)  cmd_capture_repro "$@" ;;
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
