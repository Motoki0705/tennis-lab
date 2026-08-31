#!/usr/bin/env bash
# training_queue.sh — logical two-slot GPU queue for training runs
#
# A file-backed FIFO queue. Jobs declare either half (one logical slot) or all
# (the complete logical GPU). Two half jobs may overlap; all jobs and raw CI
# users of the main lock are exclusive. This is coordination only: it neither
# changes CUDA visibility nor configures MIG or a VRAM hard cap.
#
# Subcommands:
#   add "<command>" [--name NAME] [--resource half|all] [--prune-ckpt]
#                                   Enqueue a command (runs in the current CWD).
#                                   --prune-ckpt: after the run succeeds, delete
#                                   its checkpoints iff the test-split
#                                   predictions were saved (issue #533).
#   start [--after-pid PID] [--idle-timeout S]
#                                   Launch the background worker (nohup-style).
#   serve [--after-pid PID] [--idle-timeout S]
#                                   Run the worker in the foreground (systemd).
#   status                          Show worker state + queued/running/done jobs.
#   list                            List jobs with resource/allocation state.
#   cancel <job-file>               Cancel one queued/waiting job idempotently.
#   stop                            Stop after currently active jobs finish.
#   clear                           Remove all pending (not-yet-started) jobs.
#
# State dir: $TRAINING_QUEUE_DIR (default: .training_queue under the CWD).
# Prune python: $TRAINING_QUEUE_PYTHON (default: $VIRTUAL_ENV or repo .venv).
# GPU locks: $TRAINING_QUEUE_LOCK_FILE plus derived .gate/.slot-0/.slot-1 files.
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
CANCELLED_DIR="$QUEUE_DIR/cancelled"
REPRO_DIR="$QUEUE_DIR/repro"
STATE_DIR="$QUEUE_DIR/state"
CANCEL_REQUEST_DIR="$QUEUE_DIR/cancel-requests"
CONTROL_DIR="$QUEUE_DIR/control"
EXTERNAL_ACK_DIR="$CONTROL_DIR/external-acks"
WORKER_PID_FILE="$QUEUE_DIR/worker.pid"
WORKER_LOCK_FILE="$QUEUE_DIR/worker.lock"
WORKER_LOG="$QUEUE_DIR/worker.log"
STOP_FILE="$QUEUE_DIR/stop"

# Tests redirect only the production-namespace probe through this seam; normal
# callers either use the fixed system namespace or TRAINING_QUEUE_LOCK_FILE.
SYSTEM_GPU_LOCK_FILE="${TRAINING_QUEUE_SYSTEM_LOCK_FILE:-/var/lib/tennis-lab-actions/gpu.lock}"
GPU_LOCK_FILE=""
GPU_GATE_FILE=""
GPU_SLOT_0_FILE=""
GPU_SLOT_1_FILE=""
GPU_LOCK_NAMESPACE_KIND=""

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
TERM_GRACE_SECONDS=15

_ensure_dirs() {
  if [ -L "$CONTROL_DIR" ] || [ -L "$EXTERNAL_ACK_DIR" ]; then
    echo "error: queue control directory must not be a symlink" >&2
    return 1
  fi
  mkdir -p \
    "$JOBS_DIR" "$LOGS_DIR" "$DONE_DIR" "$FAILED_DIR" "$RUNNING_DIR" \
    "$CANCELLED_DIR" "$REPRO_DIR" "$STATE_DIR" "$CANCEL_REQUEST_DIR" \
    "$EXTERNAL_ACK_DIR"
  if [ ! -d "$CONTROL_DIR" ] || [ ! -d "$EXTERNAL_ACK_DIR" ] \
    || ! chmod 0700 "$CONTROL_DIR" "$EXTERNAL_ACK_DIR"; then
    echo "error: queue control directory must be private and owner-writable" >&2
    return 1
  fi
}

_validate_term_grace() {
  TERM_GRACE_SECONDS="${TRAINING_QUEUE_TEST_TERM_GRACE_SECONDS:-15}"
  if ! [[ "$TERM_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: training queue TERM grace must be a positive integer (default: 15)" >&2
    return 1
  fi
}

_set_gpu_lock_namespace() {
  GPU_LOCK_FILE="$1"
  GPU_GATE_FILE="${GPU_LOCK_FILE}.gate"
  GPU_SLOT_0_FILE="${GPU_LOCK_FILE}.slot-0"
  GPU_SLOT_1_FILE="${GPU_LOCK_FILE}.slot-1"
  GPU_LOCK_NAMESPACE_KIND="$2"
}

_resolve_gpu_lock_namespace() {
  [ -n "$GPU_LOCK_FILE" ] && return 0

  if [ -n "${TRAINING_QUEUE_LOCK_FILE:-}" ]; then
    if [ "$TRAINING_QUEUE_LOCK_FILE" = "$SYSTEM_GPU_LOCK_FILE" ]; then
      _set_gpu_lock_namespace "$TRAINING_QUEUE_LOCK_FILE" system
    else
      _set_gpu_lock_namespace "$TRAINING_QUEUE_LOCK_FILE" explicit
    fi
    return 0
  fi

  local system_gate="${SYSTEM_GPU_LOCK_FILE}.gate"
  local system_slot_0="${SYSTEM_GPU_LOCK_FILE}.slot-0"
  local system_slot_1="${SYSTEM_GPU_LOCK_FILE}.slot-1"
  local system_parent path present=0 valid=0
  system_parent="$(dirname "$SYSTEM_GPU_LOCK_FILE")"
  if [ -L "$system_parent" ] || { [ -e "$system_parent" ] && { [ ! -d "$system_parent" ] || [ ! -x "$system_parent" ]; }; }; then
    echo "error: system GPU lock namespace directory is unsafe or unusable: $system_parent" >&2
    return 1
  fi
  for path in "$SYSTEM_GPU_LOCK_FILE" "$system_gate" "$system_slot_0" "$system_slot_1"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      present=$((present + 1))
    fi
    if [ -f "$path" ] && [ ! -L "$path" ] && [ -w "$path" ]; then
      valid=$((valid + 1))
    fi
  done
  if [ "$present" -gt 0 ]; then
    if [ "$present" -ne 4 ] || [ "$valid" -ne 4 ]; then
      echo "error: system GPU lock namespace is partial, unsafe, or unusable: $SYSTEM_GPU_LOCK_FILE" >&2
      return 1
    fi
    _set_gpu_lock_namespace "$SYSTEM_GPU_LOCK_FILE" system
    return 0
  fi

  local git_common="" git_root=""
  git_common="$(git -C "$PWD" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  if [ -n "$git_common" ]; then
    git_root="$(cd "$(dirname "$git_common")" 2>/dev/null && pwd)" || {
      echo "error: cannot resolve Git common root from $git_common" >&2
      return 1
    }
    _set_gpu_lock_namespace "$git_root/.training_queue/gpu.lock" git-common
    return 0
  fi

  _set_gpu_lock_namespace "$QUEUE_DIR/gpu.lock" queue-local
}

_ensure_gpu_lock_namespace() {
  local path probe_fd system_parent
  _resolve_gpu_lock_namespace || return 1
  if [ "$GPU_LOCK_NAMESPACE_KIND" = "system" ]; then
    system_parent="$(dirname "$GPU_LOCK_FILE")"
    if [ -L "$system_parent" ] || [ ! -d "$system_parent" ] || [ ! -x "$system_parent" ]; then
      echo "error: system GPU lock namespace directory is unsafe or unusable: $system_parent" >&2
      return 1
    fi
  fi
  if [ "$GPU_LOCK_NAMESPACE_KIND" != "system" ]; then
    mkdir -p "$(dirname "$GPU_LOCK_FILE")" || return 1
  fi
  for path in "$GPU_LOCK_FILE" "$GPU_GATE_FILE" "$GPU_SLOT_0_FILE" "$GPU_SLOT_1_FILE"; do
    if [ -L "$path" ] || { [ -e "$path" ] && [ ! -f "$path" ]; }; then
      echo "error: GPU lock path must be a regular file: $path" >&2
      return 1
    fi
    if [ "$GPU_LOCK_NAMESPACE_KIND" = "system" ]; then
      if [ ! -f "$path" ] || [ ! -w "$path" ]; then
        echo "error: system GPU lock path is unavailable: $path" >&2
        return 1
      fi
      if ! exec {probe_fd}>> "$path"; then
        echo "error: system GPU lock path cannot be opened: $path" >&2
        return 1
      fi
      exec {probe_fd}>&-
      continue
    fi
    if [ ! -e "$path" ]; then
      (umask 0117; : > "$path") || {
        echo "error: cannot create GPU lock file $path; provision the shared lock namespace" >&2
        return 1
      }
    fi
    if [ ! -w "$path" ] || ! exec {probe_fd}>> "$path"; then
      echo "error: GPU lock path is not writable: $path" >&2
      return 1
    fi
    exec {probe_fd}>&-
  done
}

_job_resource() {
  local jobfile="$1" metadata count resource
  # Parse only the leading comment header. A multiline user command must not be
  # able to masquerade as or duplicate queue metadata later in the script.
  metadata="$(awk '
    NR == 1 && /^#!/ { next }
    /^# resource: / { count += 1; value = substr($0, 13); next }
    /^#/ { next }
    { exit }
    END { printf "%d\n%s", count, value }
  ' "$jobfile" 2>/dev/null)" || return 2
  count="${metadata%%$'\n'*}"
  resource="${metadata#*$'\n'}"
  if [ "$count" -eq 0 ]; then
    printf 'all'
    return 0
  fi
  if [ "$count" -ne 1 ]; then
    return 2
  fi
  case "$resource" in
    half|all) printf '%s' "$resource" ;;
    *) return 2 ;;
  esac
}

_job_external_teardown_required() {
  local jobfile="$1" metadata count required
  metadata="$(awk '
    NR == 1 && /^#!/ { next }
    /^# external_teardown_ack: / {
      count += 1; value = substr($0, 26); next
    }
    /^#/ { next }
    { exit }
    END { printf "%d\n%s", count, value }
  ' "$jobfile" 2>/dev/null)" || return 2
  count="${metadata%%$'\n'*}"
  required="${metadata#*$'\n'}"
  if [ "$count" -eq 0 ]; then
    printf '0'
    return 0
  fi
  if [ "$count" -ne 1 ]; then
    return 2
  fi
  case "$required" in
    0|1) printf '%s' "$required" ;;
    *) return 2 ;;
  esac
}

_state_file() {
  printf '%s/%s.state' "$STATE_DIR" "${1%.job}"
}

_cancel_request_file() {
  printf '%s/%s.cancel' "$CANCEL_REQUEST_DIR" "${1%.job}"
}

_owner_file() {
  printf '%s/%s.owner' "$STATE_DIR" "${1%.job}"
}

_external_ack_file() {
  printf '%s/%s.ack' "$EXTERNAL_ACK_DIR" "${1%.job}"
}

_write_owner_record() {
  local job="$1" start_time tmp owner
  owner="$(_owner_file "$job")"
  start_time="$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null)"
  if ! [[ "$start_time" =~ ^[0-9]+$ ]]; then
    echo "error: cannot verify wrapper start time for $job" >&2
    return 1
  fi
  tmp="$STATE_DIR/.tmp.${job%.job}.owner.$$.$RANDOM"
  {
    printf 'pid=%s\n' "$$"
    printf 'start_time=%s\n' "$start_time"
    printf 'job=%s\n' "$job"
  } > "$tmp"
  mv -f "$tmp" "$owner"
}

_verified_owner_pid() {
  local job="$1" owner pid expected actual recorded_job
  owner="$(_owner_file "$job")"
  [ -f "$owner" ] && [ ! -L "$owner" ] || return 1
  pid="$(sed -n 's/^pid=//p' "$owner" | head -1)"
  expected="$(sed -n 's/^start_time=//p' "$owner" | head -1)"
  recorded_job="$(sed -n 's/^job=//p' "$owner" | head -1)"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] && [[ "$expected" =~ ^[0-9]+$ ]] || return 1
  [ "$recorded_job" = "$job" ] || return 1
  actual="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null)"
  [ "$actual" = "$expected" ] || return 1
  printf '%s' "$pid"
}

_external_ack_valid() {
  local job="$1" ack value
  ack="$(_external_ack_file "$job")"
  [ -f "$ack" ] && [ ! -L "$ack" ] || return 1
  value="$(sed -n '1p' "$ack" 2>/dev/null)"
  [ "$value" = "$job" ]
}

_waiting_cancel_requested() {
  local job="$1"
  [ -f "$(_cancel_request_file "$job")" ] || [ ! -f "$JOBS_DIR/$job" ]
}

_finish_waiting_cancellation() {
  local job="$1"
  if [ -f "$JOBS_DIR/$job" ]; then
    mv "$JOBS_DIR/$job" "$CANCELLED_DIR/$job" 2>/dev/null || true
  elif [ -f "$RUNNING_DIR/$job" ]; then
    mv "$RUNNING_DIR/$job" "$CANCELLED_DIR/$job" 2>/dev/null || true
  fi
  rm -f "$(_state_file "$job")" "$(_cancel_request_file "$job")" \
    "$(_owner_file "$job")" "$(_external_ack_file "$job")"
  if [ -f "$CANCELLED_DIR/$job" ]; then
    _record_test_teardown_event "$job" terminal-cancelled
  fi
}

_publish_cancel_request() {
  local job="$1" marker tmp
  marker="$(_cancel_request_file "$job")"
  tmp="$CANCEL_REQUEST_DIR/.tmp.${job%.job}.$$.$RANDOM"
  : > "$tmp"
  mv -f "$tmp" "$marker"
}

_write_job_state() {
  local job="$1" state="$2" resource="$3" slot="$4" pid="$5" wait_reason="$6"
  local pgid="${7:--}" state_file tmp
  state_file="$(_state_file "$job")"
  tmp="$STATE_DIR/.tmp.${job%.job}.$$.$RANDOM"
  {
    printf 'state=%s\n' "$state"
    printf 'resource=%s\n' "$resource"
    printf 'slot=%s\n' "$slot"
    printf 'pid=%s\n' "$pid"
    printf 'pgid=%s\n' "$pgid"
    printf 'wait=%s\n' "$wait_reason"
  } > "$tmp"
  mv -f "$tmp" "$state_file"
}

# Deterministic transition barrier used only by the queue E2E suite. The ready
# file is an event containing the wrapper/child PIDs; the named pipe is the
# release barrier. Production callers do not set either variable.
_test_transition_barrier() {
  local ready_file="$1" barrier_fifo="$2" payload="$3" release
  if [ -z "$ready_file" ] && [ -z "$barrier_fifo" ]; then
    return 0
  fi
  if [ -z "$ready_file" ] || [ -z "$barrier_fifo" ]; then
    echo "error: transition test barrier requires both ready file and FIFO" >&2
    return 1
  fi
  # A worker can supervise several wrappers; one ready file claims exactly one
  # transition so successors are not paused by the same fixture seam.
  [ -e "$ready_file" ] && return 0
  printf '%s\n' "$payload" > "$ready_file" || return 1
  IFS= read -r release < "$barrier_fifo" || true
}

# Optional append-only lifecycle oracle for isolated E2E tests. Production
# callers leave the seam unset. A failed oracle write is loud but must not
# alter teardown safety or release capacity early.
_record_test_teardown_event() {
  local job="$1" event="$2" event_file
  event_file="${TRAINING_QUEUE_TEST_TEARDOWN_EVENT_FILE:-}"
  [ -n "$event_file" ] || return 0
  if ! printf '%s %s\n' "$job" "$event" >> "$event_file"; then
    echo "error: could not record teardown test event: $event" >&2
  fi
}

_state_value() {
  local job="$1" key="$2"
  sed -n "s/^${key}=//p" "$(_state_file "$job")" 2>/dev/null | head -1
}

_print_job_record() {
  local directory="$1" job="$2" fallback_state="$3" fallback_wait="$4"
  local resource state slot pid pgid wait_reason
  resource="$(_job_resource "$directory/$job" 2>/dev/null || printf 'invalid')"
  state="$(_state_value "$job" state)"; state="${state:-$fallback_state}"
  slot="$(_state_value "$job" slot)"; slot="${slot:--}"
  pid="$(_state_value "$job" pid)"; pid="${pid:--}"
  pgid="$(_state_value "$job" pgid)"; pgid="${pgid:--}"
  wait_reason="$(_state_value "$job" wait)"; wait_reason="${wait_reason:-$fallback_wait}"
  printf '%s resource=%s slot=%s pid=%s pgid=%s state=%s wait=%s\n' \
    "$job" "$resource" "$slot" "$pid" "$pgid" "$state" "$wait_reason"
}

_poll_flock() {
  local job="$1" resource="$2" slot="$3" fd="$4" mode="$5" wait_reason="$6"
  while true; do
    if _waiting_cancel_requested "$job"; then
      return 125
    fi
    if [ "$wrapper_signal" -ne 0 ]; then
      return 143
    fi
    if [ "$mode" = "shared" ]; then
      if flock -n -s "$fd"; then
        return 0
      fi
    elif flock -n -x "$fd"; then
      return 0
    fi
    _write_job_state "$job" waiting "$resource" "$slot" "$$" "$wait_reason"
    sleep 0.05
  done
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
  local name="" command="" provider="" session="" issue="" resource="all"
  local prune_ckpt=0 external_teardown_ack=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --name|--provider|--session|--issue|--resource)
        if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
          echo "error: $1 requires a value" >&2
          exit 2
        fi
        case "$1" in
          --name) name="$2" ;;
          --provider) provider="$2" ;;
          --session) session="$2" ;;
          --issue) issue="$2" ;;
          --resource) resource="$2" ;;
        esac
        shift 2
        ;;
      --prune-ckpt) prune_ckpt=1; shift ;;
      --require-external-teardown-ack)
        external_teardown_ack=1
        shift
        ;;
      --*) echo "error: unknown add option: $1" >&2; exit 2 ;;
      *)
        if [ -n "$command" ]; then
          echo "error: add accepts exactly one command string" >&2
          exit 2
        fi
        command="$1"
        shift
        ;;
    esac
  done
  if [ -z "$command" ]; then
    echo "error: add requires a command string" >&2
    exit 2
  fi
  case "$resource" in
    half|all) ;;
    *) echo "error: --resource must be half or all (got: $resource)" >&2; exit 2 ;;
  esac
  _ensure_dirs || return 1
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
  local queue_abs repro_job_abs external_ack_abs
  queue_abs="$(cd "$QUEUE_DIR" && pwd)"
  repro_job_abs="$queue_abs/repro/$jobid"
  external_ack_abs="$queue_abs/control/external-acks/$jobid.ack"
  {
    echo "#!/usr/bin/env bash"
    echo "# name: ${name}"
    echo "# added: $(date -Iseconds)"
    echo "# provider: ${provider}"
    echo "# session: ${session}"
    echo "# issue: ${issue}"
    echo "# resource: ${resource}"
    echo "# external_teardown_ack: ${external_teardown_ack}"
    echo "# prune_ckpt: ${prune_ckpt}"
    echo "cd $(printf '%q' "$PWD")"
    echo "export TENNIS_RUN_ID=$(printf '%q' "$jobid")"
    echo "export TENNIS_REPRO_DIR=$(printf '%q' "$repro_job_abs")"
    echo "export TENNIS_GPU_RESOURCE=$(printf '%q' "$resource")"
    if [ "$external_teardown_ack" -eq 1 ]; then
      echo "export TRAINING_QUEUE_EXTERNAL_TEARDOWN_ACK=$(printf '%q' "$external_ack_abs")"
    fi
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
  local resource="${TENNIS_GPU_RESOURCE:-all}" slot="${TENNIS_GPU_SLOT:-all}"
  while [ $# -gt 0 ]; do
    case "$1" in
      --dir) dir="$2"; shift 2 ;;
      --name) name="$2"; shift 2 ;;
      --provider) provider="$2"; shift 2 ;;
      --session) session="$2"; shift 2 ;;
      --issue) issue="$2"; shift 2 ;;
      --cmd) cmd="$2"; shift 2 ;;
      --resource) resource="$2"; shift 2 ;;
      --slot) slot="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  if [ -z "$dir" ]; then
    echo "error: __capture-repro requires --dir" >&2
    exit 2
  fi
  case "$resource" in
    half|all) ;;
    *) echo "error: invalid repro GPU resource: $resource" >&2; exit 2 ;;
  esac
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
    printf '  "resource": "%s",\n' "$(_json_str "$resource")"
    printf '  "logical_gpu_slot": "%s",\n' "$(_json_str "$slot")"
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
    echo "# provider=${provider} session=${session} issue=${issue} resource=${resource} logical_slot=${slot}"
    echo "set -uo pipefail"
    printf 'export TENNIS_GPU_RESOURCE=%q\n' "$resource"
    printf 'export TENNIS_GPU_SLOT=%q\n' "$slot"
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

# Internal: acquire one logical allocation, execute the job, and retain the
# lock descriptors for the complete child lifecycle. One wrapper is created per
# admitted FIFO head so two half jobs can overlap without queue-local counters.
cmd_run_job() {
  local job="${1:-}" jobfile="$JOBS_DIR/${1:-}" log resource slot="all"
  local external_required=0 gate_fd="" slot_fd="" main_fd=""
  local job_pid="" job_pgid="" last_pid="-" last_pgid="-" rc=0
  local leader_rc=0 leader_rc_known=0 wrapper_signal=0 finalized=0 user_released=0
  local launch_gate="" transition_rc=0 group_probe_failed_logged=0
  if [ -z "$job" ] || [ ! -f "$jobfile" ]; then
    return 0
  fi
  _validate_term_grace || return 2
  log="$LOGS_DIR/${job%.job}.log"
  if ! resource="$(_job_resource "$jobfile")" \
    || ! external_required="$(_job_external_teardown_required "$jobfile")"; then
    echo "error: invalid or duplicate resource or teardown metadata in $job" > "$log"
    mv "$jobfile" "$FAILED_DIR/$job"
    _write_job_state "$job" failed invalid - - invalid-resource
    echo "[worker] $(date -Iseconds) FAILED (invalid metadata): $job"
    return 2
  fi
  if _waiting_cancel_requested "$job"; then
    _finish_waiting_cancellation "$job"
    return 0
  fi
  if ! _ensure_gpu_lock_namespace; then
    echo "error: GPU lock namespace is unavailable for $job" > "$log"
    mv "$jobfile" "$FAILED_DIR/$job"
    _write_job_state "$job" failed "$resource" - - lock-namespace
    return 1
  fi

  _job_wrapper_signal() {
    wrapper_signal=143
  }

  _owned_group_exists() {
    [ -n "$job_pgid" ] || return 1
    kill -0 -- "-$job_pgid" 2>/dev/null
  }

  _owned_group_has_live_members() {
    local process_table member_pgid member_state
    [ -n "$job_pgid" ] || return 1
    if ! process_table="$(ps -e -o pgid=,stat= 2>/dev/null)"; then
      if [ "$group_probe_failed_logged" -ne 1 ]; then
        echo "error: cannot inspect owned process-group members; retaining capacity" >> "$log"
        group_probe_failed_logged=1
      fi
      return 0
    fi
    while read -r member_pgid member_state; do
      [ "$member_pgid" = "$job_pgid" ] || continue
      case "$member_state" in
        Z*|X*) ;;
        *) return 0 ;;
      esac
    done <<< "$process_table"
    return 1
  }

  _reap_direct_leader() {
    local wait_rc=0
    [ -n "$job_pid" ] || return 0
    _record_test_teardown_event "$job" leader-reap-called
    while true; do
      wait "$job_pid"
      wait_rc=$?
      if ! kill -0 "$job_pid" 2>/dev/null; then
        break
      fi
      # A trapped TERM can interrupt wait(1) without reaping the child.
      sleep 0.02
    done
    leader_rc="$wait_rc"
    leader_rc_known=1
    job_pid=""
  }

  _publish_terminating() {
    local wait_reason="$1"
    _write_job_state \
      "$job" terminating "$resource" "$slot" "$last_pid" "$wait_reason" \
      "$last_pgid"
  }

  _teardown_owned_group() {
    local started_ns deadline_ns now_ns
    [ -n "$job_pgid" ] || return 0
    _publish_terminating process-group-teardown
    kill -TERM -- "-$job_pgid" 2>/dev/null || true
    _record_test_teardown_event "$job" term-sent
    started_ns="$(date +%s%N 2>/dev/null)"
    if ! [[ "$started_ns" =~ ^[0-9]+$ ]]; then
      echo "error: cannot read the teardown grace clock; retaining capacity" >> "$log"
      while _owned_group_has_live_members; do
        _publish_terminating process-group-teardown
        sleep 0.10
      done
    else
      deadline_ns=$((started_ns + TERM_GRACE_SECONDS * 1000000000))
      while _owned_group_has_live_members; do
        now_ns="$(date +%s%N 2>/dev/null)"
        if ! [[ "$now_ns" =~ ^[0-9]+$ ]]; then
          echo "error: teardown grace clock failed; retaining capacity" >> "$log"
          while _owned_group_has_live_members; do
            _publish_terminating process-group-teardown
            sleep 0.10
          done
          break
        fi
        [ "$now_ns" -ge "$deadline_ns" ] && break
        sleep 0.05
      done
      if _owned_group_has_live_members; then
        kill -KILL -- "-$job_pgid" 2>/dev/null || true
        _record_test_teardown_event "$job" kill-sent
      fi
      # A supported descendant in D state can survive SIGKILL. Keep the
      # wrapper, allocation FDs, and nonterminal state forever rather than
      # oversubscribe. Z/X members are exited and may remain until their parent
      # is reaped, so they do not block this first phase.
      while _owned_group_has_live_members; do
        _publish_terminating process-group-teardown
        sleep 0.10
      done
    fi

    _record_test_teardown_event "$job" live-members-absent
    _reap_direct_leader
    _record_test_teardown_event "$job" leader-reaped
    # Reaping the direct zombie leader permits the kernel PGID itself to
    # disappear. Retain capacity until that final identity proof succeeds.
    while _owned_group_exists; do
      _publish_terminating process-group-teardown
      sleep 0.10
    done
    _record_test_teardown_event "$job" pgid-absent
    job_pgid=""
  }

  _wait_for_external_ack() {
    [ "$external_required" -eq 1 ] && [ "$user_released" -eq 1 ] || return 0
    while ! _external_ack_valid "$job"; do
      _publish_terminating external-teardown
      sleep 0.10
    done
  }

  _finalize_terminal() {
    local terminal_state="$1" wait_reason="$2" terminal_dir
    case "$terminal_state" in
      done) terminal_dir="$DONE_DIR" ;;
      failed) terminal_dir="$FAILED_DIR" ;;
      cancelled) terminal_dir="$CANCELLED_DIR" ;;
      *) return 2 ;;
    esac
    if [ -f "$RUNNING_DIR/$job" ]; then
      mv "$RUNNING_DIR/$job" "$terminal_dir/$job" || return 1
    elif [ -f "$JOBS_DIR/$job" ] && [ "$terminal_state" = "cancelled" ]; then
      mv "$JOBS_DIR/$job" "$terminal_dir/$job" || return 1
    fi
    _write_job_state \
      "$job" "$terminal_state" "$resource" "$slot" "$last_pid" "$wait_reason" \
      "$last_pgid"
    _record_test_teardown_event "$job" "terminal-$terminal_state"
    rm -f "$(_cancel_request_file "$job")" "$(_owner_file "$job")" \
      "$(_external_ack_file "$job")"
    finalized=1
  }

  _finalize_after_teardown() {
    local terminal_rc="$1"
    _teardown_owned_group
    _wait_for_external_ack
    if [ -f "$(_cancel_request_file "$job")" ]; then
      printf 'exit_code=%s\n' "$terminal_rc" >> "$log"
      _finalize_terminal cancelled none
      echo "[worker] $(date -Iseconds) cancelled: $job resource=$resource slot=$slot"
    else
      printf 'exit_code=%s\n' "$terminal_rc" >> "$log"
      _finalize_terminal failed signal
      echo "[worker] $(date -Iseconds) FAILED (signal): $job resource=$resource slot=$slot"
    fi
  }

  _job_wrapper_cleanup() {
    local exit_rc=$?
    trap - EXIT
    trap _job_wrapper_signal TERM INT HUP
    [ -n "$launch_gate" ] && rm -f "$launch_gate"
    if [ "$finalized" -ne 1 ]; then
      if [ -n "$job_pgid" ]; then
        if [ "$wrapper_signal" -ne 0 ]; then
          _finalize_after_teardown "$wrapper_signal"
        else
          _finalize_after_teardown "$exit_rc"
        fi
      elif [ -n "$job_pid" ]; then
        # User code is still behind the launch gate and no private group was
        # verified. Signal only the known direct child; never an inherited PGID.
        kill -TERM "$job_pid" 2>/dev/null || true
        _reap_direct_leader
      fi
      if [ "$finalized" -ne 1 ]; then
        if [ -f "$(_cancel_request_file "$job")" ]; then
          _finish_waiting_cancellation "$job"
        elif [ -f "$RUNNING_DIR/$job" ]; then
          printf 'exit_code=%s\n' "$exit_rc" >> "$log"
          _finalize_terminal failed signal
        elif [ -f "$JOBS_DIR/$job" ]; then
          rm -f "$(_state_file "$job")" "$(_owner_file "$job")"
        fi
      fi
    fi
    rm -f "$(_owner_file "$job")"
    exit "$exit_rc"
  }

  _terminate_prelaunch_child() {
    if [ -n "$job_pid" ]; then
      if [ -n "$job_pgid" ]; then
        _teardown_owned_group
      else
        # The private PGID was not verified, so preserve the existing bounded
        # direct-child fallback and never signal an inherited group.
        kill -TERM "$job_pid" 2>/dev/null || true
        _reap_direct_leader
      fi
    fi
    [ -n "$launch_gate" ] && rm -f "$launch_gate"
    launch_gate=""
  }
  _prelaunch_checkpoint() {
    if [ "$wrapper_signal" -ne 0 ]; then
      return 143
    fi
    if [ -f "$(_cancel_request_file "$job")" ]; then
      _terminate_prelaunch_child
      if [ "$wrapper_signal" -ne 0 ]; then
        return 143
      fi
      _finish_waiting_cancellation "$job"
      finalized=1
      return 125
    fi
    if [ -f "$STOP_FILE" ]; then
      _terminate_prelaunch_child
      if [ "$wrapper_signal" -ne 0 ]; then
        return 143
      fi
      if [ -f "$(_cancel_request_file "$job")" ]; then
        _finish_waiting_cancellation "$job"
        finalized=1
        return 125
      fi
      if ! mv "$RUNNING_DIR/$job" "$JOBS_DIR/$job"; then
        return 1
      fi
      rm -f "$(_state_file "$job")" "$(_owner_file "$job")" \
        "$(_external_ack_file "$job")"
      finalized=1
      return 75
    fi
    return 0
  }
  _handle_prelaunch_checkpoint() {
    transition_rc=0
    _prelaunch_checkpoint || transition_rc=$?
    case "$transition_rc" in
      0) return 0 ;;
      75|125)
        trap - EXIT TERM INT HUP
        return "$transition_rc"
        ;;
      143) exit 143 ;;
      *) exit "$transition_rc" ;;
    esac
  }
  trap _job_wrapper_signal TERM INT HUP
  trap _job_wrapper_cleanup EXIT

  if [ "$resource" = "half" ]; then
    slot="-"
    while [ "$wrapper_signal" -eq 0 ]; do
      exec {gate_fd}> "$GPU_GATE_FILE" || exit 1
      if ! flock -n -s "$gate_fd"; then
        _write_job_state "$job" waiting "$resource" - "$$" admission-gate
        _poll_flock "$job" "$resource" - "$gate_fd" shared admission-gate
        case "$?" in
          0) ;;
          125) _finish_waiting_cancellation "$job"; finalized=1; trap - EXIT TERM INT HUP; return 0 ;;
          *) exit 143 ;;
        esac
      fi

      if _waiting_cancel_requested "$job"; then
        _finish_waiting_cancellation "$job"
        finalized=1
        trap - EXIT TERM INT HUP
        return 0
      fi
      _write_job_state "$job" waiting "$resource" - "$$" slot-capacity
      exec {slot_fd}> "$GPU_SLOT_0_FILE" || exit 1
      if flock -n "$slot_fd"; then
        slot="0"
      else
        exec {slot_fd}>&-
        exec {slot_fd}> "$GPU_SLOT_1_FILE" || exit 1
        if flock -n "$slot_fd"; then
          slot="1"
        else
          exec {slot_fd}>&-
          flock -u "$gate_fd" || true
          exec {gate_fd}>&-
          sleep 0.05
          continue
        fi
      fi

      exec {main_fd}> "$GPU_LOCK_FILE" || exit 1
      if ! flock -n -s "$main_fd"; then
        _write_job_state "$job" waiting "$resource" "$slot" "$$" main-shared-lock
        _poll_flock "$job" "$resource" "$slot" "$main_fd" shared main-shared-lock
        case "$?" in
          0) ;;
          125) _finish_waiting_cancellation "$job"; finalized=1; trap - EXIT TERM INT HUP; return 0 ;;
          *) exit 143 ;;
        esac
      fi
      flock -u "$gate_fd" || true
      exec {gate_fd}>&-
      break
    done
  else
    exec {gate_fd}> "$GPU_GATE_FILE" || exit 1
    if ! flock -n -x "$gate_fd"; then
      _write_job_state "$job" waiting "$resource" all "$$" admission-gate
      _poll_flock "$job" "$resource" all "$gate_fd" exclusive admission-gate
      case "$?" in
        0) ;;
        125) _finish_waiting_cancellation "$job"; finalized=1; trap - EXIT TERM INT HUP; return 0 ;;
        *) exit 143 ;;
      esac
    fi
    exec {main_fd}> "$GPU_LOCK_FILE" || exit 1
    if ! flock -n -x "$main_fd"; then
      _write_job_state "$job" waiting "$resource" all "$$" main-exclusive-lock
      _poll_flock "$job" "$resource" all "$main_fd" exclusive main-exclusive-lock
      case "$?" in
        0) ;;
        125) _finish_waiting_cancellation "$job"; finalized=1; trap - EXIT TERM INT HUP; return 0 ;;
        *) exit 143 ;;
      esac
    fi
    flock -u "$gate_fd" || true
    exec {gate_fd}>&-
  fi

  if [ "$wrapper_signal" -ne 0 ]; then
    exit 143
  fi
  # Queued cancellation may have moved the file while the wrapper waited. A
  # cooperative stop also forbids a waiting head from becoming newly active.
  if _waiting_cancel_requested "$job"; then
    _finish_waiting_cancellation "$job"
    finalized=1
    trap - EXIT TERM INT HUP
    return 0
  fi
  if [ -f "$STOP_FILE" ]; then
    rm -f "$(_state_file "$job")" "$(_cancel_request_file "$job")"
    finalized=1
    trap - EXIT TERM INT HUP
    return 0
  fi
  if ! mv "$jobfile" "$RUNNING_DIR/$job"; then
    rm -f "$(_state_file "$job")"
    finalized=1
    trap - EXIT TERM INT HUP
    return 0
  fi
  _write_owner_record "$job" || exit 1
  rm -f "$(_external_ack_file "$job")"

  _test_transition_barrier \
    "${TRAINING_QUEUE_TEST_POST_MOVE_READY_FILE:-}" \
    "${TRAINING_QUEUE_TEST_POST_MOVE_BARRIER_FIFO:-}" "$$" || exit 1
  # Reinspect the complete transition state immediately after jobs/ -> running/.
  # TERM is checked before cancellation and STOP because worker TERM publishes
  # STOP as part of its own teardown.
  _handle_prelaunch_checkpoint
  transition_rc=$?
  [ "$transition_rc" -eq 0 ] || return 0

  # Reinspect once more immediately before creating the child. The child starts
  # behind a private launch gate, so a TERM in the fork/$! assignment gap cannot
  # execute the user job before the wrapper captures and reaps its PID.
  _handle_prelaunch_checkpoint
  transition_rc=$?
  [ "$transition_rc" -eq 0 ] || return 0
  launch_gate="$STATE_DIR/.launch.${job%.job}.$$.$RANDOM"
  (umask 0177; mkfifo "$launch_gate") || exit 1

  if ! command -v setsid >/dev/null 2>&1; then
    echo "error: setsid is required for private queue workload ownership" > "$log"
    exit 1
  fi
  echo "[worker] $(date -Iseconds) running: $job resource=$resource slot=$slot (log: $log)"
  setsid bash -c '
    main_fd="$1"
    slot_fd="$2"
    launch_gate="$3"
    jobfile="$4"
    resource="$5"
    slot="$6"
    exec {main_fd}>&-
    if [ -n "$slot_fd" ]; then exec {slot_fd}>&-; fi
    if [ "${TRAINING_QUEUE_TEST_PRELAUNCH_IGNORE_TERM:-}" = "1" ]; then
      trap "" TERM
      if [ -n "${TRAINING_QUEUE_TEST_PRELAUNCH_READY_FILE:-}" ]; then
        printf "%s\n" "$$" > "$TRAINING_QUEUE_TEST_PRELAUNCH_READY_FILE"
      fi
    fi
    release=""
    IFS= read -r release < "$launch_gate" || exit 143
    [ "$release" = "launch" ] || exit 143
    TENNIS_GPU_RESOURCE="$resource" TENNIS_GPU_SLOT="$slot" exec bash "$jobfile"
  ' queue-job-launcher \
    "$main_fd" "$slot_fd" "$launch_gate" "$RUNNING_DIR/$job" \
    "$resource" "$slot" > "$log" 2>&1 &
  job_pid=$!
  last_pid="$job_pid"
  _test_transition_barrier \
    "${TRAINING_QUEUE_TEST_PRE_PID_READY_FILE:-}" \
    "${TRAINING_QUEUE_TEST_PRE_PID_BARRIER_FIFO:-}" "$$ $job_pid" || exit 1

  local identity="" verified_group=0
  for _ in $(seq 1 100); do
    identity="$(ps -o pgid=,sid= -p "$job_pid" 2>/dev/null | xargs)"
    if [ "$identity" = "$job_pid $job_pid" ]; then
      verified_group=1
      break
    fi
    kill -0 "$job_pid" 2>/dev/null || break
    sleep 0.01
  done
  if [ "$verified_group" -ne 1 ]; then
    echo "error: workload private session/PGID could not be verified" >> "$log"
    kill -TERM "$job_pid" 2>/dev/null || true
    _reap_direct_leader
    exit 1
  fi
  job_pgid="$job_pid"
  last_pgid="$job_pgid"

  # A trapped signal may have run after fork while job_pid was still empty.
  # Capture first, then immediately re-read every prelaunch transition before
  # allowing the gated child to execute.
  _handle_prelaunch_checkpoint
  transition_rc=$?
  [ "$transition_rc" -eq 0 ] || return 0
  _handle_prelaunch_checkpoint
  transition_rc=$?
  [ "$transition_rc" -eq 0 ] || return 0
  _write_job_state "$job" running "$resource" "$slot" "$job_pid" none "$job_pgid"
  user_released=1
  printf 'launch\n' > "$launch_gate" || exit 1
  rm -f "$launch_gate"
  launch_gate=""
  if [ "$wrapper_signal" -ne 0 ]; then
    exit 143
  fi
  while [ -n "$job_pid" ]; do
    wait "$job_pid"
    rc=$?
    if [ "$wrapper_signal" -ne 0 ] || [ -f "$(_cancel_request_file "$job")" ]; then
      rc=143
      _finalize_after_teardown "$rc"
      break
    fi
    if ! kill -0 "$job_pid" 2>/dev/null; then
      leader_rc="$rc"
      leader_rc_known=1
      job_pid=""
      break
    fi
  done

  if [ "$finalized" -eq 1 ]; then
    trap - EXIT TERM INT HUP
    return 0
  fi

  # A shell leader may return while an in-group background descendant remains.
  # Treat that as teardown work and keep capacity until the whole group is gone.
  if _owned_group_exists; then
    _teardown_owned_group
  fi
  _wait_for_external_ack

  if [ "$rc" -eq 0 ]; then
    _finalize_terminal done none
    echo "[worker] $(date -Iseconds) done: $job resource=$resource slot=$slot"
    _maybe_prune_ckpt "$job"
  else
    echo "exit_code=$rc" >> "$log"
    _finalize_terminal failed none
    echo "[worker] $(date -Iseconds) FAILED (rc=$rc): $job resource=$resource slot=$slot"
  fi
  trap - EXIT TERM INT HUP
}

_running_worker_children() {
  jobs -pr
}

_signal_wrapper() {
  local wrapper_pid="$1"
  kill -TERM "$wrapper_pid" 2>/dev/null || true
}

_terminate_worker_children() {
  local pid
  while read -r pid; do
    [ -n "$pid" ] && _signal_wrapper "$pid"
  done < <(_running_worker_children)
  wait 2>/dev/null || true
}

_wait_worker_children() {
  wait 2>/dev/null || true
}

# Internal: supervise wrappers. Only the current FIFO head may wait for
# admission; the next local job is considered only after that head starts.
cmd_worker() {
  local after_pid="" idle_timeout=30 worker_signal=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --after-pid) after_pid="$2"; shift 2 ;;
      --idle-timeout) idle_timeout="$2"; shift 2 ;;
      *) echo "error: unknown worker option: $1" >&2; return 2 ;;
    esac
  done
  _ensure_dirs || return 1
  _validate_term_grace || return 2
  rm -f "$STOP_FILE"

  _worker_signal_handler() {
    worker_signal=1
    # Further service TERM requests must not kill the worker while its wrappers
    # still own allocation FDs and are proving process/container absence.
    trap '' TERM INT HUP
    # Close admission before signalling wrappers. A wrapper that was already
    # waiting may acquire capacity during teardown and must leave its job queued
    # for the replacement worker, which clears this marker on startup.
    touch "$STOP_FILE"
    echo "[worker] termination requested; stopping supervised jobs."
    _terminate_worker_children
  }
  trap _worker_signal_handler TERM INT HUP

  if [ -n "$after_pid" ]; then
    echo "[worker] waiting for PID $after_pid to finish..."
    tail --pid="$after_pid" -f /dev/null 2>/dev/null || true
    echo "[worker] PID $after_pid finished; starting queue."
  fi

  local idle_elapsed=0
  while [ "$worker_signal" -eq 0 ]; do
    if [ -f "$STOP_FILE" ]; then
      echo "[worker] stop requested; waiting for active jobs."
      _wait_worker_children
      rm -f "$STOP_FILE"
      break
    fi

    local job active active_pid
    job="$(find "$JOBS_DIR" -maxdepth 1 -name '*.job' ! -name '.tmp.*' -printf '%f\n' 2>/dev/null | sort | head -1)"
    if [ -z "$job" ]; then
      # Keep jobs(1) in a process substitution that inherits this shell's job
      # table. A command-substitution pipeline reports zero and can let an
      # idle-timeout=0 worker exit before its moved-to-running wrapper finishes.
      active=0
      while read -r active_pid; do
        [ -n "$active_pid" ] && active=$((active + 1))
      done < <(_running_worker_children)
      if [ "$active" -gt 0 ]; then
        idle_elapsed=0
        sleep 0.05
        continue
      fi
      wait 2>/dev/null || true
      if [ "$idle_elapsed" -ge "$idle_timeout" ]; then
        echo "[worker] queue empty for ${idle_timeout}s; exiting."
        break
      fi
      sleep 1
      idle_elapsed=$((idle_elapsed + 1))
      continue
    fi
    idle_elapsed=0

    # worker.lock belongs only to serve; wrappers must not keep it alive after
    # a supervised worker termination.
    bash "$SCRIPT_PATH" __run-job "$job" 9>&- &
    local admission_pid=$!
    # Preserve strict FIFO: do not inspect the next job until this head has
    # actually moved to running (or failed/cancelled before execution).
    while [ -f "$JOBS_DIR/$job" ] && kill -0 "$admission_pid" 2>/dev/null; do
      if [ -f "$STOP_FILE" ] || [ "$worker_signal" -ne 0 ]; then
        _signal_wrapper "$admission_pid"
        wait "$admission_pid" 2>/dev/null || true
        break
      fi
      sleep 0.05
    done
  done
  if [ "$worker_signal" -ne 0 ]; then
    return 143
  fi
  echo "[worker] $(date -Iseconds) worker exited."
}

# Foreground worker entry point for supervisors such as systemd. The lock is
# held for the process lifetime, so start/serve cannot create duplicate workers.
cmd_serve() {
  _ensure_dirs || return 1
  _validate_term_grace || exit 2
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
  _ensure_dirs || return 1
  _validate_term_grace || exit 2
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
  _ensure_dirs || return 1
  if _worker_running; then
    echo "worker: RUNNING (PID $(cat "$WORKER_PID_FILE"))"
  else
    echo "worker: stopped"
  fi
  local q q_total r r_executing waiting d f c job state
  q_total=$(find "$JOBS_DIR" -maxdepth 1 -name '*.job' ! -name '.tmp.*' 2>/dev/null | wc -l)
  r_executing=$(find "$RUNNING_DIR" -maxdepth 1 -name '*.job' 2>/dev/null | wc -l)
  waiting=0
  while read -r job; do
    [ -n "$job" ] || continue
    state="$(_state_value "$job" state)"
    [ "$state" = "waiting" ] && waiting=$((waiting + 1))
  done < <(find "$JOBS_DIR" -maxdepth 1 -name '*.job' ! -name '.tmp.*' -printf '%f\n' 2>/dev/null)
  q=$((q_total - waiting))
  r=$((r_executing + waiting))
  d=$(find "$DONE_DIR" -maxdepth 1 -name '*.job' 2>/dev/null | wc -l)
  f=$(find "$FAILED_DIR" -maxdepth 1 -name '*.job' 2>/dev/null | wc -l)
  c=$(find "$CANCELLED_DIR" -maxdepth 1 -name '*.job' 2>/dev/null | wc -l)
  echo "gpu-capacity: slots=2 mode=logical-only mig=unchanged vram-hard-cap=none"
  echo "queued=$q running=$r done=$d failed=$f cancelled=$c"
  echo "executing=$r_executing waiting=$waiting"
  if [ "$r_executing" -gt 0 ]; then
    echo "running:"
    while read -r job; do
      [ -n "$job" ] && _print_job_record "$RUNNING_DIR" "$job" running none | sed 's/^/  /'
    done < <(find "$RUNNING_DIR" -maxdepth 1 -name '*.job' -printf '%f\n' | sort)
  fi
  if [ "$q_total" -gt 0 ]; then
    echo "pending (run order):"
    local index=0 job wait_reason
    while read -r job; do
      [ -n "$job" ] || continue
      wait_reason="fifo"
      [ "$index" -eq 0 ] && wait_reason="next"
      _print_job_record "$JOBS_DIR" "$job" queued "$wait_reason" | sed 's/^/  /'
      index=$((index + 1))
    done < <(find "$JOBS_DIR" -maxdepth 1 -name '*.job' ! -name '.tmp.*' -printf '%f\n' | sort)
  fi
  if [ "$d" -gt 0 ]; then
    echo "completed (latest 10):"
    while read -r job; do
      [ -n "$job" ] && _print_job_record "$DONE_DIR" "$job" done none | sed 's/^/  /'
    done < <(find "$DONE_DIR" -maxdepth 1 -name '*.job' -printf '%f\n' | sort | tail -10)
  fi
  if [ "$f" -gt 0 ]; then
    echo "failed (latest 10):"
    while read -r job; do
      [ -n "$job" ] && _print_job_record "$FAILED_DIR" "$job" failed none | sed 's/^/  /'
    done < <(find "$FAILED_DIR" -maxdepth 1 -name '*.job' -printf '%f\n' | sort | tail -10)
  fi
  if [ "$c" -gt 0 ]; then
    echo "cancelled (latest 10):"
    while read -r job; do
      [ -n "$job" ] && _print_job_record "$CANCELLED_DIR" "$job" cancelled none | sed 's/^/  /'
    done < <(find "$CANCELLED_DIR" -maxdepth 1 -name '*.job' -printf '%f\n' | sort | tail -10)
  fi
}

cmd_list() {
  _ensure_dirs || return 1
  local directory state job index=0 wait_reason
  while read -r job; do
    [ -n "$job" ] || continue
    wait_reason="fifo"
    [ "$index" -eq 0 ] && wait_reason="next"
    _print_job_record "$JOBS_DIR" "$job" queued "$wait_reason"
    index=$((index + 1))
  done < <(find "$JOBS_DIR" -maxdepth 1 -name '*.job' ! -name '.tmp.*' -printf '%f\n' 2>/dev/null | sort)
  for directory in "$RUNNING_DIR" "$DONE_DIR" "$FAILED_DIR" "$CANCELLED_DIR"; do
    case "$directory" in
      "$RUNNING_DIR") state=running ;;
      "$DONE_DIR") state=done ;;
      "$FAILED_DIR") state=failed ;;
      *) state=cancelled ;;
    esac
    while read -r job; do
      [ -n "$job" ] && _print_job_record "$directory" "$job" "$state" none
    done < <(find "$directory" -maxdepth 1 -name '*.job' -printf '%f\n' 2>/dev/null | sort)
  done
}

cmd_cancel() {
  local job="${1:-}" marker state attempt owner_pid
  if [ -z "$job" ] || [ "$job" != "${job##*/}" ] || [[ "$job" != *.job ]] || [[ "$job" == .tmp.* ]]; then
    echo "error: cancel requires one published job filename" >&2
    return 2
  fi
  if [ $# -ne 1 ]; then
    echo "error: cancel accepts exactly one job filename" >&2
    return 2
  fi
  _ensure_dirs || return 1
  marker="$(_cancel_request_file "$job")"

  if [ -f "$CANCELLED_DIR/$job" ]; then
    rm -f "$marker" "$(_owner_file "$job")" "$(_external_ack_file "$job")"
    echo cancelled
    return 0
  fi
  if [ -f "$DONE_DIR/$job" ]; then
    rm -f "$marker"
    echo done
    return 0
  fi
  if [ -f "$FAILED_DIR/$job" ]; then
    rm -f "$marker"
    echo failed
    return 0
  fi

  _publish_cancel_request "$job"
  for attempt in $(seq 1 100); do
    if [ -f "$JOBS_DIR/$job" ]; then
      if mv "$JOBS_DIR/$job" "$CANCELLED_DIR/$job" 2>/dev/null; then
        # A waiting wrapper observes the missing source path on its next bounded
        # poll. Removing the marker is safe because file absence is also a
        # cancellation intent until that wrapper exits.
        rm -f "$(_state_file "$job")" "$marker" "$(_owner_file "$job")" \
          "$(_external_ack_file "$job")"
        echo cancelled
        return 0
      fi
      continue
    fi
    if [ -f "$CANCELLED_DIR/$job" ]; then
      rm -f "$(_state_file "$job")" "$marker" "$(_owner_file "$job")" \
        "$(_external_ack_file "$job")"
      echo cancelled
      return 0
    fi
    if [ -f "$RUNNING_DIR/$job" ]; then
      state="$(_state_value "$job" state)"
      if [ "$state" = "running" ] || [ "$state" = "terminating" ]; then
        if owner_pid="$(_verified_owner_pid "$job")"; then
          kill -TERM "$owner_pid" 2>/dev/null || true
          echo terminating
          return 0
        fi
        # The file/state/owner publication is atomic per file, but these three
        # publications can briefly be observed between transitions.
        if [ "$state" = "terminating" ]; then
          echo terminating
          return 0
        fi
        sleep 0.02
        continue
      fi
      if [ "$state" = "cancelled" ]; then
        rm -f "$marker"
        echo cancelled
        return 0
      fi
      # The wrapper has moved the file but has not launched the command yet.
      # Keep the marker durable until its post-move re-read transitions the job.
      sleep 0.02
      continue
    fi
    if [ -f "$DONE_DIR/$job" ]; then
      rm -f "$marker"
      echo done
      return 0
    fi
    if [ -f "$FAILED_DIR/$job" ]; then
      rm -f "$marker"
      echo failed
      return 0
    fi
    sleep 0.02
  done

  # A running cancellation marker is an ownership request and must remain
  # durable until the wrapper has verified supported teardown.
  if [ -f "$RUNNING_DIR/$job" ]; then
    echo terminating
    return 0
  fi
  rm -f "$marker"
  echo "error: job state did not stabilize during cancellation: $job" >&2
  return 1
}

cmd_stop() {
  _ensure_dirs || return 1
  touch "$STOP_FILE"
  echo "stop requested; worker will exit after currently active jobs finish."
}

cmd_clear() {
  _ensure_dirs || return 1
  local job
  while read -r job; do
    [ -n "$job" ] || continue
    rm -f "$JOBS_DIR/$job" "$(_state_file "$job")" \
      "$(_cancel_request_file "$job")" "$(_owner_file "$job")" \
      "$(_external_ack_file "$job")"
  done < <(find "$JOBS_DIR" -maxdepth 1 -name '*.job' -printf '%f\n' 2>/dev/null)
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
    __run-job)        cmd_run_job "$@" ;;
    __capture-repro)  cmd_capture_repro "$@" ;;
    status)   cmd_status "$@" ;;
    list)     cmd_list "$@" ;;
    cancel)   cmd_cancel "$@" ;;
    stop)     cmd_stop "$@" ;;
    clear)    cmd_clear "$@" ;;
    -h|--help|help|"")
      sed -n '2,40p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//' ;;
    *)
      echo "unknown subcommand: $sub (try --help)" >&2; exit 2 ;;
  esac
}

main "$@"
