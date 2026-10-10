# Heartbeat an open local CLI

Use when the requested receiver is the currently open Codex CLI. This route was investigated against `codex-cli 0.161.0`. Check `codex queue --help` on the actual host; do not assume an older CLI has it.

## Delivery contract

`codex queue` submits to the native persistent thread queue. The existing local runtime polls the shared queue (10 seconds in 0.161.0) and consumes messages when the target thread is idle. It does not interrupt an active turn. A user-interrupted thread may suspend automatic consumption. Closing an in-process CLI leaves queued messages pending until a suitable runtime resumes the thread.

Use the **same runtime environment**, particularly `CODEX_HOME` and `CODEX_SQLITE_HOME`. In Windows/WSL setups these may deliberately select different state databases even when conversation history is shared. Forward the current values; do not change the global shell configuration. A child operator needs the owning thread ID, original cwd, verified rollout path and environment values from its parent.

The helper captures only the routing environment needed for delivery, calls the existing CLI with an argument array, and does not edit native databases or start another model session. The sender's process is allowed to exit after queue acceptance. Receiving and rendering the next turn happens in the already-open CLI.

Implementation references for the observed version: [CLI queue submission](https://github.com/openai/codex/blob/979011409de0a60b52f179721948e65531d26144/codex-rs/tui/src/session_queue_commands.rs), [queue consumer](https://github.com/openai/codex/blob/979011409de0a60b52f179721948e65531d26144/codex-rs/ext/queue/src/service.rs), and [cross-runtime tests](https://github.com/openai/codex/blob/979011409de0a60b52f179721948e65531d26144/codex-rs/ext/queue/tests/queue_service.rs). These are source-level evidence; they do not replace a receiver-side live check.

## Prepare once

Use Python 3.11+ and an available user systemd manager (`systemctl --user is-system-running`). The helper supports Linux/WSL, interval schedules and existing JSONL rollouts. It rejects unsupported setups rather than switching to desktop scheduling. Timers are transient: they survive the creating shell, but are not boot-persistent.

Resolve absolute values from the actual session. A rollout under `sessions/` must match the verified thread ID in its `session_meta`; do not choose the latest file. The cwd must belong to the receiving session, not the delegated operator's worktree.

```bash
TASK_SKILL_DIR="/absolute/path/to/codex-scheduled-followup"
TASK_PY="/absolute/path/to/.venv/bin/python"
TASK_THREAD_ID="verified-owning-thread-uuid"
TASK_CWD="/absolute/path/to/owning/session/cwd"
TASK_ROLLOUT="/absolute/path/to/the/matching/rollout.jsonl"
TASK_PROMPT="/absolute/path/to/followup-prompt.txt"
TASK_ID="training-followup"
```

Write a UTF-8 prompt naming the real worktree, jobs and result locations, authorized next steps, and completion/stop condition. For long work, reference a durable memory file. Do not embed a generic instruction to keep working forever or create new work when none remains.

## Register the requested interval

For an explicitly requested 30-minute interval:

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" create \
  --id "$TASK_ID" --thread-id "$TASK_THREAD_ID" \
  --cwd "$TASK_CWD" --rollout "$TASK_ROLLOUT" \
  --prompt-file "$TASK_PROMPT" --interval-minutes 30
```

Use the returned `task_dir` in every subsequent operation. The default state location is under the Linux user's `~/.local/state/codex-followups/`; `--state-dir` can select an isolated Linux directory for a test. Do not store lock/control state on a Windows-mounted directory. New tasks default to 60 minutes only when no cadence was given. Calendar schedules need a route that can express the requested calendar/timezone; do not round them to interval schedules.

`create` registers a user systemd timer and reports its identity and next-run information. Confirm the returned active timer, interval and target. A saved configuration alone is not success. Do not silently overwrite an existing task with different instructions; use the helper's supported lifecycle and preserve its delivery evidence.

## Inspect or send one authorized probe

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" status \
  --task-dir "/absolute/returned/task_dir"

# Only for a user-authorized immediate delivery or bounded live test:
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" tick \
  --task-dir "/absolute/returned/task_dir"
```

The helper admits at most one outstanding delivery per task. Queue submission has a separate 180-second timeout (configurable with `create --queue-timeout-seconds`); systemd queries keep their 45-second timeout. It records a unique marker before invoking `codex queue`, then distinguishes queue acceptance, receipt in the target's user-message history, and the associated turn's completion. A timeout/crash/ambiguous sender result must not cause automatic repeated submissions. Inspect and resolve the uncertain delivery before retrying.

To exercise the clock as well as delivery during an explicitly authorized live test, a separate one-shot user timer may invoke the same `tick` command after a few seconds. Keep the requested recurring interval unchanged, give the probe its own unit name, and clean up both test units afterward. A successful one-shot is not proof that a 30-minute recurrence has already fired.

History verification requires the same, untruncated JSONL rollout. Unsupported migration or replacement is an explicit limitation, not permission to edit native state. Completion evidence means that the associated turn ended; assess the response/artifacts to decide whether the requested work succeeded.

For a probe addressed to the parent while it is working, the child should return the accepted/pending receipt immediately. The parent saves the verification command and yields. The queued prompt should name the task directory and tell the receiver how to record the test result and stop the test timer. The receiver can confirm arrival from its own input, but its completed-turn receipt becomes observable after that turn ends. Never wait indefinitely for this condition inside the same active turn.

## Stop

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" pause \
  --task-dir "/absolute/returned/task_dir"
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" status \
  --task-dir "/absolute/returned/task_dir"
```

Confirm the timer is inactive with no next run. Preserve receipts. A previously accepted message can still be consumed; pausing the clock is not queue cancellation. Do not delete queue rows or interrupt the user's work to hide it.

The helper has no general update/resume command. For changed instructions or a resumed schedule, first stop the old timer and resolve its outstanding delivery, then create a distinct task ID while retaining the old receipts. Do not bypass an uncertain delivery by making another task for the same work.

## Recover delivery without discarding evidence

`status` distinguishes timer registration from delivery health. `attention_required=true` and
`delivery_health=needs_recovery` mean continuations cannot advance even if the timer is active.
A timeout preserves stdout/stderr and the timeout duration. An unambiguous queue acknowledgement
for the target thread remains accepted even if the sending process subsequently times out.
Acceptance still does not prove receipt or completion.

If no acknowledgement exists, automatic resend remains disabled: the CLI has no verified
caller-controlled idempotency key. Inspect the exact marker in the target rollout and, when
available, inspect the same runtime's pending queue read-only. No pending item does not prove
that the first submission was never accepted.

For a user-authorized restoration where the remaining duplicate risk is accepted, recover the
**existing task**, specifying the exact unresolved marker and the reason. An explicit user request
to repair and restore this monitoring authorizes this operation; do not ask the user to approve
the same restoration again. Otherwise explain the remaining uncertainty before seeking authorization.

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" recover \
  --task-dir "/absolute/returned/task_dir" \
  --delivery-marker "[codex-heartbeat:task-id:exact-uuid]" \
  --reason "User requested repair and restoration; target queue and history inspected" \
  --acknowledge-possible-duplicate
```

Recovery reconciles late receipts first and refuses queued/started deliveries, mismatched markers,
or changed history. It records the old phase and reason, marks that attempt `superseded`, and keeps
its error/logs. It does not delete a queue item, fabricate completion, send a message, create another
timer, or change the cadence. The next tick can send once with a new marker; an authorized immediate
verification may invoke `tick` once. Ensure the continuation itself checks durable job identities
so a late old message cannot start duplicate training. Verify registration, acceptance and actual
receiver execution separately. Do not repeatedly recover/resend an unresolved failure automatically.

Run portable unit checks with `scripts/test_cli_heartbeat.py`. Those checks use temporary histories and mocked commands. A separate live test must verify the actual receiving CLI and should remove its scheduling side effects when complete.
