# Local heartbeat quickstart

Use only for a **desktop** follow-up when the native Codex automation tool is unavailable. For an already-open CLI, use [CLI queue heartbeats](cli-heartbeat.md). This file-based fallback was observed in desktop app 26.930.7945.0; app registration must still be verified. The helper never updates the live SQLite database.

## Resolve once, then use absolute paths

Set these task-specific variables from the loaded skill, the active project's Python 3.11+ interpreter, and the desktop app's actual Codex home. Do not change HOME or CODEX_HOME. If the home/app location or Windows–WSL boundary is unclear, read [the platform notes](windows-wsl.md); otherwise skip them.

```bash
TASK_SKILL_DIR="/absolute/path/to/codex-scheduled-followup"
TASK_PY="/absolute/path/to/.venv/bin/python"
TASK_CODEX_DIR="/absolute/path/to/the/desktop/codex-home"
TASK_THREAD_ID="verified-owning-chat-id"
TASK_ID="training-followup"
TASK_PROMPT="/absolute/path/to/followup-prompt.txt"
TASK_RECEIPT="/absolute/path/to/registration-receipt.json"
```

The helper's common options, including --codex-home, go **after the subcommand**. All examples work from outside the repository when these paths are absolute. Reuse the successful command/argument list for retries instead of retyping long paths. Shell variables do not persist across separate shell/tool invocations: include the bindings in each new shell or reuse a saved command. Use the intended parent/chat ID for delegated setup, not a child session's environment ID. Inspect existing task IDs/names in this home first and reuse a matching task.

Write the prompt as UTF-8. It must identify the actual job/status source, authorized work after success, behavior on failure, and completion condition. For detailed context, write memory.md beside the task's automation.toml and reference its absolute path. Do not schedule setup-only instructions such as “do not finalize yet” as an indefinite restriction on future turns.

## Create, or update an existing task

New task, using the user's example request of 30 minutes:

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/heartbeat.py" create \
  --codex-home "$TASK_CODEX_DIR" --id "$TASK_ID" \
  --name '処理完了後の作業を再開' --prompt-file "$TASK_PROMPT" \
  --thread-id "$TASK_THREAD_ID" --interval-minutes 30
```

Choose exactly one schedule option: --interval-minutes, --interval-hours, or --rrule. Only a **new** task with no schedule defaults to one hour. A custom RFC 5545 rule is preserved verbatim, including DTSTART/TZID when supplied; the app decides whether it supports it. Prefer the native tool for calendar/timezone schedules. verify --timezone only formats receipt timestamps and never changes the schedule.

Change an existing task's interval and resume it in one operation:

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/heartbeat.py" update \
  --codex-home "$TASK_CODEX_DIR" --id "$TASK_ID" \
  --interval-minutes 15 --status ACTIVE
```

update preserves omitted cadence/status and all other fields, including name, prompt, thread, model/effort, creation time and operator notes. It requires at least one explicit change request. A repeated identical create/update returns changed=false and the same config_sha256 without changing updated_at. A conflicting create fails: use update for an authorized change, not another task ID. Neither successful file creation nor mutation alone proves registration.

## Confirm app registration

Ensure the desktop app is running and allow it to import the configuration. Reuse a running app; changed=false does not require another app launch/import. For launch troubleshooting only, see the linked platform notes.

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/heartbeat.py" verify \
  --codex-home "$TASK_CODEX_DIR" --id "$TASK_ID" \
  --wait-seconds 20 --timezone Asia/Tokyo --receipt "$TASK_RECEIPT"
```

This compares the intended configuration with a stable temporary copy of the app's database and WAL and runs quick_check. Its default database is <codex-home>/sqlite/codex-dev.db; --database is an explicit override after locating another actual app database. The live database is never opened by SQLite in place.

Success requires one matching ID (or migrated legacy ID), matching name/prompt/kind/thread/recurrence/status, and a valid next_run_at for ACTIVE or no next run for PAUSED. File/account migration can disable this route. A missing, inconsistent or changed entry fails; do not write native database rows or repeatedly retry a permanent mismatch. A short startup/import wait is bounded to 30 seconds.

The receipt proves registration, not execution. Report its actual next-run timestamp: the observed app can add up to ±300 seconds of jitter. Do not promise execution exactly on the hour. If account migration removed the local file, use native management rather than recreating it from stale notes.

## Finish

Use the native tool to delete a completed heartbeat when available. When only a file-managed task can be controlled, pause and verify:

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/heartbeat.py" update \
  --codex-home "$TASK_CODEX_DIR" --id "$TASK_ID" --status PAUSED
"$TASK_PY" "$TASK_SKILL_DIR/scripts/heartbeat.py" verify \
  --codex-home "$TASK_CODEX_DIR" --id "$TASK_ID" --wait-seconds 20
```

set-status remains an alias for existing status-only callers. Preserve evidence, and describe a fallback pause as a pause rather than claiming deletion. Portable tests use temporary homes: run "$TASK_PY" "$TASK_SKILL_DIR/scripts/test_heartbeat.py". They do not exercise the live app or execute future scheduled turns.
