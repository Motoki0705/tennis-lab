# Local heartbeat file procedure

This is the fallback used on 2026-10-07 in a Windows/WSL environment. The installed desktop app was `OpenAI.Codex 26.930.7945.0`. Its local implementation reads `automations/<id>/automation.toml`, imports accepted entries into its scheduler, and calculates the next run. The file format is an observed implementation detail, not a public API guarantee.

## 1. Resolve the correct home and target

Use the desktop app's Codex home. In the observed environment, `CODEX_HOME` was `/mnt/c/Users/kamim/.codex`, shared with Windows `C:\Users\kamim\.codex`. Do not assume every WSL installation shares this location. Pass `--codex-home` explicitly if the environment points elsewhere.

Use `CODEX_THREAD_ID` for the current thread. If it is unavailable, obtain the intended thread ID from the app or user context; do not select the most recent session globally. The helper defaults to this environment variable and fails when neither it nor `--thread-id` is supplied.

Choose one descriptive task ID and keep it across retries. Check existing `automations/*/automation.toml` for this thread/purpose before creating a different ID.

## 2. Write the prompt and optional continuation notes

Use a UTF-8 prompt file. Include the actual completion condition and authorized post-completion work. When detailed context is needed, save it as `automations/<id>/memory.md` and mention its absolute path in the prompt; the helper does not invent or copy continuation notes.

For example, a monitoring prompt should say which job to inspect, where its durable status/logs live, what to do while it is running, what to do on failure, what work follows success, and when to stop this schedule. Do not copy a previous task's branch, job ID, or thread ID into a new task.

## 3. Create a configuration with the requested schedule

Use the active project's Python 3.11+ executable. In tennis-lab that is `.venv/bin/python`. The following commands run from this repository root. If using a personal installation, set `TASK_SKILL_DIR` to the directory containing the loaded `SKILL.md` instead.

```bash
TASK_SKILL_DIR=".agents/skills/codex-scheduled-followup"

.venv/bin/python "$TASK_SKILL_DIR/scripts/heartbeat.py" create \
  --id training-followup \
  --name '学習終了後の作業を再開' \
  --prompt-file /absolute/path/to/followup-prompt.txt \
  --thread-id "$CODEX_THREAD_ID"
```

This example leaves the schedule unspecified, so it defaults to one hour. For a user-specified schedule, add exactly one of the following options to the create command:

| User request | Option |
| --- | --- |
| Every 30 minutes | `--interval-minutes 30` |
| Every 2 hours | `--interval-hours 2` |
| A calendar-based recurrence | `--rrule 'FREQ=WEEKLY;BYDAY=MO,FR;BYHOUR=9;BYMINUTE=0'` |

Preserve an explicitly requested timezone when constructing a calendar recurrence; prefer the native tool for timezone-aware calendars. `--rrule` passes the supplied RFC 5545 text through unchanged, including a supplied DTSTART/TZID line, and the app must validate it. `verify --timezone` below changes only timestamp display, not the task's timezone. Empty explicit rules, nonpositive intervals, or multiple schedule options fail rather than reverting to one hour. An unsupported requested recurrence is a registration failure, not permission to substitute the default.

This writes the following shape, with actual values and Unix timestamps in milliseconds:

```toml
version = 1
id = "training-followup"
kind = "heartbeat"
name = "学習終了後の作業を再開"
prompt = "具体的な監視対象・残作業・停止条件"
status = "ACTIVE"
rrule = "FREQ=HOURLY;INTERVAL=1"
target_thread_id = "current-thread-id"
created_at = 1791334759537
updated_at = 1791334759537
```

`kind = "heartbeat"` returns to the existing thread. Use the native tool for standalone tasks. This helper does not create a Linux cron job, a Windows Task Scheduler job, or a new Codex process loop.

The helper refuses a conflicting existing configuration. Repeating exactly the same create request is a no-op, including when the existing task has been paused: creation does not silently reactivate it. For an authorized cadence change to an existing task, update that task through the native tool or deliberately edit its `rrule`/`updated_at` and verify it; do not create a second ID to evade the conflict. Its printed `configuration_saved` is not a registration receipt.

## 4. Ensure the desktop app is running

On the observed Windows package, the visible app executable is **ChatGPT.exe**, despite the package name **OpenAI.Codex**. Looking only for `codex.exe` would miss the GUI; that process can also be just an app server.

From WSL, the executable may not be on `PATH`. Inspect installed identity/processes using the full PowerShell path:

```bash
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  'Get-AppxPackage -Name OpenAI.Codex | Select-Object PackageFamilyName,InstallLocation | ConvertTo-Json -Compress'

/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  'Get-Process ChatGPT -ErrorAction SilentlyContinue | Select-Object ProcessName,Id,Path | ConvertTo-Json -Compress'
```

In this installation, `AppxManifest.xml` identifies application ID `App`, so opening the app used:

```bash
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  'Start-Process explorer.exe -ArgumentList "shell:AppsFolder\OpenAI.Codex_2p2nqsd0c76g0!App"'
```

Check package identity before reusing that command on another machine. On other platforms use the installed app's normal launch route. Starting the GUI is an operational dependency of local scheduling, not a way to bypass missing account access or permissions.

## 5. Confirm native registration

```bash
.venv/bin/python "$TASK_SKILL_DIR/scripts/heartbeat.py" verify \
  --id training-followup \
  --wait-seconds 20 \
  --timezone Asia/Tokyo \
  --receipt /absolute/path/to/registration-receipt.json
```

The default database is `<codex-home>/sqlite/codex-dev.db`; override it with `--database` only after locating the app's real database. The helper never opens the live database with SQLite. It copies the database and WAL to a temporary directory, checks that their file identities/sizes/mtime/ctime were stable during the copy, reads that copy, and runs `PRAGMA quick_check`. This avoids the cross-OS SQLite `disk I/O error` seen when WSL accessed the live Windows database.

Acceptance requires matching name, prompt, heartbeat kind, thread ID, recurrence and status. An active task must have a positive `next_run_at`; a paused task must have no next run. The receipt includes both nominal/actual timestamps when the app exposes them. A native ID migrated from the local ID can be found via `legacy_automation_id`, but the helper still requires the local config as the intended-state reference. If migration removed the local file, use the native management tool/UI instead.

Verification exits nonzero if the app has not accepted the file, the database snapshot is unstable/corrupt/incompatible, or the native entry differs. A short bounded wait is allowed for startup/import; do not poll indefinitely or insert/update rows in the live database. Account migration can cause this local-file route to be ignored, even when its TOML is valid.

This proves **registration**, not execution of the first scheduled turn. In the original task the app's receipt showed nominal `2026-10-07T11:00:37+09:00`, actual `11:05:17+09:00`. The observed implementation can add up to ±300 seconds of deterministic jitter to qualifying schedules. Preserve the requested recurrence and report the actual timestamp rather than promising execution exactly at the nominal time.

## 6. Stop when complete

Use the native tool when available. For a task still managed by the local file:

```bash
.venv/bin/python "$TASK_SKILL_DIR/scripts/heartbeat.py" set-status \
  --id training-followup --status PAUSED
.venv/bin/python "$TASK_SKILL_DIR/scripts/heartbeat.py" verify \
  --id training-followup --wait-seconds 20
```

Only `status` and `updated_at` are changed; other TOML content is retained. Verify the app's pause before claiming it stopped. Resuming uses the same command with `--status ACTIVE`, when the user authorizes resumption.

## Evidence and maintenance

The original live task is `blcs-physics-v3-retrain-hourly`. Its `automation.toml`, `memory.md`, and `registration-receipt.json` are under the observed Codex home. They are examples of an actual run; use new task-specific values when reproducing.

The installed app's `.vite/build/bootstrap-C8gUBg5L.js` supplied the format/import/next-run behavior. The app was allowed to import the TOML itself; no SQLite rows or app binaries were modified. A native database row was inspected after the GUI started. This reference records those observations so ordinary reuse does not require unpacking the application again.

[Official documentation](https://learn.chatgpt.com/docs/automations?surface=app) supports scheduled continuation in the same chat and the requirement that local tasks keep the computer/app running. It does not document this file-write fallback. If verification fails after an app upgrade, use the native tool/UI or inspect the changed implementation before updating this procedure.

Test the helper without registering any real tasks:

```bash
.venv/bin/python "$TASK_SKILL_DIR/scripts/test_heartbeat.py"
```
