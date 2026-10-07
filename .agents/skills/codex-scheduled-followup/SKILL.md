---
name: codex-scheduled-followup
description: Schedule recurring follow-ups in an existing Codex thread at the user's requested cadence, defaulting to one hour only when unspecified. Monitor a long process and continue already-authorized work afterward. Prefer the native Codex automation tool; otherwise use the verified local heartbeat-file workflow and confirm app registration. Use for Codex scheduling requests, not OS cron, Sites schedules, or merely running a long command.
---

# Codex Scheduled Follow-up

Create a recurring return to the intended thread and verify that the scheduler accepted it. A scheduled prompt describes the work; it does not grant permission for additional work.

## Preserve the user's schedule

Use the interval, calendar times, days and timezone specified by the user. Only when no cadence was given, default to once per hour and state that choice. The one-hour default must not override an explicit request or replace a requested schedule that the current tool cannot represent. If a specified schedule is ambiguous or unsupported, clarify the missing detail or report the limitation; do not silently round it to hours or discard its timezone.

## Choose the route

1. Discover the actual tools exposed in this session. If a native Codex `automation_update` tool is available, use its provided schema and create/update the task there. Use an existing task when it already covers this thread and purpose. A Sites schedule or a shell cron job is a different product.
2. If the native tool is unavailable and this is a local desktop follow-up, read [the local file procedure](references/local-heartbeat.md). This fallback was verified on Windows/WSL with app `26.930.7945.0`; it is not a documented public API and may not be accepted by other versions or account migration states.
3. If neither route produces a confirmed scheduler entry, report the saved configuration and the registration problem separately. Do not claim that a file's existence guarantees execution. Do not write the native SQLite database or change security settings to force acceptance.

Use the current thread ID from `CODEX_THREAD_ID`, or an explicitly supplied and verified target ID. Do not infer it from the newest rollout file. Keep the task's model/effort unspecified unless the user requested a change.

## Make the continuation useful

Keep durable instructions in the prompt or an explicitly referenced `memory.md`. Include the actual worktree, process/job identifiers, output locations, already-completed checks, remaining authorized actions, and a clear completion condition. Reuse stable identifiers across runs.

For monitoring, define three branches:

- Still running: inspect once, report only useful progress, and finish this scheduled turn. Do not start another polling loop or duplicate the job.
- Finished: validate the result, continue the already-authorized remaining work, and retain evidence of partial completion so retries can resume.
- Failed: inspect the failure; repair within scope where possible and report a real blocker if one remains. Never promote failure to completion.

When the requested outcome is complete, pause this task using the native tool, or the file helper's `set-status` command when the task is still file-managed. Verify the pause. A monitoring request alone does not authorize new training, deployment, messages to others, or destructive cleanup.

## Verify and report

Confirm the task identity, intended thread, recurrence, status, and actual next run. The desktop app may apply jitter; report the scheduler's timestamp, not a calculated promise of execution exactly on the hour. Registration and a successful future run are separate facts.

For local tasks, the computer and desktop app must remain running. The CLI alone does not provide the Scheduled management interface. See [official scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app).

The bundled Python 3.11+ helper accepts an explicit minute interval, hour interval or RFC 5545 recurrence; omitting all three defaults to one hour. It creates a heartbeat configuration without overwriting a different task, changes its status, and checks a stable read-only snapshot of the app database. Command examples and verification failures are documented in the linked procedure. Run its tests against temporary directories; do not create test schedules in the real Codex home.
