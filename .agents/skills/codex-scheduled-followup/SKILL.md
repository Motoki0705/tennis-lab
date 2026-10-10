---
name: codex-scheduled-followup
description: Schedule recurring follow-ups in the intended Codex session. For an open local CLI, deliver through codex queue so the existing runtime continues; for desktop chats, use native automations. Honor the requested cadence and verify receipt as well as registration. Use for Codex heartbeats and follow-ups, not generic OS scheduling, Sites schedules, or merely running a long command.
---

# Codex Scheduled Follow-up

Deliver periodic work to the intended conversation **and runtime**. Sharing a thread ID or history file does not by itself make a desktop run appear in an already-open CLI. A scheduled prompt describes authorized work; it does not grant additional permissions.

## Preserve the user's schedule

Use the interval, calendar times, days and timezone specified by the user. Only when no cadence was given, default to once per hour and state that choice. The one-hour default must not override an explicit request or replace a requested schedule that the current tool cannot represent. If a specified schedule is ambiguous or unsupported, clarify the missing detail or report the limitation; do not silently round it to hours or discard its timezone.

## Choose the receiving runtime

1. **The user wants this open CLI session to continue:** read [CLI queue heartbeats](references/cli-heartbeat.md). Use the existing runtime's `codex queue` delivery path and environment. The provided Linux/WSL helper uses a user systemd timer as its clock; this is not a desktop Scheduled entry. Do not substitute `codex exec resume`, a new agent, or a desktop heartbeat for same-runtime delivery. If the CLI route is unavailable, report the specific limitation before changing the receiving product.
2. **The user wants a desktop chat or explicitly requests desktop Scheduled:** search once for native `automation_update` unless availability is already known. Create/update with its schema and view the result. If unavailable, read [the desktop file procedure](references/local-heartbeat.md). That fallback is version-dependent and requires app registration verification; a file alone proves no execution.

Reuse an existing task for the same purpose and target. Use an explicitly supplied, verified thread ID, otherwise the owning chat's `CODEX_THREAD_ID`. A delegated operator must receive that owning ID and runtime environment explicitly, not substitute its own. Match the rollout's session ID; do not select the newest rollout. Keep model/effort unchanged unless requested. Never edit native SQLite rows or relax security settings to force delivery.

## Make the continuation useful

Keep durable instructions in the prompt or an explicitly referenced `memory.md`. Include the actual worktree, host/shell/interpreter, process/job identifiers, output locations, already-completed checks, remaining authorized actions, and a clear completion condition. Use absolute paths valid on the execution host and reuse stable identifiers across runs. Separate setup-only restrictions from instructions for future monitoring turns.

For monitoring, define three branches:

- Still running or unchanged: inspect once and stay quiet unless the user explicitly requested periodic status reports. Do not start another polling loop or duplicate the job. Notify only on a meaningful change, completion, failure, or required user action.
- Finished: validate the result, continue the already-authorized remaining work, and retain evidence of partial completion so retries can resume.
- Failed: inspect the failure; repair within scope where possible and report a real blocker if one remains. Never promote failure to completion. For CLI delivery failures, distinguish the active timer from `delivery_health` and use the documented evidence-preserving `recover` route for an authorized restoration; never silently reset or automatically resend an uncertain delivery.

When the requested outcome is complete, stop the chosen route: pause the CLI helper's timer, delete a native desktop automation, or pause a file-managed desktop task. Verify that there is no next scheduled run. Pausing a CLI timer does not retract a message already queued; report that separately. Do not archive the chat unless requested. Monitoring alone does not authorize new training, deployment, or messages to other people.

## Verify and report

Distinguish these observable stages:

- **Scheduled:** task identity, target, interval, active timer/automation and its reported next run.
- **Accepted:** the queue/automation accepted a delivery. This does not prove that the receiving runtime ran it.
- **Received and completed:** the intended thread received the unique delivery marker and completed the associated turn. Check the receiving context and output, not just a sender's successful exit.

For a live test aimed at your **own active turn**, the message normally waits until that turn ends. Save a handoff with the task directory and verification command, yield once, then finish verification from the queued continuation. Do not keep the sending turn busy polling for its own queued successor, or claim receipt before it happens. If the user explicitly requests a live test, use a bounded, identifiable test and clean up its timer; otherwise use temporary state and mocked delivery for script tests.

Local CLI delivery requires the computer and receiving CLI runtime to remain running. A user interruption can suspend automatic queue consumption. Desktop scheduling instead requires the desktop app. See [official desktop scheduling scope](https://learn.chatgpt.com/docs/automations?surface=app). Do not present either route as waking a powered-off computer or guaranteeing exactly-once task completion.
