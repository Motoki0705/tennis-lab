---
name: codex-scheduled-followup
description: Schedule recurring follow-ups into an open Codex CLI session through codex queue, so the existing runtime continues the work. Honor the requested cadence and verify receipt as well as registration. Use for Codex CLI heartbeats and follow-ups, not generic OS scheduling, desktop automations, or merely running a long command.
---

# Codex Scheduled Follow-up

Deliver periodic work to an already-open Codex CLI session. A user systemd timer is the clock, `codex queue` is the delivery path, and the existing runtime consumes the message when its thread is idle. Do not substitute `codex exec resume`, a new agent, or a desktop automation. If this route is unavailable on the host, report the specific limitation instead of switching products. A scheduled prompt describes authorized work; it does not grant additional permissions.

Commands, prerequisites, delivery phases and recovery: [CLI queue heartbeats](references/cli-heartbeat.md).

## Preserve the user's schedule

Use the interval specified by the user. Only when no cadence was given, default to 60 minutes and state that choice. The helper supports fixed minute intervals only: if the user asked for calendar times, days or a timezone, report that limitation; do not round it into an interval.

## Target the owning session

Reuse an existing task for the same purpose and target; resume it if it is paused. Use an explicitly supplied, verified thread ID, otherwise the owning session's `CODEX_THREAD_ID`. A delegated operator must receive that ID, the original cwd, the matching rollout path and the runtime environment from its parent, not substitute its own. Never select the newest rollout. Keep model/effort unchanged unless requested. Never edit native state or relax security settings to force delivery.

## Make the continuation useful

Keep durable instructions in the prompt or an explicitly referenced `memory.md`. Include the actual worktree, host/shell/interpreter, process/job identifiers, output locations, already-completed checks, remaining authorized actions, and a clear completion condition. Use absolute paths valid on the execution host and reuse stable identifiers across runs. Separate setup-only restrictions from instructions for future monitoring turns. Do not embed a generic instruction to keep working forever or create new work when none remains.

For monitoring, define three branches:

- Still running or unchanged: inspect once and stay quiet unless the user explicitly requested periodic status reports. Do not start another polling loop or duplicate the job. Notify only on a meaningful change, completion, failure, or required user action.
- Finished: validate the result, continue the already-authorized remaining work, and retain evidence of partial completion so retries can resume.
- Failed: inspect the failure; repair within scope where possible and report a real blocker if one remains. Never promote failure to completion. For delivery failures, distinguish the active timer from `delivery_health` and use the evidence-preserving `recover` route for an authorized restoration; never silently reset or automatically resend an uncertain delivery.

Monitoring alone does not authorize new training, deployment, or messages to other people.

## Verify and report

Distinguish these observable stages:

- **Scheduled:** task identity, target, interval, active timer and its reported next run.
- **Accepted:** the queue accepted the delivery. This does not prove that the receiving runtime ran it.
- **Received and completed:** the target rollout contains the unique delivery marker as user input and the associated turn completed. Check the receiving side, not a sender's successful exit; then assess the response/artifacts to decide whether the work itself succeeded.

For a live test aimed at your **own active turn**, the message waits until that turn ends. Save a handoff with the task directory and verification command, yield once, then finish verification from the queued continuation. Do not keep the sending turn busy polling for its own queued successor, or claim receipt before it happens. Run a live test only when the user explicitly requests one; keep it bounded and identifiable and clean up its timer. Otherwise use the helper's unit tests, which mock all delivery.

Delivery requires the computer and the receiving CLI runtime to keep running; a user interruption can suspend automatic queue consumption, and the transient timer does not survive a reboot. Do not present this route as waking a powered-off computer or guaranteeing exactly-once task completion.

## Stop

When the requested outcome is complete, pause the task and verify that there is no next scheduled run. To change the interval or prompt later, pause and `resume` the same task rather than creating a new one. Pausing does not retract a message already queued; report that separately. Do not archive the chat unless requested.
