# CLI queue heartbeats

Operating reference for `scripts/cli_heartbeat.py`. The decision rules live in [SKILL.md](../SKILL.md).

## Delivery contract

This route was investigated against `codex-cli 0.161.0`; check `codex queue --help` on the actual host. `codex queue` submits to the native persistent thread queue. The existing local runtime polls it (10 seconds in 0.161.0) and consumes a message when the target thread is idle. It does not interrupt an active turn. Closing an in-process CLI leaves queued messages pending until a suitable runtime resumes the thread. Implementation references for the observed version: [submission](https://github.com/openai/codex/blob/979011409de0a60b52f179721948e65531d26144/codex-rs/tui/src/session_queue_commands.rs), [consumer](https://github.com/openai/codex/blob/979011409de0a60b52f179721948e65531d26144/codex-rs/ext/queue/src/service.rs), [cross-runtime tests](https://github.com/openai/codex/blob/979011409de0a60b52f179721948e65531d26144/codex-rs/ext/queue/tests/queue_service.rs).

The receiving session's **rollout** is its append-only JSONL log under `$CODEX_HOME/sessions/`. Its first record is `session_meta` (thread ID and cwd); later records are user input, responses, tool calls and turn events. The helper reads it to prove receipt and completion.

The helper calls `codex` with an argument array (never a shell) and with the `CODEX_HOME`, `CODEX_SQLITE_HOME` and `PATH` captured at creation. In Windows/WSL setups these can deliberately select different state databases even when conversation history is shared, so forward the current values and do not change the global shell configuration. It passes `--disable daemon_auto_start`, does not edit native databases, and does not start another model session.

## Prepare

Requirements: Linux/WSL, Python 3.11+, a user systemd manager (`systemctl --user is-system-running`), and an existing JSONL rollout. Unsupported setups are rejected rather than switched to another scheduler.

Resolve absolute values from the actual session. The rollout's `session_meta` must match the thread ID and cwd; the cwd belongs to the receiving session, not a delegated operator's worktree.

```bash
TASK_SKILL_DIR="/absolute/path/to/codex-scheduled-followup"
TASK_PY="/absolute/path/to/.venv/bin/python"
TASK_THREAD_ID="verified-owning-thread-uuid"
TASK_CWD="/absolute/path/to/owning/session/cwd"
TASK_ROLLOUT="/absolute/path/to/the/matching/rollout.jsonl"
TASK_PROMPT="/absolute/path/to/followup-prompt.txt"
TASK_ID="training-followup"
```

## Create

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" create \
  --id "$TASK_ID" --thread-id "$TASK_THREAD_ID" \
  --cwd "$TASK_CWD" --rollout "$TASK_ROLLOUT" \
  --prompt-file "$TASK_PROMPT" --interval-minutes 30
```

`--interval-minutes` defaults to 60 only when omitted. State lives under `~/.local/state/codex-followups/<id>/` (`--state-dir` selects another Linux directory; Windows `/mnt/<drive>` paths are rejected). An existing ID is never overwritten. `create` registers a transient user systemd timer and verifies its unit, both interval properties and next run; a saved configuration alone is not success. Use the returned `task_dir` in every later command.

## Inspect

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" status --task-dir "<task_dir>"
```

`registration_verified` reports the timer. `delivery_health` reports the latest delivery separately: `ready`, `waiting_for_runtime` (queued), `in_progress` (started), or `needs_recovery`, in which case continuations cannot advance even with an active timer. `execution_verified` is true only for a completed delivery.

For a user-authorized immediate delivery or bounded live test, `tick --task-dir "<task_dir>"` sends once. To exercise the clock too, a separate one-shot user timer with its own unit name may invoke the same `tick`; clean up both afterwards. A successful one-shot does not prove the recurring interval has fired.

## Delivery phases

Each `tick` holds a per-task lock, reconciles the rollout, and sends at most one outstanding delivery with a unique marker `[codex-heartbeat:<id>:<uuid>]`. The delivery is saved as `uncertain` **before** `codex queue` runs, so a crash cannot cause a silent resend.

| Phase | Meaning | Next tick sends? |
|---|---|---|
| `uncertain` | No unambiguous queue acknowledgement (timeout, non-zero exit, crash) | No: inspect, then `recover` |
| `queued` | `Queued message <id> for thread <target>.` was printed exactly once | No |
| `started` | The marker appeared as user input in a turn after the delivery boundary | No |
| `completed` | That turn emitted `task_complete` | Yes |
| `failed` | That turn was aborted | No: `recover` |
| `superseded` | An operator recovered it; the old record is kept | Yes |

Queue submission has its own timeout (`create --queue-timeout-seconds`, default 180); an acknowledgement printed before a timeout still counts as accepted. Assistant echoes and tool output containing the marker never count as receipt.

Each delivery records a rollout anchor: byte offset, file identity (device/inode) and a SHA-256 of the 64 KiB before the offset. A replaced or truncated rollout, or a rewrite inside that window, sets `blocked_reason` and stops delivery. Codex only appends to rollouts, so older history outside the window is deliberately not rehashed; each tick reads only the new records. Migration is a limitation to report, not permission to edit native state. State files from the earlier full-prefix helper (version 1) are verified once against their old hashes and upgraded automatically.

## Pause and resume

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" pause --task-dir "<task_dir>"
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" resume --task-dir "<task_dir>" \
  [--interval-minutes 15] [--prompt-file "/absolute/path/to/new-prompt.txt"]
```

`pause` stops the timer and verifies it is inactive with no next run. Receipts are preserved. A previously accepted message can still be consumed: pausing the clock is not queue cancellation, so do not delete queue rows or interrupt the user's work to hide it.

`resume` re-registers a paused task's timer and verifies it like `create`. Omitted options keep the current interval and prompt; to change either on a running task, pause first. It refuses a task that is not paused, has a `blocked_reason`, or whose latest delivery is `uncertain`/`failed` (recover it first). A queued or started delivery may remain; the next tick still waits for it. If registration does not verify, the task stays paused with its previous settings; run `pause` to clear any partially registered timer. Do not bypass an unresolved delivery by creating another task for the same work.

## Recover without discarding evidence

The CLI has no caller-controlled idempotency key, so an unacknowledged delivery is never resent automatically. Inspect the exact marker in the target rollout and, when available, the same runtime's pending queue read-only. No pending item does not prove that the first submission was rejected.

For a user-authorized restoration where the remaining duplicate risk is accepted, recover the **existing task**. An explicit user request to repair and restore this monitoring authorizes this operation; otherwise explain the remaining uncertainty first.

```bash
"$TASK_PY" "$TASK_SKILL_DIR/scripts/cli_heartbeat.py" recover \
  --task-dir "<task_dir>" \
  --delivery-marker "[codex-heartbeat:task-id:exact-uuid]" \
  --reason "User requested repair and restoration; target queue and history inspected" \
  --acknowledge-possible-duplicate
```

`recover` reconciles late receipts first and refuses queued/started deliveries, mismatched markers or a blocked history. It marks the attempt `superseded` with the reason and keeps its logs. It does not delete a queue item, send a message, or change the timer. The next tick sends once with a new marker. Make the continuation check durable job identities so a late old message cannot start duplicate work. Do not recover/resend an unresolved failure repeatedly.

## Tests

`"$TASK_PY" "$TASK_SKILL_DIR/scripts/test_cli_heartbeat.py"` uses temporary rollouts and mocked commands; `tests/e2e/skills/test_codex_scheduled_followup.py` runs it under pytest. Neither touches a live queue or timer.
