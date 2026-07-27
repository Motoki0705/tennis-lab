# Hermes Agent

Use `scripts/hermes-auto.sh` for one non-interactive Hermes Agent run.

## Commands

```bash
.agents/skills/agent-auto/scripts/hermes-auto.sh \
  --dir /path/to/repo \
  "Review the repository and report risks"
```

Continue a specific previous run only by passing its session ID from the prior
run's `summary.txt`:

```bash
.agents/skills/agent-auto/scripts/hermes-auto.sh \
  --resume 20260727_181249_0d18c2 \
  "Continue the previous task and run the tests"
```

`--continue` can resume the latest session or a named Hermes session, but
`--resume SESSION_ID` is preferred for reproducible automation. `--resume` and
`--continue` cannot be combined.

## Success detection

The wrapper captures Hermes stdout and stderr independently. It requires all of
the following:

- Hermes exits with status 0;
- stderr contains a machine-readable `session_id: ...` line;
- stdout contains a non-empty final response;
- no terminal reasoning display block is left unterminated.

The cleaned response is written to `result.txt`. `summary.txt` records
`status`, `session_id`, whether the run was resumed, the Hermes exit code, and a
failure reason when applicable. The raw stdout and stderr remain in the run
directory for diagnosis.

## Options and safety

- The wrapper uses `hermes chat --query ... --quiet --pass-session-id`.
- Hermes `--yolo` is enabled by default so a headless coding task does not stop
  at a dangerous-command approval prompt. Use `--no-yolo` when the task must
  fail instead of receiving unrestricted command approval.
- `--worktree` asks Hermes to isolate its changes in a git worktree. Prefer it
  for unattended write tasks when the caller does not need the current checkout
  modified directly.
- `--ignore-user-config` is enabled by default to make display and behavioral
  output reproducible; credentials from Hermes' `.env` remain available.
  Pass `--use-user-config` when an explicitly configured Hermes model/provider
  is required.
- Repository `AGENTS.md` and rules are loaded by default. Pass `--ignore-rules`
  only for an intentionally isolated run.

This wrapper performs one native Hermes invocation. It never implements an
outer retry or resume loop; a follow-up must explicitly supply `--resume` or
`--continue`.

## Verified interface

The command shape was verified against Hermes Agent v0.15.1. The initial run
and a second run using the emitted session ID were both exercised against the
local CLI. The separate `hermes-delegation` skill remains the appropriate path
when Codex should automatically retain a conversation per Codex thread and
relay the answer verbatim.
