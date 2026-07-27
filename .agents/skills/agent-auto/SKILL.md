---
name: agent-auto
description: Run Claude Code, Codex CLI, or Antigravity CLI non-interactively to carry a task through one autonomous headless invocation. Use for unattended coding-agent runs, CI jobs, cron jobs, or batch work that needs provider-specific permissions, structured logging, and machine-checkable success detection.
---

# Autonomous coding agents

Run one headless coding-agent invocation and let that CLI's native agent loop
work until it exits. Do not add an outer resume loop.

## Workflow

1. Select the requested provider.
2. Read only the matching provider reference:
   - [Claude Code](references/claude.md)
   - [Codex CLI](references/codex.md)
   - [Antigravity CLI](references/agy.md)
   - [Hermes Agent](references/hermes.md)
3. Confirm that the provider CLI is installed and authenticated.
4. Run the matching wrapper under `scripts/`.
5. Treat the wrapper exit status as the machine-readable outcome and inspect
   its run directory when diagnosing a failure.

## Safety

- Prefer the least-privileged provider mode that can complete the task.
- Use a sandbox, container, or disposable worktree for unattended write access.
- Scope the working directory to the target repository.
- Keep production credentials and unrelated writable directories out of scope.
- Reserve unrestricted modes for externally isolated environments.

## Outputs

Each wrapper writes a timestamped directory below `logs/agent-auto/<provider>/`.
The directory contains the prompt, rendered command, raw structured output,
final response, stderr when buffered, and a normalized summary.

Exit status `0` means the provider emitted its documented success signal.
Exit status `1` means the provider run or structured result failed. Exit status
`2` means wrapper usage or setup failed.
