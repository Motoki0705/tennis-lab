# Claude Code

Use `scripts/claude-auto.sh` for one non-interactive Claude Code run.

## Commands

```bash
.agents/skills/agent-auto/scripts/claude-auto.sh \
  "Run the test suite and fix any failures"
```

Use a restricted permission mode and explicit tool allowlist when full bypass is
not required:

```bash
.agents/skills/agent-auto/scripts/claude-auto.sh \
  --mode dontAsk \
  --allow "Read,Bash(pytest *)" \
  --model sonnet \
  --stream \
  --name fix-tests \
  "Investigate and fix the failing tests"
```

Read a prompt from a file, select a working directory, or inspect the rendered
command without starting Claude:

```bash
.agents/skills/agent-auto/scripts/claude-auto.sh \
  --file task.md \
  --dir /path/to/repo \
  --dry-run
```

## Success detection

Claude's structured result contains `is_error`, `result`, `num_turns`, and
`session_id`. The wrapper derives its exit status from `is_error`; it does not
trust the Claude process exit status alone.

Buffered runs store one JSON result. Streaming runs store JSONL and use the last
`type: result` event.

## Permissions

- `bypassPermissions` skips all approval checks and is the wrapper default.
- `dontAsk` denies tools that are not already allowed.
- `acceptEdits` auto-approves edits but can still restrict other operations.
- Use `--allow` and `--disallow` to scope tools.
- Use bypass mode only in an externally isolated environment.

The wrapper uses existing Claude authentication. Do not use bare mode with
OAuth credentials because bare mode skips keychain authentication.

## Verified interface

Verified locally with Claude Code 2.1.185.

Sources:

- https://code.claude.com/docs/en/headless
- https://code.claude.com/docs/en/cli-reference
