# Codex CLI

Use `scripts/codex-auto.sh` for one non-interactive Codex run.

## Commands

The default uses a writable workspace sandbox and disables approval prompts:

```bash
.agents/skills/agent-auto/scripts/codex-auto.sh \
  "Run the test suite and fix any failures"
```

Use a read-only sandbox for analysis or unrestricted mode only inside an
external sandbox:

```bash
.agents/skills/agent-auto/scripts/codex-auto.sh \
  --sandbox read-only \
  "Review the repository and report risks"

.agents/skills/agent-auto/scripts/codex-auto.sh \
  --dangerous \
  "Complete the requested migration and verify it"
```

Pass provider config overrides repeatedly, avoid session persistence, stream
events, or inspect the rendered command:

```bash
.agents/skills/agent-auto/scripts/codex-auto.sh \
  --config 'model_reasoning_effort="high"' \
  --ephemeral \
  --stream \
  --name migrate-api \
  "Migrate the deprecated API and run tests"

.agents/skills/agent-auto/scripts/codex-auto.sh \
  --file task.md \
  --dir /path/to/repo \
  --dry-run
```

## Success detection

Codex emits JSONL events in non-interactive JSON mode. The wrapper requires all
of the following:

- the Codex process exits zero;
- a `turn.completed` event is present;
- no `turn.failed` or `error` event is present.

The final completed `agent_message` is written to `result.txt`, and the
`thread_id` is written to `summary.txt`.

## Permissions

- `workspace-write` is the wrapper default.
- `read-only` is appropriate for inspection-only tasks.
- `danger-full-access` removes filesystem sandboxing but is distinct from the
  explicit dangerous bypass flag.
- The wrapper sets `approval_policy="never"` for sandboxed runs so unattended
  executions fail within their boundary instead of waiting for user input.
- `--dangerous` bypasses both approvals and sandboxing. Use it only in an
  externally isolated runner.

The wrapper reuses existing Codex authentication.

## Verified interface

Verified locally with codex-cli 0.138.0.

Sources:

- https://developers.openai.com/codex/noninteractive
- https://developers.openai.com/codex/cli/reference#codex-exec
