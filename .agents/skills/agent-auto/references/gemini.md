# Gemini CLI

Use `scripts/gemini-auto.sh` for one non-interactive Gemini CLI run.

## Commands

The default uses YOLO approval mode so the headless run does not pause for tool
confirmation:

```bash
.agents/skills/agent-auto/scripts/gemini-auto.sh \
  "Run the test suite and fix any failures"
```

Select another approval mode, model, sandbox, or live JSONL output when needed:

```bash
.agents/skills/agent-auto/scripts/gemini-auto.sh \
  --mode plan \
  --model pro \
  "Review the repository and produce an implementation plan"

.agents/skills/agent-auto/scripts/gemini-auto.sh \
  --sandbox \
  --stream \
  --name fix-tests \
  "Investigate and fix the failing tests"
```

Read a prompt from a file, select a working directory, or inspect the rendered
command without starting Gemini:

```bash
.agents/skills/agent-auto/scripts/gemini-auto.sh \
  --file task.md \
  --dir /path/to/repo \
  --dry-run
```

## Success detection

Buffered JSON output contains `response`, `stats`, and an optional `error`.
The wrapper requires a zero process exit status and no structured error.
Gemini can emit setup failures as structured JSON on stderr; the wrapper also
parses that channel so `session_id` and `error_code` remain available.

Streaming JSONL ends with a `result` event. The wrapper requires a zero process
exit status and `status: success`, and concatenates assistant message chunks
into `result.txt`.

Documented Gemini CLI headless exit codes include zero for success, one for a
general or API failure, 42 for invalid input, and 53 for a turn-limit failure.
Other failures can use dedicated codes; for example, Gemini CLI 0.47.0 emitted
41 when no authentication method was configured.

## Permissions

- `yolo` auto-approves all tools and is the wrapper default.
- `auto_edit` auto-approves edit tools only.
- `plan` is read-only.
- Gemini enables sandboxing by default with YOLO mode; use `--sandbox` to request
  it explicitly.
- The wrapper passes session-only workspace trust by default to avoid an
  unattended trust prompt.

The wrapper uses existing Gemini CLI authentication.

## Verified interface

The command shape and output contracts were verified against Gemini CLI 0.47.0
and the Gemini CLI documentation updated March 10, 2026.

Sources:

- https://geminicli.com/docs/cli/headless/
- https://geminicli.com/docs/cli/cli-reference/
- https://geminicli.com/docs/reference/configuration/
