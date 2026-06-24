# Antigravity CLI (agy)

Use `scripts/agy-auto.sh` for one non-interactive Antigravity (agy) run.

## Commands

The default uses YOLO approval mode so the headless run does not pause for tool
confirmation:

```bash
.agents/skills/agent-auto/scripts/agy-auto.sh \
  "Run the test suite and fix any failures"
```

Select another approval mode, model, sandbox, or live JSONL output when needed:

```bash
.agents/skills/agent-auto/scripts/agy-auto.sh \
  --mode plan \
  --model pro \
  "Review the repository and produce an implementation plan"

.agents/skills/agent-auto/scripts/agy-auto.sh \
  --sandbox \
  --name fix-tests \
  "Investigate and fix the failing tests"
```

Read a prompt from a file, select a working directory, or inspect the rendered
command without starting Gemini:

```bash
.agents/skills/agent-auto/scripts/agy-auto.sh \
  --file task.md \
  --dir /path/to/repo \
  --dry-run
```

## Success detection

The wrapper reads the raw text output from agy and considers a zero exit status as a success.
agy does not currently emit structured JSON, so `result.txt` receives the plain output.

Streaming is not natively supported by `agy --print`.
Documented exit codes include zero for success and non-zero for failures.

## Permissions

- `yolo` maps to `--dangerously-skip-permissions` to auto-approve all tools.
- `auto_edit` also maps to `--dangerously-skip-permissions`.
- `plan` is read-only (doesn't pass the skip flag).
- agy supports `--sandbox` for terminal restrictions.

The wrapper uses existing agy configuration.

## Verified interface

The command shape and output contracts were verified against agy CLI.
