# Antigravity CLI (agy)

Use `scripts/agy-auto.sh` for one non-interactive Antigravity (agy) run.

## Commands

The default uses YOLO approval mode so the headless run does not pause for tool
confirmation:

```bash
.agents/skills/agent-auto/scripts/agy-auto.sh \
  "Run the test suite and fix any failures"
```

Select another approval mode, model, sandbox, or a longer print timeout for
long-running tasks:

```bash
.agents/skills/agent-auto/scripts/agy-auto.sh \
  --mode plan \
  --model pro \
  "Review the repository and produce an implementation plan"

.agents/skills/agent-auto/scripts/agy-auto.sh \
  --sandbox \
  --print-timeout 60m \
  --name fix-tests \
  "Investigate and fix the failing tests"
```

Read a prompt from a file, select a working directory, or inspect the rendered
command without starting agy:

```bash
.agents/skills/agent-auto/scripts/agy-auto.sh \
  --file task.md \
  --dir /path/to/repo \
  --dry-run
```

## Success detection

The wrapper reads the raw text output from agy. agy does not emit structured
JSON, so `result.txt` receives the plain output.

Caveat (agy v1.0.11): a zero exit status is **not** a reliable success signal.
agy `--print` returns rc 0 even when the run fails — on timeout it prints
`Error: timed out waiting for response` and on a bad invocation
`Error: empty prompt ...`, both with rc 0. The wrapper therefore also rejects a
stdout that starts with `Error: ` and marks a timeout as
`provider_status=timed_out` so the run is correctly reported as `status=failed`
(wrapper exit 1).

Streaming is not natively supported by `agy --print`.

## Timeouts (important for long runs)

`agy --print` has its own `--print-timeout`, whose **default is only 5m**. Any
autonomous task that runs longer is aborted at that point. The wrapper raises
the default to `30m` and exposes `-t/--print-timeout` so long batch/cron runs
can extend it (for example `--print-timeout 60m`).

## Permissions

- `yolo` maps to `--dangerously-skip-permissions` to auto-approve all tools.
- `auto_edit` also maps to `--dangerously-skip-permissions`.
- `plan` is read-only (doesn't pass the skip flag).
- agy supports `--sandbox` for terminal restrictions.

The wrapper uses existing agy configuration.

## Verified interface

The command shape and output contracts were verified against agy CLI.
