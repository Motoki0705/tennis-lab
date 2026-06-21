---
name: claude-auto
description: Use this skill when running Claude Code itself non-interactively (headless `claude -p`) to carry a task through to completion without approval prompts. Covers the verified flags, the wrapper scripts under scripts/, success detection, and the safety rails for unattended runs.
---

# Autonomous Claude Code (no-approval, run-to-completion)

## Scope

Use this skill when you want Claude Code to **run a task end-to-end without
stopping for permission prompts** — CI jobs, cron jobs, batch refactors, or an
unattended "fix it until it's done" run. It documents the headless `claude -p`
workflow and ships two wrapper scripts that add logging and a machine-checkable
result.

This is for invoking the `claude` CLI from a shell. It is **not** about the
interactive `/`-skills inside a session.

Verified against **Claude Code 2.1.185**.

## The core command

A single `claude -p` call already runs the **full agent loop to completion** —
it keeps acting until the task is done, then prints the final result and exits.
To make it run without any approval prompts:

```bash
claude -p "Run the test suite and fix any failures" \
  --permission-mode bypassPermissions \
  --output-format json
```

`--permission-mode bypassPermissions` and `--dangerously-skip-permissions` are
equivalent — both skip every approval.

## Key flags

| Flag | Purpose |
| --- | --- |
| `-p, --print` | Headless mode; runs the agent loop and exits. Required for all of the below. |
| `--permission-mode bypassPermissions` | No approval prompts at all (full autonomy). |
| `--dangerously-skip-permissions` | Same effect; the explicit "I know" form. |
| `--permission-mode dontAsk` | Denies anything **not** in `--allowedTools` / `permissions.allow` (locked-down CI). |
| `--permission-mode acceptEdits` | Auto-approves file writes + common fs commands; other shell/network still need an allow rule. |
| `--allowedTools "Bash(git *),Read,Edit"` | Scope tools (space before `*` matters: `Bash(git diff *)` ≠ `Bash(git diff*)`). |
| `--output-format json` | One JSON object with `result`, `is_error`, `num_turns`, `session_id`. |
| `--output-format stream-json` `--verbose` | Live newline-delimited events; last `type:result` line is the outcome. |
| `--model opus\|sonnet\|haiku` | Pick the model. |
| `--resume <session_id>` | Continue a prior headless session (drives the loop wrapper). |

## Detecting success — do NOT trust the exit code

`claude` exits **0 even when the run failed** (`is_error:true`). Always read the
JSON `is_error` field:

```bash
out=$(claude -p "…" --permission-mode bypassPermissions --output-format json)
echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(1 if d["is_error"] else 0)'
```

The wrapper scripts do exactly this and re-derive a correct exit code (0 ok, 1 failed).

## Wrapper scripts (reproducibility)

Both live in `scripts/` next to this file, are dependency-free (bash + python3),
and write a timestamped run log under `logs/claude-auto/` (gitignored).

### `claude-auto.sh` — one autonomous run

```bash
# simplest: hand it a task, it runs to completion with bypass mode
.agents/skills/claude-auto/scripts/claude-auto.sh "Run the test suite and fix any failures"

# scoped + locked-down: restricted tools, a named run, live output
.agents/skills/claude-auto/scripts/claude-auto.sh \
  -m dontAsk -a "Read,Bash(pytest *)" --model sonnet --stream \
  --name fix-flaky "Investigate and fix the flaky test in tests/test_foo.py"

# from a prompt file, against another directory; dry-run prints the command only
.agents/skills/claude-auto/scripts/claude-auto.sh -f task.md -d /path/to/repo --dry-run
```

Each run dir holds `prompt.txt`, `command.txt`, `output.json` (or `stream.jsonl`),
`result.txt`, and `summary.txt`. Exit code is 0 only when `is_error=false`.

### `claude-auto-loop.sh` — keep going until truly done

A single `-p` call can stop early on a very large task. This wrapper resumes the
same session and re-prompts it to continue until it prints a completion
**sentinel**, bounded by `--max-iters`.

```bash
.agents/skills/claude-auto/scripts/claude-auto-loop.sh \
  -i 8 -m bypassPermissions \
  "Migrate every call site off the deprecated api(); run the tests after each batch."
```

Exit 0 only when the sentinel (`TASK_COMPLETE` by default) is observed; exit 1 on
the iteration cap or a turn error.

## Safety rails (read before unattended use)

Bypassing permissions removes every guardrail, so constrain the blast radius:

- **Prefer the least-privileged mode that works.** `dontAsk` + an explicit
  `--allowedTools` allowlist is far safer than full `bypassPermissions`; reserve
  bypass for sandboxes / disposable environments.
- **Run in a sandbox or throwaway worktree** with no production credentials and,
  ideally, no outbound internet — Anthropic's own guidance recommends bypass mode
  "only for sandboxes with no internet access."
- **Scope the working directory** with `-d` so the run can't wander out of the repo.
- For a middle ground between manual review and full bypass, consider Claude
  Code's **auto mode** (`--permission-mode auto`); inspect its allow/deny rules
  with `claude auto-mode defaults`.

## Auth & environment notes

- These scripts use your normal (OAuth / subscription) login. **Do not add
  `--bare`** unless you provide `ANTHROPIC_API_KEY` — bare mode skips OAuth and
  keychain and will fail with `Not logged in`.
- In a git **worktree** there is usually no local `.venv`; the scripts fall back
  to `python3` automatically (they try `.venv/bin/python` first).
- `--bare` is otherwise recommended for CI for reproducible context, but only with
  an API key.

## Sources

- Run Claude Code programmatically (headless): https://code.claude.com/docs/en/headless
- How we built Claude Code auto mode: https://www.anthropic.com/engineering/claude-code-auto-mode
- Verified locally with `claude --help` and live `claude -p` runs on v2.1.185.
