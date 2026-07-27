---
name: hermes-delegation
description: Delegate repository file and folder explanation requests to the Hermes CLI. Use when a user asks Codex to explain, investigate, summarize, or map files under a directory and the requested work should be performed by Hermes, with the final Hermes response relayed unchanged and follow-up requests kept in the same Hermes conversation.
---

# Hermes Delegation

Use Hermes as the repository-reading worker. Codex must turn the user's request
into a precise prompt, send it through the bundled adapter, and relay the
adapter's stdout as the answer. Do not read the target files yourself to write
an independent explanation or merge your own analysis into Hermes' response.

## Delegate a request

1. Identify the target path and make it absolute. Preserve the user's language,
   requested depth, audience, and output format.
2. Expand the request into a self-contained Hermes prompt. Include:
   - the absolute file or directory path;
   - the questions Hermes must answer (purpose, responsibilities, data flow,
     entry points, configuration, constraints, and relevant tests as
     applicable);
   - whether the task is read-only (use read-only unless the user explicitly
     requests a change);
   - the language and concise formatting expected in the final answer.
3. Invoke the adapter from the repository root. Use stdin for long prompts to
   avoid shell quoting and argument-length problems:

   ```bash
   prompt='...the decomposed Hermes prompt...'
   printf '%s' "$prompt" | .venv/bin/python .agents/skills/hermes-delegation/scripts/hermes_delegate.py --prompt-file -
   ```

   The adapter calls `hermes chat --query ... --quiet --pass-session-id` with
   `--ignore-user-config` by default, keeps Hermes' final response on stdout,
   and stores the session ID separately. Hermes still loads repository
   `AGENTS.md` and credentials from `.env`; use `--use-user-config` only when a
   user explicitly needs their personal Hermes display/model settings.
4. Return stdout verbatim. Do not add a preface, postscript, translation, or
   claims that Codex independently verified the answer. If the adapter exits
   non-zero, surface its stderr diagnostic and do not start a replacement
   session silently.

## Preserve the conversation

The adapter stores a small private JSON state file outside the repository. It
uses `CODEX_THREAD_ID` when available, so every Codex task gets an independent
state key. Without that environment variable it hashes the current working
directory; use `--session-key` or `--session-file` when running it from another
host that has multiple conversations in one directory.

For the first delegated request, invoke the adapter normally. For every
follow-up in the same user conversation, pass `--resume-required`:

```bash
printf '%s' "$prompt" | .venv/bin/python .agents/skills/hermes-delegation/scripts/hermes_delegate.py --prompt-file - --resume-required
```

This requires the saved session and prevents a missing state file from being
silently replaced by a new conversation. The adapter passes the saved ID to
`hermes chat --resume`; when Hermes creates a continuation ID during context
compression, it atomically replaces the saved ID with the latest one. Use
`--new-session` only when the user explicitly asks to reset the Hermes
conversation. A resume failure is an error to report, not a reason to fall back
to a fresh session.

## Hermes CLI contract

Use `hermes chat`, not `hermes send` (messaging delivery) and not
`hermes -z/--oneshot`: `-z` produces clean answer-only stdout but intentionally
does not emit a session ID, so it cannot support follow-ups. The relevant
arguments are:

- `--query` / `-q`: one non-interactive prompt;
- `--quiet` / `-Q`: suppress interactive presentation and keep the final text
  suitable for piping;
- `--resume SESSION_ID`: restore the existing conversation;
- `--pass-session-id`: make the ID available to Hermes when useful;
- `--source tool`: identify adapter-created sessions;
- `--ignore-user-config`: keep machine output independent of personal display
  settings (the adapter's default);
- optional `--model`, `--provider`, `--toolsets`, and `--max-turns` overrides.

The adapter captures stdout and stderr independently because Hermes writes the
final response to stdout and the machine-readable `session_id: ...` footer to
stderr. Never parse a session ID from the answer text.

## Prompt and response boundaries

Keep Codex's role as prompt manager only:

- Do not open, search, or summarize the target directory before delegation.
- Do include enough task detail for Hermes to choose the right files and tools.
- Tell Hermes not to modify files for explanation/investigation requests.
- On follow-ups, rely on Hermes' restored history instead of repeating the
  entire first prompt.
- Relay the response exactly, including Markdown and code formatting.

The workflow was smoke-tested against
`src/synthetic_data_generation/alignment`: Hermes successfully produced a
module report, and a second query resumed the same session ID and used the
previous report as context.

## Resource

Use [`scripts/hermes_delegate.py`](scripts/hermes_delegate.py) for all calls.
It validates state, serializes concurrent access with a lock, writes state
atomically with private permissions, and fails loudly on corrupt state,
missing session IDs, Hermes errors, or a required-but-missing session.
