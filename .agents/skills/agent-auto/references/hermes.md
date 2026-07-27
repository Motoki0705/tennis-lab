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

## Model selection

Hermes Agent supports model and provider specification at several levels.

### CLI flags

The wrapper accepts `--model` and `--provider` flags and forwards them to
`hermes chat` unchanged:

```bash
.agents/skills/agent-auto/scripts/hermes-auto.sh \
  --model anthropic/claude-sonnet-4 \
  "Run the test suite"

.agents/skills/agent-auto/scripts/hermes-auto.sh \
  --provider openrouter \
  "Review the code"
```

**`--model MODEL`** specifies the model in `provider/model-name` format (e.g.
`anthropic/claude-sonnet-4`, `deepseek/deepseek-v4-flash`). When the provider
prefix is included, it overrides the provider as well. This flag is also
settable via the `HERMES_INFERENCE_MODEL` environment variable.

**`--provider PROVIDER`** overrides the inference provider independently (e.g.
`openrouter`, `anthropic`, `deepseek`). The built-in default is `auto`
(confirmed by `hermes chat --help`). The persistent default lives in
`config.yaml` under `model.provider` (confirmed by `hermes --help`). Built-in
provider names and custom user-defined names from the `providers:` section of
`config.yaml` are both valid (confirmed by `hermes chat --help`).

*Inference:* `auto` likely means auto-detection from available credentials
(e.g. the first provider with a valid API key in `~/.hermes/.env`), but the
exact detection logic is not documented in CLI help.

Unlike `--model`, there is no `HERMES_INFERENCE_PROVIDER` environment variable
— provider can only be set via `--provider` (or `model.provider` in config).

### Config-based defaults

Persistent model settings live in `~/.hermes/config.yaml`:

- `model.default` — default model name
- `model.provider` — default provider
- `model.base_url` — custom API base URL for self-hosted endpoints

These are set interactively with `hermes model` or imperatively with
`hermes config set model.default <name>`.

### Dynamic model discovery

Available models are not hard-coded in the CLI. Each configured provider
exposes its models via a `/v1/models` endpoint. Use `hermes model` for an
interactive picker (requires TTY). Pass `hermes model --refresh` to clear the
disk cache and re-fetch every provider's live model list.

Known models can therefore only be enumerated by running the interactive
picker or querying each provider's API directly.

### Resolution order

Based on the CLI help text (`hermes --help`, `hermes chat --help`):

- `--model MODEL` is documented as "a model override for this invocation" — it
  takes highest precedence over config. The `HERMES_INFERENCE_MODEL` env var is
  described as "also settable via" — an alternative at the same level, not a
  higher-priority override. In practice the CLI flag takes precedence in
  argparse-based resolution.
- When neither `--model` nor the env var is set, `model.default` from
  `config.yaml` is used (inferred from the "override" semantics of `--model`
  and `HERMES_INFERENCE_MODEL`).
- Provider resolution: if the `--model` value includes a `provider/` prefix
  (e.g. `anthropic/claude-sonnet-4`), that prefix overrides the provider
  (inferred from the format convention shown in `hermes --help` examples).
  Otherwise, `--provider` (if given) takes precedence, then
  `model.provider` from config, then the built-in `auto` (confirmed by
  `hermes chat --help`: "(default: auto)").

### Fallback providers

When the primary model fails with rate-limit, overload, or connection errors,
Hermes tries fallback providers in order. Managed via `hermes fallback`:

```bash
hermes fallback list       # Show the fallback chain
hermes fallback add        # Pick a provider + model to append
hermes fallback remove     # Remove an entry
hermes fallback clear      # Clear all fallback entries
```

The fallback chain lives in `config.yaml`. *Inference:* it is likely independent
of per-invocation `--model` / `--provider` overrides, since fallback is
configured separately via `hermes fallback add/remove`.

### Effect on `--ignore-user-config`

When `--ignore-user-config` is active (the wrapper default), the configured
`model.default` and `model.provider` from `~/.hermes/config.yaml` are **not
loaded**. In this mode:

- If neither `--model` nor `--provider` is passed, Hermes falls back to its
  built-in defaults.
- Pass `--use-user-config` (the wrapper's inverse of `--ignore-user-config`)
  to restore config-based model selection.
- Credentials from `~/.hermes/.env` remain available regardless, so a
  self-hosted `base_url` + API key in `.env` works even under
  `--ignore-user-config`.

### Wrapper script behaviour

`scripts/hermes-auto.sh` simply forwards `--model` and `--provider` to
`hermes chat` as-is when those options are provided. No transformation or
validation is applied in the wrapper. Verified against the script source
(lines 79–80, 137–138).

## Verified interface

The command shape was verified against Hermes Agent v0.15.1. The initial run
and a second run using the emitted session ID were both exercised against the
local CLI. The separate `hermes-delegation` skill remains the appropriate path
when Codex should automatically retain a conversation per Codex thread and
relay the answer verbatim.
