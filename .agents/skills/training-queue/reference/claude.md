# Session id — Claude Code

Status: **validated** (Claude Code 2.x, this repo).

## Your own session id

Claude Code exports it as an environment variable:

```bash
echo "$CLAUDE_CODE_SESSION_ID"     # e.g. d22b7d68-7d91-4a6f-862d-434085e5d2d9
```

It equals the transcript filename under
`~/.claude/projects/<cwd-with-slashes-as-dashes>/<session-id>.jsonl`. As a
fallback (e.g. if the env var is unset), the newest transcript for this repo:

```bash
ls -t ~/.claude/projects/-home-kamimura-projects-tennis-lab/*.jsonl \
  | head -1 | xargs -n1 basename | sed 's/\.jsonl$//'
```

## Enqueue training with attribution

```bash
Q=.agents/skills/training-queue/scripts/training_queue.sh
bash "$Q" add "python -m src.tasks.plcs.scripts.train model=... loss=... data=..." \
    --name i525_asym --provider claude --session "$CLAUDE_CODE_SESSION_ID" --issue 525
```

## Delegating to a fresh sub-session (orchestration)

Sub-agents spawned via the Agent tool are themselves Claude Code sessions and
each has its own `$CLAUDE_CODE_SESSION_ID`; the sub-agent reads it the same way
and passes it to `training_queue.sh add`.
