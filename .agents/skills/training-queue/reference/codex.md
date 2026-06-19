# Session id — Codex CLI

Status: **retrieval mechanism validated live** (codex-cli 0.138). Please confirm
in your own session and refine this file if anything differs.

## Non-interactive task delegation

```bash
codex exec [--json] [-m MODEL] [-C DIR] [-s read-only|workspace-write] "PROMPT"
# alias: codex e ...   |   resume later: codex exec resume <id>   /   codex resume
```

## Your own session id (a.k.a. thread id)

With `--json`, the **first** emitted event carries it:

```bash
codex exec --json "..." | head -1       # {"type":"thread.started","thread_id":"019ed...-..."}
```

So to capture it while still running the task, tee the stream:

```bash
codex exec --json "DO THE TASK" | tee /tmp/codex.jsonl
SID=$(head -1 /tmp/codex.jsonl | python3 -c 'import sys,json;print(json.load(sys.stdin)["thread_id"])')
```

Equivalently, sessions persist as rollout files and the id is the uuid in the
filename (use after a run; `--ephemeral` skips persistence):

```bash
ls -t ~/.codex/sessions/*/*/*/rollout-*.jsonl | head -1 \
  | sed -E 's#.*rollout-[0-9T:-]+-([0-9a-f-]+)\.jsonl#\1#'
```

The id also appears as `payload.id` in the leading `session_meta` record of that
rollout file.

## Enqueue training with attribution

```bash
Q=.agents/skills/training-queue/scripts/training_queue.sh
bash "$Q" add "python -m src.tasks.plcs.scripts.train model=... loss=... data=..." \
    --name i525_param --provider codex --session "$SID" --issue 525
```
