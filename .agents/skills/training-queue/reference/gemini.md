# Session id (conversation id) — Gemini / Antigravity CLI (`agy`)

Status: **retrieval mechanism validated live** (agy 1.0.10). Please confirm in
your own session and refine this file if anything differs.

## Non-interactive task delegation

```bash
agy -p "PROMPT" [--model M] [--add-dir DIR] [--dangerously-skip-permissions]
# -p / --print / --prompt : single non-interactive prompt
# resume later: agy --conversation <id>   |   most recent: agy -c
```

## Your own conversation id

Conversations are SQLite files named `<conversationId>.db` under
`~/.gemini/antigravity-cli/conversations/`. **`agy -p` creates a new `.db` but
does *not* append to `history.jsonl`** (interactive mode does). So the robust
way after a print run is the newest `.db` stem:

```bash
CID=$(ls -t ~/.gemini/antigravity-cli/conversations/*.db \
        | head -1 | xargs -n1 basename | sed 's/\.db$//')
echo "$CID"     # e.g. 13e7d30e-241c-49cf-b453-0c225c83afcc
```

For an interactive session, the latest `history.jsonl` line scoped to this
workspace also holds it:

```bash
grep '"workspace":"'"$PWD"'"' ~/.gemini/antigravity-cli/history.jsonl \
  | tail -1 | python3 -c 'import sys,json;print(json.loads(sys.stdin.readline())["conversationId"])'
```

## Enqueue training with attribution

```bash
Q=.agents/skills/training-queue/scripts/training_queue.sh
bash "$Q" add "python -m src.tasks.plcs.scripts.train model=... loss=... data=..." \
    --name i525_param --provider gemini --session "$CID" --issue 525
```
