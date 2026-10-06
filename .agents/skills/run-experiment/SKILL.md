---
name: run-experiment
description: Submit and track one planned Tennis Lab experiment through the repository
  shared training queue.
---

> **Tennis Lab integration:** Read [INTEGRATION.md](../../aris/INTEGRATION.md) first. It defines the execution, review, storage and artifact-path rules for this installation.

# Run one ARIS experiment through Tennis Lab's queue

Read training-queue first. Resolve a concrete command, task worktree, unique job
name, actual session ID, resource declaration and timeout from the campaign plan.

From the execution checkout root, call:

```bash
.venv/bin/python .agents/aris/aris.py queue add '<concrete command with timeout>' \
  --name '<campaign-run>' --provider codex --session '<actual session id>' \
  --resource all
.venv/bin/python .agents/aris/aris.py queue status
```

Keep the exact returned job ID, command and cwd in the tracker. Start the existing
queue worker with `queue start` only if `status` shows no active worker. Use
`queue list` for individual lifecycle/capacity state.

The adapter resolves the shared queue from Git's common directory and records
this worktree as the job cwd. Never launch local GPU work through screen, nohup,
a second scheduler or a direct CUDA command. Respect explicit remote/Colab scope
using the project's corresponding workflow; this local adapter does not provision
cloud instances or select another machine automatically.

A successful enqueue is not a successful experiment. After terminal execution,
inspect the saved logs and validate the expected metrics/artifacts. Register the
exact bundle through knowledge-control. For cancellation, use `queue cancel` for
this job and wait for teardown to finish before reporting cancellation complete.

Adapted from [ARIS Codex run-experiment](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d/skills/skills-codex/run-experiment/SKILL.md); [MIT license](../../aris/LICENSE).
