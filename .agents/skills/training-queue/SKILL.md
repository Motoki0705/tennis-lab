---
name: training-queue
description: Use this skill to run multiple training/experiment commands serially in the background via a file-backed FIFO queue. Enqueue jobs (optionally from several agents), then start a single worker that runs them one at a time and can wait for an already-running native run (a PID) to finish first.
---

# Training Queue

## When to use

Use this when several training runs (or any long commands) must run **one at a
time** on a shared GPU, and you want to queue them up front — possibly from
multiple sub-agents — instead of manually chaining `tail --pid` waits. The first
run may already be launched natively; the queue can wait for that PID and then
drain the rest.

## Script

`scripts/training_queue.sh` — a single self-contained bash script. State lives
under `$TRAINING_QUEUE_DIR` (default `.training_queue/` in the current
directory): `jobs/` (pending), `running/`, `done/`, `failed/`, `logs/`,
`worker.pid`, `worker.log`.

Always run it from the repository root so queued commands inherit the right CWD
(the enqueue CWD is recorded in each job and restored at run time).

## Subcommands

```bash
Q=.agents/skills/training-queue/scripts/training_queue.sh

# 1. Enqueue jobs (FIFO; safe to call concurrently from multiple agents).
bash "$Q" add "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.tasks.plcs.scripts.train data=... model=... loss=..." --name exp_a
bash "$Q" add "python -m src.tasks.plcs.scripts.train ..." --name exp_b

# 2. Start the worker. --after-pid makes it wait for a native first run to
#    finish before draining the queue. --idle-timeout S (default 30) is how long
#    it polls for late additions once the queue is empty before exiting.
bash "$Q" start --after-pid 12345
bash "$Q" start                       # start immediately

# 3. Observe / control.
bash "$Q" status      # worker state + queued/running/done/failed counts
bash "$Q" list        # pending jobs in run order
bash "$Q" stop        # exit after the current job finishes
bash "$Q" clear       # drop pending (not-yet-started) jobs
```

## Typical flow (first run native, rest queued)

1. Launch the first training natively (e.g. `nohup python -m ... &`) and note its PID.
2. `add` the remaining runs to the queue.
3. `start --after-pid <PID>` — the worker waits for the native run, then runs the
   queued jobs serially, logging each to `.training_queue/logs/<jobid>.log`.

## Spawning a sub-agent to enqueue

A sub-agent can be told to read this SKILL.md and append jobs:

> Read `.agents/skills/training-queue/SKILL.md`, then enqueue these commands with
> `training_queue.sh add "<cmd>" --name <name>` (one per run). Do not start the
> worker.

Because enqueue is atomic (unique nanosecond+PID job ids) and lock-free,
multiple agents can add concurrently; a single worker consumes them in order.

## Notes

- Exactly one worker runs at a time (`start` refuses if one is already alive).
- A job's exit code is appended to its log; non-zero moves it to `failed/`.
- Put per-run env vars (e.g. `PYTORCH_CUDA_ALLOC_CONF=...`) inside the command string.
- This is a developer workflow tool under `.agents/skills/`; it is not a
  `src/**/scripts` script and intentionally does not use Hydra.
