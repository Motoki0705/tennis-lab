---
name: training-queue
description: Use this skill to enqueue training or experiment commands in a file-backed FIFO queue with logical half/all GPU reservations shared across workers and CUDA CI.
---

# Training Queue

## When to use

Use this when training runs (or other long GPU commands) must share one GPU.
Jobs declare `half` (one of two logical slots) or `all` (the complete logical
capacity). Up to two `half` jobs can overlap. `all` jobs remain exclusive.

## Script

`scripts/training_queue.sh` — a single self-contained bash script. State lives
under `$TRAINING_QUEUE_DIR` (default `.training_queue/` in the current
directory): `jobs/` (pending), `running/`, `done/`, `failed/`, `cancelled/`,
`logs/`, `worker.pid`, `worker.log`.

Always run it from the repository root so queued commands inherit the right CWD
(the enqueue CWD is recorded in each job and restored at run time).

## Subcommands

```bash
Q=.agents/skills/training-queue/scripts/training_queue.sh

# 1. Enqueue jobs (strict FIFO within each queue; safe for concurrent callers).
#    AI callers MUST declare --provider and --session (so the run is
#    attributable in the knowledge graph). --issue is recommended. Humans
#    launching by hand may omit them.
bash "$Q" add "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.tasks.plcs.scripts.train data=... model=... loss=..." \
    --name exp_a --provider claude --session "$CLAUDE_CODE_SESSION_ID" --issue 525 \
    --resource half --prune-ckpt
bash "$Q" add "python -m src.tasks.plcs.scripts.train ..." --name exp_b

# 2. Start the worker. --after-pid makes it wait for a native first run to
#    finish before draining the queue. --idle-timeout S (default 30) is how long
#    it polls for late additions once the queue is empty before exiting.
bash "$Q" start --after-pid 12345
bash "$Q" start                       # start immediately

# Supervisors such as systemd use the foreground entry point instead.
bash "$Q" serve --idle-timeout 2147483647

# 3. Observe / control.
bash "$Q" status      # worker state + queued/running/done/failed counts
bash "$Q" list        # all jobs with resource/slot/PID/state/wait fields
bash "$Q" cancel <job-file>  # cancel queued/waiting or request running teardown
bash "$Q" stop        # exit after currently active jobs finish
bash "$Q" clear       # drop pending (not-yet-started) jobs
```

`--resource` accepts exactly `half` or `all`; an unknown value or missing value
is an error. Omission defaults to `all`, as do legacy job files without a
resource header. A blocked FIFO head is never bypassed by later local jobs.

## Typical flow (first run native, rest queued)

1. Launch the first training natively (e.g. `nohup python -m ... &`) and note its PID.
2. `add` the remaining runs to the queue.
3. `start --after-pid <PID>` — the worker waits for the native run, then admits
   strict-FIFO jobs as capacity allows, logging each under `.training_queue/logs/`.

## Spawning a sub-agent to enqueue

A sub-agent can be told to read this SKILL.md and append jobs:

> Read `.agents/skills/training-queue/SKILL.md`, then enqueue these commands with
> `training_queue.sh add "<cmd>" --name <name>` (one per run). Do not start the
> worker.

Because enqueue is atomic (unique nanosecond+PID job ids) and lock-free,
multiple agents can add concurrently; a single worker consumes them in order.

## Reproducibility bundle (auto-saved, issue #533)

When a queued job starts, the worker captures the git/repro state **in the job's
own CWD** (worktree-aware) into `$TRAINING_QUEUE_DIR/repro/<jobid>/`:

- `run.json` — name, command, provider, session, issue, resource, logical slot,
  commit, branch, remote, and cwd.
- `uncommitted.patch` — `git diff HEAD` (apply-able), plus `git_status.txt`.
- `repro.sh` — one-shot reproduction: cd repo → checkout commit → apply patch → run command.

The job also gets `TENNIS_RUN_ID` and `TENNIS_REPRO_DIR` in its environment. The
training **lightning module** writes test-split inference results to
`$TENNIS_REPRO_DIR/predictions/{pred_test.npz, metrics.json}` (so checkpoints can
be deleted and metrics still recomputed). This replaces keeping `*.ckpt` around.

`add --prune-ckpt` opts a job into post-run checkpoint deletion. After a
successful job, the runner-written `$TENNIS_REPRO_DIR/output_dir.txt` identifies
that run's checkpoint directory. The worker deletes only its `*.ckpt` files and
only when the same repro bundle contains a verified
`predictions/pred_test.npz`. Failed jobs, jobs without the flag, missing/invalid
prediction bundles, and missing pointers keep their checkpoints. Pruning is
non-fatal and never removes the prediction bundle.

After a run finishes, register it into the git-tracked knowledge graph with the
**knowledge-control** skill, which promotes the repro bundle + predictions into
`knowledge/runs/<run-id>/`:

```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_register.py <job-name> \
    --issue 525 --provider claude
```

### Declaring your provider + session id

`--provider`/`--session` make the run attributable. To find your own session id,
read the per-provider workflow under [`reference/`](./reference/):
`reference/claude.md`, `reference/codex.md`, `reference/gemini.md`.

## Notes

- Exactly one worker runs per queue directory (`start` refuses if one is already
  alive). Workers using the same lock namespace share capacity.
- `serve` holds the same singleton lock as `start` and is the supported
  foreground entry point for systemd or another process supervisor.
- The main capacity lock is resolved in this order: an explicit non-empty
  `TRAINING_QUEUE_LOCK_FILE`; the fully provisioned production namespace at
  `/var/lib/tennis-lab-actions/gpu.lock`; a lock under the Git common root's
  `.training_queue/` (shared by linked worktrees); or, outside Git, the queue's
  own `gpu.lock`. A partial, symlinked, non-regular, or unwritable production
  namespace is a configuration error and is never repaired or bypassed.
  `half` takes the main lock shared plus one exclusive derived slot; `all` and
  raw CUDA CI take it exclusive. Cooperating queues also use `.gate`, `.slot-0`,
  and `.slot-1` beside the main lock. Production installers provision all four
  as narrowly writable regular files.
- `status` and `list` show declaration, allocation slot, leader PID, owned PGID,
  lifecycle state, and any wait reason. The capacity marker says `logical-only`.
- Every admitted workload runs in a queue-created private session/process group.
  Its wrapper is the sole allocation and process owner. On worker TERM or a
  running cancellation it publishes nonterminal `state=terminating`, sends TERM
  to the verified PGID, waits exactly 15 seconds, escalates with KILL, proves no
  non-zombie member remains, reaps its direct leader, and then proves the kernel
  PGID absent before publishing a terminal state or releasing capacity.
  `wait=process-group-teardown` identifies this hold.
  `start`, foreground `serve`, MCP-started workers, and systemd all use this
  lifecycle; the worker PID/lock remains live until every wrapper finishes.
- `half` is only a scheduling declaration. The queue never changes
  `CUDA_VISIBLE_DEVICES`, never configures MIG, and does not enforce a VRAM hard
  cap. The caller must ensure that two overlapping jobs fit on the shared GPU.
- A job's exit code is appended to its log; non-zero moves it to `failed/`.
- `cancel` publishes cancellation intent atomically. A queued or capacity-waiting
  job moves to `cancelled/` without launching its command. A running job remains
  `terminating` and capacity-consuming until verified teardown, then moves to
  `cancelled/`.
- MCP queue jobs opt into a queue-derived acknowledgement below the private
  queue control directory. Their deterministic container must be stopped,
  escalated if needed, and observed non-running before an atomic regular-file
  acknowledgement is published. Missing or invalid acknowledgement is exposed
  as `wait=external-teardown` and retains capacity.
- The supported ordinary-workload boundary is descendants that remain in the
  queue-created PGID. Self-daemonizing processes, `setsid`/double-fork escapees,
  and work launched through arbitrary host daemons are unsupported. If an
  in-group uninterruptible task cannot be removed, the wrapper intentionally
  remains safely stuck in `terminating` with its locks held; it never reports a
  false terminal state or silently releases capacity.
- Put per-run env vars (e.g. `PYTORCH_CUDA_ALLOC_CONF=...`) inside the command string.
- This is a developer workflow tool under `.agents/skills/`; it is not a
  `src/**/scripts` script and intentionally does not use Hydra.
