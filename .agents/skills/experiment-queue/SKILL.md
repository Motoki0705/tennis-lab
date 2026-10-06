---
name: experiment-queue
description: Schedule a planned batch of Tennis Lab experiments with dependencies
  and budgets through the existing shared training queue.
allowed-tools: Bash(*), Read, Grep, Glob, Edit, Write, Skill(run-experiment), Skill(monitor-experiment)
---

> **Tennis Lab integration:** Read [INTEGRATION.md](../../aris/INTEGRATION.md) first. It defines the execution, review, storage and artifact-path rules for this installation.

# Queue an ARIS experiment batch in Tennis Lab

Read training-queue. Expand the supplied experiment plan into concrete runs with
unique names, dependencies, fixed seeds/conditions, per-run timeout and required
outputs. Track the proposed and actual resource budget.

For each ready run use `run-experiment`, which calls the repository's shared
training-queue through `.agents/aris/aris.py queue`. Save each returned job ID.
The existing FIFO worker owns concurrency and resource allocation. Do not use
ARIS's SSH scheduler, screen process launcher or queue_manager.py for local GPUs.

Only enqueue a dependent phase after all required predecessor jobs completed and
their outputs passed the plan's checks. A failed prerequisite remains a visible
failure; record dependent runs as not executed. Do not rely on FIFO order alone
to establish a successful dependency.

On resume, reconcile recorded job IDs against `queue list` and the saved queue
artifacts before adding work. Count failed attempts and retries in the budget.
Use `monitor-experiment` to collect outcomes and knowledge-control to record them.
Do not cancel another campaign's jobs or globally stop/clear the shared queue.

Adapted from [ARIS Codex experiment-queue](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d/skills/skills-codex/experiment-queue/SKILL.md); [MIT license](../../aris/LICENSE).
