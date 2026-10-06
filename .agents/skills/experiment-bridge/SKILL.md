---
name: experiment-bridge
description: Implement an ARIS experiment plan in Tennis Lab, run sanity and comparison
  jobs through training-queue, and collect measured results.
---

> **Tennis Lab integration:** Read [INTEGRATION.md](../../aris/INTEGRATION.md) first. It defines the execution, review, storage and artifact-path rules for this installation.

# ARIS experiment bridge for Tennis Lab

Turn the campaign's `refine-logs/EXPERIMENT_PLAN.md` into measured experiments.

- Read the comparison protocol, tracker, data/split, metrics, seeds, run order,
  per-run timeout and total compute budget. Reuse the existing repository; inspect
  relevant READMEs before changing implementation.
- Implement the smallest executable path and run ordinary CPU checks first.
  Missing data, weights or unsupported runtime are explicit preflight failures.
  Save measurement outputs as JSON/CSV and their actual resolved configuration.
- Submit the smallest GPU sanity job through `run-experiment`. Wait for the queue's
  terminal result and validate artifacts before submitting the dependent phase.
- Submit baseline/main/ablation batches through `experiment-queue`. Every local
  command goes to the same training-queue, including pilots, inference and retries.
  Scheduler concurrency comes from that queue, not upstream MAX_PARALLEL_RUNS.
- Record the exact queue job IDs and execution cwd in EXPERIMENT_TRACKER.md.
  Use `monitor-experiment`; distinguish waiting, running, teardown, failure,
  cancellation, successful execution and valid measured outputs.
- Inspect real metrics and logs, then follow knowledge-control to register each
  terminal run using the exact reproduction bundle. Preserve failures with missing
  metrics rather than synthetic values. Update the canonical summary as required.
- Produce `refine-logs/EXPERIMENT_RESULTS.md` from those records, linking canonical
  node IDs and artifacts. State which required runs are complete and what remains.

Independent code reviews, rescue reviewers and result reviews share the user's
single task-level validator allocation. No allocation means ordinary author checks,
not an automatic reviewer. Handle failures with logs and bounded retries; a new
condition such as smaller resolution is a separate experiment.

Adapted from [ARIS Codex experiment-bridge](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d/skills/skills-codex/experiment-bridge/SKILL.md); [MIT license](../../aris/LICENSE).
