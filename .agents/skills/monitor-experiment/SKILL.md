---
name: monitor-experiment
description: Monitor recorded ARIS job IDs in Tennis Lab training-queue and collect
  terminal outcomes and measurement artifacts.
---

> **Tennis Lab integration:** Read [INTEGRATION.md](../../aris/INTEGRATION.md) first. It defines the execution, review, storage and artifact-path rules for this installation.

# Monitor ARIS experiments in Tennis Lab

Read training-queue and the current campaign tracker. Use:

```bash
.venv/bin/python .agents/aris/aris.py paths
.venv/bin/python .agents/aris/aris.py queue list
```

Match exact recorded job IDs to the shared queue. Inspect their logs, terminal
status and reproduction bundle. Running capacity-wait entries have not necessarily
started computation; terminating entries still hold their resource reservation.
Do not infer progress or success from a process disappearing.

When a job finishes, verify the output contract and recorded configuration.
A zero exit status without required measurements is an invalid experiment result,
not evidence of improvement. Keep failed/cancelled/unrun cases in the comparison.
Use knowledge-control's registration route with the exact `--repro-dir` and task.

Update ARIS's tracker with status and links to canonical knowledge nodes rather
than creating another metrics database. Continue observing within the campaign's
budget and report blocked dependencies with their actual failure evidence.

Adapted from [ARIS Codex monitor-experiment](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d/skills/skills-codex/monitor-experiment/SKILL.md); [MIT license](../../aris/LICENSE).
