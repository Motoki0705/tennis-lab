---
name: research-pipeline
description: Run or resume an ARIS research campaign in Tennis Lab, from literature
  and comparison planning through queued experiments, knowledge registration and a
  Japanese report.
---

> **Tennis Lab integration:** Read [INTEGRATION.md](../../aris/INTEGRATION.md) first. It defines the execution, review, storage and artifact-path rules for this installation.

# ARIS research pipeline for Tennis Lab

Use ARIS to carry the requested research campaign through literature, experiment
planning, implementation, measurements and a Japanese comparison report.

1. Run `.venv/bin/python .agents/aris/aris.py setup` and `doctor` from the active
   checkout. Read knowledge-control's planning route and relevant prior evidence.
2. Resolve the user's brief and the campaign directory using INTEGRATION.md.
   Preserve already chosen methods. Obtain missing dataset/evaluation/budget
   decisions before dependent execution; continue independent investigation.
3. Use `research-lit`, `novelty-check` and, when ideas are needed, `idea-creator`
   or `research-refine`. Use `experiment-plan` to write a concrete plan and tracker:
   baseline, data/split, metrics, seeds, allowed edits, commands, run order,
   per-run timeout, total budget and stopping conditions.
4. Use `experiment-bridge` for the implemented plan. Keep implementation in this
   task's worktree. Run sanity, baseline and candidates through training-queue.
   Reuse existing data loaders, commands and evaluators.
5. Use `analyze-results`, `experiment-audit` and `result-to-claim` as appropriate
   to inspect actual saved evidence. Record completed and failed runs using
   knowledge-control. Its nodes and summary are the canonical research record.
6. Once artifacts and ordinary checks are complete, apply `auto-review-loop`
   only for the user's explicit independent-review allocation. If no allocation
   exists, record the phase as skipped; ordinary evidence checks still apply.
7. Write `NARRATIVE_REPORT.md` below the campaign directory with links to knowledge
   IDs, metrics, representative images, failure reasons, costs and reproduction
   commands. Render it with `render-html` using Japanese language metadata.
   Report the findings, remaining uncertainty and any unrun comparisons.

## Continuation and state

Resolve upstream tools through `.aris/installed-skills-codex.txt` or the explicit
`.agents/aris/tools` path. Use `run_state.py` with the campaign directory as root
and phases `planning,experiments,analysis,review,report`. Record the actual executor.
On resume, read the saved phase artifacts and queue job IDs before issuing work;
never duplicate jobs just because the agent session ended.

`done` means the artifact was produced. Record deterministic acceptance only with
an actual check and its saved evidence, and only for what it checked. For an
existing `done` phase, recheck its artifacts instead of repeating experiments.
Independent review is `skipped` when not requested. A requested same-model
validator result remains explicitly same-family/provisional under upstream state
semantics. Do not invent cross-provider reviewers to close a phase.

Continue within the agreed budget. Changed hypotheses or retry conditions remain
visible in the tracker. End at the comparison report unless the user asked for a
paper or another deliverable. Do not schedule periodic agent sessions implicitly.

Adapted from [ARIS Codex research-pipeline](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d/skills/skills-codex/research-pipeline/SKILL.md); [MIT license](../../aris/LICENSE).
