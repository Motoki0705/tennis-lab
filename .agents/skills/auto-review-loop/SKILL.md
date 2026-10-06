---
name: auto-review-loop
description: Handle an ARIS research review handoff using only the user-authorized
  Tennis Lab validator allocation; record an unrequested review as skipped.
---

> **Tennis Lab integration:** Read [INTEGRATION.md](../../aris/INTEGRATION.md) first. It defines the execution, review, storage and artifact-path rules for this installation.

# ARIS review handoff under Tennis Lab's validator policy

Read the applicable common AGENTS.md validator policy before using this skill.
This entry replaces ARIS's fixed four-round, score-threshold review loop.

If the current user task has no explicit independent-review allocation, record
`independent review not run` and skip this phase. The parent still checks real
outputs, metrics, plots, tests and claim support. Do not spawn a reviewer, ask the
user to choose a review count, or invent an accepted score to advance the workflow.

If an allocation exists, prepare the complete artifacts and ordinary verification
first. Execute only the still-authorized rounds with the configured validator
role, model/provider/effort and fresh context as required by the common policy.
Count each attempted launch before calling it, preserve consumption on resume,
retain original findings and record the parent's disposition and checks.
Use the existing task-level allocation; nested code, figure and report reviews
must not create extra rounds.

Write `review-stage/AUTO_REVIEW.md` with actual review status and evidence.
Same-model/family evaluation remains provisional in upstream ARIS terminology;
it is not proof of statistical independence or a measured scientific score.
Return to the research report without external-provider routing or automatic
GPU follow-up beyond the user's authorized experiment plan and budget.

Adapted from [ARIS Codex auto-review-loop](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d/skills/skills-codex/auto-review-loop/SKILL.md); [MIT license](../../aris/LICENSE).
