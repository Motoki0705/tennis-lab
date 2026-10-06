---
name: idea-discovery
description: Develop research hypotheses and a comparison plan from literature and
  prior Tennis Lab evidence using ARIS.
---

> **Tennis Lab integration:** Read [INTEGRATION.md](../../aris/INTEGRATION.md) first. It defines the execution, review, storage and artifact-path rules for this installation.

# ARIS idea discovery for Tennis Lab

Read the user's research direction, any supplied papers/code, and the relevant
knowledge-control summary before proposing experiments. Preserve explicit
candidate methods and constraints.

Use `research-lit` to gather primary sources, `idea-creator` to form hypotheses,
`novelty-check` to check the claimed contribution, and `research-refine` to narrow
what will be tested. Save sources, competing explanations and ranked candidates
in the campaign's `idea-stage/IDEA_REPORT.md`. Rankings are planning judgments,
not measured results. If methods were already selected, compare their hypotheses
rather than replacing them with a single new idea.

Use `experiment-plan` to produce the comparison protocol and run tracker under
`refine-logs/`. Any pilot belongs to that protocol, consumes the agreed budget
and goes through `experiment-bridge` and training-queue. Without a completed
pilot, state that the hypothesis is untested.

The parent performs ordinary evidence checks. Independent assessment, if
explicitly allocated by the user, follows the common validator policy after the
artifacts are ready. The upstream jury gate is not an extra review allocation.
Report the selected plan and continue within existing authorization.

Adapted from [ARIS Codex idea-discovery](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d/skills/skills-codex/idea-discovery/SKILL.md); [MIT license](../../aris/LICENSE).
