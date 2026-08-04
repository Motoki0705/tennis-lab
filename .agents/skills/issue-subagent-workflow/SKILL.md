---
name: issue-subagent-workflow
description: Orchestrate a tennis-lab GitHub issue through codebase exploration, parent-authored planning, implementation, independent test authoring, and issue-only validation using custom Codex subagents and .codex/tasks artifacts. Use for issue-driven implementation, fixes, or refactors that require a reproducible PASS/RETURN loop; do not use for ad hoc edits without an issue.
---

# Issue subagent workflow

Run one issue through a documented state machine. GitHub is input only: do not write workflow progress to the issue.

## Start

1. Initialize or refresh the frozen issue snapshot:
   `python .agents/skills/issue-subagent-workflow/scripts/init_issue_task.py <issue>`
2. Use `.codex/tasks/issue-<number>/` as the single artifact directory.
3. Read [workflow](references/workflow.md), [document contracts](references/document-contracts.md), and [spawn contracts](references/spawn-contracts.md) before delegating.

## Required loop

1. Spawn `codebase_explorer` with `fork_turns = "none"`; it replaces `01-exploration/exploration.md`.
2. Independently verify the explorer's critical claims in the codebase. Re-spawn it for unresolved questions.
3. The parent orchestrator replaces `02-planning/plan.md`. The parent alone owns decomposition, file ownership, and acceptance mapping.
4. Spawn `issue_implementer` and `test_writer` with explicit non-overlapping ownership. The test writer must not read `implementation.md`.
5. Spawn `issue_validator` with `fork_turns = "none"`. Its task-specific input may identify only `issue.md`, the repository state or diff to inspect, and the destination `04-validation/validation.md`. Never mention or expose exploration, plan, implementation, or test artifacts.
6. Apply the verdict:
   - `PASS`: mark complete and open a PR.
   - `RETURN`: increment the attempt, return to exploration, and overwrite the same artifact files. Never create `*-v2.md` files.

Use the state helper for transitions and validation:

```bash
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py transition .codex/tasks/issue-<number> <planning|implementation|validation>
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py verdict .codex/tasks/issue-<number> <PASS|RETURN>
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py check .codex/tasks/issue-<number>
```

Do not open a PR before validator PASS. Treat artifacts as current-state documents; Git history preserves prior attempts.
