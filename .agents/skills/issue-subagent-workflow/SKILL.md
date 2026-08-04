---
name: issue-subagent-workflow
description: Orchestrate one tennis-lab GitHub Issue through optional Luna scouting, formal codebase exploration, parent-authored planning, implementation, independent test authoring, checklist-based validation, and an optional persistent /goal loop using custom Codex subagents and .codex/tasks artifacts. Use for issue-driven implementation, fixes, or refactors that require a reproducible PASS/RETURN loop; do not use for ad hoc edits without an Issue.
---

# Issue subagent workflow

Run one Issue through a documented state machine. GitHub is input only: do not write workflow progress to the Issue.

## Start

1. Confirm the Issue body contains a `## Acceptance checklist` section with at least one concrete Markdown task-list item such as `- [ ] observable requirement`. The initializer rejects Issues without this exact section or with an empty checklist.
2. Initialize or refresh the frozen Issue snapshot:
   `python .agents/skills/issue-subagent-workflow/scripts/init_issue_task.py <issue>`
3. Use `.codex/tasks/issue-<number>/` as the single artifact directory.
4. Read [workflow](references/workflow.md), [document contracts](references/document-contracts.md), [spawn contracts](references/spawn-contracts.md), and [goal integration](references/goal-integration.md) before delegating.

## `/goal` integration

When the user explicitly starts this work with `/goal`, use the goal as the outer continuation loop and `state.toml` as the inner workflow state. The goal objective must name one Issue and must not complete until every normalized AC checklist item is independently verified PASS, the state helper accepts PASS, required checks pass, and the requested PR exists. On validator RETURN, keep the goal active and restart formal exploration. Do not treat RETURN, hard work, or an incomplete attempt as a blocked goal.

## Scout versus explorer

- Use `codebase_scout` for fast, bounded lookups: named files or symbols, direct references, nearby tests, or several independent candidate searches. It returns an advisory report to the parent and never owns a workflow artifact.
- Skip the scout and use `codebase_explorer` directly when the Issue crosses modules, depends on dynamic dispatch or configuration resolution, changes schemas or data/interface contracts, performs deletion or broad refactoring, or follows a validator RETURN.
- Promote any scout question to `codebase_explorer` when evidence is ambiguous, the impact radius expands, or the scout cannot establish a complete call path.
- A formal `codebase_explorer` run is mandatory before planning. Scout output is only a lead; the explorer and parent must verify material claims independently.

## Required loop

1. Optionally spawn one or more `codebase_scout` agents with `fork_turns = "none"` for independent bounded questions.
2. Spawn `codebase_explorer` with `fork_turns = "none"`; it replaces `01-exploration/exploration.md`. Pass useful scout findings as unverified leads, not authority.
3. Independently verify the explorer's critical claims in the codebase. Re-spawn it for unresolved questions.
4. The parent orchestrator replaces `02-planning/plan.md`. The parent alone owns decomposition, file ownership, and the mapping for every AC checklist item.
5. Spawn `issue_implementer` and `test_writer` with explicit non-overlapping ownership. The test writer must not read `implementation.md` and must account for every AC ID, including items that require non-test evidence.
6. Spawn `issue_validator` with `fork_turns = "none"`. Its task-specific input may identify only `issue.md`, the repository state or diff to inspect, and the destination `04-validation/validation.md`. Never mention or expose exploration, plan, implementation, or test artifacts.
7. Require the validator to emit exactly one ordered PASS/FAIL/NOT VERIFIED row for every normalized AC item. Source checkbox state is not evidence.
8. Apply the verdict:
   - `PASS`: run the state helper; only after it accepts the checklist hash, each exact checklist item, and every verdict as PASS may the task be marked complete and the PR opened.
   - `RETURN`: increment the attempt, return to exploration, and overwrite the same artifact files. Never create `*-v2.md` files.

Use the state helper for transitions and validation:

```bash
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py transition .codex/tasks/issue-<number> <planning|implementation|validation>
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py verdict .codex/tasks/issue-<number> <PASS|RETURN>
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py check .codex/tasks/issue-<number>
```

Do not open a PR before validator PASS. Do not mark an active `/goal` complete before the state helper and final check succeed. Treat artifacts as current-state documents; Git history preserves prior attempts.
