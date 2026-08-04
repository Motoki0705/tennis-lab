---
name: issue-subagent-workflow
description: Orchestrate one tennis-lab GitHub Issue through optional Luna scouting, formal codebase exploration, parent-authored planning, implementation, independent test gating, checklist-based validation, and an optional persistent /goal loop using custom Codex subagents and .codex/tasks artifacts. Use for issue-driven implementation, fixes, or refactors that require reproducible tester and validator PASS/RETURN loops; do not use for ad hoc edits without an Issue.
---

# Issue subagent workflow

Run one Issue through a documented state machine. GitHub is input only: do not write workflow progress to the Issue.

## Start

1. Confirm the Issue body contains a `## Acceptance checklist` section with at least one concrete Markdown task-list item such as `- [ ] observable requirement`.
2. Initialize or refresh the frozen Issue snapshot:
   `python .agents/skills/issue-subagent-workflow/scripts/init_issue_task.py <issue>`
3. Use `.codex/tasks/issue-<number>/` as the single artifact directory.
4. Read [workflow](references/workflow.md), [document contracts](references/document-contracts.md), [spawn contracts](references/spawn-contracts.md), and [goal integration](references/goal-integration.md) before delegating.

## `/goal` integration

When the user explicitly starts this work with `/goal`, use the goal as the outer continuation loop and `state.toml` as the inner workflow state. The goal remains active through both tester RETURN and validator RETURN. Complete it only after the independent test gate passes, every normalized AC item is independently verified PASS, the state helper accepts validator PASS, required checks pass, and the requested PR exists.

## Scout versus explorer

- Use `codebase_scout` for fast, bounded lookups: named files or symbols, direct references, nearby tests, or independent candidate searches.
- Use `codebase_explorer` directly for cross-module work, dynamic dispatch or configuration resolution, schema or interface changes, deletion or broad refactoring, or any validator RETURN.
- Promote Scout work when evidence is ambiguous, the impact radius expands, or a complete call path cannot be established.
- A formal `codebase_explorer` run is mandatory before planning.

## Required loop

1. Optionally run independent `codebase_scout` tasks.
2. Run `codebase_explorer`; it replaces `01-exploration/exploration.md`.
3. The parent independently verifies critical exploration claims.
4. The parent replaces `02-planning/plan.md` and maps every AC item.
5. Run one or more `issue_implementer` agents with disjoint production ownership. Join all implementation work before testing.
6. Run `test_writer` only after the implementation work is integrated. It must not read `implementation.md`.
7. Apply the independent test verdict:
   - `PASS`: record it with `test-verdict ... PASS`, then transition to validation.
   - `RETURN`: record it with `test-verdict ... RETURN`, keep the phase at implementation, send the concrete failures to `issue_implementer`, and repeat implementation then testing with the next test-cycle number.
8. Run `issue_validator` with `fork_turns = "none"` and Issue-only task context.
9. Apply the validator verdict:
   - `PASS`: only after the helper verifies every exact AC row as PASS may the task complete and the PR be opened.
   - `RETURN`: increment the attempt, return to formal exploration, and overwrite the same artifacts.

Use the state helper:

```bash
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py transition .codex/tasks/issue-<number> <planning|implementation|validation>
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py test-verdict .codex/tasks/issue-<number> <PASS|RETURN>
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py verdict .codex/tasks/issue-<number> <PASS|RETURN>
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py check .codex/tasks/issue-<number>
```

Do not run Tester and Implementer concurrently. Do not enter validation without a recorded tester PASS. Do not open a PR before validator PASS. Never create `*-v2.md`; Git history is the audit trail.
