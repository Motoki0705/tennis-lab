---
name: issue-subagent-workflow
description: Orchestrate one tennis-lab GitHub Issue through proactive parallel Luna scouting, formal codebase exploration, parent-authored planning, parallel implementation, independent test gating, checklist-based validation, and an optional persistent /goal loop using custom Codex subagents and .codex/tasks artifacts. Use for issue-driven implementation, fixes, or refactors that require reproducible tester and validator PASS/RETURN loops; do not use for ad hoc edits without an Issue.
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

## Proactive multi-agent delegation

Before exploration and before each implementation cycle, decompose the current work into independent questions or disjoint production ownership units.

- When two or more independent repository questions exist, spawn multiple `codebase_scout` agents concurrently by default.
- When the plan contains two or more non-overlapping production work units, spawn multiple `issue_implementer` agents concurrently by default.
- Do not assign the whole available scope to one agent merely because one agent could eventually complete it.
- Use as much safe parallelism as provides independent evidence or reduces the critical path; concurrency is not a quota and redundant agents are forbidden.
- Every parallel assignment must have a unique question or explicit, non-overlapping file/module ownership.
- Join every agent from the current wave before advancing the workflow or starting a dependent wave.
- If the parent uses only one Scout or one Implementer despite decomposable scope, it must record why additional parallel delegation would be unsafe, redundant, or artifact-conflicting.
- Keep one authoritative `codebase_explorer`, one `test_writer`, and one `issue_validator` per cycle. The Validator should proactively spawn multiple bounded child Explorers when independent AC evidence questions can be checked in parallel.

## Scout versus explorer

- Use `codebase_scout` for fast, bounded lookups: named files or symbols, direct references, nearby tests, or independent candidate searches.
- Prefer several narrowly scoped Scouts over one broad Scout for cross-cutting Issues.
- Use `codebase_explorer` directly for cross-module work, dynamic dispatch or configuration resolution, schema or interface changes, deletion or broad refactoring, or any validator RETURN.
- Promote Scout work when evidence is ambiguous, the impact radius expands, or a complete call path cannot be established.
- A formal `codebase_explorer` run is mandatory before planning.

## Required loop

1. Build a delegation map and proactively run multiple independent `codebase_scout` tasks when the Issue can be partitioned.
2. Run one authoritative `codebase_explorer`; it replaces `01-exploration/exploration.md` and independently verifies the joined Scout evidence.
3. The parent independently verifies critical exploration claims.
4. The parent replaces `02-planning/plan.md`, maps every AC item, and defines non-overlapping implementation ownership.
5. Proactively run multiple `issue_implementer` agents when two or more disjoint production work units exist. Join all implementation work before testing and use one artifact integrator.
6. Run one `test_writer` only after the implementation work is integrated. It must not read `implementation.md`.
7. Apply the independent test verdict:
   - `PASS`: record it with `test-verdict ... PASS`, then transition to validation.
   - `RETURN`: record it with `test-verdict ... RETURN`, keep the phase at implementation, decompose the failures into disjoint repair units where possible, send them to one or more `issue_implementer` agents, and repeat implementation then testing with the next test-cycle number.
8. Run one `issue_validator` with `fork_turns = "none"` and Issue-only task context. It may proactively delegate independent AC evidence questions to multiple bounded child Explorers.
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
