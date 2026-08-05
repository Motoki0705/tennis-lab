---
name: issue-subagent-workflow
description: Orchestrate one tennis-lab GitHub Issue through feasibility checking, bounded parallel scouting, formal exploration, parent-authored planning, parallel implementation, deterministic preflight, independent test gating, checklist validation, and an optional persistent /goal loop. Use for issue-driven implementation, fixes, or refactors that need reproducible PASS/RETURN/BLOCKED state transitions; do not use for ad hoc edits without an Issue.
---

# Issue subagent workflow

Run one Issue through a documented, fail-closed state machine. GitHub is input only: do not write workflow progress to the Issue.

## Start

1. Confirm the Issue body contains a `## Acceptance checklist` section with at least one concrete Markdown task-list item.
2. Initialize or refresh the frozen Issue snapshot:
   `python .agents/skills/issue-subagent-workflow/scripts/init_issue_task.py <issue>`
3. Use `.codex/tasks/issue-<number>/` as the single artifact directory.
4. Read [workflow](references/workflow.md), [document contracts](references/document-contracts.md), [spawn contracts](references/spawn-contracts.md), and [goal integration](references/goal-integration.md) before delegating.

## Feasibility gate

Before formal exploration or production changes, the parent replaces `00-feasibility/feasibility.md` and verifies that the Issue is satisfiable.

- Record allowed and prohibited write scopes, breaking-change requirements, required checks, and a baseline result where practical.
- Inspect whether existing tests or required checks encode behavior that the Issue requires removing.
- Map every AC item to the files and evidence needed to satisfy it.
- Use deterministic searches or bounded Scouts only when repository evidence is needed; do not start the full implementation workflow yet.
- If an AC cannot be satisfied without violating another Issue constraint, record `BLOCKED`. Do not compensate with compatibility code, weakened tests, repeated Tester cycles, or an out-of-scope edit.

Apply the gate with:

```bash
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py \
  feasibility-verdict .codex/tasks/issue-<number> PASS

python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py \
  feasibility-verdict .codex/tasks/issue-<number> BLOCKED \
  --kind constraint_conflict --reason "<specific conflict>"
```

A blocked task is paused, not complete. Refresh the upstream Issue and reinitialize it after the constraint is resolved.

## Delegation and context economy

Before exploration and each implementation wave, decompose the work into independent questions or disjoint production ownership.

- Use multiple Scouts or Implementers only when their work is genuinely independent.
- Prefer deterministic commands, AST checks, and repository scripts over an agent for complete mechanical inventories.
- Keep one authoritative Explorer, Test Writer, and Validator per cycle.
- Join the entire wave once; do not repeatedly poll unchanged agent state with short waits.
- Keep raw command output under `.codex/tasks/issue-<number>/logs/` when useful. Return compact summaries with paths, commands, outcomes, and unresolved risks instead of pasting full logs into the parent thread.
- Retry prompts carry only the new failure bundle and ownership delta. The artifacts remain the authority; do not restate the full Issue and workflow narrative.

## Required loop

1. Pass the feasibility gate or stop as `BLOCKED`.
2. Build a delegation map and run bounded `codebase_scout` tasks when the Issue can be partitioned.
3. Run one authoritative `codebase_explorer`; it replaces `01-exploration/exploration.md` and independently verifies joined Scout evidence.
4. The parent verifies load-bearing exploration claims.
5. The parent replaces `02-planning/plan.md`, maps every AC item, defines non-overlapping implementation ownership, and records the canonical required checks.
6. Run one or more `issue_implementer` agents. Join all production work through one artifact integrator.
7. Before spawning the Test Writer, the integrator replaces `03-implementation/preflight.md` and runs deterministic policy checks, focused checks, then the canonical required checks.
   - `preflight RETURN`: return directly to implementation without spending an independent test cycle.
   - `preflight PASS`: record it and spawn the Test Writer.
8. Run one `test_writer`. It must not read `implementation.md` or repair production code.
   - `PASS`: record it, then transition to validation.
   - First `RETURN`: repair the bounded production failures and repeat preflight.
   - Second Tester `RETURN` since the previous review: the state helper requires an explicit `return-review`. Choose continued implementation only for demonstrably independent omissions; otherwise restart formal exploration or block the task.
9. Run one `issue_validator` with `fork_turns = "none"` and Issue-only task context.
   - `PASS`: complete only after every exact AC row is PASS.
   - `RETURN`: increment the attempt and restart formal exploration.

## State helper

```bash
TASK=.codex/tasks/issue-<number>
MANAGE=.agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py

python $MANAGE transition $TASK planning
python $MANAGE transition $TASK implementation
python $MANAGE preflight-verdict $TASK <PASS|RETURN>
python $MANAGE test-verdict $TASK <PASS|RETURN>
python $MANAGE return-review $TASK <implementation|exploration> --reason "<classification>"
python $MANAGE block $TASK <constraint_conflict|external_dependency|missing_authority|environment> \
  --reason "<specific blocker>"
python $MANAGE transition $TASK validation
python $MANAGE verdict $TASK <PASS|RETURN>
python $MANAGE check $TASK
```

Do not run Tester and Implementer concurrently. Do not record a test verdict without a matching preflight PASS. Do not enter validation without Tester PASS. Do not open a PR before Validator PASS. Never create `*-v2.md`; Git history is the audit trail.
