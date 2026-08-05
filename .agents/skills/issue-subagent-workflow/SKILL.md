---
name: issue-subagent-workflow
description: Orchestrate one tennis-lab GitHub Issue through feasibility checking, bounded scouting, formal exploration, parent-authored planning, user-selected implementation topology, deterministic preflight, independent test gating, checklist validation, and an optional persistent /goal loop. Use for issue-driven implementation, fixes, or refactors that need reproducible PASS/RETURN/BLOCKED state transitions; do not use for ad hoc edits without an Issue.
---

# Issue subagent workflow

Run one Issue through a documented, fail-closed state machine. GitHub is input only: do not write workflow progress to the Issue.

## Start

1. Confirm the Issue body contains a concrete `## Acceptance checklist`.
2. Initialize or refresh the frozen snapshot:
   `python .agents/skills/issue-subagent-workflow/scripts/init_issue_task.py <issue>`
3. Use `.codex/tasks/issue-<number>/` as the single artifact directory.
4. Read [workflow](references/workflow.md), [document contracts](references/document-contracts.md), [spawn contracts](references/spawn-contracts.md), [completion hardening](references/completion-hardening.md), and [goal integration](references/goal-integration.md) before delegating.

## Feasibility gate

Before broad exploration or production changes, the parent replaces `00-feasibility/feasibility.md` and proves that every AC can be satisfied inside the allowed write scope and required checks.

- Record allowed and prohibited paths, breaking-change requirements, baseline failures, and canonical required checks.
- Inspect whether existing tests or checks encode behavior the Issue requires removing.
- If requirements conflict, record `BLOCKED`; do not invent compatibility code, weaken tests, or loop.

```bash
TASK=.codex/tasks/issue-<number>
MANAGE=.agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py

python $MANAGE artifact-check $TASK feasibility
python $MANAGE feasibility-verdict $TASK PASS

python $MANAGE feasibility-verdict $TASK BLOCKED \
  --kind constraint_conflict --reason "<specific conflict>"
```

A blocked task is paused, not complete. Refresh the upstream Issue and reinitialize it after the constraint is resolved.

## Delegation topology and context economy

Parallelism is a default optimization, not a compliance requirement.

- Use multiple agents only for genuinely independent evidence or non-overlapping ownership.
- An explicit user request for one Implementer or sequential execution is compliant. Record the topology in `plan.md` and preserve all evidence gates.
- Prefer deterministic repository commands or AST scripts over agents for complete mechanical inventories.
- Join a wave once with a long or event-driven wait; do not poll unchanged state repeatedly.
- Keep raw logs in child threads or `logs/`; return compact handoffs.
- Start a fresh bounded session after a Validator RETURN when a long-running child has accumulated several cycles. Artifacts carry the authority.

## Required loop

1. Pass feasibility or stop as `BLOCKED`.
2. Run bounded Scouts when independent semantic questions exist.
3. Run one authoritative Explorer and have it replace `exploration.md`.
4. Run `artifact-check ... exploration`; repair formatting before transitioning.
5. The parent verifies load-bearing claims and replaces `plan.md`, including ownership, any user topology override, exact canonical commands, and validation methods.
6. Run `artifact-check ... plan`, then transition to implementation.
7. Run the user-selected Implementer topology and join all work through one artifact integrator.
8. The integrator replaces `implementation.md` and `preflight.md`, runs deterministic checks in fail-fast order, and runs targeted artifact checks.
   - Preflight RETURN returns directly to implementation without spending a Test Writer cycle.
   - Preflight PASS permits one independent Test Writer.
9. The Test Writer uses the exact canonical commands. A non-canonical command mismatch is corrected and rerun in the same cycle; it is not a Tester RETURN by itself.
10. Run `artifact-check ... tests`, then apply PASS or RETURN.
    - First RETURN: repair the bounded findings.
    - Second RETURN: classify with `return-review` before continuing.
11. Run one Issue-only Validator. It runs `artifact-check ... validation` before returning.
12. Apply Validator PASS only after the complete artifact set validates. Then run final `check`, bind the validated content to the final PR diff, and create the PR.

## State helper

```bash
TASK=.codex/tasks/issue-<number>
MANAGE=.agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py

python $MANAGE artifact-check $TASK <artifact>
python $MANAGE transition $TASK <planning|implementation|validation>
python $MANAGE preflight-verdict $TASK <PASS|RETURN>
python $MANAGE test-verdict $TASK <PASS|RETURN>
python $MANAGE return-review $TASK <implementation|exploration> --reason "<classification>"
python $MANAGE block $TASK <constraint_conflict|external_dependency|missing_authority|environment> \
  --reason "<specific blocker>"
python $MANAGE verdict $TASK <PASS|RETURN>
python $MANAGE check $TASK
```

Do not run Tester and Implementer concurrently. Do not record a test verdict without matching preflight PASS. Do not treat a user-directed single-Implementer topology as noncompliance. Do not open a PR before Validator PASS and final check. Never create `*-v2.md`; Git history is the audit trail.
