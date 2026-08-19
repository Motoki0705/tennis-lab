---
name: issue-subagent-workflow
description: Orchestrate one tennis-lab GitHub Issue through feasibility, evidence-focused exploration, parent-authored planning, implementation, independent preflight review, independent test authoring, independent candidate sealing, Issue-only validation, and PR-bound completion. Use for Issue-driven implementation, fixes, or refactors that require reproducible PASS/RETURN/BLOCKED/VALIDATED state transitions.
---

# Issue subagent workflow

Run one Issue through the fail-closed state machine in `state.toml`. GitHub is the upstream specification and final delivery surface; do not use Issue comments as workflow storage.

## Load only the contracts needed now

Always read [workflow](references/workflow.md) and [document contracts](references/document-contracts.md). Read [spawn contracts](references/spawn-contracts.md) immediately before delegation, [completion hardening](references/completion-hardening.md) before a final seal or PR packaging, and [goal integration](references/goal-integration.md) only when `/goal` is active.

## Initialize

```bash
TASK=.codex/tasks/issue-<number>
MANAGE=.agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py

python .agents/skills/issue-subagent-workflow/scripts/init_issue_task.py <issue>
```

Initialization freezes canonical `issue.json`, renders `issue.md`, records both hashes, creates schema-v5 state, and scaffolds every formal artifact. Refresh the Issue only after the upstream specification changes; refresh restarts feasibility and replaces stale formal artifacts.

## Required loop

1. Complete feasibility. A breaking change that cannot satisfy immutable tests or another Issue constraint is `BLOCKED`, not an implementation loop.
2. Run bounded Scouts only for independent semantic questions, then one authoritative Explorer.
3. Transition `exploration -> planning`; the mutating command automatically validates `exploration.md`.
4. The parent writes `plan.md` and machine-readable `02-planning/checks.json`, including exact argv, cwd, environment, stage, and AC authority for every canonical check.
5. Transition `planning -> implementation`; the command validates both plan and check manifest.
6. Run the user-selected implementation topology. An explicit request for one Implementer or sequential execution is compliant. Only the parent or an explicitly designated implementation integrator writes `implementation.md`.
7. After implementation is integrated, run one independent `preflight_reviewer` on the current candidate. It may write only preflight evidence, executes canonical checks through `run-check`, and returns PASS or RETURN; the parent applies `preflight-verdict` after validating the artifact, machine results, and candidate fingerprint.
8. After Preflight Reviewer PASS, run one independent Test Writer. It may add or update allowed tests but may not modify production. `test-verdict` binds its result to the post-test candidate.
9. After Tester PASS, run one independent `seal_reviewer` without editing source or tests. It re-runs seal-stage canonical checks over the complete candidate, inspects scope, and returns PASS or RETURN; the parent applies `seal-verdict`.
10. Transition to validation only after Tester PASS and Seal Reviewer PASS. The Validator receives the frozen Issue and sealed candidate identity, not prior narratives.
11. Validator PASS produces `status = "validated"`, `phase = "packaging"`; it does not complete the task.
12. Create or update the PR, check out its final head, then run `capture-pr`. The helper queries the real PR metadata, paginates every changed-file page, records remote checks, and binds that evidence to state. Write `packaging.md` with the captured evidence digest and run `finalize-pr`. Only this command sets `status = "complete"`.

## Delegation communication and waiting

Every `spawn_agent` call must set `fork_turns = "none"` exactly, including retries, post-compaction work, packaging repairs, Preflight Reviewers, Test Writers, Seal Reviewers, Validators, and bounded Validator children. Never use a numeric or inherited turn window as a shortcut for writing a focused assignment. Required context travels through frozen artifacts, explicit artifact paths, AC IDs, ownership, and the concrete failure bundle. A Validator spawned with inherited parent turns is not independent and must be replaced before its verdict is accepted. The same independence rule applies to a Preflight Reviewer, Test Writer, or Seal Reviewer.

Every `spawn_agent` assignment must end with the exact terminal-only footer from [spawn contracts](references/spawn-contracts.md), even when the selected custom agent already contains the same policy. The duplicate boundary is intentional: the versioned agent config is auditable, while the assignment-level footer overrides generic progress-update defaults for that concrete child turn.

Child agents work silently and return exactly one compact terminal handoff. Before that handoff they may interrupt the parent only for missing authority, an ownership collision, or a blocker that cannot be resolved inside the assigned scope. New evidence, completed milestones, commands in progress, and percentage updates stay in the child thread or formal artifact.

After spawning a wave, do independent parent work first, then call `wait_agent` once with `timeout_ms = 3_600_000` when supported, otherwise the maximum accepted timeout. The timeout is an upper bound, not a polling cadence: `wait_agent` may wake for any child message. A nonterminal message is not completion; unless it is an allowed escalation, do not answer it, do not call `list_agents`, and immediately resume the same long wait. Aggregate user-visible status at workflow phase boundaries instead of forwarding child chatter.

## Canonical commands

```bash
python $MANAGE candidate-fingerprint $TASK
python $MANAGE run-check $TASK <preflight|test|seal> <check-id>
python $MANAGE artifact-check $TASK <artifact>
python $MANAGE transition $TASK <planning|implementation|validation>
python $MANAGE preflight-verdict $TASK <PASS|RETURN>
python $MANAGE test-verdict $TASK <PASS|RETURN>
python $MANAGE seal-verdict $TASK <PASS|RETURN>
python $MANAGE return-review $TASK <implementation|exploration> --reason "<classification>"
python $MANAGE block $TASK <constraint_conflict|external_dependency|missing_authority|environment> --reason "<blocker>"
python $MANAGE verdict $TASK <PASS|RETURN>
python $MANAGE capture-pr $TASK --pr-number <n>
python $MANAGE finalize-pr $TASK --pr-number <n> --head-sha <40-char-sha>
python $MANAGE check $TASK
```

Artifact checks are also enforced inside every mutating transition; manual checks are an early feedback tool, not an optional safety boundary.

## Non-negotiable boundaries

- Implementer, Preflight Reviewer, Test Writer, and Seal Reviewer run sequentially and never concurrently.
- Implementers never write `preflight.md` or `seal.md`; the dedicated Reviewers own those artifacts and the parent owns verdict state transitions.
- A Test Writer may change tests after Preflight Reviewer PASS, so independent final sealing is mandatory before validation.
- Any content change after Tester PASS invalidates the seal; any content change after the seal invalidates validation and packaging.
- A non-canonical diagnostic failure is not a Tester RETURN. Run the command ID from `checks.json`; the helper records and verifies its normalized invocation.
- Do not treat a user-directed single-Implementer topology as noncompliance.
- Do not open or update the delivery PR before Validator PASS unless the user explicitly requires an earlier draft; completion still requires final-head binding.
- Never create `*-v2.md`; replace the authoritative artifact in place.
- Never use routine child progress messages as a substitute for artifacts, terminal handoffs, or state transitions.
- Never accept a child verdict produced from `fork_turns` other than `"none"`.
