---
name: issue-subagent-workflow
description: Orchestrate Issue-driven implementation, fixes, or refactors in tennis-lab through feasibility, exploration, parent planning, implementation, independent preflight/test/seal, Issue-only validation, and PR-bound completion with reproducible PASS/RETURN/BLOCKED/VALIDATED transitions.
---

# Issue subagent workflow

Drive one Issue through the fail-closed `state.toml` machine. GitHub is the upstream specification and delivery surface; Issue comments are not workflow storage.

Always read [workflow](references/workflow.md) and [document contracts](references/document-contracts.md). Read [spawn contracts](references/spawn-contracts.md) immediately before delegation, [completion hardening](references/completion-hardening.md) before sealing/packaging, and [goal integration](references/goal-integration.md) only with `/goal`.

## Initialize

```bash
TASK=.codex/tasks/issue-<number>
MANAGE=.agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py
python .agents/skills/issue-subagent-workflow/scripts/init_issue_task.py <issue>
```

This freezes canonical `issue.json`, renders `issue.md`, records both hashes, creates schema-v5 state, and scaffolds all formal artifacts. Refresh only after the upstream Issue changes; it restarts feasibility and replaces stale artifacts.

## Required loop

1. Complete feasibility. A breaking change incompatible with immutable tests or another Issue constraint is `BLOCKED`, not an implementation loop.
2. Use bounded Scouts only for independent semantic questions, then one authoritative Explorer; `transition ... planning` validates `exploration.md`.
3. The parent writes `plan.md` and `02-planning/checks.json`, whose checks define exact argv, cwd, environment, stages, and AC authority; `transition ... implementation` validates both.
4. Execute the selected topology; one Implementer or sequential execution is compliant. Only the parent or explicit implementation integrator writes `implementation.md`.
5. After integration, run one independent `preflight_reviewer`. It edits only preflight evidence, uses `run-check`, and returns PASS/RETURN; the parent verifies artifact, machine results, and fingerprint before `preflight-verdict`.
6. After Preflight PASS, run one independent Test Writer. It may change allowed tests, never production; `test-verdict` binds the post-test candidate.
7. After Tester PASS, run one independent `seal_reviewer` with no source/test edits. It inspects full scope, reruns canonical seal checks, and returns PASS/RETURN; the parent applies `seal-verdict`.
8. Enter validation only after Tester and Seal PASS. The Validator receives the frozen Issue and sealed candidate identity, not prior narratives.
9. Validator PASS sets `status = "validated"`, `phase = "packaging"`—not completion. Create/update the PR, check out its final head, run `capture-pr`, write `packaging.md` with the evidence digest, then run `finalize-pr`. `capture-pr` records real PR metadata, all paginated changed files, and remote checks in state; only `finalize-pr` sets `status = "complete"`.

## Delegation

Every `spawn_agent` call must set `fork_turns = "none"` exactly, including retries, post-compaction work, packaging repairs, all reviewers, Test Writers, Validators, and bounded Validator children. Pass context only through frozen artifacts, explicit paths, AC IDs, ownership, and focused failure bundles. A Validator spawned with inherited parent turns is not independent; neither are Preflight Reviewers, Test Writers, or Seal Reviewers—discard and respawn them.

Every assignment ends with the exact terminal-only footer in [spawn contracts](references/spawn-contracts.md), even if custom-agent instructions repeat it. Children return one compact terminal handoff and may interrupt earlier only for missing authority, ownership collision, or an unresolved in-scope blocker.

After independent parent work, call `wait_agent` once with `timeout_ms = 3_600_000` when supported, otherwise the maximum. Treat it as an event-driven upper bound: for a nonterminal non-escalation, neither answer nor call `list_agents`; resume the same wait. Update users at phase boundaries.

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

Mutations enforce artifact checks; manual `artifact-check` is early feedback, never a bypass.

## Boundaries

- Implementer → Preflight Reviewer → Test Writer → Seal Reviewer is strictly sequential. Implementers never write `preflight.md`/`seal.md`; Reviewers own those artifacts, the parent owns verdict transitions.
- Test Writer edits make final sealing mandatory. Content changes after Tester PASS invalidate the seal; post-seal changes invalidate validation and packaging.
- Stage verdicts require canonical `checks.json` IDs. A non-canonical diagnostic failure alone is not Tester RETURN.
- Single-Implementer execution does not weaken evidence gates. Do not open/update the delivery PR before Validator PASS unless the user explicitly requests an earlier draft; completion still requires final-head binding.
- Replace authoritative artifacts in place; never create `*-v2.md`.
- Progress messages never replace artifacts, terminal handoffs, or transitions; never accept a verdict from `fork_turns` other than `"none"`.
