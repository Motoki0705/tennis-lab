---
name: issue-subagent-workflow
description: Orchestrate Issue-driven implementation, fixes, or refactors in tennis-lab through feasibility, exploration, parent planning, implementation, bounded independent preflight/test/seal, Issue-only validation, and PR-bound completion with reproducible PASS/RETURN/BLOCKED/VALIDATED transitions.
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

This freezes canonical `issue.json`, renders `issue.md`, records both hashes, creates schema-v6 state, and scaffolds all formal artifacts. Refresh only after the upstream Issue changes; it restarts feasibility and replaces stale artifacts. Loaded schema-v5 tasks retain their legacy Tester contract while new tasks enforce adversarial testing.

## Required loop

1. Complete feasibility. A breaking change incompatible with immutable tests or another Issue constraint is `BLOCKED`, not an implementation loop.
2. Use bounded Scouts only for independent semantic questions, then one authoritative Explorer; `transition ... planning` validates `exploration.md`.
3. The parent writes `plan.md` and `02-planning/checks.json`, whose checks define the mandatory baseline argv, cwd, environment, stages, and AC authority. They are a floor for the Test Writer, not an exhaustive ceiling. The plan still freezes any bounded diagnostic categories permitted during the initial Preflight and the evidence required for closure after RETURN; `transition ... implementation` validates both.
4. Execute the selected topology; one Implementer or sequential execution is compliant. Only the parent or explicit implementation integrator writes `implementation.md`.
5. After integration, run one independent discovery `preflight_reviewer`. It edits only preflight evidence, uses `run-check`, and returns PASS/RETURN. One RETURN permits one bounded repair and one fresh closure Reviewer; a second consecutive Preflight RETURN requires `return-review` before any further Preflight.
6. After Preflight PASS, run one independent adversarial Test Writer. It derives a bounded risk model from the frozen Issue, public contracts, repository invariants, baseline behavior, current code, and complete diff; the plan supplies only the mandatory minimum. It may change impacted tests/fixtures, never production or planning authority, and may add candidate-bound `AT-*` probes. `test-verdict` binds the post-test candidate. Two consecutive Tester RETURNs require `return-review`.
7. After Tester PASS, run one independent `seal_reviewer` with no source/test edits. Seal verifies candidate identity, approved scope, repository rules, evidence completeness, and canonical seal checks; it is not another open-ended semantic review. Any Seal RETURN requires `return-review`, and every content repair requires fresh Preflight/Test before resealing.
8. Enter validation only after Tester and Seal PASS. The Validator receives the frozen Issue and sealed candidate identity, not prior narratives.
9. Validator PASS sets `status = "validated"`, `phase = "packaging"`—not completion. Create/update the PR, check out its final head, run `capture-pr`, write `packaging.md` with the evidence digest, then run `finalize-pr`. `capture-pr` records real PR metadata, all paginated changed files, and remote checks in state; only `finalize-pr` sets `status = "complete"`.
10. If packaging exposes a content defect, run `packaging-repair ... --reason "..."` before editing further. It preserves the frozen Issue, attempt, exploration, and plan, invalidates all candidate/PR bindings, returns to in-progress implementation, and requires fresh Preflight/Test/Seal/Validation. History-only packaging does not use this transition.

## Delegation

Every `spawn_agent` call must set `fork_turns = "none"` exactly, including retries, post-compaction work, packaging repairs, all reviewers, Test Writers, Validators, and bounded Validator children. Pass context only through frozen artifacts, explicit paths, AC IDs, ownership, and focused failure bundles. A Validator spawned with inherited parent turns is not independent; neither are Preflight Reviewers, Test Writers, or Seal Reviewers—discard and respawn them.

Every assignment ends with the exact terminal-only footer in [spawn contracts](references/spawn-contracts.md), even if custom-agent instructions repeat it. Children return one compact terminal handoff and may interrupt earlier only for missing authority, ownership collision, or an unresolved in-scope blocker.

After independent parent work, call `wait_agent` once with `timeout_ms = 3_600_000` when supported, otherwise the maximum. Treat it as an event-driven upper bound: for a nonterminal non-escalation, neither answer nor call `list_agents`; resume the same wait. Update users at phase boundaries.

## Canonical commands

```bash
python $MANAGE candidate-fingerprint $TASK
python $MANAGE run-check $TASK <preflight|test|seal> <check-id>
python $MANAGE run-test-probe $TASK <AT-NNN> --authority <kind> --authority-ref "<source>" --argv <exact argv...>
python $MANAGE artifact-check $TASK <artifact>
python $MANAGE transition $TASK <planning|implementation|validation>
python $MANAGE preflight-verdict $TASK <PASS|RETURN>
python $MANAGE test-verdict $TASK <PASS|RETURN>
python $MANAGE seal-verdict $TASK <PASS|RETURN>
python $MANAGE return-review $TASK <implementation|exploration> --reason "<classification>"
python $MANAGE packaging-repair $TASK --reason "<content defect requiring repair>"
python $MANAGE block $TASK <constraint_conflict|external_dependency|missing_authority|environment> --reason "<blocker>"
python $MANAGE verdict $TASK <PASS|RETURN>
python $MANAGE capture-pr $TASK --pr-number <n>
python $MANAGE finalize-pr $TASK --pr-number <n> --head-sha <40-char-sha>
python $MANAGE check $TASK
```

Mutations enforce artifact checks; manual `artifact-check` is early feedback, never a bypass.

## Boundaries

- Implementer → Preflight Reviewer → Test Writer → Seal Reviewer is strictly sequential. Implementers never write `preflight.md`/`seal.md`; Reviewers own those artifacts, the parent owns verdict transitions.
- The first Preflight may run only the bounded diagnostic categories frozen in `plan.md` from the Issue ACs, planned risks, and `checks.json` authority. A closure Preflight reads the prior findings and verifies only those findings, canonical checks, and repair-local regressions; it must not invent a new mutation category or restart full exploratory review.
- A second consecutive Preflight RETURN and every Seal RETURN stop the ordinary repair loop. The parent must call `return-review`, classify implementation versus exploration, and update `plan.md`/`checks.json` when the return exposed a coverage-design gap.
- Seal verifies post-test identity and completeness. It does not fuzz, design new semantic mutations, repeat Preflight exploration, or pre-empt the Issue-only Validator.
- Test Writer edits make final sealing mandatory. Content changes after Tester PASS invalidate the seal; post-seal changes invalidate validation and packaging.
- Tester freedom applies to test perspectives and inputs, not to inventing product requirements. An `AT-*` RETURN must use `ISSUE_CONTRACT`, `PUBLIC_CONTRACT`, `REPO_INVARIANT`, or `BASELINE_REGRESSION` authority and machine-recorded evidence. Ambiguous expectations are missing authority, not defects.
- Tester PASS requires every planned baseline check and every recorded adversarial test to PASS. Tester RETURN may rely on an AC failure or a reproducible `AT-*` failure even when every AC row passes; an unrecorded, irreproducible, unsupported, or unrelated observation alone is not RETURN.
- After Tester RETURN, the next Test Writer closes the frozen findings and may add only repair-local adversarial coverage. A second consecutive RETURN requires `return-review` to classify implementation versus exploration before continuing.
- Single-Implementer execution does not weaken evidence gates. Do not open/update the delivery PR before Validator PASS unless the user explicitly requests an earlier draft; completion still requires final-head binding.
- Replace authoritative artifacts in place; never create `*-v2.md`.
- Progress messages never replace artifacts, terminal handoffs, or transitions; never accept a verdict from `fork_turns` other than `"none"`.
