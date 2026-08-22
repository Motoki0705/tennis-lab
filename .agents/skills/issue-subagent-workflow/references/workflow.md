# Normative workflow

This is the sole normative state-machine prose; scripts win on disagreement.

## State machine

```text
feasibility -- BLOCKED ----------------------------------------------> blocked
     |
     PASS
     v
scouting -> exploration -> planning -> implementation
                                      ^        |
                                      |        v
                                      |  discovery Preflight Reviewer
                                      |     | RETURN #1 -> bounded repair/closure review
                                      |     | RETURN #2 -> return-review
                                      |     ` PASS
                                      |        v
                                      |  independent Test Writer
                                      |     | RETURN #1 -> repair
                                      |     | RETURN #2 -> return-review
                                      |     ` PASS
                                      |        v
                                      |  bounded Seal Reviewer
                                      |     | RETURN -> return-review
                                      |     ` PASS
                                      |        v
                                      +---- validation
                                             | RETURN -> next attempt/exploration
                                             ` PASS
                                                v
                                      packaging / status=validated
                                             | PR head or checks fail -> remain validated
                                             ` finalize-pr PASS
                                                v
                                             complete
```

Preflight RETURN spends no Tester cycle. The first Preflight RETURN permits one bounded repair and one closure Reviewer; a second consecutive RETURN sets `return_review_required`. Test Writer RETURN increments `test_cycle`; return #2 also requires `return-review`. Seal is a post-test identity/evidence gate rather than another semantic exploration phase, so any Seal RETURN requires `return-review`; every content repair then requires fresh Preflight/Test. Validator RETURN increments `attempt`, clears candidate evidence, and returns to formal exploration.

`return-review` is the explicit strategy gate for repeated or late failures. The parent classifies `implementation` when the scope is understood and frozen checks/plan can be repaired, or `exploration` when the finding changes the semantic model or reveals an unknown impact radius. When the failure exposed a coverage-design gap, update `plan.md` and `checks.json` before continuing instead of launching another unchanged Reviewer.

## Frozen specification and identity

Schema-v5 initialization writes canonical raw `issue.json`, deterministic `issue.md`, and `state.toml` containing hashes, phase, verdicts, candidate bindings, and PR binding. Every mutation revalidates payload hash, exact rendering, Issue number/URL, and normalized checklist hash; editing `issue.md` alone fails.

`base_revision` is frozen. The candidate fingerprint content-hashes every changed/untracked path relative to it except `.codex/tasks/`, so history-only recommits/squashes may retain identity. Separate fields bind `preflight_candidate_sha256`, `test_candidate_sha256`, `sealed_candidate_sha256`, `validation_candidate_sha256`, and `packaging_candidate_sha256`.

Preflight may precede test edits; Tester PASS therefore binds the post-test candidate. Seal PASS must equal it and cover complete canonical checks; validation/packaging must equal the seal. Stale artifacts or machine results fail.

## Canonical checks

`02-planning/checks.json` is command authority. Each unique ID fixes `argv`, repository-relative `cwd`, environment additions, authorized `preflight`/`test`/`seal` stages, required/optional status, and AC authority. Only `manage_issue_task.py run-check` may execute it; generated stage JSON/raw logs bind invocation digest, candidate fingerprint, exit code, and verdict. Changed argv/cwd/environment/candidate or a missing required check invalidates the stage.

The first Preflight may supplement these commands only with bounded diagnostic categories explicitly frozen in `plan.md` from the ACs and planned risks. The Reviewer does not create new categories during execution. After RETURN, the next Reviewer is closure-only: the prior findings, canonical checks, and direct repair regressions are the frozen scope. Seal never creates a new mutation/fuzzing program; semantic acceptance remains owned by the frozen checks, Test Writer evidence, and final Issue-only Validator.

## Artifact tree

```text
.codex/tasks/issue-<number>/
├── issue.json
├── issue.md
├── state.toml
├── 00-feasibility/feasibility.md
├── 01-exploration/exploration.md
├── 02-planning/{plan.md,checks.json}
├── 03-implementation/
│   ├── implementation.md
│   ├── preflight.md
│   ├── preflight-checks.json
│   ├── tests.md
│   ├── test-checks.json
│   ├── seal.md
│   └── seal-checks.json
├── 04-validation/validation.md
├── 05-packaging/{pr-evidence.json,packaging.md}
└── logs/
```

One logical artifact has one replace-in-place path. Raw logs attach evidence; they are not verdicts.

## Topology and ownership

Parallelism is a latency optimization, not an acceptance criterion. `plan.md` records the user's topology; one sequential Implementer never weakens scope, checks, sealing, Validator independence, or PR-head binding. Default Implementers return handoffs. Only the parent or one named implementation integrator writes `implementation.md`; independent Preflight/Seal Reviewers own `preflight.md`/`seal.md`. A sole Implementer may be named integrator.

## Automatic gates

| Mutation | Validated evidence |
|---|---|
| feasibility verdict | feasibility |
| planning transition | exploration |
| implementation transition | plan + manifest |
| preflight verdict | implementation + preflight + results + candidate; second consecutive RETURN forces return-review |
| test verdict | tests + results + candidate; second RETURN forces return-review |
| seal verdict | seal + results + Tester-candidate equality; every RETURN forces return-review |
| validation transition | all implementation artifacts + sealed candidate |
| Validator verdict | complete artifact set + sealed candidate |
| `capture-pr` | real metadata + all paginated files + final head + status rollup |
| `finalize-pr` | evidence digest + packaging + local HEAD + revision content + remote checks |

These checks run inside mutations; manual `artifact-check` only provides earlier feedback.

## Delegation and retry discipline

- Prefer deterministic scripts for mechanical inventories.
- Every `spawn_agent` call uses `fork_turns = "none"` exactly. Numeric/inherited turns invalidate the handoff and require respawn; for Preflight Reviewers, Test Writers, Seal Reviewers, and Validators they also break independence.
- Supply frozen artifacts, paths, AC IDs, ownership, and focused failures; append the exact `Communication mode: terminal-only.` footer from `spawn-contracts.md`.
- A discovery Preflight gets the complete approved scope. A closure Preflight gets the previous findings as its frozen worklist and does not broaden mutation categories. A Seal assignment explicitly excludes open-ended semantic exploration.
- Children produce one terminal handoff; earlier contact is limited to missing authority, ownership collision, or an unresolvable assigned-scope blocker.
- Do parent work, then event-driven `wait_agent` with `timeout_ms = 3_600_000` or the maximum. Only `FINAL_ANSWER` completes; resume the long wait after other non-escalations.
- Do not pair waiting with repeated `list_agents`, short-timeout polling, or status-request `send_message`.
- Join once; keep raw output in child threads/`logs/`. Handoffs contain terminal status, changed files/evidence, exact command IDs/outcomes, and risks; user updates occur at phase boundaries.
- After `return-review` or Validator RETURN, use fresh bounded sessions. Retries contain only affected AC IDs, frozen findings, authorized ownership, artifact paths, and the mandatory footer.
