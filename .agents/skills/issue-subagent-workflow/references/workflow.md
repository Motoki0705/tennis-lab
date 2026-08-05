# Workflow contract

## Preconditions

The GitHub Issue body must contain a concrete Markdown task-list under `## Acceptance checklist`. The initializer normalizes each item to an ordered AC ID and fails closed when the checklist is absent, empty, or duplicated.

## State machine

```text
optional scouting -> exploration -> planning -> implementation -> independent testing
                                              ^                         |
                                              |---- tester RETURN ------|
                                                                        |
                                                         tester PASS -> validation
                                                                        |
                                                     validator PASS -> complete
                                                     validator RETURN -> exploration
```

The persisted phase remains `implementation` during the Implementer–Tester loop. `state.toml.test_cycle` counts completed independent test evaluations, and `state.toml.test_verdict` stores the latest tester verdict. Tester RETURN does not increment the Issue attempt because the codebase model and plan remain in force. Validator RETURN increments the attempt and restarts formal exploration because it may expose a flawed model or plan.

An optional active `/goal` surrounds this state machine. It keeps the parent working across turns but does not replace phase transitions.

## Artifact tree

```text
.codex/tasks/issue-<number>/
├── issue.md
├── state.toml
├── 01-exploration/exploration.md
├── 02-planning/plan.md
├── 03-implementation/implementation.md
├── 03-implementation/tests.md
└── 04-validation/validation.md
```

One logical artifact has one path. Replace files in place. Do not create attempt-, date-, final-, revised-, or v2-suffixed copies.

## Acceptance and test gates

- `issue.md` and `state.toml` contain the same normalized checklist hash and count.
- `plan.md` contains one exact, ordered mapping row for every AC item.
- `tests.md` contains one exact, ordered mapping row for every AC item.
- An AC ID mentioned only in prose does not satisfy either mapping contract.
- The Test Writer emits a standalone `PASS` or `RETURN` and records the current test cycle.
- Tester PASS means the independent test work and relevant commands succeeded; requirements requiring non-test evidence remain explicitly identified for the Validator.
- Tester RETURN must contain actionable implementation findings and routes to the Implementer without entering validation.
- The Validator independently emits exactly one ordered PASS/FAIL/NOT VERIFIED row for every AC item.
- Source GitHub checkbox state is never implementation evidence.

## Exploration routing

Use `codebase_scout` only for bounded, high-volume lookup work such as locating named symbols, direct references, nearby tests, configuration keys, or candidate entry points. Multiple Scouts may run in parallel when their questions are independent.

Use `codebase_explorer` directly when the Issue spans packages or execution stages, depends on registries or dynamic configuration, changes schemas or public contracts, deletes or moves code broadly, follows validator RETURN, or when Scout evidence is ambiguous.

Formal exploration is mandatory before planning. The Explorer independently verifies Scout leads, and the parent independently verifies high-impact Explorer claims.

## Responsibilities

### Parent orchestrator

- Freeze the Issue and reject unusable checklists.
- Preserve the full `/goal` objective through tester and validator RETURN cycles.
- Own `plan.md`, decomposition, AC mapping, and file ownership.
- Join all Implementer work before spawning the Test Writer.
- Never run Implementer and Test Writer concurrently.
- On tester RETURN, pass only the concrete failing tests, observed behavior, affected AC IDs, and authorized production ownership back to the Implementer.
- Enforce the Validator context firewall.
- Apply all transitions and verdicts through the state helper.

### Implementer

- Modify only explicitly owned production files.
- Do not modify independently authored tests.
- On an initial cycle, read Issue, exploration, and plan.
- On tester RETURN, additionally read the current `tests.md` failure evidence or the equivalent focused failure bundle supplied by the parent.
- Replace `implementation.md` with the current attempt and next test-cycle number.

### Test writer

- Run only after integrated implementation is available.
- Derive tests independently from Issue, plan, public behavior, and current code or diff.
- Do not read `implementation.md`.
- Never repair production code or weaken tests to obtain PASS.
- On production failure, emit RETURN with exact commands, failures, affected AC IDs, and required observable behavior.
- Replace `tests.md` with the current test-cycle number and final tester verdict.

### Validator

- Treat `issue.md` as the sole task specification.
- Do not read plan, implementation, tests artifact, prior validation, or Issue comments.
- Inspect repository state and tests directly.
- Emit exactly one ordered verdict row per AC item and final PASS or RETURN.

## Context firewall

The Validator still receives system instructions, repository guidance, tool definitions, repository state, and its output path. “Issue only” means `issue.md` is its only task specification and workflow narrative. Spawn it without parent history and never supply expected conclusions.

## RETURN discipline

Tester RETURN:

1. Run `test-verdict ... RETURN`.
2. Keep `phase = "implementation"` and the same Issue attempt.
3. Increment the next document test-cycle number.
4. Re-run Implementer on production ownership only.
5. Re-run the independent Test Writer.
6. Enter validation only after `test-verdict ... PASS`.

Validator RETURN:

1. Run `verdict ... RETURN`.
2. Increment the Issue attempt and reset tester state.
3. Keep an active `/goal` active.
4. Re-run formal Explorer with the Validator’s unresolved questions and affected AC IDs.
5. Rewrite the existing plan, then repeat implementation, testing, and validation.
