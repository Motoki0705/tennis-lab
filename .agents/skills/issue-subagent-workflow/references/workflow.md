# Workflow contract

## Preconditions

The GitHub Issue body must contain a concrete Markdown task-list under `## Acceptance checklist`. The initializer normalizes each item to an ordered AC ID and fails closed when the checklist is absent, empty, or duplicated.

## State machine

```text
parallel scouting -> exploration -> planning -> parallel implementation -> independent testing
                                                   ^                           |
                                                   |------ tester RETURN ------|
                                                                               |
                                                                tester PASS -> validation
                                                                               |
                                                            validator PASS -> complete
                                                            validator RETURN -> exploration
```

Scouting is optional only when the Issue truly has no useful independent lookup questions. Before exploration and before every implementation or repair cycle, the parent must actively test whether the work can be decomposed. When at least two independent units exist, multiple agents are the default.

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

## Proactive delegation waves

A delegation wave is a set of agents that may run concurrently because their questions or ownership do not overlap.

Before each wave, the parent records:

- the independent question or production unit;
- its assigned agent and unique task name;
- explicit file/module ownership when code may change;
- the expected evidence or handoff;
- dependencies on earlier or later waves.

Rules:

- Prefer several narrow, independent assignments over one broad assignment.
- Spawn at least two agents by default when two or more independent units exist.
- Do not use concurrency for overlapping files, duplicate questions, or shared artifact writes.
- Join all agents in a wave before consuming their combined result or advancing the state.
- The concurrency limit is available capacity, not a target.
- When decomposable work is assigned to a single Scout or Implementer, the parent records why parallel delegation would be unsafe, redundant, sequentially blocked, or artifact-conflicting.

One authoritative Explorer, Test Writer, and Validator owns each formal artifact per cycle. Parallelism around those roles is achieved through Scout waves, Implementer waves, and bounded Validator child Explorers.

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

Use `codebase_scout` for bounded, high-volume lookup work such as locating named symbols, direct references, nearby tests, configuration keys, candidate entry points, stale aliases, or independent AC evidence. Actively partition cross-cutting Issues by subsystem, execution stage, configuration domain, or evidence question, and run the resulting Scouts concurrently.

Use one authoritative `codebase_explorer` after the Scout wave when the Issue spans packages or execution stages, depends on registries or dynamic configuration, changes schemas or public contracts, deletes or moves code broadly, follows validator RETURN, or when Scout evidence is ambiguous.

Formal exploration is mandatory before planning. The Explorer independently verifies the joined Scout leads, and the parent independently verifies high-impact Explorer claims.

## Responsibilities

### Parent orchestrator

- Freeze the Issue and reject unusable checklists.
- Preserve the full `/goal` objective through tester and validator RETURN cycles.
- Before exploration and implementation, create a delegation map and actively identify safe parallel work.
- Spawn multiple Scouts for independent questions and multiple Implementers for disjoint production ownership by default.
- Own `plan.md`, decomposition, AC mapping, and the file ownership matrix.
- Join every agent in the current wave before starting dependent work.
- Designate one implementation artifact integrator and prevent concurrent writes to `implementation.md`.
- Join all Implementer work before spawning the Test Writer.
- Never run Implementer and Test Writer concurrently.
- On tester RETURN, partition independent failures into disjoint repair units where possible and pass only concrete failing tests, observed behavior, affected AC IDs, and authorized production ownership back to the retry Implementers.
- Enforce the Validator context firewall.
- Apply all transitions and verdicts through the state helper.

### Scout

- Answer one bounded question with direct repository evidence.
- Do not broaden into a second agent's assigned question.
- Report ambiguity, likely impact expansion, and the need for formal exploration.
- Do not modify code, GitHub, or formal workflow artifacts.

### Explorer

- Own the single formal `exploration.md` for the current attempt.
- Join and independently verify Scout evidence rather than concatenate summaries.
- Trace entry points, execution paths, contracts, tests, risks, and unresolved questions across the full Issue scope.

### Implementer

- Modify only explicitly owned production files.
- Do not modify independently authored tests.
- Do not touch another concurrent Implementer's ownership.
- On an initial cycle, read Issue, exploration, and plan.
- On tester RETURN, additionally read the current `tests.md` failure evidence or the equivalent focused failure bundle supplied by the parent.
- Report implementation evidence to the designated artifact integrator; only the integrator replaces `implementation.md`.

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
- Proactively split independent AC evidence questions among bounded child Explorers when useful.
- Join child evidence, independently verify load-bearing claims, and retain sole ownership of the final verdict.
- Emit exactly one ordered verdict row per AC item and final PASS or RETURN.

## Context firewall

The Validator still receives system instructions, repository guidance, tool definitions, repository state, and its output path. “Issue only” means `issue.md` is its only task specification and workflow narrative. Spawn it without parent history and never supply expected conclusions.

Validator child Explorers receive only the relevant AC IDs and bounded evidence question. They do not receive the plan, implementation narrative, tester conclusion, or expected verdict.

## RETURN discipline

Tester RETURN:

1. Run `test-verdict ... RETURN`.
2. Keep `phase = "implementation"` and the same Issue attempt.
3. Increment the next document test-cycle number.
4. Decompose independent failures into disjoint production repair units.
5. Spawn one or more retry Implementers, using multiple agents by default when two or more safe units exist.
6. Join repairs through one artifact integrator.
7. Re-run the single independent Test Writer.
8. Enter validation only after `test-verdict ... PASS`.

Validator RETURN:

1. Run `verdict ... RETURN`.
2. Increment the Issue attempt and reset tester state.
3. Keep an active `/goal` active.
4. Create a new exploration delegation map from the Validator’s unresolved questions and affected AC IDs.
5. Spawn multiple bounded Scouts when those questions are independent.
6. Re-run the authoritative Explorer with the joined evidence.
7. Rewrite the existing plan, then repeat parallel implementation, testing, and validation.
