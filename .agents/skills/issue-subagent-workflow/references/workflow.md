# Workflow contract

## Preconditions

The GitHub Issue body must contain a concrete Markdown task-list under `## Acceptance checklist`. The initializer normalizes each item to an ordered AC ID and fails closed when the checklist is absent, empty, or duplicated.

## State machine

```text
feasibility -- BLOCKED --------------------------------------------> blocked
     |
     PASS
     v
parallel scouting -> exploration -> planning -> parallel implementation
                                                  ^          |
                                                  | preflight RETURN
                                                  |          v
                                                  +------ preflight
                                                             |
                                                             PASS
                                                             v
                                                    independent testing
                                                  ^          |
                                                  | tester RETURN #1
                                                  |          |
                                                  +----------+
                                                             |
                                          tester RETURN #2 -> return review
                                                             | implementation
                                                             | exploration
                                                             ` block

independent testing -- PASS -> validation
validation -- PASS -> complete
validation -- RETURN -> exploration with the next attempt
```

`preflight RETURN` is an integration failure and does not increment `test_cycle`. Tester RETURN increments `test_cycle`. Two Tester RETURNs since the previous explicit review set `return_review_required = true`; another preflight or test verdict is rejected until the parent classifies the loop.

A task can enter `blocked` from any in-progress phase when an Issue constraint, missing authority, external dependency, or environment condition prevents valid progress. A blocked task is neither failed nor complete. The parent must not keep an active `/goal` auto-continuing against an unresolved blocker.

## State semantics

Schema version 4 records:

- frozen Issue and checklist identity;
- `feasibility_verdict`;
- current attempt and phase;
- `preflight_cycle` and `preflight_verdict`;
- completed independent `test_cycle` and `test_verdict`;
- `test_return_count` and mandatory return-review state;
- final status, verdict, and blocker details.

Schema version 3 tasks are normalized in memory to version 4 with `feasibility_verdict = "LEGACY"`. They do not retroactively repeat feasibility, but an in-progress implementation must use the new preflight gate before its next Tester verdict. New tasks always start in `feasibility`.

## Artifact tree

```text
.codex/tasks/issue-<number>/
├── issue.md
├── state.toml
├── 00-feasibility/feasibility.md
├── 01-exploration/exploration.md
├── 02-planning/plan.md
├── 03-implementation/implementation.md
├── 03-implementation/preflight.md
├── 03-implementation/tests.md
├── 04-validation/validation.md
└── logs/                              # optional, non-authoritative raw output
```

One logical artifact has one path. Replace files in place. Do not create attempt-, date-, final-, revised-, or v2-suffixed copies. Raw logs are not verdict evidence by themselves; formal artifacts cite the command, exit status, summary, and log path.

## Feasibility gate

The parent owns `feasibility.md`. Before broad exploration or implementation it must establish:

- exact allowed and prohibited write scopes;
- whether the Issue requires a breaking or compatibility-preserving change;
- which existing tests and required checks encode the current contract;
- baseline failures that predate the work;
- whether each AC can be satisfied inside the allowed scope;
- whether the required checks can pass without contradicting another Issue requirement.

A breaking change combined with immutable tests that require the removed behavior is a constraint conflict, not an implementation task. The correct outcome is `BLOCKED` until the Issue grants a coherent exception or changes its completion criteria.

The feasibility matrix contains one exact ordered row per AC item with `FEASIBLE`, `BLOCKED`, or `UNKNOWN`. PASS requires every row to be FEASIBLE. BLOCKED requires at least one BLOCKED or UNKNOWN row plus concrete conflict and resolution sections.

## Proactive delegation waves

Before each wave, the parent records the question or production unit, unique task name, ownership, expected handoff, and dependencies.

- Prefer narrow independent assignments over a broad duplicate assignment.
- Use multiple agents only when this reduces the critical path or creates independent evidence.
- Do not delegate a deterministic complete scan when one repository command or script can produce the result.
- Join every agent in a wave before consuming combined results.
- Use one long or event-driven join where available. Repeated short waits with no state change are prohibited.
- The configured concurrency limit is capacity, not a target.

One authoritative Explorer, Test Writer, and Validator owns each formal artifact per cycle. Parallelism around those roles comes from Scout waves, Implementer waves, and bounded Validator child Explorers.

## Planning and ownership

The parent owns `plan.md`, exact AC mapping, the ownership matrix, and the validation strategy. The plan must name:

- deterministic policy checks;
- focused behavior checks per work unit;
- the canonical repository required-check command;
- any baseline failure that must be distinguished from a new regression;
- the test paths the independent Test Writer may change, including any Issue prohibition.

Production ownership must be non-overlapping within a wave. Only the designated integrator replaces `implementation.md` and `preflight.md`.

## Deterministic preflight

Preflight is an integration gate, not an independent acceptance verdict. Run it after all Implementers join and before the Test Writer.

Run checks in fail-fast order:

1. changed-file and prohibited-scope inspection;
2. deterministic source-policy checks such as AST inventories, stale keys, aliases, or forbidden defaults;
3. focused tests and smoke commands for changed units;
4. compose, schema, non-CWD, persistence, or mutation checks required by the Issue;
5. lint and type checking for the changed scope;
6. the canonical required-check command once the focused checks pass.

On failure, stop the remaining expensive work when its result cannot change the verdict, write actionable production findings, record `preflight RETURN`, and return to Implementers. Do not spend a Test Writer cycle finding a failure that a deterministic command already proves.

## Acceptance and test gates

- `issue.md` and `state.toml` contain the same normalized checklist hash and count.
- `feasibility.md`, `plan.md`, `tests.md`, and `validation.md` preserve the exact ordered AC rows required by their contracts.
- A Test Writer verdict is accepted only when the same test-cycle number has a recorded preflight PASS.
- Tester PASS means relevant independent tests and commands succeeded with no known in-scope production failure.
- Tester RETURN contains exact commands, observed behavior, affected AC IDs, and actionable production findings.
- Validator PASS requires every AC row to be PASS.
- Source GitHub checkbox state and agent narratives are never implementation evidence.

## Context and token discipline

Subagents consume additional model work, so parallelism must buy latency reduction or independent evidence.

- Keep requirements, decisions, and verdicts in the parent thread; keep exploration notes, stack traces, and raw command output in child threads or log files.
- Child handoffs are compact: status, changed files or evidence, commands and outcomes, unresolved risks, and artifact/log paths.
- Retry messages contain the delta: affected AC IDs, exact new failures, and authorized ownership. Agents read the authoritative artifacts instead of receiving duplicated narrative.
- Do not send progress messages to running agents unless constraints or evidence changed.
- Do not repeatedly rerun the full suite after a known focused failure. Fix the failure, rerun its focused reproducer, then advance through preflight.

## Responsibilities

### Parent orchestrator

- Freeze the Issue and run the feasibility gate.
- Block unsatisfiable work instead of inventing compatibility or looping.
- Own delegation maps, `plan.md`, ownership, canonical checks, and return review.
- Prefer deterministic tools for mechanical inventories.
- Join waves without polling churn and keep the parent context concise.
- Enforce preflight before Test Writer, Validator context isolation, and all state-helper transitions.

### Scout

- Answer one bounded semantic repository question with direct evidence.
- Do not modify code, GitHub, or formal artifacts.
- Do not paste raw broad-search output when a compact candidate list is sufficient.

### Explorer

- Own the single `exploration.md` for the attempt.
- Independently verify Scout leads and trace real execution paths, contracts, tests, risks, and unresolved questions.

### Implementer

- Modify only assigned production files and allowed test files, if any.
- Do not weaken or silently rewrite independent tests.
- Run focused checks and return a compact handoff to the integrator.

### Artifact integrator

- Join all implementation work.
- Replace `implementation.md` and `preflight.md`.
- Run deterministic preflight and return failures directly to Implementers.
- Do not declare acceptance PASS.

### Test writer

- Run only after preflight PASS.
- Derive expected behavior from Issue, plan, public behavior, current code, and diff; do not read `implementation.md`.
- Do not repair production code or weaken a valid failing test.
- Emit one ordered AC mapping and a standalone PASS or RETURN.

### Validator

- Treat `issue.md` as the sole task specification.
- Do not read plan, implementation, preflight, tests, prior validation, or Issue comments.
- Inspect repository state and tests directly, using bounded child Explorers only for independent evidence collection.

## RETURN discipline

### Preflight RETURN

1. Run `preflight-verdict ... RETURN`.
2. Keep the same attempt and `test_cycle`.
3. Partition production failures into disjoint repairs.
4. Join repairs, overwrite implementation and preflight for the same next test-cycle number, and rerun focused checks.
5. Spawn the Test Writer only after preflight PASS.

### Tester RETURN

1. Run `test-verdict ... RETURN`.
2. Keep the same attempt and implementation phase.
3. Repair the exact independent findings and repeat preflight.
4. After the second Tester RETURN since the previous review, classify with `return-review` before any further preflight:
   - `implementation`: only for bounded, independent omissions under the valid plan;
   - `exploration`: when the model, scope, or plan is incomplete;
   - `block`: when constraints or authority make the task unsatisfiable.

### Validator RETURN

1. Run `verdict ... RETURN`.
2. Increment the attempt and reset preflight/test state.
3. Keep an active goal active.
4. Rebuild the exploration delegation map from the Validator's unresolved AC questions.
5. Rewrite the same artifacts and repeat the gated workflow.
